"""Prepare a fixed, offline execution packet from verified public inputs."""

from __future__ import annotations

import argparse
import csv
import io
import json
import random
import re
from pathlib import Path
from typing import Any

from fetch import verify_inputs, write_exclusive
from protocol import (
    M_POLICY,
    RULES,
    STAGES,
    digest,
    positive,
    protocol_digest,
    requests,
    require,
    sha256,
    source_digest,
    sources,
)


def unit(stage: str, cohort: str, original_id: str, data: dict[str, Any]) -> dict[str, Any]:
    return {
        "stage": stage,
        "cohort_sha256": cohort,
        "original_id": original_id,
        "base_id": digest({"stage": stage, "cohort_sha256": cohort, "original_id": original_id}),
        "input_digest": digest(data),
        "data": data,
    }


def populations(root: Path) -> dict[str, list[dict[str, Any]]]:
    files = verify_inputs(root)  # all bytes checked before parsing or sampling
    result: dict[str, list[dict[str, Any]]] = {stage: [] for stage in STAGES}
    body = files["G/toxicchat.csv"]
    for index, row in enumerate(csv.DictReader(io.StringIO(body.decode("utf-8"), newline=""))):
        if row["human_annotation"] == "True":
            result["G"].append(unit("G", sha256(body), f"toxicchat:{index}", {"text": row["user_input"]}))
    body = files["G/aegis.json"]
    seen = set()
    for index, row in enumerate(json.loads(body)):
        text = str(row["prompt"])
        if row["prompt_label_source"] == "human" and text != "REDACTED" and text not in seen:
            seen.add(text)
            result["G"].append(unit("G", sha256(body), f"aegis:{index}", {"text": text}))
    body = files["M/gretel-main.jsonl"]
    for row in (json.loads(line) for line in body.splitlines()):
        result["M"].append(unit("M", sha256(body), str(row["uid"]), {"text": row["text"]}))
    body = files["E/questions.json"]
    for row in json.loads(body)["questions"]:
        result["E"].append(unit("E", sha256(body), str(row["id"]), {"text": row["question"]}))
    body = files["R/cases_test.json"]
    for row in json.loads(body)["cases"]:
        candidates = [{"id": c["id"], "text": c["text"]} for c in row["candidates"]]
        require(len(candidates) == 20 and len({c["id"] for c in candidates}) == 20, "Expected 20 distinct candidates")
        result["R"].append(unit("R", sha256(body), str(row["id"]), {"query": row["query"], "candidates": candidates}))
    for stage, count in {"G": 4768, "M": 660, "E": 440, "R": 499}.items():
        require(
            len(result[stage]) == count and len({u["base_id"] for u in result[stage]}) == count,
            "Incomplete eligible population",
        )
    return result


def validate_config(config: dict[str, Any]) -> None:
    require(type(config["n"]) is int and config["n"] > 0, "n must be a positive base-unit count")
    require(type(config["warmup"]) is int and config["warmup"] >= 0, "warmup must be nonnegative")
    require(type(config["seed"]) is int, "Declare an integer seed")
    require(
        config["phase"] in ("pilot", "confirmatory") and config["arms"] in ("sie", "paired"),
        "Unknown phase or arm mode",
    )
    require(
        bool(config["stages"])
        and len(set(config["stages"])) == len(config["stages"])
        and set(config["stages"]) <= set(STAGES),
        "Invalid stage order",
    )
    require(
        all(positive(config[k]) for k in ("request_timeout_s", "wall_limit_s")), "Timeouts must be finite and positive"
    )
    require(
        bool(re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", config["placement_label"])),
        "Use a nonsecret placement label of 1-64 letters/digits/._-",
    )
    bootstrap = config["bootstrap"]
    require(
        type(bootstrap["seed"]) is int and type(bootstrap["resamples"]) is int and bootstrap["resamples"] >= 100,
        "Invalid bootstrap seed/count",
    )
    require(
        bootstrap["quantile"] == "linear-(n-1)p" and bootstrap["minimum_clusters"] == 2, "Unknown statistical protocol"
    )


def observation(selected: dict[str, Any], phase: str, rule: str | None, arms: list[str]) -> dict[str, Any]:
    identity = {"base_id": selected["base_id"], "phase": phase, "rule": rule, "repetition": 0}
    return {
        **identity,
        "observation_id": digest(identity),
        "stage": selected["stage"],
        "unit": selected,
        "arm_order": arms,
        "requests": {arm: requests(selected["stage"], selected["data"], arm, rule) for arm in arms},
    }


def build_packet(
    population: dict[str, list[dict[str, Any]]], config: dict[str, Any], prior: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    validate_config(config)
    prior = prior or []
    excluded_ids = {o["base_id"] for p in prior for o in p["observations"]}
    excluded_inputs = {(o["stage"], o["unit"]["input_digest"]) for p in prior for o in p["observations"]}
    picked: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for stage in config["stages"]:
        eligible = sorted(population[stage], key=lambda u: u["base_id"])
        random.Random(digest({"seed": config["seed"], "stage": stage})).shuffle(eligible)
        chosen = []
        for candidate in eligible:
            key = (stage, candidate["input_digest"])
            if candidate["base_id"] not in excluded_ids and key not in excluded_inputs:
                chosen.append(candidate)
                excluded_ids.add(candidate["base_id"])
                excluded_inputs.add(key)
                if len(chosen) == config["n"] + config["warmup"]:
                    break
        require(len(chosen) == config["n"] + config["warmup"], "n exceeds the remaining distinct eligible population")
        picked[("warmup", stage)] = chosen[: config["warmup"]]
        picked[(config["phase"], stage)] = chosen[config["warmup"] :]
    observations = []
    for phase in ("warmup", config["phase"]):
        for stage in config["stages"]:
            for selected in picked[(phase, stage)]:
                for rule in RULES if stage == "R" else [None]:
                    arms = ["sie", "rival"] if config["arms"] == "paired" else ["sie"]
                    random.Random(
                        digest({"seed": config["seed"], "base_id": selected["base_id"], "rule": rule})
                    ).shuffle(arms)
                    observations.append(observation(selected, phase, rule, arms))
    calls = {
        stage: {
            arm: sum(len(o["requests"].get(arm, [])) for o in observations if o["stage"] == stage)
            for arm in (["sie", "rival"] if config["arms"] == "paired" else ["sie"])
        }
        for stage in config["stages"]
    }
    content = {
        "schema": 1,
        "source_digest": source_digest(),
        "protocol_digest": protocol_digest(),
        "config": config,
        "m_policy": M_POLICY,
        "excluded_packet_digests": sorted(p["packet_digest"] for p in prior),
        "planned_semantic_calls": calls,
        "observations": observations,
    }
    return {**content, "packet_digest": digest(content)}


def load_packet(path: Path) -> dict[str, Any]:
    packet = json.loads(path.read_bytes())
    content = {k: v for k, v in packet.items() if k != "packet_digest"}
    require(packet["packet_digest"] == digest(content), "Packet digest differs")
    require(
        packet["schema"] == 1
        and packet["source_digest"] == source_digest()
        and packet["protocol_digest"] == protocol_digest(),
        "Unknown source or protocol binding",
    )
    config = packet["config"]
    validate_config(config)
    require(packet["m_policy"] == M_POLICY, "Composition policy differs")
    seen_ids, seen_inputs, seen_observations = {}, {}, set()
    cohorts = {
        stage: {
            e["sha256"]
            for e in sources()["files"]
            if e["path"].startswith(stage + "/") and not e["path"].endswith("manifest.json")
        }
        for stage in STAGES
    }
    for obs in packet["observations"]:
        selected = obs["unit"]
        require(
            selected == unit(obs["stage"], selected["cohort_sha256"], selected["original_id"], selected["data"]),
            "Unit identity differs",
        )
        require(selected["cohort_sha256"] in cohorts[obs["stage"]], "Unknown input cohort")
        require(
            obs["stage"] in config["stages"] and obs["phase"] in ("warmup", config["phase"]),
            "Unknown observation phase/stage",
        )
        require(obs["base_id"] == selected["base_id"] and obs["repetition"] == 0, "Observation identity differs")
        for key, seen in ((obs["base_id"], seen_ids), ((obs["stage"], selected["input_digest"]), seen_inputs)):
            require(
                key not in seen or seen[key] == (obs["base_id"], obs["phase"]),
                "Base/input overlaps phases or identities",
            )
            seen[key] = (obs["base_id"], obs["phase"])
        require(obs["observation_id"] not in seen_observations, "Duplicate observation")
        seen_observations.add(obs["observation_id"])
        require(
            obs == observation(selected, obs["phase"], obs["rule"], obs["arm_order"]),
            "Request envelope or observation digest differs",
        )
        arms = ["sie", "rival"] if config["arms"] == "paired" else ["sie"]
        random.Random(digest({"seed": config["seed"], "base_id": selected["base_id"], "rule": obs["rule"]})).shuffle(
            arms
        )
        require(obs["arm_order"] == arms, "Frozen arm order differs")
    for phase, count in (("warmup", config["warmup"]), (config["phase"], config["n"])):
        for stage in config["stages"]:
            rows = [o for o in packet["observations"] if o["phase"] == phase and o["stage"] == stage]
            require(
                len({o["base_id"] for o in rows}) == count and len(rows) == count * (2 if stage == "R" else 1),
                "Fixed phase population differs",
            )
    calls = {
        stage: {
            arm: sum(len(o["requests"].get(arm, [])) for o in packet["observations"] if o["stage"] == stage)
            for arm in (["sie", "rival"] if config["arms"] == "paired" else ["sie"])
        }
        for stage in config["stages"]
    }
    require(packet["planned_semantic_calls"] == calls, "Planned call counts differ")
    return packet


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n", type=int, required=True, help="base evidence units per selected stage")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--stages", nargs="+", choices=STAGES, required=True)
    parser.add_argument("--arms", choices=("sie", "paired"), required=True)
    parser.add_argument("--phase", choices=("pilot", "confirmatory"), required=True)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--exclude-packet", type=Path, action="append", default=[])
    parser.add_argument(
        "--request-timeout", type=float, required=True, help="connect, read and SDK provision budget in seconds"
    )
    parser.add_argument("--wall-limit", type=float, required=True)
    parser.add_argument("--placement-label", required=True, help="nonsecret label for this declared trial placement")
    parser.add_argument("--bootstrap-seed", type=int, default=1729)
    parser.add_argument("--bootstrap-resamples", type=int, default=5000)
    args = parser.parse_args()
    config = {
        "n": args.n,
        "seed": args.seed,
        "stages": args.stages,
        "arms": args.arms,
        "phase": args.phase,
        "warmup": args.warmup,
        "request_timeout_s": args.request_timeout,
        "wall_limit_s": args.wall_limit,
        "placement_label": args.placement_label,
        "bootstrap": {
            "seed": args.bootstrap_seed,
            "resamples": args.bootstrap_resamples,
            "quantile": "linear-(n-1)p",
            "minimum_clusters": 2,
        },
    }
    try:
        require(not args.out.exists(), "Packet destination already exists")
        packet = build_packet(populations(args.inputs), config, [load_packet(p) for p in args.exclude_packet])
        write_exclusive(args.out, packet)
    except (OSError, ValueError, KeyError, TypeError):
        raise SystemExit(
            "Preparation failed; check verified inputs, fixed trial settings, exclusions and the fresh output path."
        ) from None
    print(
        json.dumps(
            {"packet_digest": packet["packet_digest"], "planned_semantic_calls": packet["planned_semantic_calls"]},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
