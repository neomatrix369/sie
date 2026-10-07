"""Strict offline accounting and paired base-cluster bootstrap; no network or credentials."""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from fetch import write_exclusive
from prepare import load_packet
from protocol import MODELS, TIMING_POLICY, compose, digest, masked, positive, require, sha256, validate_reply
from run import operation_identity, subcall_identity


def quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    at = (len(ordered) - 1) * probability
    lower = int(at)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (at - lower)


def elapsed_summary(values: list[float]) -> dict[str, Any]:
    return (
        {"count": len(values), "p50_s": quantile(values, 0.5), "p90_s": quantile(values, 0.9)}
        if values
        else {"count": 0, "p50_s": None, "p90_s": None}
    )


def stage_qualifies(stage: dict[str, Any], paired: bool) -> bool:
    if paired:
        return bool(stage["paired_latency"].get("qualifying", False))
    return stage["accounting"]["sie"]["successful"] > 0


def paired_bootstrap(clusters: list[list[tuple[float, float]]], settings: dict[str, Any]) -> dict[str, Any]:
    require(all(positive(s) and positive(r) for cluster in clusters for s, r in cluster), "Invalid paired timing")
    cohort = "successful complete paired base-case clusters"
    if len(clusters) < settings["minimum_clusters"]:
        rows = [pair for cluster in clusters for pair in cluster]
        return {
            "status": "insufficient",
            "cohort": cohort,
            "base_clusters": len(clusters),
            "pairs": len(rows),
            "sie_median_s": statistics.median(s for s, _ in rows) if rows else None,
            "rival_median_s": statistics.median(r for _, r in rows) if rows else None,
            "sie_median_percentile_95": None,
            "rival_median_percentile_95": None,
        }

    def estimate(rows: list[tuple[float, float]]) -> tuple[float, float, float, float]:
        sie = statistics.median(s for s, _ in rows)
        rival = statistics.median(r for _, r in rows)
        return rival - sie, rival / sie, sie, rival

    point = estimate([pair for cluster in clusters for pair in cluster])
    randomizer = random.Random(settings["seed"])
    samples = [
        estimate([pair for _ in clusters for pair in randomizer.choice(clusters)]) for _ in range(settings["resamples"])
    ]
    return {
        "status": "estimated",
        "cohort": cohort,
        "base_clusters": len(clusters),
        "pairs": sum(map(len, clusters)),
        "difference_rival_minus_sie_s": point[0],
        "ratio_rival_over_sie": point[1],
        "sie_median_s": point[2],
        "rival_median_s": point[3],
        "sie_median_percentile_95": [quantile([s[2] for s in samples], p) for p in (0.025, 0.975)],
        "rival_median_percentile_95": [quantile([s[3] for s in samples], p) for p in (0.025, 0.975)],
        "difference_percentile_95": [quantile([s[0] for s in samples], p) for p in (0.025, 0.975)],
        "ratio_percentile_95": [quantile([s[1] for s in samples], p) for p in (0.025, 0.975)],
        "bootstrap": settings,
    }


def read_journal(path: Path, packet: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any] | None, str | None]:
    body = path.read_bytes()
    rows, previous, truncated = [], None, None
    lines = body.splitlines(keepends=True)
    for index, line in enumerate(lines):
        if not line.endswith(b"\n"):
            require(index == len(lines) - 1, "Malformed journal boundary")
            truncated = sha256(line)
            break
        try:
            row = json.loads(line)
        except (ValueError, UnicodeDecodeError):
            require(index == len(lines) - 1 and not line.endswith(b"\n"), "Malformed journal row")
            truncated = sha256(line)
            break
        content = {k: v for k, v in row.items() if k != "entry_digest"}
        require(
            row["entry_digest"] == digest(content) and row["previous_digest"] == previous and row["sequence"] == index,
            "Journal hash chain differs",
        )
        require(
            all(row[k] == packet[k] for k in ("packet_digest", "protocol_digest", "source_digest")),
            "Mixed packet/source/protocol",
        )
        previous = row["entry_digest"]
        rows.append(row)
    end_path = path.with_name(path.name + ".end.json")
    end = json.loads(end_path.read_bytes()) if end_path.exists() else None
    if end is not None:
        require(
            end["end_digest"] == digest({k: v for k, v in end.items() if k != "end_digest"}),
            "Terminal record digest differs",
        )
        require(
            all(end[k] == packet[k] for k in ("packet_digest", "protocol_digest", "source_digest")),
            "Mixed terminal record",
        )
        require(
            end["journal_digest"] == sha256(body)
            and end["reason"] in ("finished", "deadline", "interrupted", "child_failure"),
            "Terminal journal binding differs",
        )
        require(positive(end["elapsed_s"]), "Invalid total duration")
        require(truncated is None or end["reason"] != "finished", "Truncated journal cannot be finished")
    return rows, end, truncated


def accounting(
    keys: list[Any], intents: dict[Any, Any], outcomes: dict[Any, Any], unknown: set[Any] | None = None
) -> dict[str, Any]:
    unknown = unknown or set()
    terminal = [outcomes[k] for k in keys if k in outcomes]
    successful = [r for r in terminal if r["status"] == "success"]
    return {
        "planned": len(keys),
        "attempted": sum(k in intents for k in keys),
        "attempted_count_exact": not any(k in unknown for k in keys),
        "successful": len(successful),
        "failed": len(terminal) - len(successful),
        "unresolved": sum(k in intents and k not in outcomes for k in keys),
        "unattempted": sum(k not in intents and k not in unknown for k in keys),
        "attempt_status_unknown": sum(k in unknown for k in keys),
        "successful_operation_latency": elapsed_summary([r["elapsed_s"] for r in successful]),
        "all_terminal_attempt_elapsed": elapsed_summary([r["elapsed_s"] for r in terminal]),
        "failures": [
            {"identity": str(k), "error": outcomes[k]["error"], "elapsed_s": outcomes[k]["elapsed_s"]}
            for k in keys
            if k in outcomes and outcomes[k]["status"] == "failed"
        ],
    }


def score_trial(packet_path: Path, journal_path: Path) -> dict[str, Any]:
    packet = load_packet(packet_path)
    rows, end, truncated = read_journal(journal_path, packet)
    require(
        bool(rows) and rows[0]["event"] == "run_start" and sum(r["event"] == "run_start" for r in rows) == 1,
        "Missing/duplicate run metadata",
    )
    config = packet["config"]
    require(
        rows[0]["wall_limit_s"] == config["wall_limit_s"] and rows[0]["placement_label"] == config["placement_label"],
        "Run trial settings differ",
    )
    require(
        rows[0]["timeout_budgets_s"] == {k: config["request_timeout_s"] for k in ("connect", "read", "provision")},
        "Run request budgets differ",
    )
    operations, calls = {}, {}
    model_paths: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    for obs in packet["observations"]:
        for arm in obs["arm_order"]:
            operations[(obs["observation_id"], arm)] = (obs, operation_identity(obs, arm))
            for index in range(len(obs["requests"][arm])):
                identity = subcall_identity(obs, arm, index)
                calls[identity["call_id"]] = (obs, identity)
                model_paths[(obs["observation_id"], arm, identity["model"])].append(identity["call_id"])
    op_intents, op_results, call_intents, call_results = {}, {}, {}, {}
    dispatches: dict[str, list[dict[str, Any]]] = defaultdict(list)
    responses: dict[str, list[dict[str, Any]]] = defaultdict(list)
    discovery_intent, discovery_result = None, None
    active_operation = None
    expected_op_order = list(operations)
    next_operation = 0
    active_calls = set()
    next_in_path: dict[tuple[str, str, str], int] = defaultdict(int)
    failed_paths = set()
    captured_counts: dict[tuple[str, str], int] = defaultdict(int)
    for row in rows[1:]:
        event = row["event"]
        if event in ("discovery_intent", "discovery_result"):
            require(row["call_id"] == "metadata" and not op_intents, "Unexpected discovery")
            if event == "discovery_intent":
                require(discovery_intent is None and discovery_result is None, "Duplicate discovery intent")
                discovery_intent = row
            else:
                require(discovery_intent is not None and discovery_result is None, "Unexpected discovery terminal")
                require(
                    row["status"] in ("success", "failed") and positive(row["elapsed_s"]), "Invalid discovery outcome"
                )
                discovery_result = row
            continue
        if event in ("operation_intent", "operation_result"):
            key = (row["observation_id"], row["arm"])
            require(
                key in operations and all(row[k] == v for k, v in operations[key][1].items()),
                "Unexpected operation identity/request",
            )
            if event == "operation_intent":
                require(discovery_result is not None, "Operation before completed discovery")
                require(
                    key not in op_intents
                    and active_operation is None
                    and next_operation < len(expected_op_order)
                    and key == expected_op_order[next_operation],
                    "Duplicate, overlapping or reordered operation",
                )
                op_intents[key] = row
                active_operation = key
                next_operation += 1
            else:
                require(
                    key in op_intents and key not in op_results and active_operation == key and not active_calls,
                    "Unexpected operation terminal",
                )
                require(
                    row["status"] in ("success", "failed") and positive(row["elapsed_s"]),
                    "Invalid operation duration/status",
                )
                require(
                    row.get("evidence_sealed") is True
                    and type(row.get("captured_event_count")) is int
                    and row["captured_event_count"] == captured_counts[key],
                    "Missing or incomplete operation evidence seal",
                )
                if row["status"] == "failed":
                    require(isinstance(row.get("error"), dict), "Missing failure evidence")
                expected_calls = [
                    cid for cid, (_, identity) in calls.items() if (identity["observation_id"], identity["arm"]) == key
                ]
                if row["status"] == "success":
                    require(
                        all(cid in call_results and call_results[cid]["status"] == "success" for cid in expected_calls),
                        "Success hides missing/failed constituent calls",
                    )
                    obs = operations[key][0]
                    validated = [
                        validate_reply(
                            obs["stage"],
                            row["arm"],
                            call_results[cid]["reply"],
                            obs["unit"]["data"],
                            obs["requests"][row["arm"]][calls[cid][1]["request_index"]],
                        )
                        for cid in expected_calls
                    ]
                    if obs["stage"] == "M" and row["arm"] == "sie":
                        paths = [
                            [
                                entity
                                for cid, entities in zip(expected_calls, validated, strict=True)
                                if calls[cid][1]["model"] == model
                                for entity in entities
                            ]
                            for model in MODELS["M"]
                        ]
                        spans = compose(obs["unit"]["data"]["text"], paths)
                        require(row["result"]["spans"] == spans, "Caller composition result differs")
                        require(
                            "[REDACTED_CREDENTIAL]" in row["result"]["masked"]
                            or row["result"]["masked"] == masked(obs["unit"]["data"]["text"], spans),
                            "Caller mask differs",
                        )
                    else:
                        require(row["result"] == validated[0], "Operation result differs from its constituent reply")
                completed = [call_results[cid] for cid in expected_calls if cid in call_results]
                if row["stage"] == "M" and row["arm"] == "sie":
                    lower_bound = max(
                        math.fsum(call["elapsed_s"] for call in completed if call["model"] == model)
                        for model in MODELS["M"]
                    )
                else:
                    lower_bound = max((call["elapsed_s"] for call in completed), default=0)
                # One microsecond or 1e-9 relative covers floating timer subtraction/summation,
                # not omitted serial work. Failed completed calls remain inside the bound.
                require(
                    row["elapsed_s"] >= lower_bound
                    or math.isclose(row["elapsed_s"], lower_bound, rel_tol=1e-9, abs_tol=1e-6),
                    "Operation timer excludes completed serial call time",
                )
                op_results[key] = row
                active_operation = None
            continue
        if event in ("call_intent", "call_result", "dispatch", "response"):
            cid = row["call_id"]
            if cid == "metadata":
                require(
                    event in ("dispatch", "response") and discovery_intent is not None and discovery_result is None,
                    "Unexpected metadata HTTP event",
                )
            else:
                require(
                    cid in calls and all(row[k] == v for k, v in calls[cid][1].items()),
                    "Unexpected subcall identity/request",
                )
                op_key = (row["observation_id"], row["arm"])
                require(active_operation == op_key, "Subcall outside its operation")
                captured_counts[op_key] += 1
            if event == "call_intent":
                require(cid not in call_intents, "Duplicate call intent")
                obs, identity = calls[cid]
                if obs["stage"] == "M" and row["arm"] == "sie":
                    path = (row["observation_id"], row["arm"], identity["model"])
                    require(path not in failed_paths, "Continuation after model path failure")
                    require(
                        next_in_path[path] < len(model_paths[path]) and cid == model_paths[path][next_in_path[path]],
                        "Skipped or reordered model path window",
                    )
                    next_in_path[path] += 1
                call_intents[cid] = row
                active_calls.add(cid)
                limit = 2 if obs["stage"] == "M" and row["arm"] == "sie" else 1
                require(len(active_calls) <= limit, "Exceeded dispatch concurrency")
                require(
                    not any(calls[other][1]["model"] == identity["model"] for other in active_calls if other != cid),
                    "Overlapping windows in one model path",
                )
            elif event == "call_result":
                require(
                    cid in call_intents and cid not in call_results and cid in active_calls,
                    "Duplicate/unstarted call terminal",
                )
                require(
                    row["status"] in ("success", "failed") and positive(row["elapsed_s"]), "Invalid subcall outcome"
                )
                require(
                    row["physical_dispatches"] == len(dispatches[cid])
                    and type(row["sdk_retry_count"]) is int
                    and row["sdk_retry_count"] >= 0,
                    "Retry dispatch count differs",
                )
                if row["status"] == "success":
                    obs, identity = calls[cid]
                    validate_reply(
                        obs["stage"],
                        row["arm"],
                        row["reply"],
                        obs["unit"]["data"],
                        obs["requests"][row["arm"]][identity["request_index"]],
                    )
                    require(
                        len(responses[cid]) == len(dispatches[cid])
                        and bool(dispatches[cid])
                        and 200 <= responses[cid][-1]["status"] < 300,
                        "Successful call lacks completed HTTP evidence",
                    )
                else:
                    require(isinstance(row.get("error"), dict), "Missing subcall failure evidence")
                    if row["stage"] == "M" and row["arm"] == "sie":
                        failed_paths.add((row["observation_id"], row["arm"], row["model"]))
                call_results[cid] = row
                active_calls.remove(cid)
            elif event == "dispatch":
                require(cid == "metadata" or cid in active_calls, "Dispatch before captured call intent")
                require(row["dispatch_index"] == len(dispatches[cid]) + 1, "Physical dispatch sequence differs")
                require(len(dispatches[cid]) == len(responses[cid]), "Overlapping physical retries")
                dispatches[cid].append(row)
            else:
                require(cid == "metadata" or cid in active_calls, "Response after terminal")
                require(
                    row["dispatch_index"] == len(responses[cid]) + 1 and len(dispatches[cid]) == row["dispatch_index"],
                    "Unexpected physical response",
                )
                require(
                    type(row["status"]) is int and 100 <= row["status"] <= 599 and positive(row["headers_elapsed_s"]),
                    "Invalid HTTP result",
                )
                responses[cid].append(row)
            continue
        raise ValueError("Unknown journal event")
    complete = (
        end is not None
        and end["reason"] == "finished"
        and truncated is None
        and discovery_result is not None
        and len(op_results) == len(operations)
        and len(call_results) == len(calls)
    )
    result: dict[str, Any] = {
        "schema": 1,
        "packet_digest": packet["packet_digest"],
        "protocol_digest": packet["protocol_digest"],
        "timing_policy": TIMING_POLICY,
        "complete": complete,
        "qualifying_confirmatory": complete and config["phase"] == "confirmatory",
        "ending": end,
        "truncated_tail_sha256": truncated,
        "discovery": discovery_result,
        "discovery_attempted": discovery_intent is not None,
        "discovery_unresolved": discovery_intent is not None and discovery_result is None,
        "statistics_scope": "client operation time excluding journal replay; successful complete base-case pairs",
        "phases": {},
        "stages": {},
    }
    for phase in ("warmup", config["phase"]):
        result["phases"][phase] = {}
        for stage in config["stages"]:
            arms = ["sie", "rival"] if config["arms"] == "paired" else ["sie"]
            result["phases"][phase][stage] = {}
            for arm in arms:
                op_keys = [
                    key
                    for key, (obs, _) in operations.items()
                    if obs["stage"] == stage and obs["phase"] == phase and key[1] == arm
                ]
                call_keys = [
                    cid
                    for cid, (obs, identity) in calls.items()
                    if obs["stage"] == stage and obs["phase"] == phase and identity["arm"] == arm
                ]
                summary = accounting(op_keys, op_intents, op_results)
                unknown_calls = {
                    cid
                    for cid in call_keys
                    if cid not in call_intents
                    and (calls[cid][1]["observation_id"], arm) in op_intents
                    and (calls[cid][1]["observation_id"], arm) not in op_results
                }
                summary["semantic_calls"] = accounting(call_keys, call_intents, call_results, unknown_calls)
                summary["semantic_calls"]["successful_call_latency"] = summary["semantic_calls"].pop(
                    "successful_operation_latency"
                )
                summary["physical_dispatches"] = sum(len(dispatches[cid]) for cid in call_keys)
                summary["physical_dispatches_exact"] = all(
                    key not in op_intents or key in op_results for key in op_keys
                )
                summary["sdk_retries"] = sum(
                    call_results[cid]["sdk_retry_count"] for cid in call_keys if cid in call_results
                )
                summary["sdk_retries_exact"] = summary["physical_dispatches_exact"]
                result["phases"][phase][stage][arm] = summary
            if phase != "confirmatory":
                continue
            selected = [o for o in packet["observations"] if o["stage"] == stage and o["phase"] == phase]
            by_base: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for obs in selected:
                by_base[obs["base_id"]].append(obs)
            clusters = []
            if config["arms"] == "paired":
                for group in by_base.values():
                    if all(
                        (o["observation_id"], arm) in op_results
                        and op_results[(o["observation_id"], arm)]["status"] == "success"
                        for o in group
                        for arm in ("sie", "rival")
                    ):
                        clusters.append(
                            [
                                (
                                    op_results[(o["observation_id"], "sie")]["elapsed_s"],
                                    op_results[(o["observation_id"], "rival")]["elapsed_s"],
                                )
                                for o in group
                            ]
                        )
            comparison = (
                paired_bootstrap(clusters, config["bootstrap"])
                if config["arms"] == "paired"
                else {"status": "sie_only", "pairs": 0, "base_clusters": 0}
            )
            comparison["excluded_base_clusters"] = len(by_base) - len(clusters) if config["arms"] == "paired" else 0
            comparison["excluded_pairs"] = len(selected) - sum(map(len, clusters)) if config["arms"] == "paired" else 0
            comparison["qualifying"] = complete and comparison["status"] == "estimated"
            result["stages"][stage] = {"accounting": result["phases"][phase][stage], "paired_latency": comparison}
    if config["phase"] == "pilot":
        result["stages"] = {
            stage: {
                "accounting": result["phases"]["pilot"][stage],
                "paired_latency": {"status": "pilot_only", "qualifying": False},
            }
            for stage in config["stages"]
        }
    result["qualifying_confirmatory"] = (
        complete
        and config["phase"] == "confirmatory"
        and all(stage_qualifies(stage, config["arms"] == "paired") for stage in result["stages"].values())
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--journal", type=Path, required=True)
    parser.add_argument("--out", type=Path, help="new report destination; otherwise prints the report")
    args = parser.parse_args()
    try:
        if args.out is not None:
            require(not args.out.exists(), "Report already exists")
        report = score_trial(args.packet, args.journal)
        if args.out is not None:
            write_exclusive(args.out, report)
        else:
            print(json.dumps(report, indent=2))
    except (OSError, ValueError, KeyError, TypeError):
        raise SystemExit(
            "Scoring rejected the packet or journal. "
            "Preserve the original files; check their bindings and completeness."
        ) from None


if __name__ == "__main__":
    main()
