"""Frozen public request semantics, canonical identities, and caller composition."""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path, PurePosixPath
from typing import Any

HERE = Path(__file__).resolve().parent
STAGES = ("G", "M", "E", "R")
MODELS = {
    "G": ["Qwen/Qwen3Guard-Gen-4B"],
    "M": ["urchade/gliner_multi_pii-v1", "numind/NuNER_Zero"],
    "E": ["Qwen/Qwen3-Embedding-4B"],
    "R": ["Qwen/Qwen3-Reranker-4B"],
}
RIVALS = {"G": "anthropic", "M": "anthropic", "E": "openai", "R": "cohere"}
REQUEST_LABELS = [
    "person", "street address", "city", "county", "postal code", "gps coordinates", "date of birth",
    "phone number", "fax number", "email address", "url", "ip address", "mac address", "social security number",
    "national id number", "tax id number", "medical record number", "health plan beneficiary number",
    "account number", "credit card number", "cvv", "pin", "password", "api key", "http cookie", "username",
    "customer id", "employee id", "certificate or license number", "license plate",
    "vehicle identification number", "device identifier", "biometric identifier", "unique identifier",
    "passport number", "driver's license number",
]  # fmt: skip
RULES = {
    "in-force": (
        "Return the final rule the agency has adopted on this subject, not a proposed rule. "
        "A proposal that is still open for comment is not relevant."
    ),
    "proposed": (
        "Return the proposed rule the agency has published for comment on this subject, not a final rule. "
        "A rule already adopted is not relevant."
    ),
}
GUARD_PROMPT = (
    "Is the following user message harmful? Answer with exactly one word: harmful or unharmful.\n\nMessage:\n{prompt}"
)
PII_PROMPT = (
    "Find every piece of personally identifiable information in the text. Use these types: {labels}. "
    "Copy each mention exactly as it appears in the text. Return an empty list if there is none."
)
SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["entities"],
    "properties": {
        "entities": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["text", "type"],
                "properties": {"text": {"type": "string"}, "type": {"type": "string", "enum": REQUEST_LABELS}},
            },
        },
    },
}
WORD = re.compile(r"\w+(?:[-_]\w+)*|\S")
NAME_TOKEN = re.compile(r"[^\W\d_]{3,}")
M_POLICY = {"window_units": 300, "overlap": 50, "score_floor": 0.6, "model_paths": 2, "serial_windows": True}
TIMING_POLICY = {
    "version": 2,
    "clock": "perf_counter",
    "stage": "request-building-through-joined-caller-composition",
    "call": "after-intent-capture-through-body-validation",
    "headers": "after-dispatch-capture-through-response-headers",
    "checkpoint": "durable-operation-intent-buffered-events-sealed-operation-result",
    "replay": "after-stage-timer-before-next-operation",
    "unsealed_missing_calls": "attempt-status-unknown",
}
PROVIDER_URLS = {
    "anthropic": "https://api.anthropic.com",
    "openai": "https://api.openai.com",
    "cohere": "https://api.cohere.com",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def canonical(value: Any) -> bytes:
    """UTF-8 JSON, sorted keys, compact separators, unescaped Unicode, no NaN, no newline."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def digest(value: Any) -> str:
    return sha256(canonical(value))


def sources() -> dict[str, Any]:
    return json.loads((HERE / "sources.json").read_bytes())


def source_digest() -> str:
    return digest(sources())


def protocol_digest() -> str:
    # These are behavior bindings, never endpoint eligibility requirements.
    return digest(
        {
            "timing_policy": TIMING_POLICY,
            "files": {name: sha256((HERE / name).read_bytes()) for name in ("protocol.py", "prepare.py", "run.py")},
        }
    )


def safe_path(value: str) -> str:
    path = PurePosixPath(value)
    require(
        bool(value)
        and not path.is_absolute()
        and ".." not in path.parts
        and str(path) == value
        and "\\" not in value
        and all(ord(c) >= 32 for c in value),
        "Unsafe manifest path",
    )
    return value


def positive(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0


def windows(text: str) -> list[tuple[int, int]]:
    words = [match.span() for match in WORD.finditer(text)]
    if len(words) <= 300:
        return [(0, len(text))]
    out = []
    for first in range(0, len(words), 250):
        last = min(first + 300, len(words)) - 1
        out.append((words[first][0], words[last][1]))
        if last == len(words) - 1:
            break
    return out


def propagated(text: str, spans: list[dict[str, Any]]) -> list[dict[str, Any]]:
    tokens = sorted(
        {t for s in spans if s["label"] == "person" for t in NAME_TOKEN.findall(text[s["start"] : s["end"]])}
    )
    added = []
    for token in tokens:
        pattern = re.compile(r"(?<![^\W\d_])" + re.escape(token) + r"(?![^\W\d_])")
        for match in pattern.finditer(text):
            if not any(s["start"] <= match.start() and s["end"] >= match.end() for s in spans):
                added.append({"start": match.start(), "end": match.end(), "label": "person", "derived": True})
    return added


def compose(text: str, paths: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    merged: dict[tuple[int, int, str], dict[str, Any]] = {}
    for entities in paths:
        for entity in entities:
            if (entity.get("score") or 0) < 0.6:
                continue
            key = (entity["start"], entity["end"], entity["label"])
            if key not in merged or entity["score"] > (merged[key].get("score") or 0):
                merged[key] = dict(entity)
    spans = sorted(merged.values(), key=lambda s: (s["start"], s["end"]))
    return sorted(spans + propagated(text, spans), key=lambda s: (s["start"], s["end"]))


def llm_spans(text: str, output: dict[str, Any]) -> tuple[list[dict[str, Any]], int]:
    require(set(output) == {"entities"} and isinstance(output["entities"], list), "Malformed PII schema")
    spans, seen, unmatched = [], set(), 0
    for entity in output["entities"]:
        require(isinstance(entity, dict) and set(entity) == {"text", "type"}, "Malformed PII entity")
        require(isinstance(entity["text"], str) and entity["type"] in REQUEST_LABELS, "Malformed PII entity")
        needle = entity["text"]
        at = text.find(needle) if needle else -1
        unmatched += at == -1
        while at != -1:
            end = at + len(needle)
            if (at, end) not in seen:
                seen.add((at, end))
                spans.append({"start": at, "end": end, "label": entity["type"]})
            at = text.find(needle, at + 1)
    return spans, unmatched


def masked(text: str, spans: list[dict[str, Any]]) -> str:
    labels: dict[tuple[int, int], str] = {}
    merged: list[list[int]] = []
    for span in spans:
        labels.setdefault((span["start"], span["end"]), span["label"])
    for span in sorted(spans, key=lambda s: (s["start"], s["end"])):
        if merged and span["start"] < merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], span["end"])
        else:
            merged.append([span["start"], span["end"]])
    out, at = [], 0
    for start, end in merged:
        label = next((label for (s, _), label in labels.items() if s == start), "pii")
        out.extend([text[at:start], f"[{label.upper()}]"])
        at = end
    return "".join([*out, text[at:]])


def requests(stage: str, data: dict[str, Any], arm: str, rule: str | None = None) -> list[dict[str, Any]]:
    """Semantic calls; SDK-added wire IDs and physical retries are recorded separately."""
    require(stage in STAGES and arm in ("sie", "rival"), "Unknown stage or arm")
    text = data.get("text", "")
    if arm == "sie":
        if stage == "G":
            calls = [
                {
                    "method": "chat_completions",
                    "model": MODELS[stage][0],
                    "body": {
                        "messages": [{"role": "user", "content": text}],
                        "temperature": 0,
                        "max_tokens": 64,
                    },
                }
            ]
        elif stage == "M":
            calls = [
                {
                    "method": "extract",
                    "model": model,
                    "window": index,
                    "offset": start,
                    "body": {
                        "item": {"text": text[start:end]},
                        "labels": REQUEST_LABELS,
                    },
                }
                for model in MODELS[stage]
                for index, (start, end) in enumerate(windows(text))
            ]
        elif stage == "E":
            calls = [
                {"method": "encode", "model": MODELS[stage][0], "body": {"item": {"text": text}, "is_query": True}}
            ]
        else:
            require(rule in RULES, "Unknown rerank rule")
            calls = [
                {
                    "method": "score",
                    "model": MODELS[stage][0],
                    "body": {
                        "query": {"text": data["query"]},
                        "items": data["candidates"],
                        "instruction": RULES[rule or ""],
                    },
                }
            ]
    elif stage in ("G", "M"):
        body: dict[str, Any] = {
            "model": "claude-haiku-4-5",
            "temperature": 0,
            "max_tokens": 10 if stage == "G" else 4096,
            "messages": [{"role": "user", "content": GUARD_PROMPT.format(prompt=text) if stage == "G" else text}],
        }
        if stage == "M":
            body.update(
                {
                    "system": PII_PROMPT.format(labels=", ".join(REQUEST_LABELS)),
                    "output_config": {"format": {"type": "json_schema", "schema": SCHEMA}},
                }
            )
        calls = [
            {
                "method": "messages",
                "provider": "anthropic",
                "model": body["model"],
                "path": "/v1/messages",
                "body": body,
            }
        ]
    elif stage == "E":
        body = {"model": "text-embedding-3-large", "input": text, "dimensions": 3072, "encoding_format": "float"}
        calls = [
            {
                "method": "embeddings",
                "provider": "openai",
                "model": body["model"],
                "path": "/v1/embeddings",
                "body": body,
            }
        ]
    else:
        require(rule in RULES, "Unknown rerank rule")
        body = {
            "model": "rerank-v4.0-pro",
            "query": f"Instruction: {RULES[rule or '']}\nQuery: {data['query']}",
            "documents": [c["text"] for c in data["candidates"]],
            "top_n": 20,
            "max_tokens_per_doc": 4096,
        }
        calls = [{"method": "rerank", "provider": "cohere", "model": body["model"], "path": "/v2/rerank", "body": body}]
    return [{**call, "request_digest": digest(call)} for call in calls]


def call_id(observation_id: str, arm: str, index: int) -> str:
    return digest({"observation_id": observation_id, "arm": arm, "index": index})


def validate_reply(stage: str, arm: str, reply: dict[str, Any], data: dict[str, Any], call: dict[str, Any]) -> Any:
    """Validate actual projected replies before treating a call as successful."""
    require(
        isinstance(reply, dict) and not reply.get("item_error") and not reply.get("malformed_reply"),
        "Per-item or malformed reply",
    )
    returned_model = reply.get("returned_model")
    if returned_model is not None:
        # The Haiku alias can resolve to a dated snapshot; SIE IDs and the other
        # frozen provider IDs have no alias-to-snapshot substitution in this trial.
        require(
            returned_model == call["model"]
            or (
                arm == "rival"
                and call.get("provider") == "anthropic"
                and isinstance(returned_model, str)
                and re.fullmatch(re.escape(call["model"]) + r"-[0-9]{8}", returned_model) is not None
            ),
            "Response model differs from the frozen request",
        )
    if stage == "G":
        text = reply["text"]
        require(isinstance(text, str) and not reply.get("refusal"), "Malformed guard reply")
        pattern = r"Safety:\s*(Safe|Unsafe|Controversial)" if arm == "sie" else r"^\s*(harmful|unharmful)\s*$"
        match = re.search(pattern, text, re.IGNORECASE)
        if match is None:
            raise ValueError("Malformed guard verdict")
        return {"text": text, "verdict": match.group(1).lower()}
    if stage == "M":
        if arm == "rival":
            require(not reply.get("refusal"), "PII refusal")
            spans, unmatched = llm_spans(data["text"], json.loads(reply["text"]))
            return {"spans": spans, "unmatched_mentions": unmatched, "masked": masked(data["text"], spans)}
        require(isinstance(reply.get("entities"), list), "Missing entities")
        length = len(call["body"]["item"]["text"])
        for entity in reply["entities"]:
            start, end = entity["start"], entity["end"]
            require(type(start) is int and type(end) is int and 0 <= start < end <= length, "Invalid Unicode span")
            require(entity["label"] in REQUEST_LABELS, "Unknown entity label")
            score = entity.get("score")
            require(
                score is None
                or (
                    isinstance(score, (int, float))
                    and not isinstance(score, bool)
                    and math.isfinite(score)
                    and 0 <= score <= 1
                ),
                "Invalid entity score",
            )
        return [
            {**entity, "start": entity["start"] + call["offset"], "end": entity["end"] + call["offset"]}
            for entity in reply["entities"]
        ]
    if stage == "E":
        vector = reply["dense"]
        expected = 2560 if arm == "sie" else 3072
        require(isinstance(vector, list) and len(vector) == expected, "Embedding dimension differs")
        require(
            all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) for x in vector),
            "Invalid embedding value",
        )
        return {"dimensions": len(vector)}
    scores = reply["scores"]
    ids = [c["id"] for c in data["candidates"]]
    if arm == "rival":
        results = reply["results"]
        require(
            isinstance(results, list)
            and len(results) == 20
            and all(isinstance(r, dict) and type(r.get("index")) is int for r in results)
            and {r["index"] for r in results} == set(range(20)),
            "Invalid provider rerank indices",
        )
    require(
        isinstance(scores, list)
        and len(scores) == 20
        and all(isinstance(s, dict) and isinstance(s.get("item_id"), str) for s in scores)
        and len({s["item_id"] for s in scores}) == 20
        and {s["item_id"] for s in scores} == set(ids),
        "Rerank candidate set differs",
    )
    require(
        all(
            isinstance(s["score"], (int, float)) and not isinstance(s["score"], bool) and math.isfinite(s["score"])
            for s in scores
        ),
        "Invalid rerank score",
    )
    return {"scores": scores}
