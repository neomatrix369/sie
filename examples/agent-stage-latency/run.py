"""Explicitly execute a frozen packet against a supplied SIE URL within a hard wall deadline."""

from __future__ import annotations

import argparse
import contextlib
import copy
import importlib.metadata
import json
import logging
import math
import multiprocessing
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from fetch import fsync_directory, mkdir_synced, write_exclusive
from prepare import load_packet
from protocol import (
    MODELS,
    PROVIDER_URLS,
    RIVALS,
    call_id,
    canonical,
    compose,
    digest,
    masked,
    requests,
    require,
    sha256,
    validate_reply,
)

ERROR_CODES = frozenset(
    {
        "ADMISSION_QUEUE_FULL",
        "MODEL_LOADING",
        "MODEL_LOAD_FAILED",
        "RESOURCE_EXHAUSTED",
        "INVALID_ARGUMENT",
        "UNAUTHENTICATED",
        "PERMISSION_DENIED",
        "NOT_FOUND",
        "UNAVAILABLE",
        "INTERNAL",
        "DEADLINE_EXCEEDED",
        "RATE_LIMITED",
        "ITEM_ERROR",
        "MALFORMED_REPLY",
        "HTTP_ERROR",
        "WALL_DEADLINE",
        "CALL_FAILED",
    }
)
ERROR_CLASSES = frozenset(
    {
        "RequestError",
        "ServerError",
        "SIEConnectionError",
        "ProvisioningError",
        "ReadTimeout",
        "ConnectTimeout",
        "WriteTimeout",
        "PoolTimeout",
        "ConnectError",
        "ReadError",
        "RemoteProtocolError",
        "HTTPStatusError",
        "ValueError",
        "KeyError",
        "TypeError",
        "JSONDecodeError",
        "CallFailure",
        "TimeoutError",
        "AssertionError",
        "AttributeError",
        "SIEError",
        "PoolError",
        "LoraLoadingError",
        "ModelLoadingError",
        "ModelLoadFailedError",
        "InputTooLongError",
        "RateLimitError",
        "InsufficientCreditsError",
        "SpendLimitError",
        "AccountInactiveError",
        "AccountStateUnavailableError",
        "ResourceExhaustedError",
        "IncompleteBatchError",
    }
)


def endpoint(value: str) -> tuple[str, str]:
    """Reject secret-bearing URLs before constructing a client or displaying an error."""
    try:
        parsed = urlsplit(value)
        require(parsed.scheme in ("http", "https") and bool(parsed.hostname), "Invalid endpoint")
        require(
            not parsed.username and not parsed.password and not parsed.query and not parsed.fragment, "Invalid endpoint"
        )
        require(
            not any(ord(c) <= 32 for c in value) and not any(c in value for c in ("\\", "%", "?", "#", "@")),
            "Invalid endpoint",
        )
        require(parsed.port is None or 1 <= parsed.port <= 65535, "Invalid endpoint")
        require(bool(re.fullmatch(r"[A-Za-z0-9./_:-]*", parsed.path)), "Invalid endpoint path")
        origin = f"{parsed.scheme}://{parsed.netloc}"
        return value.rstrip("/"), origin
    except (ValueError, TypeError):
        raise ValueError("Use an absolute HTTP(S) endpoint without userinfo, query or fragment.") from None


def public_id(value: Any) -> str | None:
    return value if isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_./:-]{1,180}", value) else None


def revision(value: Any) -> str | None:
    return value if isinstance(value, str) and re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", value) else None


def sanitize(value: Any, credentials: list[str]) -> Any:
    if isinstance(value, str):
        for secret in credentials:
            if secret:
                value = value.replace(secret, "[REDACTED_CREDENTIAL]")
        return value
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            redacted_key = sanitize(str(key), credentials)
            unique_key = redacted_key
            suffix = 2
            # Retain every value when distinct keys redact to the same name.
            while unique_key in result:
                unique_key = f"{redacted_key}#{suffix}"
                suffix += 1
            result[unique_key] = sanitize(item, credentials)
        return result
    if isinstance(value, (list, tuple)):
        return [sanitize(v, credentials) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return "[NONFINITE]"
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if hasattr(value, "tolist"):
        return sanitize(value.tolist(), credentials)
    return "[UNSERIALIZABLE]"


def stable_error(error: BaseException) -> dict[str, Any]:
    name = type(error).__name__
    result: dict[str, Any] = {"class": name if name in ERROR_CLASSES else "UnexpectedError"}
    code = getattr(error, "code", getattr(error, "error_code", None))
    if isinstance(code, str) and code in ERROR_CODES:
        result["code"] = code
    status = getattr(error, "status_code", None)
    if type(status) is int and 100 <= status <= 599:
        result["status"] = status
    if isinstance(error, (ValueError, KeyError, TypeError)):
        result.setdefault("code", "MALFORMED_REPLY")
    return result


class CallFailure(Exception):
    def __init__(self, code: str, status_code: int | None = None) -> None:
        super().__init__(code)
        self.code = code
        self.status_code = status_code


class Journal:
    """One serialized, fsynced hash chain shared by both M model paths."""

    def __init__(self, path: Path, packet: dict[str, Any], credentials: list[str], *, create: bool = False) -> None:
        self.path = path
        self.packet = packet
        self.credentials = credentials
        self.lock = threading.Lock()
        mkdir_synced(path.parent)
        self.output = path.open("xb" if create else "ab")
        self.output.flush()
        os.fsync(self.output.fileno())
        fsync_directory(path.parent)
        rows = [] if create else [json.loads(line) for line in path.read_bytes().splitlines()]
        self.sequence = len(rows)
        self.previous = rows[-1]["entry_digest"] if rows else None

    def append(self, event: str, **fields: Any) -> None:
        with self.lock:
            content = sanitize(
                {
                    "event": event,
                    "sequence": self.sequence,
                    "previous_digest": self.previous,
                    "packet_digest": self.packet["packet_digest"],
                    "protocol_digest": self.packet["protocol_digest"],
                    "source_digest": self.packet["source_digest"],
                    **fields,
                },
                self.credentials,
            )
            row = {**content, "entry_digest": digest(content)}
            self.output.write(canonical(row) + b"\n")
            self.output.flush()
            os.fsync(self.output.fileno())
            self.previous = row["entry_digest"]
            self.sequence += 1

    def close(self) -> None:
        self.output.close()


class EvidenceCaptureError(Exception):
    pass


class OperationEvents:
    """Snapshot mutable event metadata; call_result transfers its owned sanitized reply."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.events: list[tuple[str, dict[str, Any]]] = []
        self.failed = False

    def append(self, event: str, **fields: Any) -> None:
        try:
            require(event in ("call_intent", "dispatch", "response", "call_result"), "Unexpected buffered event")
            reply = fields.get("reply")
            memo = {id(reply): reply} if event == "call_result" else None
            captured = copy.deepcopy(fields, memo)
            with self.lock:
                self.events.append((event, captured))
        except Exception:
            self.failed = True
            raise EvidenceCaptureError from None

    def flush(self, journal: Journal) -> None:
        # Both M paths have joined before replay. A capture or flush failure must
        # escape to the child supervisor without a misleading operation seal.
        if self.failed:
            raise EvidenceCaptureError
        for event, fields in self.events:
            journal.append(event, **fields)


def operation_identity(obs: dict[str, Any], arm: str) -> dict[str, Any]:
    return {k: obs[k] for k in ("observation_id", "base_id", "stage", "phase", "rule", "repetition", "arm_order")} | {
        "arm": arm,
        "request_digest": digest(obs["requests"][arm]),
    }


def subcall_identity(obs: dict[str, Any], arm: str, index: int) -> dict[str, Any]:
    call = obs["requests"][arm][index]
    return operation_identity(obs, arm) | {
        "call_id": call_id(obs["observation_id"], arm, index),
        "request_digest": call["request_digest"],
        "request_index": index,
        "model": call["model"],
        "window": call.get("window"),
        "offset": call.get("offset"),
    }


class TransportEvidence:
    def __init__(self, journal: Journal) -> None:
        self.journal = journal
        self.local = threading.local()

    def begin(self, identity: dict[str, Any], events: OperationEvents | None = None) -> None:
        self.local.identity = identity
        self.local.dispatches = 0
        self.local.sink = events if events is not None else self.journal

    def request(self, request: Any) -> None:
        self.local.dispatches += 1
        self.local.sink.append("dispatch", **self.local.identity, dispatch_index=self.local.dispatches)
        self.local.dispatched_at = time.perf_counter()

    def response(self, response: Any) -> None:
        self.local.sink.append(
            "response",
            **self.local.identity,
            dispatch_index=self.local.dispatches,
            status=response.status_code,
            headers_elapsed_s=time.perf_counter() - self.local.dispatched_at,
            execution_revision=revision(response.headers.get("X-SIE-Model-Revision")),
        )


def project_reply(stage: str, arm: str, raw: Any, data: dict[str, Any]) -> dict[str, Any]:
    """Keep workload replies and allowlisted execution evidence; omit arbitrary telemetry."""
    if not isinstance(raw, dict):
        return {"malformed_reply": True}
    reply: dict[str, Any] = {"returned_model": public_id(raw.get("model"))}
    if raw.get("model") is not None and reply["returned_model"] is None:
        reply["malformed_reply"] = True
    request = raw.get("request")
    if isinstance(request, dict):
        reply["execution_identity_sha256"] = revision(request.get("execution_identity_sha256"))
        reply["execution_binding_sha256"] = revision(request.get("execution_binding_sha256"))
    if raw.get("error"):
        reply["item_error"] = {"code": "ITEM_ERROR"}
    if stage == "G" or (stage == "M" and arm == "rival"):
        if arm == "sie":
            choices = raw.get("choices")
            if not isinstance(choices, list):
                reply["malformed_reply"] = True
                choices = []
            malformed = [index for index, choice in enumerate(choices) if not isinstance(choice, dict)]
            if malformed:
                reply.update(malformed_reply=True, malformed_choice_indices=malformed, choice_count=len(choices))
            first = choices[0] if choices and isinstance(choices[0], dict) else {}
            message = first.get("message", {})
            if not isinstance(message, dict):
                reply["malformed_reply"] = True
                message = {}
            reply.update(
                {
                    "text": message.get("content"),
                    "refusal": message.get("refusal"),
                    "finish_reason": public_id(first.get("finish_reason")),
                }
            )
        else:
            blocks = raw.get("content")
            if isinstance(blocks, list):
                reply["content"] = [
                    {"type": public_id(b.get("type")), "text": b.get("text")} if isinstance(b, dict) else None
                    for b in blocks
                ]
                if any(not isinstance(b, dict) for b in blocks):
                    reply["malformed_reply"] = True
                reply["text"] = "".join(
                    b.get("text", "")
                    for b in blocks
                    if isinstance(b, dict) and b.get("type") == "text" and isinstance(b.get("text"), str)
                )
            reply["refusal"] = raw.get("stop_reason") == "refusal"
            reply["stop_reason"] = public_id(raw.get("stop_reason"))
    elif stage == "M":
        entities = raw.get("entities")
        reply["entities"] = (
            [{k: e.get(k) for k in ("start", "end", "label", "score", "text")} for e in entities if isinstance(e, dict)]
            if isinstance(entities, list)
            else None
        )
        if isinstance(entities, list) and sum(isinstance(e, dict) for e in entities) != len(entities):
            reply["entities"] = None
    elif stage == "E":
        if arm == "sie":
            reply["dense"] = raw.get("dense")
        else:
            rows = raw.get("data")
            if (
                isinstance(rows, list)
                and len(rows) == 1
                and isinstance(rows[0], dict)
                and type(rows[0].get("index")) is int
                and rows[0]["index"] == 0
            ):
                reply["dense"] = rows[0].get("embedding")
            else:
                reply["dense"] = None
                reply["malformed_reply"] = True
    elif arm == "sie":
        scores = raw.get("scores")
        reply["scores"] = (
            [
                {"item_id": s.get("item_id"), "score": s.get("score"), "rank": s.get("rank")}
                for s in scores
                if isinstance(s, dict)
            ]
            if isinstance(scores, list)
            else []
        )
        if isinstance(scores, list) and len(reply["scores"]) != len(scores):
            reply["malformed_reply"] = True
    else:
        results = raw.get("results")
        reply["results"] = (
            [
                {"index": r.get("index"), "relevance_score": r.get("relevance_score")} if isinstance(r, dict) else None
                for r in results
            ]
            if isinstance(results, list)
            else []
        )
        if not isinstance(results, list) or any(not isinstance(r, dict) for r in results):
            reply["malformed_reply"] = True
        reply["scores"] = []
        for row in reply["results"]:
            if not isinstance(row, dict):
                continue
            index = row["index"]
            if type(index) is int and 0 <= index < len(data["candidates"]):
                reply["scores"].append({"item_id": data["candidates"][index]["id"], "score": row["relevance_score"]})
    usage = raw.get("usage")
    if isinstance(usage, dict):
        reply["usage"] = {
            k: usage[k]
            for k in ("input_tokens", "output_tokens", "prompt_tokens", "total_tokens")
            if type(usage.get(k)) is int and usage[k] >= 0
        }
    return reply


class Executor:
    def __init__(self, packet: dict[str, Any], journal: Journal, config: dict[str, Any]) -> None:
        # Optional live dependencies are imported only in the opted-in child.
        import httpx
        from sie_sdk import SIEClient

        self.packet = packet
        self.journal = journal
        self.config = config
        self.clients: dict[str, Any] = {}
        self.evidence: dict[str, TransportEvidence] = {}
        timeout = min(packet["config"]["request_timeout_s"], self.remaining())
        for stage in packet["config"]["stages"]:
            for model in MODELS[stage]:
                hooks = TransportEvidence(journal)
                transport = httpx.Client(
                    base_url=config["sie_url"],
                    timeout=timeout,
                    follow_redirects=False,
                    trust_env=False,
                    event_hooks={"request": [hooks.request], "response": [hooks.response]},
                )
                client = SIEClient(
                    config["sie_url"],
                    api_key=config["credentials"].get("sie", ""),
                    timeout_s=timeout,
                    connect_timeout_s=timeout,
                    read_timeout_s=timeout,
                    http_client=transport,
                )
                self.clients[model] = client
                self.evidence[model] = hooks
        if packet["config"]["arms"] == "paired":
            for provider in {RIVALS[s] for s in packet["config"]["stages"]}:
                hooks = TransportEvidence(journal)
                headers = (
                    {"x-api-key": config["credentials"][provider], "anthropic-version": "2023-06-01"}
                    if provider == "anthropic"
                    else {"Authorization": f"Bearer {config['credentials'][provider]}"}
                )
                self.clients[provider] = httpx.Client(
                    base_url=config["provider_urls"][provider],
                    headers=headers,
                    timeout=timeout,
                    follow_redirects=False,
                    trust_env=False,
                    event_hooks={"request": [hooks.request], "response": [hooks.response]},
                )
                self.evidence[provider] = hooks

    def remaining(self) -> float:
        remaining = self.config["deadline"] - time.monotonic()
        if remaining <= 0:
            raise CallFailure("WALL_DEADLINE")
        return remaining

    def discover(self) -> None:
        model = next(iter(self.clients))
        self.journal.append("discovery_intent", call_id="metadata")
        self.evidence[model].begin({"call_id": "metadata"})
        start = time.perf_counter()
        try:
            listed = self.clients[model].list_models()
            selected = {m for s in self.packet["config"]["stages"] for m in MODELS[s]}
            safe = [
                {
                    "name": m.get("name"),
                    "weights_revision": revision(m.get("revision")),
                    "dims": m.get("dims") if type(m.get("dims")) is int else None,
                }
                for m in listed
                if isinstance(m, dict) and m.get("name") in selected
            ]
            self.journal.append(
                "discovery_result",
                call_id="metadata",
                status="success",
                models=safe,
                elapsed_s=time.perf_counter() - start,
            )
        except Exception as error:
            self.journal.append(
                "discovery_result",
                call_id="metadata",
                status="failed",
                error=stable_error(error),
                elapsed_s=time.perf_counter() - start,
            )

    def call(self, obs: dict[str, Any], arm: str, index: int, events: OperationEvents) -> Any:
        from sie_sdk import Item

        call = obs["requests"][arm][index]
        identity = subcall_identity(obs, arm, index)
        client_key = call["model"] if arm == "sie" else call["provider"]
        client = self.clients[client_key]
        hooks = self.evidence[client_key]
        events.append("call_intent", **identity)
        hooks.begin(identity, events)
        start = time.perf_counter()
        reply = None
        try:
            timeout = min(self.packet["config"]["request_timeout_s"], self.remaining())
            body = call["body"]
            if arm == "rival":
                response = client.post(call["path"], json=body, timeout=timeout)
                if not 200 <= response.status_code < 300:
                    raise CallFailure("HTTP_ERROR", response.status_code)
                raw = response.json()
            else:
                options = {"wait_for_capacity": False, "max_oom_retries": 0, "provision_timeout_s": timeout}
                if call["method"] == "chat_completions":
                    raw = client.chat_completions(call["model"], **body, **options)
                elif call["method"] == "extract":
                    raw = client.extract(call["model"], Item(**body["item"]), labels=body["labels"], **options)
                elif call["method"] == "encode":
                    raw = client.encode(call["model"], Item(**body["item"]), is_query=True, **options)
                else:
                    raw = client.score(
                        call["model"],
                        Item(**body["query"]),
                        [Item(**item) for item in body["items"]],
                        instruction=body["instruction"],
                        **options,
                    )
            reply = sanitize(project_reply(obs["stage"], arm, raw, obs["unit"]["data"]), self.journal.credentials)
            if reply.get("item_error"):
                raise CallFailure("ITEM_ERROR")
            validated = validate_reply(obs["stage"], arm, reply, obs["unit"]["data"], call)
        except (EvidenceCaptureError, MemoryError):
            raise
        except Exception as error:
            events.append(
                "call_result",
                **identity,
                status="failed",
                reply=reply,
                error=stable_error(error),
                elapsed_s=time.perf_counter() - start,
                physical_dispatches=hooks.local.dispatches,
                sdk_retry_count=client.last_retry_count if arm == "sie" else 0,
                execution_revision=revision(client.last_model_revision) if arm == "sie" else None,
            )
            raise CallFailure(stable_error(error).get("code", "CALL_FAILED")) from None
        events.append(
            "call_result",
            **identity,
            status="success",
            reply=reply,
            elapsed_s=time.perf_counter() - start,
            physical_dispatches=hooks.local.dispatches,
            sdk_retry_count=client.last_retry_count if arm == "sie" else 0,
            execution_revision=revision(client.last_model_revision) if arm == "sie" else None,
        )
        return validated

    def operation(self, obs: dict[str, Any], arm: str) -> None:
        identity = operation_identity(obs, arm)
        self.journal.append("operation_intent", **identity)
        events = OperationEvents()
        start = time.perf_counter()
        result = None
        try:
            # Regeneration includes M's tokenizer/windowing inside its measured boundary.
            expected = requests(obs["stage"], obs["unit"]["data"], arm, obs["rule"])
            require(expected == obs["requests"][arm], "Frozen request differs")
            if obs["stage"] == "M" and arm == "sie":

                def model_path(model: str) -> list[dict[str, Any]]:
                    try:
                        entities = []
                        for index, call in enumerate(expected):
                            if call["model"] == model:
                                entities.extend(self.call(obs, arm, index, events))
                        return entities
                    except (EvidenceCaptureError, MemoryError):
                        # Another future may raise an ordinary call failure first.
                        # Resource loss in either joined path forbids a seal.
                        events.failed = True
                        raise

                with ThreadPoolExecutor(max_workers=2) as pool:
                    futures = [pool.submit(model_path, model) for model in MODELS["M"]]
                    outputs = [future.result() for future in futures]
                spans = compose(obs["unit"]["data"]["text"], outputs)
                result = {"spans": spans, "masked": masked(obs["unit"]["data"]["text"], spans)}
            else:
                result = self.call(obs, arm, 0, events)
        except (EvidenceCaptureError, MemoryError):
            raise
        except Exception as error:
            outcome = {"status": "failed", "error": stable_error(error)}
        else:
            outcome = {"status": "success"}
        elapsed = time.perf_counter() - start
        events.flush(self.journal)
        self.journal.append(
            "operation_result",
            **identity,
            **outcome,
            result=result,
            elapsed_s=elapsed,
            evidence_sealed=True,
            captured_event_count=len(events.events),
        )

    def close(self) -> None:
        for client in self.clients.values():
            client.close()


def child_main(connection: Any) -> None:
    # SDK errors may include connection URLs. Only stable records leave this child.
    with Path(os.devnull).open("w") as sink, contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
        logging.disable(logging.CRITICAL)
        journal = None
        executor = None
        try:
            config = connection.recv()
            packet = load_packet(Path(config["packet_path"]))
            journal = Journal(Path(config["journal"]), packet, list(config["credentials"].values()))
            executor = Executor(packet, journal, config)
            executor.discover()
            for obs in packet["observations"]:
                for arm in obs["arm_order"]:
                    executor.remaining()
                    executor.operation(obs, arm)
            connection.send("finished")
        except CallFailure as error:
            connection.send("deadline" if error.code == "WALL_DEADLINE" else "child_failure")
        except BaseException:
            connection.send("child_failure")
        finally:
            if executor is not None:
                with contextlib.suppress(BaseException):
                    executor.close()
            if journal is not None:
                with contextlib.suppress(BaseException):
                    journal.close()
            connection.close()


def run_trial(
    packet_path: Path,
    journal_path: Path,
    sie_url: str,
    credentials: dict[str, str],
    *,
    execute: bool = False,
    provider_urls: dict[str, str] | None = None,
) -> str:
    require(execute, "Execution requires --execute")
    require(not journal_path.with_name(journal_path.name + ".end.json").exists(), "Run terminal record already exists")
    packet = load_packet(packet_path)
    sie_url, origin = endpoint(sie_url)
    selected_credentials = {"sie": credentials.get("sie", "")}
    selected_urls = {}
    provider_origins = {}
    if packet["config"]["arms"] == "paired":
        for provider in {RIVALS[s] for s in packet["config"]["stages"]}:
            require(bool(credentials.get(provider)), "A selected paired provider credential is missing")
            selected_credentials[provider] = credentials[provider]
            selected_urls[provider], provider_origins[provider] = endpoint((provider_urls or PROVIDER_URLS)[provider])
    try:
        sdk_version = importlib.metadata.version("sie-sdk")
    except importlib.metadata.PackageNotFoundError:
        raise ValueError("Install the workspace sie-sdk before execution") from None
    started = time.monotonic()
    deadline = started + packet["config"]["wall_limit_s"]
    journal = Journal(journal_path, packet, list(selected_credentials.values()), create=True)
    journal.append(
        "run_start",
        sie_origin=origin,
        provider_origins=provider_origins,
        placement_label=packet["config"]["placement_label"],
        sdk_version=sdk_version,
        timeout_budgets_s={
            "connect": packet["config"]["request_timeout_s"],
            "read": packet["config"]["request_timeout_s"],
            "provision": packet["config"]["request_timeout_s"],
        },
        wall_limit_s=packet["config"]["wall_limit_s"],
    )
    journal.close()
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe()
    process = context.Process(target=child_main, args=(child,))
    reason = "child_failure"
    try:
        process.start()
        child.close()
        # Credentials travel in a private pipe, never command-line arguments.
        parent.send(
            {
                "packet_path": str(packet_path.resolve()),
                "journal": str(journal_path.resolve()),
                "sie_url": sie_url,
                "credentials": selected_credentials,
                "provider_urls": selected_urls,
                "deadline": deadline,
            }
        )
        process.join(max(0, deadline - time.monotonic()))
        if process.is_alive():
            reason = "deadline"
        elif parent.poll():
            reason = parent.recv()
    except KeyboardInterrupt:
        reason = "interrupted"
    except Exception:
        reason = "child_failure"
    finally:
        if process.pid is not None:
            if process.is_alive():
                process.terminate()
                process.join(1)
            if process.is_alive():
                process.kill()
                process.join()
        parent.close()
        child.close()
        # A terminated write may leave a truncated final line. Preserve it and
        # append the parent's terminal record to a distinct, exclusive sidecar.
        end = {
            "packet_digest": packet["packet_digest"],
            "reason": reason,
            "protocol_digest": packet["protocol_digest"],
            "source_digest": packet["source_digest"],
            "journal_digest": sha256(journal_path.read_bytes()),
            "elapsed_s": time.monotonic() - started,
            "child_exitcode": process.exitcode,
        }
        write_exclusive(journal_path.with_name(journal_path.name + ".end.json"), {**end, "end_digest": digest(end)})
    return reason


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument(
        "--out", type=Path, required=True, help="new JSONL journal; creates a separate .end.json record"
    )
    parser.add_argument("--sie-base-url", default=None)
    parser.add_argument("--execute", action="store_true", help="send the frozen requests; may incur API charges")
    args = parser.parse_args()
    if not args.execute:
        parser.error("Use --execute to opt in to inference.")
    try:
        packet = load_packet(args.packet)
        require(
            not args.out.exists() and not args.out.with_name(args.out.name + ".end.json").exists(),
            "Run output already exists",
        )
        needed = {RIVALS[s] for s in packet["config"]["stages"]} if packet["config"]["arms"] == "paired" else set()
        keys = {"sie": os.environ.get("SIE_API_KEY", "")}
        for provider, variable in {
            "anthropic": "ANTHROPIC_API_KEY",
            "openai": "OPENAI_API_KEY",
            "cohere": "COHERE_API_KEY",
        }.items():
            if provider in needed:
                keys[provider] = os.environ.get(variable, "")
        reason = run_trial(
            args.packet, args.out, args.sie_base_url or os.environ.get("SIE_BASE_URL", ""), keys, execute=True
        )
    except Exception:
        raise SystemExit(
            "Execution could not start. Check the packet, fresh output, nonsecret endpoint and selected credentials."
        ) from None
    print(f"Run ended: {reason}. Score the preserved journal offline.")
    if reason != "finished":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
