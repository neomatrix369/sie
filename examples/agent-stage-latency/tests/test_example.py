"""Credential-free protocol, durability and accounting tests using the real SDK."""

from __future__ import annotations

import contextlib
import json
import multiprocessing
import os
import random
import struct
import subprocess
import sys
import threading
import time
from collections import Counter
from concurrent.futures import Future
from http.client import HTTPConnection
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import msgpack
import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

import fetch  # noqa: E402
import prepare  # noqa: E402
import protocol  # noqa: E402
import run  # noqa: E402
import score  # noqa: E402


def config(**changes: Any) -> dict[str, Any]:
    return {
        "n": 2,
        "seed": 19,
        "stages": list(protocol.STAGES),
        "arms": "paired",
        "phase": "confirmatory",
        "warmup": 0,
        "request_timeout_s": 3.0,
        "wall_limit_s": 15.0,
        "placement_label": "local-fixture",
        "bootstrap": {"seed": 17, "resamples": 100, "quantile": "linear-(n-1)p", "minimum_clusters": 2},
        **changes,
    }


def population(count: int = 8, *, long: bool = False) -> dict[str, list[dict[str, Any]]]:
    result = {}
    for stage in protocol.STAGES:
        cohort = next(
            e["sha256"]
            for e in protocol.sources()["files"]
            if e["path"].startswith(stage + "/") and not e["path"].endswith("manifest.json")
        )
        rows = []
        for index in range(count):
            text = f"Élodie {index} met Élodie."
            if long:
                text += " café-name " * 620 + "終わり"
            data = (
                {
                    "query": f"Question {index}",
                    "candidates": [
                        {"id": f"c{candidate}", "text": f"Candidate {candidate}"} for candidate in range(20)
                    ],
                }
                if stage == "R"
                else {"text": text}
            )
            rows.append(prepare.unit(stage, cohort, f"fixture:{index}", data))
        result[stage] = rows
    return result


def packet_file(tmp_path: Path, *, long: bool = False, **changes: Any) -> tuple[Path, dict[str, Any]]:
    packet = prepare.build_packet(population(long=long), config(**changes))
    path = tmp_path / "packet.json"
    fetch.write_exclusive(path, packet)
    assert prepare.load_packet(path) == packet
    return path, packet


def journal_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_bytes().splitlines() if line.endswith(b"}")]


def rechain(path: Path, rows: list[dict[str, Any]], *, total_elapsed_s: float | None = None) -> None:
    previous = None
    with path.open("wb") as output:
        for sequence, row in enumerate(rows):
            content = {k: v for k, v in row.items() if k != "entry_digest"} | {
                "sequence": sequence,
                "previous_digest": previous,
            }
            previous = protocol.digest(content)
            output.write(protocol.canonical({**content, "entry_digest": previous}) + b"\n")
    end_path = path.with_name(path.name + ".end.json")
    if end_path.exists():
        end = json.loads(end_path.read_bytes())
        if total_elapsed_s is not None:
            end["elapsed_s"] = total_elapsed_s
        end["journal_digest"] = protocol.sha256(path.read_bytes())
        end["end_digest"] = protocol.digest({k: v for k, v in end.items() if k != "end_digest"})
        end_path.write_bytes(protocol.canonical(end) + b"\n")


def diagnostic_journal(path: Path, packet: dict[str, Any]) -> run.Journal:
    journal = run.Journal(path, packet, [], create=True)
    settings = packet["config"]
    journal.append(
        "run_start",
        wall_limit_s=settings["wall_limit_s"],
        placement_label=settings["placement_label"],
        timeout_budgets_s={k: settings["request_timeout_s"] for k in ("connect", "read", "provision")},
    )
    journal.append("discovery_intent", call_id="metadata")
    journal.append("discovery_result", call_id="metadata", status="success", models=[], elapsed_s=0.001)
    return journal


def diagnostic_end(path: Path, packet: dict[str, Any], reason: str) -> None:
    end = {k: packet[k] for k in ("packet_digest", "protocol_digest", "source_digest")} | {
        "reason": reason,
        "journal_digest": protocol.sha256(path.read_bytes()),
        "elapsed_s": 1.0,
        "child_exitcode": -15 if reason == "interrupted" else 0,
    }
    fetch.write_exclusive(path.with_name(path.name + ".end.json"), end | {"end_digest": protocol.digest(end)})


def synthetic_executor(packet: dict[str, Any], journal: run.Journal, advance: Any = lambda _: None) -> run.Executor:
    """Exercise actual calls, hooks and composition without a socket or inference."""
    executor = object.__new__(run.Executor)
    executor.packet, executor.journal = packet, journal
    executor.config = {"deadline": time.monotonic() + 60}
    executor.clients, executor.evidence = {}, {}

    class Client:
        last_retry_count = 0
        last_model_revision = "a" * 64

        def __init__(self, model: str, hooks: run.TransportEvidence) -> None:
            self.model, self.hooks = model, hooks

        def invoke(self, reply: dict[str, Any]) -> dict[str, Any]:
            assert journal_rows(journal.path)[-1]["event"] == "operation_intent"
            self.hooks.request(None)
            advance(1)
            self.hooks.response(SimpleNamespace(status_code=200, headers={"X-SIE-Model-Revision": "a" * 64}))
            advance(2)
            return {"model": self.model, **reply}

        def encode(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
            return self.invoke({"dense": [0.125] * 2560})

        def extract(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
            return self.invoke({"entities": []})

        def close(self) -> None:
            pass

    for stage in packet["config"]["stages"]:
        for model in protocol.MODELS[stage]:
            hooks = run.TransportEvidence(journal)
            executor.evidence[model] = hooks
            executor.clients[model] = Client(model, hooks)
    return executor


def checkpoint_worker(packet_path: Path, journal_path: Path, connection: Any, checkpoint: str) -> None:
    """Pause at an acknowledged boundary so the parent can really terminate us."""
    packet = prepare.load_packet(packet_path)
    journal = run.Journal(journal_path, packet, [])
    executor = synthetic_executor(packet, journal)

    def pause() -> None:
        connection.send(checkpoint)
        connection.recv()

    original_capture, original_append = run.OperationEvents.append, journal.append

    def capture(events: run.OperationEvents, event: str, **fields: Any) -> None:
        original_capture(events, event, **fields)
        if checkpoint == "after_response" and event == "response":
            pause()

    def append(event: str, **fields: Any) -> None:
        if checkpoint == "partial_flush" and event == "response":
            journal.output.write(b'{"partial":')
            journal.output.flush()
            os.fsync(journal.output.fileno())
            pause()
        original_append(event, **fields)
        if (checkpoint, event) in {
            ("after_intent", "operation_intent"),
            ("mid_flush", "call_intent"),
            ("after_seal", "operation_result"),
        }:
            pause()

    original_result = Future.result

    def result(future: Any, *args: Any, **kwargs: Any) -> Any:
        value = original_result(future, *args, **kwargs)
        if checkpoint == "after_m_path":
            pause()
        return value

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(run.OperationEvents, "append", capture)
        patch.setattr(journal, "append", append)
        patch.setattr(Future, "result", result)
        if checkpoint == "before_flush":
            patch.setattr(run.OperationEvents, "flush", lambda *_: pause())
        executor.operation(packet["observations"][0], "sie")


class FakeServer:
    def __init__(
        self,
        journal: Path,
        *,
        delay: float = 0,
        fail_at: int | None = None,
        retry: bool = False,
        malformed: bool = False,
        chat_content: Any = "Safety: Safe",
        catalog: bool = True,
        catalog_status: int = 200,
        revision_headers: bool = True,
        loading_sleep: bool = False,
        response_model: str | None = "requested",
        malformed_ranking: bool = False,
    ) -> None:
        self.journal = journal
        self.delay = delay
        self.fail_at = fail_at
        self.retry = retry
        self.malformed = malformed
        self.chat_content = chat_content
        self.catalog = catalog
        self.catalog_status = catalog_status
        self.revision_headers = revision_headers
        self.loading_sleep = loading_sleep
        self.response_model = response_model
        self.malformed_ranking = malformed_ranking
        self.lock = threading.Lock()
        self.captures: list[dict[str, Any]] = []
        self.errors: list[str] = []
        self.active = 0
        self.maximum = 0
        self.counts: Counter[str] = Counter()
        self.observed = threading.Event()
        fixture = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, format: str, *args: Any) -> None:
                pass

            def reply(
                self, payload: Any, status: int = 200, *, native: bool = False, headers: dict[str, str] | None = None
            ) -> None:
                if isinstance(payload, dict) and "model" in payload and fixture.response_model != "requested":
                    if fixture.response_model is None:
                        payload.pop("model")
                    else:
                        payload["model"] = fixture.response_model
                body = msgpack.packb(payload, use_bin_type=True) if native else json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/msgpack" if native else "application/json")
                self.send_header("Content-Length", str(len(body)))
                for key, value in (headers or {}).items():
                    self.send_header(key, value)
                self.end_headers()
                with contextlib.suppress(BrokenPipeError, ConnectionResetError):
                    self.wfile.write(body)

            def do_GET(self) -> None:
                models = [
                    {"name": model, "revision": "e" * 40, "dims": 2560, "worker_host": "private-telemetry-sentinel"}
                    for group in protocol.MODELS.values()
                    for model in group
                ]
                self.reply({"models": models} if fixture.catalog else {"models": []}, fixture.catalog_status)

            def do_POST(self) -> None:
                body = self.rfile.read(int(self.headers["Content-Length"]))
                data = (
                    msgpack.unpackb(body, raw=False)
                    if self.headers.get("Content-Type") == "application/msgpack"
                    else json.loads(body)
                )
                model = self.path.split("/v1/extract/", 1)[-1] if "/v1/extract/" in self.path else data.get("model", "")
                with fixture.lock:
                    fixture.active += 1
                    fixture.maximum = max(fixture.maximum, fixture.active)
                    fixture.counts[self.path] += 1
                    ordinal = fixture.counts[self.path]
                    fixture.captures.append(
                        {
                            "path": self.path,
                            "body": data,
                            "model": model,
                            "auth": self.headers.get("Authorization"),
                            "api_key": self.headers.get("x-api-key"),
                            "port": self.client_address[1],
                        }
                    )
                    rows = journal_rows(fixture.journal)
                    if not rows or rows[-1]["event"] != "operation_intent":
                        fixture.errors.append("Server observed dispatch without the current durable operation intent")
                fixture.observed.set()
                try:
                    if fixture.delay:
                        time.sleep(fixture.delay)
                    if fixture.loading_sleep and self.path.startswith("/v1/encode/"):
                        self.reply(
                            {"detail": {"code": "MODEL_LOADING", "message": "loading"}},
                            503,
                            headers={"Retry-After": "10"},
                        )
                        return
                    if fixture.retry and ordinal == 1 and "NuNER_Zero" in self.path:
                        self.reply(
                            {"detail": {"code": "MODEL_LOADING", "message": "loading"}},
                            503,
                            headers={"Retry-After": "0.01"},
                        )
                        return
                    revision = "a" * 64 if "gliner" in self.path else "b" * 64
                    headers = {
                        "X-SIE-Model-Revision": revision,
                        "X-SIE-Execution-Identity-SHA256": "c" * 64,
                        "X-SIE-Execution-Binding-SHA256": "d" * 64,
                    }
                    if not fixture.revision_headers:
                        headers = {}
                    if self.path == "/v1/chat/completions":
                        text = "fake-auth-sentinel" if fixture.malformed else fixture.chat_content
                        self.reply(
                            {
                                "model": data["model"],
                                "choices": [
                                    {"message": {"role": "assistant", "content": text}, "finish_reason": "stop"}
                                ],
                            },
                            headers=headers,
                        )
                    elif self.path.startswith("/v1/extract/"):
                        text = data["items"][0]["text"]
                        at = text.find("Élodie")
                        entities = (
                            [{"text": "Élodie", "start": at, "end": at + 6, "label": "person", "score": 0.6}]
                            if at >= 0
                            else []
                        )
                        item: dict[str, Any] = {"entities": entities}
                        if fixture.fail_at is not None and "gliner" in self.path and ordinal == fixture.fail_at:
                            item = {
                                "entities": [],
                                "error": {"code": "INVALID_ARGUMENT", "message": "fake-auth-sentinel"},
                            }
                        self.reply({"model": model, "items": [item]}, native=True, headers=headers)
                    elif self.path.startswith("/v1/encode/"):
                        self.reply(
                            {
                                "model": protocol.MODELS["E"][0],
                                "items": [
                                    {
                                        "dense": {
                                            "dims": 2560,
                                            "values": {
                                                b"nd": True,
                                                b"type": "<f4",
                                                b"kind": b"",
                                                b"shape": [2560],
                                                b"data": struct.pack("<2560f", *([0.125] * 2560)),
                                            },
                                        }
                                    }
                                ],
                            },
                            native=True,
                            headers=headers,
                        )
                    elif self.path.startswith("/v1/score/"):
                        self.reply(
                            {
                                "model": protocol.MODELS["R"][0],
                                "scores": [
                                    {"item_id": item["id"], "score": 1 / (rank + 1), "rank": rank + 1}
                                    for rank, item in enumerate(reversed(data["items"]))
                                ],
                            },
                            native=True,
                            headers=headers,
                        )
                    elif self.path == "/v1/messages":
                        text = (
                            "unharmful"
                            if data["max_tokens"] == 10
                            else json.dumps({"entities": [{"text": "Élodie", "type": "person"}]})
                        )
                        self.reply(
                            {
                                "model": "claude-haiku-4-5",
                                "content": [{"type": "text", "text": text}],
                                "stop_reason": "end_turn",
                            }
                        )
                    elif self.path == "/v1/embeddings":
                        self.reply({"model": data["model"], "data": [{"index": 0, "embedding": [0.25] * 3072}]})
                    elif self.path == "/v2/rerank":
                        self.reply(
                            {
                                "results": [
                                    {"index": index, "relevance_score": index / 20} for index in reversed(range(20))
                                ]
                                + ([None] if fixture.malformed_ranking else [])
                            }
                        )
                    else:
                        fixture.errors.append("Unplanned endpoint")
                        self.reply({}, 404)
                finally:
                    with fixture.lock:
                        fixture.active -= 1

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self) -> FakeServer:
        self.thread.start()
        return self

    def __exit__(self, *args: Any) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()


def tiny_catalog(tmp_path: Path) -> tuple[Path, dict[str, Any]]:
    bodies = {
        "G/toxicchat.csv": b"fixture",
        "G/aegis.json": b"[]",
        "M/gretel-main.jsonl": b"{}\n",
        "E/questions.json": b"{}",
        "R/cases_test.json": b"{}",
    }
    for stage, task in {
        "G": "guardrails",
        "M": "redact",
        "E": "lookalike-search",
        "R": "rerank-relevance-rules",
    }.items():
        entries = {
            "inputs/" + name.split("/", 1)[1]: protocol.sha256(body)
            for name, body in bodies.items()
            if name.startswith(stage + "/")
        }
        bodies[f"{stage}/manifest.json"] = protocol.canonical({"task": task, "files_sha256": entries})
    catalog = {
        "files": [
            {"path": name, "bytes": len(body), "sha256": protocol.sha256(body), "url": (tmp_path / name).as_uri()}
            for name, body in sorted(bodies.items())
        ]
    }
    for name, body in bodies.items():
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(body)
    return tmp_path, catalog


def test_fetch_allowlist_verification_and_exclusive_staging(tmp_path: Path) -> None:
    source, catalog = tiny_catalog(tmp_path / "source")
    output = tmp_path / "downloaded"
    fetch.fetch_inputs(output, catalog)
    assert len(fetch.verify_inputs(output, catalog)) == 9
    with pytest.raises(FileExistsError):
        fetch.fetch_inputs(output, catalog)
    (output / "M/gretel-main.jsonl").write_bytes(b"tamper")
    with pytest.raises(ValueError, match="digest"):
        fetch.verify_inputs(output, catalog)
    (source / "E/questions.json").unlink()
    with pytest.raises(OSError):
        fetch.fetch_inputs(tmp_path / "failed", catalog)
    assert not (tmp_path / "failed").exists()


def inode(path: Path) -> tuple[int, int]:
    status = path.stat()
    return status.st_dev, status.st_ino


def record_syncs(monkeypatch: pytest.MonkeyPatch) -> list[tuple[int, int]]:
    synced = []
    original = os.fsync

    def record(descriptor: int) -> None:
        original(descriptor)
        status = os.fstat(descriptor)
        synced.append((status.st_dev, status.st_ino))

    monkeypatch.setattr(os, "fsync", record)
    return synced


@pytest.mark.skipif(os.name != "posix", reason="Directory fsync is a POSIX persistence step")
@pytest.mark.parametrize("journal", [False, True])
def test_output_creation_syncs_file_and_new_parent_entries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, journal: bool
) -> None:
    synced = record_syncs(monkeypatch)
    path = tmp_path / "new-parent" / "leaf" / "output.jsonl"
    if journal:
        output = run.Journal(path, {}, [], create=True)
        output.close()
    else:
        fetch.write_exclusive(path, {"fixture": True})
    if inode(path.parent) not in synced:
        pytest.skip("This filesystem does not support directory fsync")
    file_sync = synced.index(inode(path))
    assert inode(path.parent) in synced[file_sync + 1 :]
    assert all(inode(parent) in synced for parent in (tmp_path, path.parent.parent, path.parent))
    with pytest.raises(FileExistsError):
        if journal:
            run.Journal(path, {}, [], create=True)
        else:
            fetch.write_exclusive(path, {})


@pytest.mark.skipif(os.name != "posix", reason="Directory fsync is a POSIX persistence step")
def test_fetch_syncs_data_and_renames_before_publishing_marker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _, catalog = tiny_catalog(tmp_path / "source")
    output = tmp_path / "downloaded"
    synced = record_syncs(monkeypatch)
    rename = Path.rename
    last_data_rename = 0
    published = False

    def checked_rename(path: Path, target: Path) -> Path:
        nonlocal last_data_rename, published
        if path.name == "verified.json":
            if inode(output) not in synced:
                pytest.skip("This filesystem does not support directory fsync")
            for entry in catalog["files"]:
                data = output / entry["path"]
                file_sync = synced.index(inode(data))
                assert inode(data.parent) in synced[file_sync + 1 :]
            assert inode(path) in synced
            assert inode(output) in synced[last_data_rename:]
            assert inode(path.parent) in synced[last_data_rename:]
            published = True
        result = rename(path, target)
        if not published:
            last_data_rename = len(synced)
        return result

    monkeypatch.setattr(Path, "rename", checked_rename)
    fetch.fetch_inputs(output, catalog)
    assert published and len(fetch.verify_inputs(output, catalog)) == 9
    assert synced[-1] == inode(output)


@pytest.mark.parametrize("path", ["../escape", "/absolute", "a/../b", "a\\b", "a//b", "a/./b"])
def test_unsafe_paths(path: str) -> None:
    with pytest.raises(ValueError):
        protocol.safe_path(path)


@pytest.mark.parametrize("change", ["wrong_task", "unsafe_manifest", "missing_file", "wrong_source"])
def test_input_source_and_manifest_rejection(tmp_path: Path, change: str) -> None:
    root, catalog = tiny_catalog(tmp_path)
    if change in ("wrong_task", "unsafe_manifest"):
        path = root / "M/manifest.json"
        manifest = json.loads(path.read_bytes())
        if change == "wrong_task":
            manifest["task"] = "different"
        else:
            manifest["files_sha256"]["../outside"] = "0" * 64
        body = protocol.canonical(manifest)
        path.write_bytes(body)
        entry = next(e for e in catalog["files"] if e["path"] == "M/manifest.json")
        entry.update(bytes=len(body), sha256=protocol.sha256(body))
    if change == "missing_file":
        (root / "E/questions.json").unlink()
    fetch.write_exclusive(
        root / "verified.json", {"source_digest": "wrong" if change == "wrong_source" else protocol.digest(catalog)}
    )
    with pytest.raises((ValueError, FileNotFoundError)):
        fetch.verify_inputs(root, catalog)


def test_packet_determinism_phase_exclusion_counts_and_tampering(tmp_path: Path) -> None:
    settings = config(warmup=1)
    pilot = prepare.build_packet(population(), settings | {"phase": "pilot"})
    trial = prepare.build_packet(population(), settings, [pilot])
    assert protocol.canonical(trial) == protocol.canonical(prepare.build_packet(population(), settings, [pilot]))
    assert {o["base_id"] for o in pilot["observations"]}.isdisjoint(o["base_id"] for o in trial["observations"])
    assert {(o["stage"], o["unit"]["input_digest"]) for o in pilot["observations"]}.isdisjoint(
        (o["stage"], o["unit"]["input_digest"]) for o in trial["observations"]
    )
    assert trial["planned_semantic_calls"] == {
        "G": {"sie": 3, "rival": 3},
        "M": {"sie": 6, "rival": 3},
        "E": {"sie": 3, "rival": 3},
        "R": {"sie": 6, "rival": 6},
    }
    assert {tuple(o["arm_order"]) for o in trial["observations"]} == {("sie", "rival"), ("rival", "sie")}
    with pytest.raises(ValueError, match="remaining"):
        prepare.build_packet(population(4), settings, [pilot])
    path = tmp_path / "packet.json"
    fetch.write_exclusive(path, trial)
    assert prepare.load_packet(path) == trial
    trial["observations"][0]["unit"]["data"]["text"] += " tampered"
    path.write_bytes(protocol.canonical(trial))
    with pytest.raises(ValueError, match="digest"):
        prepare.load_packet(path)
    with pytest.raises(FileExistsError):
        fetch.write_exclusive(path, {})


def test_semantic_input_duplicates_cannot_enter_different_phases() -> None:
    rows = population(3)
    duplicate = dict(rows["G"][0])
    duplicate = prepare.unit("G", duplicate["cohort_sha256"], "different-id", duplicate["data"])
    rows["G"].append(duplicate)
    with pytest.raises(ValueError, match="remaining"):
        prepare.build_packet(rows, config(stages=["G"], n=3, warmup=1))


def test_exact_four_stage_request_bindings() -> None:
    data = population(1)
    guard = protocol.requests("G", data["G"][0]["data"], "sie")[0]
    rival = protocol.requests("G", data["G"][0]["data"], "rival")[0]
    assert guard["method"] == "chat_completions" and guard["body"] == {
        "messages": [{"role": "user", "content": "Élodie 0 met Élodie."}],
        "temperature": 0,
        "max_tokens": 64,
    }
    assert (
        rival["body"]["messages"][0]["content"]
        == "Is the following user message harmful? Answer with exactly one word: harmful or unharmful."
        "\n\nMessage:\nÉlodie 0 met Élodie."
    )
    assert rival["body"]["max_tokens"] == 10 and "system" not in rival["body"] and "thinking" not in rival["body"]
    pii = protocol.requests("M", data["M"][0]["data"], "rival")[0]["body"]
    assert pii["max_tokens"] == 4096 and pii["system"] == protocol.PII_PROMPT.format(
        labels=", ".join(protocol.REQUEST_LABELS)
    )
    assert pii["output_config"]["format"]["schema"] == protocol.SCHEMA and len(protocol.REQUEST_LABELS) == 36
    embed = protocol.requests("E", data["E"][0]["data"], "rival")[0]["body"]
    assert embed == {
        "model": "text-embedding-3-large",
        "input": "Élodie 0 met Élodie.",
        "dimensions": 3072,
        "encoding_format": "float",
    }
    for rule in protocol.RULES:
        sie = protocol.requests("R", data["R"][0]["data"], "sie", rule)[0]
        cohere = protocol.requests("R", data["R"][0]["data"], "rival", rule)[0]
        assert [c["id"] for c in sie["body"]["items"]] == [f"c{i}" for i in range(20)]
        assert sie["body"]["instruction"] == protocol.RULES[rule]
        assert (
            cohere["path"] == "/v2/rerank"
            and cohere["body"]["query"] == f"Instruction: {protocol.RULES[rule]}\nQuery: Question 0"
        )
        assert cohere["body"]["top_n"] == 20 and cohere["body"]["max_tokens_per_doc"] == 4096


def test_unicode_windows_composition_and_masking() -> None:
    text = "Élodie étudie. Élodie, élodie and Élodie2."
    span = {"start": 0, "end": 6, "label": "person", "score": 0.6}
    overlap = {"start": 0, "end": 13, "label": "street address", "score": 0.7}
    spans = protocol.compose(
        text, [[span], [span | {"score": 0.9}, overlap, {"start": 7, "end": 13, "label": "city", "score": 0.599}]]
    )
    assert len(spans) == 4 and spans[0]["score"] == 0.9
    assert protocol.masked(text, spans) == "[PERSON]. [PERSON], élodie and [PERSON]2."
    long = " " + "café-name " * 620 + "終わり  "
    ranges = protocol.windows(long)
    assert len(ranges) == 3 and ranges[0][0] == 1
    words = list(protocol.WORD.finditer(long))
    assert ranges == [
        (words[0].start(), words[299].end()),
        (words[250].start(), words[549].end()),
        (words[500].start(), words[-1].end()),
    ]
    calls = protocol.requests("M", {"text": long}, "sie")
    assert len(calls) == 6 and {c["model"] for c in calls} == set(protocol.MODELS["M"])
    assert [c["offset"] for c in calls[:3]] == [a for a, _ in ranges]
    entity = {"start": 0, "end": 4, "label": "city", "score": 0.6, "text": "café"}
    adjusted = protocol.validate_reply("M", "sie", {"entities": [entity]}, {"text": long}, calls[1])[0]
    assert adjusted["start"] == ranges[1][0] and long[adjusted["start"] : adjusted["end"]] == "café"
    llm, unmatched = protocol.llm_spans(
        text, {"entities": [{"text": "Élodie", "type": "person"}, {"text": "absent", "type": "person"}]}
    )
    assert len(llm) == 3 and unmatched == 1


def test_real_sdk_and_provider_adapters_durable_paired_flow(tmp_path: Path) -> None:
    path, packet = packet_file(tmp_path)
    journal = tmp_path / "run.jsonl"
    keys = {provider: "fake-auth-sentinel" for provider in ("sie", "anthropic", "openai", "cohere")}
    with FakeServer(journal, retry=True) as server:
        assert (
            run.run_trial(
                path,
                journal,
                server.url,
                keys,
                execute=True,
                provider_urls={p: server.url for p in protocol.PROVIDER_URLS},
            )
            == "finished"
        )
        assert not server.errors and 1 <= server.maximum <= 2
        captures = server.captures
    report = score.score_trial(path, journal)
    assert report["complete"] and report["qualifying_confirmatory"]
    assert all(stage["paired_latency"]["status"] == "estimated" for stage in report["stages"].values())
    assert report["stages"]["R"]["paired_latency"]["base_clusters"] == 2
    assert report["stages"]["R"]["paired_latency"]["pairs"] == 4
    rows = journal_rows(journal)
    assert "fake-auth-sentinel" not in journal.read_text() and "private-telemetry-sentinel" not in journal.read_text()
    assert {m["weights_revision"] for m in report["discovery"]["models"]} == {"e" * 40}
    model_replies = [r for r in rows if r["event"] == "call_result" and r["stage"] == "M" and r["arm"] == "sie"]
    assert all(r["execution_revision"] == ("a" * 64 if "gliner" in r["model"] else "b" * 64) for r in model_replies)
    assert sum(r["sdk_retry_count"] for r in model_replies) == 1
    retried = next(r for r in model_replies if r["sdk_retry_count"] == 1)
    assert [r["status"] for r in rows if r["event"] == "response" and r["call_id"] == retried["call_id"]] == [
        503,
        200,
    ]
    assert report["stages"]["M"]["accounting"]["sie"]["physical_dispatches"] == 5
    assert all(c["auth"] == "Bearer fake-auth-sentinel" for c in captures if c["path"] != "/v1/messages")
    assert all(c["api_key"] == "fake-auth-sentinel" for c in captures if c["path"] == "/v1/messages")
    extract_ports = {model: {c["port"] for c in captures if c["model"] == model} for model in protocol.MODELS["M"]}
    assert extract_ports[protocol.MODELS["M"][0]].isdisjoint(extract_ports[protocol.MODELS["M"][1]])
    for capture in captures:
        route, body = capture["path"], capture["body"]
        if route.startswith("/v1/encode/"):
            assert len(body["items"]) == 1 and body["params"]["options"]["is_query"] is True
        if route == "/v1/chat/completions":
            assert body["max_tokens"] == 64 and body["temperature"] == 0
            assert len(body["messages"]) == 1 and body["messages"][0]["role"] == "user"
        if route.startswith("/v1/extract/"):
            assert body["params"]["labels"] == protocol.REQUEST_LABELS
        if route.startswith("/v1/score/"):
            assert len(body["items"]) == 20 and [i["id"] for i in body["items"]] == [f"c{i}" for i in range(20)]
        if route in ("/v1/messages", "/v1/embeddings", "/v2/rerank"):
            assert any(
                c["body"] == body and c["path"] == route
                for obs in packet["observations"]
                for c in obs["requests"]["rival"]
            )
    preserved = journal.read_bytes()
    with pytest.raises((FileExistsError, ValueError)):
        run.run_trial(path, journal, "http://127.0.0.1:1", keys, execute=True)
    assert journal.read_bytes() == preserved


def test_long_m_partial_success_and_no_outer_retry(tmp_path: Path) -> None:
    path, packet = packet_file(tmp_path, long=True, stages=["M"], n=1, arms="sie")
    journal = tmp_path / "partial.jsonl"
    with FakeServer(journal, fail_at=2, delay=0.01) as server:
        assert run.run_trial(path, journal, server.url, {}, execute=True) == "finished"
        assert server.maximum == 2 and not server.errors
        assert server.counts["/v1/extract/urchade/gliner_multi_pii-v1"] == 2
        assert server.counts["/v1/extract/numind/NuNER_Zero"] == 3
    rows = journal_rows(journal)
    earlier = [r for r in rows if r["event"] == "call_result"]
    assert len(earlier) == 5 and sum(r["status"] == "failed" for r in earlier) == 1
    failed = next(r for r in earlier if r["status"] == "failed")
    assert failed["error"]["code"] == "ITEM_ERROR" and failed["reply"]["item_error"]
    assert packet["planned_semantic_calls"]["M"]["sie"] == 6
    report = score.score_trial(path, journal)
    arm = report["stages"]["M"]["accounting"]["sie"]
    assert not report["complete"] and arm["planned"] == arm["failed"] == 1
    assert arm["semantic_calls"]["unattempted"] == 1 and arm["semantic_calls"]["successful"] == 4
    assert arm["semantic_calls"]["attempt_status_unknown"] == 0 and arm["semantic_calls"]["attempted_count_exact"]
    assert arm["physical_dispatches_exact"] and arm["sdk_retries_exact"]


@pytest.mark.parametrize("with_auth,catalog_status", [(False, 200), (True, 200), (False, 404), (False, 501)])
def test_optional_auth_absent_catalog_and_sie_only(tmp_path: Path, with_auth: bool, catalog_status: int) -> None:
    path, _ = packet_file(tmp_path, stages=["E"], n=1, arms="sie")
    journal = tmp_path / "local.jsonl"
    keys = {"sie": "fake-auth-sentinel"} if with_auth else {}
    with FakeServer(
        journal, catalog=False, catalog_status=catalog_status, revision_headers=False, response_model=None
    ) as server:
        assert run.run_trial(path, journal, server.url, keys, execute=True) == "finished"
        assert server.captures[0]["auth"] == ("Bearer fake-auth-sentinel" if with_auth else None)
        assert len(server.captures) == 1
    report = score.score_trial(path, journal)
    assert report["complete"] and report["discovery_attempted"] and not report["discovery_unresolved"]
    if catalog_status == 200:
        assert report["discovery"]["status"] == "success" and report["discovery"]["models"] == []
    else:
        assert report["discovery"]["status"] == "failed" and report["discovery"]["error"]
    assert report["stages"]["E"]["accounting"]["sie"]["successful"] == 1
    assert next(r for r in journal_rows(journal) if r["event"] == "call_result")["execution_revision"] is None
    assert report["stages"]["E"]["paired_latency"]["status"] == "sie_only"


def test_caller_mask_is_inside_operation_timer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path, packet = packet_file(tmp_path, stages=["M"], n=1, arms="sie")
    journal_path = tmp_path / "timer.jsonl"
    original = run.masked

    def delayed_mask(text: str, spans: list[dict[str, Any]]) -> str:
        time.sleep(0.04)
        return original(text, spans)

    monkeypatch.setattr(run, "masked", delayed_mask)
    journal = run.Journal(journal_path, packet, [], create=True)
    with FakeServer(journal_path) as server:
        executor = run.Executor(
            packet,
            journal,
            {"sie_url": server.url, "credentials": {}, "provider_urls": {}, "deadline": time.monotonic() + 5},
        )
        executor.operation(packet["observations"][0], "sie")
        executor.close()
    journal.close()
    rows = journal_rows(journal_path)
    operation = next(r for r in rows if r["event"] == "operation_result")
    calls = [r for r in rows if r["event"] == "call_result"]
    assert operation["status"] == "success" and operation["elapsed_s"] >= max(r["elapsed_s"] for r in calls) + 0.04


def test_request_timeout_retains_failed_duration(tmp_path: Path) -> None:
    path, _ = packet_file(tmp_path, stages=["E"], n=1, arms="sie", request_timeout_s=0.1)
    journal = tmp_path / "timeout.jsonl"
    with FakeServer(journal, delay=0.4) as server:
        assert run.run_trial(path, journal, server.url, {}, execute=True) == "finished"
        assert len(server.captures) == 1
    report = score.score_trial(path, journal)
    arm = report["stages"]["E"]["accounting"]["sie"]
    assert arm["planned"] == arm["attempted"] == arm["failed"] == 1
    assert arm["all_terminal_attempt_elapsed"]["p50_s"] >= 0.1
    assert arm["successful_operation_latency"]["count"] == 0


@pytest.mark.parametrize("loading_sleep", [False, True])
def test_wall_deadline_reaps_child_and_preserves_unresolved_intent(tmp_path: Path, loading_sleep: bool) -> None:
    path, _ = packet_file(tmp_path, stages=["E"], n=1, arms="sie", wall_limit_s=1.5)
    journal = tmp_path / "deadline.jsonl"
    before = {p.pid for p in multiprocessing.active_children()}
    with FakeServer(journal, delay=0 if loading_sleep else 3, loading_sleep=loading_sleep) as server:
        start = time.monotonic()
        assert run.run_trial(path, journal, server.url, {}, execute=True) == "deadline"
        assert time.monotonic() - start < 2.5 and server.observed.is_set()
        actual_dispatches = len(server.captures)
    assert {p.pid for p in multiprocessing.active_children()} == before
    report = score.score_trial(path, journal)
    assert not report["complete"] and report["ending"]["reason"] == "deadline"
    arm = report["stages"]["E"]["accounting"]["sie"]
    assert arm["unresolved"] + arm["failed"] == 1
    calls = arm["semantic_calls"]
    assert calls["failed"] + calls["unresolved"] + calls["attempt_status_unknown"] == 1
    assert calls["successful"] == 0
    assert calls["unattempted"] == 0
    assert calls["attempted"] + calls["attempt_status_unknown"] == 1
    assert arm["physical_dispatches"] <= actual_dispatches
    assert arm["physical_dispatches_exact"] == arm["sdk_retries_exact"] == (arm["unresolved"] == 0)
    if arm["physical_dispatches_exact"]:
        assert arm["physical_dispatches"] == actual_dispatches
    if not calls["attempted"]:
        assert calls["attempt_status_unknown"] == 1 and arm["physical_dispatches"] == arm["sdk_retries"] == 0


@pytest.mark.parametrize("code,expected", [("WALL_DEADLINE", "deadline"), ("CALL_FAILED", "child_failure")])
def test_child_terminal_signal_wins_before_parent_deadline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, code: str, expected: str
) -> None:
    path, _ = packet_file(tmp_path, stages=["E"], n=1, arms="sie")
    journal = tmp_path / "child-terminal.jsonl"
    checked = []

    class ExpiredExecutor(run.Executor):
        def __init__(self, packet: Any, journal: Any, settings: dict[str, Any]) -> None:
            self.config = settings | {"deadline": time.monotonic() - 1}

        def discover(self) -> None:
            pass

        def remaining(self) -> float:
            checked.append(code)
            if code == "WALL_DEADLINE":
                return super().remaining()
            raise run.CallFailure(code)

        def close(self) -> None:
            pass

    class Connection:
        peer: Connection

        def __init__(self) -> None:
            self.messages: list[Any] = []
            self.received: list[Any] = []

        def recv(self) -> Any:
            message = self.messages.pop(0)
            self.received.append(message)
            return message

        def poll(self) -> bool:
            return bool(self.messages)

        def send(self, message: Any) -> None:
            self.peer.messages.append(message)

        def close(self) -> None:
            pass

    parent, child = Connection(), Connection()
    parent.peer, child.peer = child, parent

    # Complete the real child function during join, before the parent's live
    # budget expires, so terminal classification is independent of scheduling.
    class InlineProcess:
        pid = 1
        exitcode: int | None = None

        def __init__(self, target: Any, args: tuple[Any, ...]) -> None:
            self.target, self.args = target, args

        def start(self) -> None:
            pass

        def join(self, timeout: float | None = None) -> None:
            assert timeout is not None and timeout > 0
            self.target(*self.args)
            self.exitcode = 0

        def is_alive(self) -> bool:
            return self.exitcode is None

    class Context:
        Process = InlineProcess

        @staticmethod
        def Pipe() -> tuple[Connection, Connection]:
            return parent, child

    monkeypatch.setattr(run, "Executor", ExpiredExecutor)
    monkeypatch.setattr(run.multiprocessing, "get_context", lambda _: Context())
    assert run.run_trial(path, journal, "http://127.0.0.1:8000", {}, execute=True) == expected
    assert checked == [code] and parent.received == [expected]
    ending = json.loads(journal.with_name(journal.name + ".end.json").read_bytes())
    assert ending["reason"] == expected and ending["child_exitcode"] == 0
    assert ending["elapsed_s"] < 15
    report = score.score_trial(path, journal)
    assert not report["complete"] and report["stages"]["E"]["accounting"]["sie"]["unattempted"] == 1


def test_interrupt_preserves_intent_and_reaps_child(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path, _ = packet_file(tmp_path, stages=["E"], n=1, arms="sie")
    journal = tmp_path / "interrupt.jsonl"
    real_context = multiprocessing.get_context("spawn")
    before = {p.pid for p in multiprocessing.active_children()}
    with FakeServer(journal, delay=3) as server:

        class ProcessProxy:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                self.process = real_context.Process(*args, **kwargs)
                self.first_join = True

            def __getattr__(self, name: str) -> Any:
                return getattr(self.process, name)

            def join(self, timeout: float | None = None) -> None:
                if self.first_join:
                    self.first_join = False
                    assert server.observed.wait(3)
                    raise KeyboardInterrupt
                self.process.join(timeout)

        class ContextProxy:
            Process = ProcessProxy
            Pipe = real_context.Pipe

        monkeypatch.setattr(run.multiprocessing, "get_context", lambda _: ContextProxy())
        assert run.run_trial(path, journal, server.url, {}, execute=True) == "interrupted"
    assert {p.pid for p in multiprocessing.active_children()} == before
    report = score.score_trial(path, journal)
    assert not report["complete"] and report["ending"]["reason"] == "interrupted"
    calls = report["stages"]["E"]["accounting"]["sie"]["semantic_calls"]
    assert calls["attempt_status_unknown"] == 1 and calls["unresolved"] == calls["unattempted"] == 0


def test_truncated_tail_and_terminal_binding(tmp_path: Path) -> None:
    path, _ = packet_file(tmp_path, stages=["E"], n=1, arms="sie", wall_limit_s=1.5)
    journal = tmp_path / "tail.jsonl"
    with FakeServer(journal, delay=3) as server:
        assert run.run_trial(path, journal, server.url, {}, execute=True) == "deadline"
    with journal.open("ab") as output:
        output.write(b'{"partial":')
    end_path = journal.with_name(journal.name + ".end.json")
    end = json.loads(end_path.read_bytes())
    end["journal_digest"] = protocol.sha256(journal.read_bytes())
    end["end_digest"] = protocol.digest({k: v for k, v in end.items() if k != "end_digest"})
    end_path.write_bytes(protocol.canonical(end))
    report = score.score_trial(path, journal)
    assert not report["complete"] and report["truncated_tail_sha256"] == protocol.sha256(b'{"partial":')
    assert report["stages"]["E"]["accounting"]["sie"]["unresolved"] == 1
    end["journal_digest"] = "0" * 64
    end["end_digest"] = protocol.digest({k: v for k, v in end.items() if k != "end_digest"})
    end_path.write_bytes(protocol.canonical(end))
    with pytest.raises(ValueError, match="binding"):
        score.score_trial(path, journal)


def test_malformed_reply_keeps_actual_redacted_reply(tmp_path: Path) -> None:
    path, _ = packet_file(tmp_path, stages=["G"], n=1, arms="sie")
    journal = tmp_path / "malformed.jsonl"
    with FakeServer(journal, malformed=True) as server:
        run.run_trial(path, journal, server.url, {"sie": "fake-auth-sentinel"}, execute=True)
    text = journal.read_text()
    assert "fake-auth-sentinel" not in text and "[REDACTED_CREDENTIAL]" in text
    report = score.score_trial(path, journal)
    assert report["stages"]["G"]["accounting"]["sie"]["failed"] == 1
    assert (
        next(r for r in journal_rows(journal) if r["event"] == "call_result")["reply"]["text"]
        == "[REDACTED_CREDENTIAL]"
    )


def test_malformed_reply_redacts_nested_keys_and_preserves_collisions(tmp_path: Path) -> None:
    secret = "fake-auth-sentinel"
    marker = "[REDACTED_CREDENTIAL]"
    content = {
        secret: [{f"nested-{secret}": secret}],
        marker: "second entry",
        marker + "#2": "third entry",
    }
    path, _ = packet_file(tmp_path, stages=["G"], n=1, arms="sie")
    journal = tmp_path / "malformed-keys.jsonl"
    with FakeServer(journal, chat_content=content) as server:
        assert run.run_trial(path, journal, server.url, {"sie": secret}, execute=True) == "finished"
        assert not server.errors and len(server.captures) == 1
    assert secret not in journal.read_text() and marker in journal.read_text()
    terminal = next(r for r in journal_rows(journal) if r["event"] == "call_result")
    assert terminal["status"] == "failed" and terminal["error"]["code"] == "MALFORMED_REPLY"
    assert terminal["reply"]["text"] == {
        marker: [{"nested-" + marker: marker}],
        marker + "#2": "second entry",
        marker + "#2#2": "third entry",
    }
    report = score.score_trial(path, journal)
    assert secret not in protocol.canonical(report).decode()
    counts = report["stages"]["G"]["accounting"]["sie"]
    assert counts["planned"] == counts["attempted"] == counts["failed"] == 1
    assert counts["successful"] == counts["unattempted"] == counts["attempt_status_unknown"] == 0
    assert counts["semantic_calls"]["failed"] == counts["physical_dispatches"] == 1
    assert counts["sdk_retries"] == 0 and counts["physical_dispatches_exact"]


@pytest.mark.parametrize(
    "keys",
    [
        ["first-secret", "second-secret", "[REDACTED_CREDENTIAL]"],
        ["[REDACTED_CREDENTIAL]", "second-secret", "first-secret"],
        ["first-secret", "[REDACTED_CREDENTIAL]#2", "[REDACTED_CREDENTIAL]"],
    ],
)
def test_redacted_key_collisions_keep_all_values_and_are_stable(keys: list[str]) -> None:
    raw = dict(zip(keys, (1, 2, 3), strict=True))
    credentials = ["first-secret", "second-secret"]
    redacted = run.sanitize(raw, credentials)
    assert len(redacted) == 3 and sorted(redacted.values()) == [1, 2, 3]
    assert all(secret not in protocol.canonical(redacted).decode() for secret in credentials)
    assert run.sanitize(raw, credentials) == run.sanitize(redacted, credentials) == redacted
    assert list(raw) == keys


@pytest.mark.parametrize(
    "url",
    [
        "http://user:query-secret@localhost",
        "http://localhost?token=query-secret",
        "http://localhost#query-secret",
        "http://localhost/%71uery-secret",
    ],
)
def test_endpoint_secret_rejection_before_client_or_journal(tmp_path: Path, url: str) -> None:
    path, _ = packet_file(tmp_path, stages=["E"], n=1, arms="sie")
    output = tmp_path / "unsafe.jsonl"
    with pytest.raises(ValueError) as error:
        run.run_trial(path, output, url, {"sie": "fake-auth-sentinel"}, execute=True)
    assert "query-secret" not in str(error.value) and "fake-auth-sentinel" not in str(error.value)
    assert not output.exists()


def test_opt_in_missing_paired_keys_and_transport_sanitizing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path, _ = packet_file(tmp_path, stages=["E"], n=1)
    output = tmp_path / "run.jsonl"
    with pytest.raises(ValueError, match="execute"):
        run.run_trial(path, output, "http://127.0.0.1:1", {})
    with pytest.raises(ValueError, match="credential"):
        run.run_trial(path, output, "http://127.0.0.1:1", {}, execute=True)
    assert not output.exists()
    assert (
        run.run_trial(
            path,
            output,
            "http://127.0.0.1:1",
            {"sie": "fake-auth-sentinel", "openai": "fake-openai-sentinel"},
            execute=True,
            provider_urls={"openai": "http://127.0.0.1:1"},
        )
        == "finished"
    )
    assert all(value not in output.read_text() for value in ("fake-auth-sentinel", "fake-openai-sentinel"))
    captured = capsys.readouterr()
    assert "sentinel" not in captured.out + captured.err
    report = score.score_trial(path, output)
    assert report["stages"]["E"]["accounting"]["sie"]["failed"] == 1


@pytest.mark.parametrize(
    "change",
    [
        "missing",
        "duplicate",
        "unexpected",
        "protocol",
        "derived_mask",
        "all_503",
        "missing_discovery",
        "unresolved_discovery",
        "late_discovery",
    ],
)
def test_strict_journal_rejects_unexpected_or_nonqualifies_missing(tmp_path: Path, change: str) -> None:
    path, _ = packet_file(tmp_path, stages=["M"], n=1, arms="sie")
    journal = tmp_path / "journal.jsonl"
    with FakeServer(journal) as server:
        run.run_trial(path, journal, server.url, {}, execute=True)
    rows = journal_rows(journal)
    index = next(i for i, row in enumerate(rows) if row["event"] == "operation_result")
    if change == "missing":
        rows.pop(index)
    elif change == "duplicate":
        rows.insert(index + 1, dict(rows[index]))
    elif change == "unexpected":
        rows[index]["observation_id"] = "unexpected"
    elif change == "protocol":
        rows[index]["protocol_digest"] = "0" * 64
    elif change == "derived_mask":
        rows[index]["result"]["masked"] = "wrong mask"
    elif change == "all_503":
        for row in rows:
            if row["event"] == "response" and row["call_id"] != "metadata":
                row["status"] = 503
    elif change == "missing_discovery":
        rows = [row for row in rows if row.get("call_id") != "metadata"]
    elif change == "unresolved_discovery":
        rows = [row for row in rows if row["event"] != "discovery_result"]
    else:
        discovery = next(row for row in rows if row["event"] == "discovery_result")
        rows.remove(discovery)
        first_operation = next(i for i, row in enumerate(rows) if row["event"] == "operation_intent")
        rows.insert(first_operation + 1, discovery)
    rechain(journal, rows)
    if change == "missing":
        result = score.score_trial(path, journal)
        assert not result["complete"] and result["stages"]["M"]["accounting"]["sie"]["unresolved"] == 1
    else:
        with pytest.raises(ValueError):
            score.score_trial(path, journal)


def test_interrupted_discovery_remains_unresolved(tmp_path: Path) -> None:
    path, _ = packet_file(tmp_path, stages=["E"], n=1, arms="sie")
    journal = tmp_path / "discovery.jsonl"
    with FakeServer(journal) as server:
        run.run_trial(path, journal, server.url, {}, execute=True)
    rows = journal_rows(journal)
    terminal = next(i for i, row in enumerate(rows) if row["event"] == "discovery_result")
    rechain(journal, rows[:terminal])
    end_path = journal.with_name(journal.name + ".end.json")
    end = json.loads(end_path.read_bytes()) | {"reason": "deadline"}
    end["end_digest"] = protocol.digest({k: v for k, v in end.items() if k != "end_digest"})
    end_path.write_bytes(protocol.canonical(end) + b"\n")
    report = score.score_trial(path, journal)
    assert not report["complete"] and not report["qualifying_confirmatory"]
    assert report["discovery"] is None and report["discovery_attempted"] and report["discovery_unresolved"]
    assert report["stages"]["E"]["accounting"]["sie"]["unattempted"] == 1


@pytest.fixture
def long_m_history(tmp_path: Path) -> tuple[Path, Path, list[dict[str, Any]], list[list[dict[str, Any]]]]:
    path, packet = packet_file(tmp_path, long=True, stages=["M"], n=1, arms="sie")
    journal = tmp_path / "long.jsonl"
    with FakeServer(journal, delay=0.005) as server:
        assert run.run_trial(path, journal, server.url, {}, execute=True) == "finished"
        assert 1 <= server.maximum <= 2 and not server.errors
    rows = journal_rows(journal)
    obs = packet["observations"][0]
    blocks = [
        [row for row in rows if row.get("call_id") == protocol.call_id(obs["observation_id"], "sie", index)]
        for index in range(len(obs["requests"]["sie"]))
    ]
    assert len(blocks) == 6 and all(
        [row["event"] for row in block] == ["call_intent", "dispatch", "response", "call_result"] for block in blocks
    )
    assert score.score_trial(path, journal)["qualifying_confirmatory"]
    return path, journal, rows, blocks


def replace_m_calls(rows: list[dict[str, Any]], calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    begin = next(i for i, row in enumerate(rows) if row["event"] == "operation_intent")
    end = next(i for i, row in enumerate(rows) if row["event"] == "operation_result")
    rows[end]["captured_event_count"] = len(calls)
    return rows[: begin + 1] + calls + rows[end:]


@pytest.mark.parametrize("change", ["reversed", "skipped", "after_failure", "serial_total", "failed_total"])
def test_strict_model_path_order_and_serial_duration(
    long_m_history: tuple[Path, Path, list[dict[str, Any]], list[list[dict[str, Any]]]], change: str
) -> None:
    path, journal, rows, blocks = long_m_history
    operation = next(row for row in rows if row["event"] == "operation_result")
    if change == "reversed":
        ordered = [blocks[index] for index in (2, 1, 0, 5, 4, 3)]
    elif change == "skipped":
        ordered = blocks[1:]
    elif change == "after_failure":
        blocks[0][-1].update(status="failed", error={"code": "CALL_FAILED"})
        operation.update(status="failed", result=None, error={"code": "CALL_FAILED"})
        ordered = blocks
    else:
        for block in blocks:
            block[-1]["elapsed_s"] = 1.0
        operation["elapsed_s"] = 1.0
        ordered = blocks
        if change == "failed_total":
            # The failed second call contributes one second to its stopped path;
            # the other path finishes three much shorter serial windows.
            blocks[1][-1].update(status="failed", error={"code": "CALL_FAILED"})
            for block in blocks[3:]:
                block[-1]["elapsed_s"] = 0.1
            ordered = blocks[:2] + blocks[3:]
            operation.update(status="failed", result=None, error={"code": "CALL_FAILED"}, elapsed_s=1.5)
    rows = replace_m_calls(rows, [row for block in ordered for row in block])
    rechain(journal, rows, total_elapsed_s=5.0)
    with pytest.raises(ValueError, match="model path|serial call time"):
        score.score_trial(path, journal)
    if change == "failed_total":
        operation["elapsed_s"] = 2.0
        rechain(journal, rows)
        report = score.score_trial(path, journal)
        counts = report["stages"]["M"]["accounting"]["sie"]
        assert not report["complete"] and counts["failed"] == 1
        assert counts["all_terminal_attempt_elapsed"]["p50_s"] == 2.0
        assert counts["semantic_calls"]["failed"] == counts["semantic_calls"]["unattempted"] == 1


def test_interleaved_model_paths_cover_serial_sums_with_float_tolerance(
    long_m_history: tuple[Path, Path, list[dict[str, Any]], list[list[dict[str, Any]]]],
) -> None:
    path, journal, rows, blocks = long_m_history
    for block in blocks:
        block[-1]["elapsed_s"] = 1.0
    operation = next(row for row in rows if row["event"] == "operation_result")
    operation["elapsed_s"] = 3.0 - 0.0000005
    interleaved = [
        row
        for left, right in zip(blocks[:3], blocks[3:], strict=True)
        for row in left[:2] + right[:2] + right[2:] + left[2:]
    ]
    rechain(journal, replace_m_calls(rows, interleaved), total_elapsed_s=5.0)
    report = score.score_trial(path, journal)
    assert report["complete"] and report["qualifying_confirmatory"]
    assert report["stages"]["M"]["accounting"]["sie"]["semantic_calls"]["successful"] == 6


@pytest.mark.parametrize("stage", ["G", "E"])
def test_wrong_sie_model_is_failed_live_and_rejected_if_recast_success(tmp_path: Path, stage: str) -> None:
    path, _ = packet_file(tmp_path, stages=[stage], n=1, arms="sie")
    journal = tmp_path / "wrong-model.jsonl"
    with FakeServer(journal, response_model="unrelated/model") as server:
        assert run.run_trial(path, journal, server.url, {}, execute=True) == "finished"
    report = score.score_trial(path, journal)
    assert report["stages"][stage]["accounting"]["sie"]["failed"] == 1
    rows = journal_rows(journal)
    terminal = next(row for row in rows if row["event"] == "call_result")
    assert terminal["status"] == "failed" and terminal["reply"]["returned_model"] == "unrelated/model"
    assert terminal["error"]["class"] == "ValueError"
    terminal["status"] = "success"
    terminal.pop("error")
    rechain(journal, rows)
    with pytest.raises(ValueError, match="Response model differs"):
        score.score_trial(path, journal)


@pytest.mark.parametrize("stage", protocol.STAGES)
@pytest.mark.parametrize("returned", [None, "requested", "unrelated/model"])
def test_sie_reply_model_binding(stage: str, returned: str | None) -> None:
    data = population(1)[stage][0]["data"]
    call = protocol.requests(stage, data, "sie", "in-force" if stage == "R" else None)[0]
    reply = {
        "returned_model": call["model"] if returned == "requested" else returned,
        "text": "Safety: Safe",
        "entities": [],
        "dense": [0.0] * 2560,
        "scores": [{"item_id": c["id"], "score": 0.5} for c in data.get("candidates", [])],
    }
    if returned == "unrelated/model":
        with pytest.raises(ValueError, match="Response model differs"):
            protocol.validate_reply(stage, "sie", reply, data, call)
    else:
        protocol.validate_reply(stage, "sie", reply, data, call)


@pytest.mark.parametrize("stage", ["G", "M"])
def test_haiku_alias_accepts_dated_snapshot_identity(stage: str) -> None:
    data = population(1)[stage][0]["data"]
    call = protocol.requests(stage, data, "rival")[0]
    reply = {
        "returned_model": "claude-haiku-4-5-20260102",
        "text": "unharmful" if stage == "G" else '{"entities":[]}',
    }
    protocol.validate_reply(stage, "rival", reply, data, call)
    reply["returned_model"] = "claude-unrelated-20260102"
    with pytest.raises(ValueError, match="Response model differs"):
        protocol.validate_reply(stage, "rival", reply, data, call)


@pytest.mark.parametrize("member,extra", [(None, True), (None, False), (42, True), ("bad", False)])
def test_malformed_cohere_members_survive_projection_as_failure(member: Any, extra: bool) -> None:
    data = population(1)["R"][0]["data"]
    rankings = [{"index": i, "relevance_score": i / 20} for i in range(20)]
    if extra:
        rankings.append(member)
    else:
        rankings[5] = member
    reply = run.project_reply("R", "rival", {"results": rankings}, data)
    assert reply["malformed_reply"] and reply["results"][-1 if extra else 5] is None
    assert len(reply["results"]) == len(rankings)
    call = protocol.requests("R", data, "rival", "in-force")[0]
    with pytest.raises(ValueError, match="malformed reply"):
        protocol.validate_reply("R", "rival", reply, data, call)
    reply.pop("malformed_reply")
    with pytest.raises(ValueError, match="rerank indices"):
        protocol.validate_reply("R", "rival", reply, data, call)


def test_malformed_cohere_reply_is_durable_failed_call(tmp_path: Path) -> None:
    path, _ = packet_file(tmp_path, stages=["R"], n=1)
    journal = tmp_path / "rankings.jsonl"
    with FakeServer(journal, malformed_ranking=True) as server:
        assert (
            run.run_trial(
                path,
                journal,
                server.url,
                {"cohere": "fake-auth-sentinel"},
                execute=True,
                provider_urls={"cohere": server.url},
            )
            == "finished"
        )
    report = score.score_trial(path, journal)
    assert not report["qualifying_confirmatory"]
    assert report["stages"]["R"]["accounting"]["rival"]["failed"] == 2
    replies = [row for row in journal_rows(journal) if row["event"] == "call_result" and row["arm"] == "rival"]
    assert all(row["status"] == "failed" and row["reply"]["malformed_reply"] for row in replies)
    assert all(len(row["reply"]["results"]) == 21 and row["reply"]["results"][-1] is None for row in replies)


@pytest.mark.parametrize("index", [False, True, 0.0, "0", None])
def test_openai_embedding_projection_requires_integer_zero_index(index: Any) -> None:
    data = population(1)["E"][0]["data"]
    reply = run.project_reply("E", "rival", {"data": [{"index": index, "embedding": [0.0] * 3072}]}, data)
    assert reply["dense"] is None and reply["malformed_reply"]
    call = protocol.requests("E", data, "rival")[0]
    with pytest.raises(ValueError, match="malformed reply"):
        protocol.validate_reply("E", "rival", reply, data, call)


@pytest.mark.parametrize("stage", ["G", "M"])
@pytest.mark.parametrize("member", [None, 17])
def test_anthropic_non_object_content_is_preserved_as_malformed(stage: str, member: Any) -> None:
    data = population(1)[stage][0]["data"]
    text = "unharmful" if stage == "G" else '{"entities":[]}'
    reply = run.project_reply(stage, "rival", {"content": [{"type": "text", "text": text}, member]}, data)
    assert reply["text"] == text and reply["content"][1] is None and reply["malformed_reply"]
    call = protocol.requests(stage, data, "rival")[0]
    with pytest.raises(ValueError, match="malformed reply"):
        protocol.validate_reply(stage, "rival", reply, data, call)


@pytest.mark.parametrize("member,first", [(None, False), (17, False), (None, True)])
def test_non_object_guard_choices_keep_stable_failure_evidence(member: Any, first: bool) -> None:
    data = population(1)["G"][0]["data"]
    choices = [{"message": {"content": "Safety: Safe"}, "finish_reason": "stop"}]
    choices.insert(0 if first else 1, member)
    reply = run.project_reply("G", "sie", {"choices": choices}, data)
    assert reply["malformed_reply"] and reply["choice_count"] == 2
    assert reply["malformed_choice_indices"] == [0 if first else 1]
    assert reply["text"] == (None if first else "Safety: Safe")
    call = protocol.requests("G", data, "sie")[0]
    with pytest.raises(ValueError, match="malformed reply"):
        protocol.validate_reply("G", "sie", reply, data, call)


def test_paired_cluster_bootstrap_known_statistics() -> None:
    settings = config()["bootstrap"]
    clusters = [[(1.0, 4.0), (3.0, 6.0)], [(2.0, 5.0), (4.0, 7.0)]]
    result = score.paired_bootstrap(clusters, settings)
    assert result == score.paired_bootstrap(clusters, settings)
    assert result["difference_rival_minus_sie_s"] == 3.0 and result["ratio_rival_over_sie"] == 2.2
    assert result["difference_percentile_95"] == [3.0, 3.0]
    assert result["ratio_percentile_95"] == [2.0, 2.5]
    assert result["sie_median_s"] == 2.5 and result["rival_median_s"] == 5.5
    assert result["sie_median_percentile_95"] == [2.0, 3.0]
    assert result["rival_median_percentile_95"] == [5.0, 6.0]
    assert result["cohort"] == "successful complete paired base-case clusters"
    assert result["pairs"] == 4 and result["base_clusters"] == 2
    assert score.paired_bootstrap([clusters[0]], settings)["status"] == "insufficient"
    # Independently enumerate the same seeded whole-cluster choices.
    chooser = random.Random(settings["seed"])
    ratios = []
    for _ in range(settings["resamples"]):
        sampled = [pair for _ in clusters for pair in chooser.choice(clusters)]
        ratios.append(score.quantile([r for _, r in sampled], 0.5) / score.quantile([s for s, _ in sampled], 0.5))
    assert result["ratio_percentile_95"] == [score.quantile(ratios, p) for p in (0.025, 0.975)]


def test_offline_cli_help_and_opt_in_error(tmp_path: Path) -> None:
    for script in ("fetch.py", "prepare.py", "run.py", "score.py"):
        result = subprocess.run(
            [sys.executable, str(HERE / script), "--help"], capture_output=True, text=True, check=False
        )
        assert result.returncode == 0 and "usage:" in result.stdout
    result = subprocess.run(
        [sys.executable, str(HERE / "run.py"), "--packet", "absent.json", "--out", str(tmp_path / "out.jsonl")],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2 and "--execute" in result.stderr and not (tmp_path / "out.jsonl").exists()
    assert "API_KEY" not in result.stdout + result.stderr


def test_persistence_is_outside_timers_but_inside_wall_budget(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _, packet = packet_file(tmp_path, stages=["E"], n=1, arms="sie")
    path = tmp_path / "timing.jsonl"
    journal = diagnostic_journal(path, packet)
    clock = {"now": 1.0}

    def advance(seconds: float) -> None:
        clock["now"] += seconds

    monkeypatch.setattr(run.time, "perf_counter", lambda: clock["now"])
    monkeypatch.setattr(run.time, "monotonic", lambda: clock["now"])
    original_append, original_capture = journal.append, run.OperationEvents.append
    persisted = []

    def append(event: str, **fields: Any) -> None:
        original_append(event, **fields)
        advance(100)
        persisted.append(event)

    def capture(events: run.OperationEvents, event: str, **fields: Any) -> None:
        original_capture(events, event, **fields)
        advance(0.25)

    monkeypatch.setattr(journal, "append", append)
    monkeypatch.setattr(run.OperationEvents, "append", capture)
    executor = synthetic_executor(packet, journal, advance)
    executor.config["deadline"] = clock["now"] + 600
    executor.operation(packet["observations"][0], "sie")
    journal.close()
    rows = journal_rows(path)
    operation = rows[-1]
    assert operation["evidence_sealed"] and operation["captured_event_count"] == 4
    assert operation["elapsed_s"] == 4.0
    assert next(r for r in rows if r["event"] == "call_result")["elapsed_s"] == 3.5
    assert next(r for r in rows if r["event"] == "response")["headers_elapsed_s"] == 1.0
    assert persisted == ["operation_intent", "call_intent", "dispatch", "response", "call_result", "operation_result"]
    assert clock["now"] == 605.0
    with pytest.raises(run.CallFailure, match="WALL_DEADLINE"):
        executor.remaining()


def test_m_buffer_preserves_interleaving_without_serializing_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    packet_path, packet = packet_file(tmp_path, long=True, stages=["M"], n=1, arms="sie")
    path = tmp_path / "interleaving.jsonl"
    journal = diagnostic_journal(path, packet)
    barrier = threading.Barrier(2)
    captured = []
    original_flush = run.OperationEvents.flush
    original_append = journal.append

    def advance(seconds: float) -> None:
        if seconds == 1:
            barrier.wait(timeout=3)

    def flush(events: run.OperationEvents, target: run.Journal) -> None:
        captured.extend(events.events)
        original_flush(events, target)

    def append(event: str, **fields: Any) -> None:
        assert threading.current_thread() is threading.main_thread(), "M worker persisted during measured work"
        original_append(event, **fields)

    monkeypatch.setattr(run.OperationEvents, "flush", flush)
    monkeypatch.setattr(journal, "append", append)
    executor = synthetic_executor(packet, journal, advance)
    executor.operation(packet["observations"][0], "sie")
    journal.close()
    diagnostic_end(path, packet, "finished")
    rows = journal_rows(path)
    replayed = rows[4:-1]
    assert len(captured) == len(replayed) == 24
    for (event, fields), row in zip(captured, replayed, strict=True):
        assert event == row["event"] and all(row[k] == value for k, value in fields.items())
    for model in protocol.MODELS["M"]:
        ordered = [r for r in replayed if r["model"] == model and r["event"] in ("call_intent", "call_result")]
        assert [r["event"] for r in ordered] == ["call_intent", "call_result"] * 3
        assert [r["window"] for r in ordered] == [0, 0, 1, 1, 2, 2]
    report = score.score_trial(packet_path, path)
    counts = report["stages"]["M"]["accounting"]["sie"]
    assert report["complete"] and counts["physical_dispatches_exact"] and counts["sdk_retries_exact"]
    assert counts["semantic_calls"]["successful"] == 6


def test_buffer_owns_sanitized_reply_and_snapshots_metadata(tmp_path: Path) -> None:
    _, packet = packet_file(tmp_path, stages=["E"], n=1, arms="sie")
    path = tmp_path / "snapshot.jsonl"
    journal = run.Journal(path, packet, ["fake-auth-sentinel"], create=True)
    events = run.OperationEvents()
    reply = run.sanitize(
        {"text": "fake-auth-sentinel", "usage": {"input_tokens": 3}, "dense": [0.5]}, ["fake-auth-sentinel"]
    )
    identity = {"arm_order": ["sie", "rival"], "error": {"code": "CALL_FAILED"}}
    events.append("call_result", reply=reply, **identity, sdk_retry_count=1, execution_revision="a" * 64)
    assert events.events[0][1]["reply"] is reply
    assert events.events[0][1]["reply"]["dense"] is reply["dense"]
    identity["arm_order"].reverse()
    identity["error"]["code"] = "changed-metadata"
    events.flush(journal)
    journal.close()
    row = journal_rows(path)[0]
    assert row["reply"] == {"text": "[REDACTED_CREDENTIAL]", "usage": {"input_tokens": 3}, "dense": [0.5]}
    assert row["arm_order"] == ["sie", "rival"] and row["sdk_retry_count"] == 1
    assert row["error"] == {"code": "CALL_FAILED"} and row["execution_revision"] == "a" * 64
    assert "fake-auth-sentinel" not in path.read_text()


@pytest.mark.parametrize("arm", ["sie", "rival"])
def test_retained_owned_vector_is_independent_from_mutable_raw_reply(tmp_path: Path, arm: str) -> None:
    _, packet = packet_file(tmp_path, stages=["E"], n=1)
    path = tmp_path / "owned-vector.jsonl"
    data = packet["observations"][0]["unit"]["data"]
    call = packet["observations"][0]["requests"][arm][0]
    vector = [0.125] * (2560 if arm == "sie" else 3072)
    raw = {"model": call["model"], "usage": {"input_tokens": 3}}
    raw.update({"dense": vector} if arm == "sie" else {"data": [{"index": 0, "embedding": vector}]})
    reply = run.sanitize(run.project_reply("E", arm, raw, data), [])
    assert reply["dense"] is not vector
    assert protocol.validate_reply("E", arm, reply, data, call)["dimensions"] == len(vector)
    events = run.OperationEvents()
    events.append("call_result", reply=reply)
    assert events.events[0][1]["reply"] is reply and events.events[0][1]["reply"]["dense"] is reply["dense"]
    vector[0] = 99.0
    raw["usage"]["input_tokens"] = 99
    raw["model"] = "changed-raw-model"
    journal = run.Journal(path, packet, [], create=True)
    events.flush(journal)
    journal.close()
    recorded = journal_rows(path)[0]["reply"]
    assert recorded["dense"] == [0.125] * len(vector)
    assert recorded["usage"] == {"input_tokens": 3} and recorded["returned_model"] == call["model"]


@pytest.mark.parametrize("error_type", [MemoryError, ValueError])
def test_deepcopy_failure_in_second_m_path_prevents_replay_and_seal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error_type: type[Exception]
) -> None:
    packet_path, packet = packet_file(tmp_path, stages=["M"], n=1, arms="sie")
    path = tmp_path / "copy-failure.jsonl"
    journal = diagnostic_journal(path, packet)
    executor = synthetic_executor(packet, journal)
    original_copy = run.copy.deepcopy

    def ordinary_failure(*args: Any, **kwargs: Any) -> Any:
        raise run.CallFailure("CALL_FAILED")

    def copy(value: Any, memo: dict[int, Any] | None = None) -> Any:
        if isinstance(value, dict) and value.get("model") == protocol.MODELS["M"][1]:
            raise error_type
        return original_copy(value, memo)

    monkeypatch.setattr(executor.clients[protocol.MODELS["M"][0]], "extract", ordinary_failure)
    monkeypatch.setattr(run.copy, "deepcopy", copy)
    with pytest.raises(run.EvidenceCaptureError):
        executor.operation(packet["observations"][0], "sie")
    journal.close()
    assert [row["event"] for row in journal_rows(path)] == [
        "run_start",
        "discovery_intent",
        "discovery_result",
        "operation_intent",
    ]
    diagnostic_end(path, packet, "child_failure")
    report = score.score_trial(packet_path, path)
    counts = report["stages"]["M"]["accounting"]["sie"]
    assert not report["complete"] and counts["unresolved"] == 1
    assert counts["semantic_calls"]["attempt_status_unknown"] == 2
    assert not counts["physical_dispatches_exact"] and not counts["sdk_retries_exact"]


def test_fake_server_rejects_post_after_last_operation_seal(tmp_path: Path) -> None:
    packet_path, packet = packet_file(tmp_path, stages=["G"], n=2, arms="sie")
    path = tmp_path / "last-intent.jsonl"
    with FakeServer(path) as server:
        assert run.run_trial(packet_path, path, server.url, {}, execute=True) == "finished"
        assert not server.errors and journal_rows(path)[-1]["event"] == "operation_result"
        call = packet["observations"][0]["requests"]["sie"][0]
        connection = HTTPConnection("127.0.0.1", server.server.server_port, timeout=2)
        try:
            connection.request(
                "POST",
                "/v1/chat/completions",
                body=protocol.canonical({"model": call["model"], **call["body"]}),
                headers={"Content-Type": "application/json"},
            )
            response = connection.getresponse()
            assert response.status == 200
            response.read()
        finally:
            connection.close()
        assert server.errors == ["Server observed dispatch without the current durable operation intent"]


@pytest.mark.parametrize(
    "checkpoint,stage,attempted,unknown,physical",
    [
        ("after_intent", "E", 0, 1, 0),
        ("after_response", "E", 0, 1, 0),
        ("after_m_path", "M", 0, 6, 0),
        ("before_flush", "M", 0, 6, 0),
        ("mid_flush", "M", 1, 5, 0),
        ("partial_flush", "E", 1, 0, 1),
        ("after_seal", "E", 1, 0, 1),
    ],
)
def test_real_termination_preserves_only_durable_checkpoints(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    checkpoint: str,
    stage: str,
    attempted: int,
    unknown: int,
    physical: int,
) -> None:
    # Pytest importlib mode does not otherwise expose this namespace module to spawn.
    monkeypatch.syspath_prepend(str(HERE.parents[1]))
    packet_path, packet = packet_file(tmp_path, long=stage == "M", stages=[stage], n=2, arms="sie")
    path = tmp_path / "stopped.jsonl"
    diagnostic_journal(path, packet).close()
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe()
    process = context.Process(target=checkpoint_worker, args=(packet_path, path, child, checkpoint))
    try:
        process.start()
        child.close()
        assert parent.poll(10) and parent.recv() == checkpoint
        process.terminate()
        process.join(3)
        assert not process.is_alive()
    finally:
        if process.is_alive():
            process.kill()
            process.join()
        parent.close()
        child.close()
    diagnostic_end(path, packet, "interrupted")
    report = score.score_trial(packet_path, path)
    counts = report["stages"][stage]["accounting"]["sie"]
    calls = counts["semantic_calls"]
    assert not report["complete"] and not report["qualifying_confirmatory"]
    assert counts["planned"] == 2 and counts["unattempted"] == 1
    assert calls["planned"] == packet["planned_semantic_calls"][stage]["sie"]
    assert calls["attempted"] == attempted and calls["attempt_status_unknown"] == unknown
    assert calls["attempted_count_exact"] == (unknown == 0)
    assert counts["physical_dispatches"] == physical and counts["sdk_retries"] == 0
    assert counts["physical_dispatches_exact"] == counts["sdk_retries_exact"] == (checkpoint == "after_seal")
    assert calls["planned"] == sum(
        calls[k] for k in ("successful", "failed", "unresolved", "unattempted", "attempt_status_unknown")
    )
    if checkpoint == "after_seal":
        assert counts["successful"] == calls["successful"] == calls["unattempted"] == 1
    else:
        assert counts["unresolved"] == 1 and counts["successful_operation_latency"]["count"] == 0
        assert calls["successful"] == calls["failed"] == 0
    assert (report["truncated_tail_sha256"] is not None) == (checkpoint == "partial_flush")


@pytest.mark.parametrize("failure", ["capture", "flush", "m_memory"])
def test_observer_failure_stops_child_without_seal_or_next_operation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    stage = "M" if failure == "m_memory" else "E"
    packet_path, packet = packet_file(tmp_path, stages=[stage], n=2, arms="sie")
    path = tmp_path / "observer-failure.jsonl"
    diagnostic_journal(path, packet).close()
    original_journal = run.Journal

    class FailingJournal(original_journal):
        def append(self, event: str, **fields: Any) -> None:
            if failure == "flush" and event == "dispatch":
                self.output.write(b'{"partial":')
                self.output.flush()
                os.fsync(self.output.fileno())
                raise OSError("synthetic persistence failure")
            super().append(event, **fields)

    class Executor(run.Executor):
        def __init__(self, packet: dict[str, Any], journal: run.Journal, config: dict[str, Any]) -> None:
            self.__dict__.update(synthetic_executor(packet, journal).__dict__)
            if failure == "m_memory":

                def ordinary_failure(*args: Any, **kwargs: Any) -> Any:
                    raise run.CallFailure("CALL_FAILED")

                def memory_failure(*args: Any, **kwargs: Any) -> Any:
                    raise MemoryError

                monkeypatch.setattr(self.clients[protocol.MODELS["M"][0]], "extract", ordinary_failure)
                monkeypatch.setattr(self.clients[protocol.MODELS["M"][1]], "extract", memory_failure)

        def discover(self) -> None:
            pass

    class Connection:
        messages: list[str] = []

        def recv(self) -> dict[str, Any]:
            return {"packet_path": str(packet_path), "journal": str(path), "credentials": {}}

        def send(self, message: str) -> None:
            self.messages.append(message)

        def close(self) -> None:
            pass

    original_capture = run.OperationEvents.append

    def capture(events: run.OperationEvents, event: str, **fields: Any) -> None:
        if failure == "capture" and event == "response":
            events.failed = True
            raise run.EvidenceCaptureError
        original_capture(events, event, **fields)

    monkeypatch.setattr(run, "Journal", FailingJournal)
    monkeypatch.setattr(run, "Executor", Executor)
    monkeypatch.setattr(run.OperationEvents, "append", capture)
    connection = Connection()
    run.child_main(connection)
    assert connection.messages == ["child_failure"]
    rows = journal_rows(path)
    assert sum(r["event"] == "operation_intent" for r in rows) == 1
    assert not any(r["event"] == "operation_result" for r in rows)
    diagnostic_end(path, packet, "child_failure")
    report = score.score_trial(packet_path, path)
    assert not report["complete"]
    assert report["stages"][stage]["accounting"]["sie"]["unattempted"] == 1


@pytest.mark.parametrize("change", ["missing", "count", "premature"])
def test_invalid_evidence_seals_are_rejected(
    long_m_history: tuple[Path, Path, list[dict[str, Any]], list[list[dict[str, Any]]]], change: str
) -> None:
    packet_path, path, rows, _ = long_m_history
    terminal = next(r for r in rows if r["event"] == "operation_result")
    if change == "missing":
        terminal.pop("evidence_sealed")
    elif change == "count":
        terminal["captured_event_count"] -= 1
    else:
        rows.remove(terminal)
        terminal.update(status="failed", result=None, error={"code": "CALL_FAILED"})
        at = next(i for i, r in enumerate(rows) if r["event"] == "call_result")
        rows.insert(at, terminal)
    rechain(path, rows)
    with pytest.raises(ValueError, match="seal|operation terminal"):
        score.score_trial(packet_path, path)


def test_old_timing_binding_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path, packet = packet_file(tmp_path, stages=["E"], n=1, arms="sie")
    current = protocol.protocol_digest()
    legacy = protocol.digest(
        {name: protocol.sha256((HERE / name).read_bytes()) for name in ("protocol.py", "prepare.py", "run.py")}
    )
    assert legacy != current
    packet["protocol_digest"] = legacy
    packet["packet_digest"] = protocol.digest({k: v for k, v in packet.items() if k != "packet_digest"})
    path.write_bytes(protocol.canonical(packet))
    with pytest.raises(ValueError, match="protocol binding"):
        prepare.load_packet(path)
    monkeypatch.setitem(protocol.TIMING_POLICY, "version", 1)
    assert protocol.protocol_digest() != current


def test_source_templates_match_builders_and_no_external_runtime_defaults() -> None:
    sources = protocol.sources()
    assert len(sources["files"]) == 9
    assert all(entry["revision"] in entry["url"] and len(entry["sha256"]) == 64 for entry in sources["files"])
    for stage in protocol.STAGES:
        for arm, rules in sources["request_bindings"][stage]["requests"].items():
            for rule, calls in rules.items():
                data = {
                    "text": "{source_text}",
                    "query": "{source_query}",
                    "candidates": [
                        {"id": f"{{candidate_{i}_id}}", "text": f"{{candidate_{i}_text}}"} for i in range(20)
                    ],
                }
                assert calls == protocol.requests(stage, data, arm, None if rule == "None" else rule)
    assert "api.superlinked.com" not in (HERE / "run.py").read_text()


@pytest.mark.parametrize(
    "stage,arm,reply",
    [
        ("G", "sie", {"text": "no verdict"}),
        ("G", "rival", {"text": "unharmful extra"}),
        ("M", "sie", {"entities": [], "item_error": {"code": "ITEM_ERROR"}}),
        ("E", "sie", {"dense": [0.0] * 12}),
        ("E", "rival", {"dense": [float("nan")] * 3072}),
        ("R", "rival", {"scores": [], "results": [{"index": 0}] * 20}),
    ],
)
def test_reply_failures_never_become_success(stage: str, arm: str, reply: dict[str, Any]) -> None:
    selected = population(1)[stage][0]
    call = protocol.requests(stage, selected["data"], arm, "in-force" if stage == "R" else None)[0]
    with pytest.raises(ValueError):
        protocol.validate_reply(stage, arm, reply, selected["data"], call)
