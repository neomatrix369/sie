"""Anonymously fetch exactly nine pinned public files; never replace previous inputs."""

from __future__ import annotations

import argparse
import errno
import json
import os
import shutil
import urllib.request
from pathlib import Path
from typing import Any

from protocol import canonical, digest, require, safe_path, sha256, sources


def fsync_directory(path: Path) -> None:
    """Sync directory entries on POSIX filesystems that support directory fsync."""
    if os.name != "posix":
        return
    descriptor = None
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        os.fsync(descriptor)
    except OSError as error:
        if error.errno not in (errno.EINVAL, errno.ENOTSUP, errno.EOPNOTSUPP):
            raise
    finally:
        if descriptor is not None:
            os.close(descriptor)


def mkdir_synced(path: Path, *, exist_ok: bool = True) -> None:
    """Create missing ancestors and sync each new directory and its parent."""
    if path.parent != path and not path.parent.is_dir():
        mkdir_synced(path.parent)
    try:
        path.mkdir()
    except FileExistsError:
        if not exist_ok or not path.is_dir():
            raise
    else:
        fsync_directory(path)
        fsync_directory(path.parent)


def write_exclusive(path: Path, value: Any) -> None:
    mkdir_synced(path.parent)
    with path.open("xb") as output:
        output.write(canonical(value) + b"\n")
        output.flush()
        os.fsync(output.fileno())
    fsync_directory(path.parent)


def verify_inputs(root: Path, catalog: dict[str, Any] | None = None) -> dict[str, bytes]:
    catalog = sources() if catalog is None else catalog
    marker = json.loads((root / "verified.json").read_bytes())
    require(marker == {"source_digest": digest(catalog)}, "Wrong input source marker")
    files: dict[str, bytes] = {}
    for entry in catalog["files"]:
        name = safe_path(entry["path"])
        require(name not in files, "Duplicate allowlist file")
        body = (root / name).read_bytes()
        require(len(body) == entry["bytes"] and sha256(body) == entry["sha256"], "Input bytes or digest differ")
        files[name] = body
    for stage in ("G", "M", "E", "R"):
        manifest = json.loads(files[f"{stage}/manifest.json"])
        expected_task = {"G": "guardrails", "M": "redact", "E": "lookalike-search", "R": "rerank-relevance-rules"}[
            stage
        ]
        require(manifest.get("task") == expected_task, "Wrong-source manifest")
        entries = manifest.get("files_sha256", manifest.get("files", {}))
        if isinstance(entries, list):
            indexed = {safe_path(e["path"]): e for e in entries}
            require(len(indexed) == len(entries), "Duplicate manifest paths")
        else:
            require(isinstance(entries, dict), "Malformed manifest files")
            indexed = {safe_path(name): {"sha256": value} for name, value in entries.items()}
        for name, body in files.items():
            if name.startswith(f"{stage}/") and name != f"{stage}/manifest.json" and stage != "G":
                remote_name = "inputs/" + name.split("/", 1)[1]
                require(remote_name in indexed, "Input missing from source manifest")
                check = indexed[remote_name]
                require(
                    sha256(body) == check["sha256"] and check.get("bytes", len(body)) == len(body),
                    "Manifest input digest differs",
                )
    return files


def fetch_inputs(destination: Path, catalog: dict[str, Any] | None = None) -> None:
    catalog = sources() if catalog is None else catalog
    mkdir_synced(destination.parent)
    destination.mkdir()  # exclusive reservation, including empty existing directories
    staging = destination / ".staging"
    try:
        fsync_directory(destination)
        fsync_directory(destination.parent)
        mkdir_synced(staging, exist_ok=False)
        for entry in catalog["files"]:
            name = safe_path(entry["path"])
            request = urllib.request.Request(entry["url"], headers={"User-Agent": "sie-examples/agent-stage-latency"})
            with urllib.request.urlopen(request, timeout=60) as response:
                body = response.read(entry["bytes"] + 1)
            require(
                len(body) == entry["bytes"] and sha256(body) == entry["sha256"], "Downloaded bytes or digest differ"
            )
            target = staging / name
            mkdir_synced(target.parent)
            with target.open("xb") as output:
                output.write(body)
                output.flush()
                os.fsync(output.fileno())
            fsync_directory(target.parent)
        write_exclusive(staging / "verified.json", {"source_digest": digest(catalog)})
        verify_inputs(staging, catalog)
        for child in staging.iterdir():
            if child.name != "verified.json":
                child.rename(destination / child.name)
        fsync_directory(staging)
        fsync_directory(destination)
        (staging / "verified.json").rename(destination / "verified.json")
        fsync_directory(staging)
        fsync_directory(destination)
        staging.rmdir()
        fsync_directory(destination)
    except BaseException:
        # Only this invocation's exclusively reserved directory is removed.
        shutil.rmtree(destination)
        fsync_directory(destination.parent)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        fetch_inputs(args.out)
    except (OSError, ValueError):
        raise SystemExit(
            "Fetch failed; no verified inputs published. Choose a fresh destination and check the public pins."
        ) from None
    print("Verified the nine pinned public input files.")


if __name__ == "__main__":
    main()
