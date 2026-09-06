"""Canonical JSON and JSONL helpers for evaluation evidence."""

from __future__ import annotations

import errno
import hashlib
import json
import os
from collections.abc import Iterable, Mapping
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    rows: list[dict[str, Any]] = []
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {source}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"JSONL row at {source}:{line_number} must be an object")
            rows.append(value)
    return rows


def _fsync_directory(directory: Path) -> None:
    if os.name == "nt":
        return

    unsupported_errors = {errno.EINVAL, errno.ENOSYS}
    for name in ("ENOTSUP", "EOPNOTSUPP"):
        value = getattr(errno, name, None)
        if isinstance(value, int):
            unsupported_errors.add(value)

    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        directory_fd = os.open(directory, directory_flags)
    except OSError as error:
        if error.errno in unsupported_errors:
            return
        raise
    try:
        os.fsync(directory_fd)
    except OSError as error:
        if error.errno in unsupported_errors:
            os.close(directory_fd)
            return
        _close_directory_fd(directory_fd, primary_error=error)
        raise
    except BaseException as error:
        _close_directory_fd(directory_fd, primary_error=error)
        raise
    else:
        os.close(directory_fd)


def _close_directory_fd(directory_fd: int, *, primary_error: BaseException) -> None:
    try:
        os.close(directory_fd)
    except OSError as cleanup_error:
        primary_error.add_note(f"failed to close directory fd: {cleanup_error}")


def _write_bytes_atomic(destination: Path, payload: bytes) -> Path:
    temporary: Path | None = None
    try:
        with NamedTemporaryFile(
            dir=destination.parent, prefix=f".{destination.name}.", delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        # The source path is no longer ours after replace and may be reused.
        temporary = None
        _fsync_directory(destination.parent)
    except BaseException as error:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError as cleanup_error:
                error.add_note(f"failed to remove temporary file {temporary}: {cleanup_error}")
        raise
    return destination


def write_text_atomic(path: str | Path, content: str) -> Path:
    """Write UTF-8 text, fsync it, and atomically replace the destination."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    return _write_bytes_atomic(destination, content.encode("utf-8"))


def write_json_atomic(path: str | Path, value: Mapping[str, Any]) -> Path:
    """Write canonical, fsync'd JSON and atomically replace the destination."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = (canonical_json(value) + "\n").encode("utf-8")
    return _write_bytes_atomic(destination, payload)


def write_jsonl_atomic(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(f"{canonical_json(row)}\n" for row in rows).encode("utf-8")
    return _write_bytes_atomic(destination, payload)
