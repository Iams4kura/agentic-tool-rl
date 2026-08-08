"""Append-only, resumable JSONL evidence storage."""

from __future__ import annotations

import hashlib
import json
import os
import threading
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from agentic_tool_rl.evaluation.io import canonical_json

try:  # macOS/Linux locking; the project does not target Windows CI.
    import fcntl
except ImportError:  # pragma: no cover - defensive import isolation
    fcntl = None  # type: ignore[assignment]


class TraceConflictError(ValueError):
    """An existing case id was reused with different evidence."""


class CorruptTraceStoreError(ValueError):
    """A non-recoverable JSONL record was found."""


class TraceStore:
    """An append-only case store with idempotent resume semantics.

    A crash-truncated final line is removed on the next open. Complete records
    are never rewritten, and repeated appends of the same record are no-ops.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)
        self._thread_lock = threading.RLock()
        self._records: dict[str, dict[str, Any]] = {}
        self._checksums: dict[str, str] = {}
        self._refresh(repair_tail=True)

    @staticmethod
    def _case_id(record: Mapping[str, Any]) -> str:
        case_id = record.get("case_id")
        if not isinstance(case_id, str) or not case_id:
            raise ValueError("trace record must have a non-empty string case_id")
        return case_id

    @staticmethod
    def _checksum(payload: bytes) -> str:
        return hashlib.sha256(payload).hexdigest()

    def _refresh(self, *, repair_tail: bool) -> None:
        payload = self.path.read_bytes()
        records: dict[str, dict[str, Any]] = {}
        checksums: dict[str, str] = {}
        offset = 0
        valid_end = 0
        lines = payload.splitlines(keepends=True)
        for index, raw_line in enumerate(lines):
            is_last = index == len(lines) - 1
            complete = raw_line.endswith(b"\n")
            line = raw_line.rstrip(b"\r\n")
            next_offset = offset + len(raw_line)
            # A newline is the record's commit marker. Syntactically complete
            # JSON without it can still be the result of a crashed append.
            if is_last and not complete:
                if repair_tail:
                    with self.path.open("r+b") as handle:
                        handle.truncate(valid_end)
                        handle.flush()
                        os.fsync(handle.fileno())
                    break
                raise CorruptTraceStoreError("trace store ends with an uncommitted record")
            if not line:
                valid_end = next_offset
                offset = next_offset
                continue
            try:
                value = json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                raise CorruptTraceStoreError(
                    f"invalid JSONL record at byte offset {offset}: {exc}"
                ) from exc
            if not isinstance(value, dict):
                raise CorruptTraceStoreError(f"record at byte offset {offset} is not an object")
            case_id = self._case_id(value)
            if case_id in records:
                raise CorruptTraceStoreError(f"duplicate case_id {case_id!r} in trace store")
            records[case_id] = value
            checksums[case_id] = self._checksum(line)
            valid_end = next_offset
            offset = next_offset
        self._records = records
        self._checksums = checksums

    def __len__(self) -> int:
        return len(self._records)

    def completed_case_ids(self) -> frozenset[str]:
        return frozenset(self._records)

    def missing_case_ids(self, expected: Iterable[str]) -> list[str]:
        return [case_id for case_id in expected if case_id not in self._records]

    def record_checksum(self, case_id: str) -> str:
        try:
            return self._checksums[case_id]
        except KeyError as exc:
            raise KeyError(f"case_id {case_id!r} is not in the trace store") from exc

    def records(self) -> list[dict[str, Any]]:
        return [dict(value) for value in self._records.values()]

    def append(self, record: Mapping[str, Any]) -> bool:
        """Append once; return False for an identical already-complete case."""

        normalized = json.loads(canonical_json(record))
        if not isinstance(normalized, dict):  # canonical JSON protects this path
            raise ValueError("trace record must be a JSON object")
        case_id = self._case_id(normalized)
        line = canonical_json(normalized).encode("utf-8")

        with self._thread_lock, self.path.open("a+b") as handle:
            if fcntl is not None:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                self._refresh(repair_tail=True)
                existing = self._records.get(case_id)
                if existing is not None:
                    if existing != normalized:
                        raise TraceConflictError(
                            f"case_id {case_id!r} already exists with different evidence"
                        )
                    return False
                handle.seek(0, os.SEEK_END)
                handle.write(line + b"\n")
                handle.flush()
                os.fsync(handle.fileno())
                self._records[case_id] = normalized
                self._checksums[case_id] = self._checksum(line)
                return True
            finally:
                if fcntl is not None:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
