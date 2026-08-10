"""Append-only, resumable JSONL evidence storage."""

from __future__ import annotations

import hashlib
import json
import os
import threading
from collections.abc import Collection, Iterable, Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any, BinaryIO

from agentic_tool_rl.evaluation.io import canonical_json

try:  # macOS/Linux locking; the project does not target Windows CI.
    import fcntl
except ImportError:  # pragma: no cover - defensive import isolation
    fcntl = None  # type: ignore[assignment]

_MAX_PATH_RETRIES = 8


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
        self._thread_lock = threading.RLock()
        self._records: dict[str, dict[str, Any]] = {}
        self._checksums: dict[str, str] = {}
        self._valid_end = 0
        self._inode: tuple[int, int] | None = None
        # Tail repair must participate in the same inter-process lock as append.
        # Otherwise, an opener could mistake another process's in-flight write
        # for a crashed record and truncate it while that writer still owns the
        # file lock.
        with self._thread_lock:
            for _ in range(_MAX_PATH_RETRIES):
                with self.path.open("a+b") as handle:
                    self._lock(handle)
                    try:
                        if not self._path_matches_handle(handle):
                            continue
                        self._full_refresh(handle, repair_tail=True)
                        if not self._path_matches_handle(handle):
                            continue
                        break
                    finally:
                        self._unlock(handle)
            else:
                raise RuntimeError("trace store path changed repeatedly while opening")

    @staticmethod
    def _case_id(record: Mapping[str, Any]) -> str:
        case_id = record.get("case_id")
        if not isinstance(case_id, str) or not case_id:
            raise ValueError("trace record must have a non-empty string case_id")
        return case_id

    @staticmethod
    def _checksum(payload: bytes) -> str:
        return hashlib.sha256(payload).hexdigest()

    @staticmethod
    def _file_identity(handle: BinaryIO) -> tuple[int, int]:
        status = os.fstat(handle.fileno())
        return status.st_dev, status.st_ino

    def _path_matches_handle(self, handle: BinaryIO) -> bool:
        """Return whether ``path`` still names the opened file description."""

        try:
            status = self.path.stat()
        except OSError:
            return False
        return (status.st_dev, status.st_ino) == self._file_identity(handle)

    @staticmethod
    def _lock(handle: BinaryIO) -> None:
        if fcntl is not None:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)

    @staticmethod
    def _unlock(handle: BinaryIO) -> None:
        if fcntl is not None:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def _scan_from(
        self,
        handle: BinaryIO,
        *,
        start_offset: int,
        known_case_ids: Collection[str],
        repair_tail: bool,
    ) -> tuple[dict[str, dict[str, Any]], dict[str, str], int]:
        """Parse complete records from one known line boundary.

        Newly parsed records stay local until the complete suffix has been
        validated. A corrupt suffix therefore cannot leave the in-memory index
        partially advanced and make a later retry report false duplicates.
        """

        records: dict[str, dict[str, Any]] = {}
        checksums: dict[str, str] = {}
        offset = start_offset
        valid_end = start_offset
        handle.seek(start_offset)
        while raw_line := handle.readline():
            complete = raw_line.endswith(b"\n")
            line = raw_line.rstrip(b"\r\n")
            next_offset = offset + len(raw_line)
            # A newline is the record's commit marker. Syntactically complete
            # JSON without it can still be the result of a crashed append.
            if not complete:
                if repair_tail:
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
            if case_id in known_case_ids or case_id in records:
                raise CorruptTraceStoreError(f"duplicate case_id {case_id!r} in trace store")
            records[case_id] = value
            checksums[case_id] = self._checksum(line)
            valid_end = next_offset
            offset = next_offset
        return records, checksums, valid_end

    def _full_refresh(self, handle: BinaryIO, *, repair_tail: bool) -> None:
        records, checksums, valid_end = self._scan_from(
            handle,
            start_offset=0,
            known_case_ids=(),
            repair_tail=repair_tail,
        )
        self._records = records
        self._checksums = checksums
        self._valid_end = valid_end
        self._inode = self._file_identity(handle)

    def _synchronize(self, handle: BinaryIO) -> None:
        """Synchronize the index with bytes committed by other writers."""

        identity = self._file_identity(handle)
        size = os.fstat(handle.fileno()).st_size
        if self._inode != identity or size < self._valid_end:
            # Replacement and truncation invalidate every cached byte offset.
            # Rebuilding under the exclusive lock is safe and preserves file
            # order while still keeping the common append path incremental.
            self._full_refresh(handle, repair_tail=True)
            return
        if size == self._valid_end:
            return
        records, checksums, valid_end = self._scan_from(
            handle,
            start_offset=self._valid_end,
            known_case_ids=self._records,
            repair_tail=True,
        )
        self._records.update(records)
        self._checksums.update(checksums)
        self._valid_end = valid_end
        self._inode = identity

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
        return [deepcopy(value) for value in self._records.values()]

    def append(self, record: Mapping[str, Any]) -> bool:
        """Append once; return False for an identical already-complete case."""

        normalized = json.loads(canonical_json(record))
        if not isinstance(normalized, dict):  # canonical JSON protects this path
            raise ValueError("trace record must be a JSON object")
        case_id = self._case_id(normalized)
        line = canonical_json(normalized).encode("utf-8")

        with self._thread_lock:
            for _ in range(_MAX_PATH_RETRIES):
                with self.path.open("a+b") as handle:
                    self._lock(handle)
                    try:
                        # A non-cooperating os.replace can race open() before
                        # flock(). Never append to an inode no longer named by
                        # the configured path; close it and retry the new path.
                        if not self._path_matches_handle(handle):
                            continue
                        self._synchronize(handle)
                        if not self._path_matches_handle(handle):
                            continue
                        existing = self._records.get(case_id)
                        if existing is not None:
                            if not self._path_matches_handle(handle):
                                continue
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
                        self._valid_end = os.fstat(handle.fileno()).st_size
                        self._inode = self._file_identity(handle)
                        if not self._path_matches_handle(handle):
                            continue
                        return True
                    finally:
                        self._unlock(handle)
        raise RuntimeError("trace store path changed repeatedly during append")
