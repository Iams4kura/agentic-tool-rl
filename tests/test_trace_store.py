from __future__ import annotations

import json
import multiprocessing
import os
import threading
from pathlib import Path
from queue import Empty, Queue
from types import SimpleNamespace
from typing import Any

import pytest

import agentic_tool_rl.evaluation.trace_store as trace_store_module
from agentic_tool_rl.evaluation.io import canonical_json
from agentic_tool_rl.evaluation.trace_store import TraceStore

_PROCESS_START_TIMEOUT_S = 30
_WRITER_RELEASE_TIMEOUT_S = 60


def _record(case_id: str, **values: object) -> dict[str, object]:
    return {"case_id": case_id, **values}


def _write_half_record_under_lock(
    path: str,
    prefix: bytes,
    suffix: bytes,
    ready: Any,
    release: Any,
) -> None:
    import fcntl

    with Path(path).open("a+b", buffering=0) as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            handle.write(prefix)
            os.fsync(handle.fileno())
            ready.set()
            if not release.wait(_WRITER_RELEASE_TIMEOUT_S):
                raise TimeoutError("test writer was not released")
            handle.write(suffix + b"\n")
            os.fsync(handle.fileno())
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _open_store(
    path: str,
    started: Any,
    finished: Any,
    result: Any,
) -> None:
    started.set()
    try:
        result.put(("ok", TraceStore(path).records()))
    except Exception as exc:  # pragma: no cover - asserted in the parent process
        result.put(("error", repr(exc)))
    finally:
        finished.set()


def test_append_parses_records_in_linear_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = 0
    real_loads = json.loads

    def counted_loads(value: str | bytes | bytearray, *args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        return real_loads(value, *args, **kwargs)

    monkeypatch.setattr(
        trace_store_module,
        "json",
        SimpleNamespace(loads=counted_loads, JSONDecodeError=json.JSONDecodeError),
    )
    store = TraceStore(tmp_path / "linear.jsonl")
    count = 200

    for index in range(count):
        assert store.append(_record(f"case-{index:04d}", value=index))

    # Each input is normalized once. Already indexed records are never parsed
    # again on the single-writer append path.
    assert calls == count
    assert len(store) == count


def test_two_instances_incrementally_absorb_each_others_appends(tmp_path: Path) -> None:
    path = tmp_path / "alternating.jsonl"
    first = TraceStore(path)
    second = TraceStore(path)
    records = (
        _record("case-1", writer="first"),
        _record("case-2", writer="second"),
        _record("case-3", writer="first"),
    )

    assert first.append(records[0])
    assert second.append(records[1])
    assert first.append(records[2])
    # This append synchronizes case-3 into the second instance before applying
    # the normal idempotency rule.
    assert second.append(records[2]) is False

    assert first.records() == list(records)
    assert second.records() == list(records)
    assert path.read_bytes() == b"".join(
        f"{canonical_json(record)}\n".encode() for record in records
    )


@pytest.mark.skipif(
    getattr(trace_store_module, "fcntl", None) is None,
    reason="cross-process trace locking requires fcntl",
)
def test_opener_waits_for_locked_half_record_without_truncating(tmp_path: Path) -> None:
    path = tmp_path / "locked.jsonl"
    record = _record("case-locked", payload="committed by the writer")
    line = canonical_json(record).encode()
    split_at = len(line) // 2
    context = multiprocessing.get_context("spawn")
    writer_ready = context.Event()
    release_writer = context.Event()
    opener_started = threading.Event()
    opener_finished = threading.Event()
    result: Queue[tuple[str, object]] = Queue()
    writer = context.Process(
        target=_write_half_record_under_lock,
        args=(
            str(path),
            line[:split_at],
            line[split_at:],
            writer_ready,
            release_writer,
        ),
    )
    opener = threading.Thread(
        target=_open_store,
        args=(str(path), opener_started, opener_finished, result),
        daemon=True,
    )

    writer.start()
    try:
        assert writer_ready.wait(_PROCESS_START_TIMEOUT_S), (
            "writer did not acquire the file lock"
        )
        opener.start()
        assert opener_started.wait(_PROCESS_START_TIMEOUT_S), (
            "opener process did not start"
        )
        assert not opener_finished.wait(0.25), (
            "TraceStore opened before the active writer released its lock"
        )
        release_writer.set()
        assert opener_finished.wait(_PROCESS_START_TIMEOUT_S), (
            "opener remained blocked after commit"
        )
        opener.join(_PROCESS_START_TIMEOUT_S)
        writer.join(_PROCESS_START_TIMEOUT_S)
        assert writer.exitcode == 0
        assert not opener.is_alive()
        try:
            status, payload = result.get(timeout=2)
        except Empty as exc:  # pragma: no cover - diagnostic failure path
            raise AssertionError("opener returned no result") from exc
        assert status == "ok", payload
        assert payload == [record]
        assert path.read_bytes() == line + b"\n"
    finally:
        release_writer.set()
        opener.join(1)
        writer.join(1)
        if writer.is_alive():
            writer.terminate()
            writer.join(5)


def test_append_rebuilds_index_after_external_truncation(tmp_path: Path) -> None:
    path = tmp_path / "truncated.jsonl"
    store = TraceStore(path)
    assert store.append(_record("old", payload="x" * 1_024))
    replacement = _record("replacement")
    path.write_bytes(f"{canonical_json(replacement)}\n".encode())

    fresh = _record("fresh")
    assert store.append(fresh)

    assert store.records() == [replacement, fresh]


def test_append_rebuilds_index_after_inode_replacement(tmp_path: Path) -> None:
    path = tmp_path / "replaced.jsonl"
    store = TraceStore(path)
    assert store.append(_record("old"))
    replacement = _record("replacement")
    replacement_path = tmp_path / "replacement.tmp"
    replacement_path.write_bytes(f"{canonical_json(replacement)}\n".encode())
    os.replace(replacement_path, path)

    fresh = _record("fresh")
    assert store.append(fresh)

    assert store.records() == [replacement, fresh]


def test_append_reopens_when_path_is_replaced_before_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "replace-before-lock.jsonl"
    store = TraceStore(path)
    assert store.append(_record("old"))
    replacement = _record("replacement")
    appended = _record("appended")
    replacement_path = tmp_path / "replacement-before-lock.tmp"
    replacement_path.write_bytes(f"{canonical_json(replacement)}\n".encode())
    first_lock_opened = threading.Event()
    allow_first_lock = threading.Event()
    append_result: Queue[tuple[str, object]] = Queue()
    real_lock = TraceStore._lock
    lock_calls = 0

    def delayed_first_lock(handle: Any) -> None:
        nonlocal lock_calls
        lock_calls += 1
        if lock_calls == 1:
            first_lock_opened.set()
            if not allow_first_lock.wait(_PROCESS_START_TIMEOUT_S):
                raise TimeoutError("replacement test did not release the first lock")
        real_lock(handle)

    monkeypatch.setattr(TraceStore, "_lock", staticmethod(delayed_first_lock))

    def append_in_thread() -> None:
        try:
            append_result.put(("ok", store.append(appended)))
        except Exception as exc:  # pragma: no cover - asserted below
            append_result.put(("error", repr(exc)))

    worker = threading.Thread(target=append_in_thread, daemon=True)
    worker.start()
    try:
        assert first_lock_opened.wait(_PROCESS_START_TIMEOUT_S)
        os.replace(replacement_path, path)
        allow_first_lock.set()
        worker.join(_PROCESS_START_TIMEOUT_S)
        assert not worker.is_alive()
        status, result = append_result.get(timeout=2)
        assert status == "ok", result
        assert result is True
        assert store.records() == [replacement, appended]
        assert TraceStore(path).records() == [replacement, appended]
    finally:
        allow_first_lock.set()
        worker.join(1)
