from __future__ import annotations

import errno
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

import agentic_tool_rl.evaluation.io as evaluation_io
from agentic_tool_rl.evaluation.recompute import recompute_metrics


@pytest.mark.parametrize(
    "row",
    [
        '{"success":true,"success":false}',
        '{"success":false,"success":true}',
        '{"final_state":{"success":true,"success":false}}',
        '{"steps":[{"accepted":true,"accepted":false}]}',
        '{"success":true,"\\u0073uccess":false}',
    ],
)
def test_read_jsonl_rejects_duplicate_object_keys(tmp_path: Path, row: str) -> None:
    source = tmp_path / "traces.jsonl"
    source.write_text('{}\n\n' + row + '\n', encoding="utf-8")

    with pytest.raises(ValueError) as error:
        evaluation_io.read_jsonl(source)

    assert f"invalid JSONL at {source}:3:" in str(error.value)
    assert "duplicate object key" in str(error.value)


def test_read_jsonl_allows_repeated_keys_in_separate_objects(tmp_path: Path) -> None:
    rows = [
        {"success": True, "steps": [{"accepted": True}, {"accepted": False}]},
        {"success": False, "final_state": {"success": True}},
    ]
    source = evaluation_io.write_jsonl_atomic(tmp_path / "traces.jsonl", rows)

    assert evaluation_io.read_jsonl(source) == rows


def test_recompute_rejects_ambiguous_trace_before_replacing_report(tmp_path: Path) -> None:
    source = tmp_path / "traces.jsonl"
    source.write_text(
        '{"case_id":"one","family":"test","success":false,"success":true,'
        '"steps":1,"optimal_steps":1,"simulated_latency_s":1.0}\n',
        encoding="utf-8",
    )
    output = tmp_path / "report.json"
    original = b'{"previous_report":true}\n'
    output.write_bytes(original)

    with pytest.raises(ValueError, match="duplicate object key 'success'"):
        recompute_metrics(source, output_path=output)

    assert output.read_bytes() == original


def _write_json(path: Path) -> Path:
    return evaluation_io.write_json_atomic(path, {"value": "new"})


def _write_jsonl(path: Path) -> Path:
    return evaluation_io.write_jsonl_atomic(path, [{"value": "new"}])


def _write_text(path: Path) -> Path:
    return evaluation_io.write_text_atomic(path, "new\n")


@pytest.mark.parametrize(
    "writer", (_write_json, _write_jsonl, _write_text), ids=("json", "jsonl", "text")
)
@pytest.mark.parametrize("failure_point", ("fsync", "replace"))
def test_atomic_writers_remove_temporary_file_after_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    writer: Callable[[Path], Path],
    failure_point: str,
) -> None:
    destination = tmp_path / "evidence.json"
    original = b'"original"\n'
    destination.write_bytes(original)

    def fail(*_args: object) -> None:
        raise OSError(f"injected {failure_point} failure")

    monkeypatch.setattr(evaluation_io.os, failure_point, fail)

    with pytest.raises(OSError, match=f"injected {failure_point} failure"):
        writer(destination)

    assert destination.read_bytes() == original
    assert list(tmp_path.iterdir()) == [destination]


def test_atomic_writer_removes_temporary_file_after_write_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "evidence.json"
    original = b'"original"\n'
    destination.write_bytes(original)
    real_named_temporary_file = evaluation_io.NamedTemporaryFile

    class WriteFailingTemporaryFile:
        def __init__(self, handle: Any) -> None:
            self._handle = handle
            self.name = handle.name

        def __enter__(self) -> WriteFailingTemporaryFile:
            self._handle.__enter__()
            return self

        def __exit__(self, *args: object) -> object:
            return self._handle.__exit__(*args)

        def write(self, _payload: bytes) -> int:
            raise OSError("injected write failure")

    def failing_named_temporary_file(*args: object, **kwargs: object) -> object:
        return WriteFailingTemporaryFile(real_named_temporary_file(*args, **kwargs))

    monkeypatch.setattr(evaluation_io, "NamedTemporaryFile", failing_named_temporary_file)

    with pytest.raises(OSError, match="injected write failure"):
        evaluation_io.write_json_atomic(destination, {"value": "new"})

    assert destination.read_bytes() == original
    assert list(tmp_path.iterdir()) == [destination]


def test_cleanup_failure_does_not_mask_replace_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "evidence.json"
    real_unlink = Path.unlink

    def fail_replace(*_args: object) -> None:
        raise OSError("primary replace failure")

    def fail_temporary_unlink(path: Path, *, missing_ok: bool = False) -> None:
        if path.name.startswith(f".{destination.name}."):
            raise PermissionError("secondary cleanup failure")
        real_unlink(path, missing_ok=missing_ok)

    monkeypatch.setattr(evaluation_io.os, "replace", fail_replace)
    monkeypatch.setattr(Path, "unlink", fail_temporary_unlink)

    with pytest.raises(OSError, match="primary replace failure"):
        evaluation_io.write_json_atomic(destination, {"value": "new"})


def test_atomic_writer_does_not_unlink_reused_temporary_path_after_replace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "evidence.json"
    recreated_paths: list[Path] = []
    real_replace = evaluation_io.os.replace

    def replace_and_reuse(source: str | Path, target: str | Path) -> None:
        real_replace(source, target)
        reused = Path(source)
        reused.write_bytes(b"owned by another writer\n")
        recreated_paths.append(reused)

    monkeypatch.setattr(evaluation_io.os, "replace", replace_and_reuse)

    evaluation_io.write_json_atomic(destination, {"value": "new"})

    assert len(recreated_paths) == 1
    assert recreated_paths[0].read_bytes() == b"owned by another writer\n"


def test_atomic_writer_fsyncs_parent_after_replace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "evidence.json"
    events: list[object] = []
    real_replace = evaluation_io.os.replace

    def replace(source: str | Path, target: str | Path) -> None:
        events.append("replace")
        real_replace(source, target)

    def fsync_directory(directory: Path) -> None:
        events.append(("fsync_directory", directory))

    monkeypatch.setattr(evaluation_io.os, "replace", replace)
    monkeypatch.setattr(evaluation_io, "_fsync_directory", fsync_directory)

    evaluation_io.write_json_atomic(destination, {"value": "new"})

    assert events == ["replace", ("fsync_directory", tmp_path)]


def test_directory_fsync_ignores_unsupported_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail_open(*_args: object) -> int:
        raise OSError(errno.EINVAL, "directory open unsupported")

    monkeypatch.setattr(evaluation_io.os, "open", fail_open)

    evaluation_io._fsync_directory(tmp_path)


def test_directory_fsync_ignores_unsupported_fsync_and_closes_fd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory_fd = 71
    closed: list[int] = []

    def fail_fsync(_fd: int) -> None:
        raise OSError(errno.EINVAL, "directory fsync unsupported")

    monkeypatch.setattr(evaluation_io.os, "open", lambda *_args: directory_fd)
    monkeypatch.setattr(evaluation_io.os, "fsync", fail_fsync)
    monkeypatch.setattr(evaluation_io.os, "close", closed.append)

    evaluation_io._fsync_directory(tmp_path)

    assert closed == [directory_fd]


def test_directory_fsync_propagates_open_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail_open(*_args: object) -> int:
        raise OSError(errno.EMFILE, "too many open files")

    monkeypatch.setattr(evaluation_io.os, "open", fail_open)

    with pytest.raises(OSError) as error:
        evaluation_io._fsync_directory(tmp_path)

    assert error.value.errno == errno.EMFILE


def test_directory_fsync_propagates_io_errors_and_closes_fd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory_fd = 71
    closed: list[int] = []

    def fail_fsync(_fd: int) -> None:
        raise OSError(errno.EIO, "directory fsync failed")

    monkeypatch.setattr(evaluation_io.os, "open", lambda *_args: directory_fd)
    monkeypatch.setattr(evaluation_io.os, "fsync", fail_fsync)
    monkeypatch.setattr(evaluation_io.os, "close", closed.append)

    with pytest.raises(OSError) as error:
        evaluation_io._fsync_directory(tmp_path)

    assert error.value.errno == errno.EIO
    assert closed == [directory_fd]


def test_directory_close_failure_does_not_mask_fsync_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory_fd = 71

    def fail_fsync(_fd: int) -> None:
        raise OSError(errno.EIO, "directory fsync failed")

    def fail_close(_fd: int) -> None:
        raise OSError(errno.EBADF, "directory close failed")

    monkeypatch.setattr(evaluation_io.os, "open", lambda *_args: directory_fd)
    monkeypatch.setattr(evaluation_io.os, "fsync", fail_fsync)
    monkeypatch.setattr(evaluation_io.os, "close", fail_close)

    with pytest.raises(OSError) as error:
        evaluation_io._fsync_directory(tmp_path)

    assert error.value.errno == errno.EIO
    assert error.value.__notes__ == [
        "failed to close directory fd: [Errno 9] directory close failed"
    ]
