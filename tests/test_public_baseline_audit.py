from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentic_tool_rl.contracts import Split
from agentic_tool_rl.envs.benchmark import generate_tasks
from agentic_tool_rl.evaluation.io import read_jsonl
from agentic_tool_rl.evaluation.public_baselines import (
    MANIFEST_FILENAME,
    TRACE_FILENAME,
    V13_DEFAULT_BASE_SEED,
    VERIFICATION_FILENAME,
    PublicReadyGreedy,
    verify_public_ready_audit,
    write_public_ready_audit,
)
from agentic_tool_rl.policy_input import PolicyInput


def test_public_ready_audit_defaults_to_the_frozen_canonical_seed() -> None:
    assert V13_DEFAULT_BASE_SEED == 2_006_549_735


def test_public_ready_baseline_accepts_only_policy_input() -> None:
    baseline = PublicReadyGreedy()

    with pytest.raises(TypeError, match="only PolicyInput"):
        baseline.select(object())  # type: ignore[arg-type]

    assert list(baseline.select.__annotations__) == ["policy_input", "return"]
    assert baseline.select.__annotations__["policy_input"] in {PolicyInput, "PolicyInput"}


def test_v13_public_ready_audit_is_deterministic_and_independently_replayable(
    tmp_path: Path,
) -> None:
    base_seed = 1402
    tasks = generate_tasks(Split.TEST, 4, base_seed=base_seed)
    first = tmp_path / "first"
    second = tmp_path / "second"

    write_public_ready_audit(first, tasks, base_seed=base_seed)
    verification = verify_public_ready_audit(first, tasks, base_seed=base_seed)
    write_public_ready_audit(second, list(reversed(tasks)), base_seed=base_seed)
    verify_public_ready_audit(second, list(reversed(tasks)), base_seed=base_seed)

    assert verification["passed"] is True
    assert verification["task_success_rate"] == 1.0
    for filename in (TRACE_FILENAME, MANIFEST_FILENAME, VERIFICATION_FILENAME):
        assert (first / filename).read_bytes() == (second / filename).read_bytes()

    manifest = json.loads((first / MANIFEST_FILENAME).read_text(encoding="utf-8"))
    traces = read_jsonl(first / TRACE_FILENAME)
    assert manifest["case_count"] == manifest["successful_cases"] == len(tasks)
    assert manifest["mean_steps"] == sum(task.optimal_steps for task in tasks) / len(tasks)
    assert all(trace["steps"] == trace["optimal_steps"] for trace in traces)
    assert all(
        decision["selection_reason"] == "selectable+policy_allowed+mutating+unfinished"
        for trace in traces
        for decision in trace["decisions"]
    )
    assert "call_id" not in (first / TRACE_FILENAME).read_text(encoding="utf-8")
    assert all(task.entity_id not in (first / TRACE_FILENAME).read_text() for task in tasks)


def test_public_ready_audit_verification_fails_closed_on_trace_tampering(tmp_path: Path) -> None:
    base_seed = 1403
    tasks = generate_tasks(Split.TEST, 1, base_seed=base_seed)
    write_public_ready_audit(tmp_path, tasks, base_seed=base_seed)
    traces = read_jsonl(tmp_path / TRACE_FILENAME)
    traces[0]["success"] = False
    (tmp_path / TRACE_FILENAME).write_text(
        "".join(f"{json.dumps(row, sort_keys=True)}\n" for row in traces),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="trace replay mismatch"):
        verify_public_ready_audit(tmp_path, tasks, base_seed=base_seed)


def test_public_ready_audit_rejects_misdeclared_base_seed(tmp_path: Path) -> None:
    tasks = generate_tasks(Split.TEST, 1, base_seed=1404)

    with pytest.raises(ValueError, match=r"declared frozen v1\.3 base seed"):
        write_public_ready_audit(tmp_path, tasks, base_seed=9999)
