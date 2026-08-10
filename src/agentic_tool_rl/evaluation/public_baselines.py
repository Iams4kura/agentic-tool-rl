"""Public-only rule baselines and replayable post-hoc audit evidence."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agentic_tool_rl.contracts import Split, WorkflowTask
from agentic_tool_rl.envs.benchmark import generate_tasks
from agentic_tool_rl.envs.workflow import TransactionalWorkflowEnv
from agentic_tool_rl.evaluation.io import (
    canonical_json,
    read_jsonl,
    sha256_json,
    write_json_atomic,
    write_jsonl_atomic,
)
from agentic_tool_rl.grounding.action_mask import ActionMask
from agentic_tool_rl.policy_input import PolicyInput

PUBLIC_READY_GREEDY_ID = "R1-public-ready-greedy-v1"
AUDIT_SCHEMA_VERSION = "v1.3-public-ready-audit-v1"
# Frozen by docs/protocol/preregistration-v1.3.md and configs/cpu_full.yaml.
V13_DEFAULT_BASE_SEED = 2_006_549_735
V13_GENERATOR_VERSION = "benchmark-v1.3.0"
TRACE_FILENAME = "v1.3-public-ready-audit.jsonl"
MANIFEST_FILENAME = "v1.3-public-ready-audit.manifest.json"
VERIFICATION_FILENAME = "v1.3-public-ready-audit.verification.json"


@dataclass(frozen=True, slots=True)
class PublicBaselineDecision:
    action_index: int
    reason: str


class PublicReadyGreedy:
    """The v1.3 shortcut, intentionally limited to one ``PolicyInput`` value."""

    baseline_id = PUBLIC_READY_GREEDY_ID

    def select(self, policy_input: PolicyInput) -> PublicBaselineDecision:
        if not isinstance(policy_input, PolicyInput):
            raise TypeError("public baselines accept only PolicyInput")
        completed = set(policy_input.observation.visible_state.get("completed_nodes", []))
        for index, candidate in enumerate(policy_input.candidates):
            schema = candidate.tool_schema
            if (
                candidate.action_mask
                and schema.policy_allowed
                and schema.mutating
                and schema.operation_id is not None
                and schema.operation_id not in completed
            ):
                return PublicBaselineDecision(
                    action_index=index,
                    reason="selectable+policy_allowed+mutating+unfinished",
                )
        for index, candidate in enumerate(policy_input.candidates):
            if candidate.action_mask:
                return PublicBaselineDecision(
                    action_index=index,
                    reason="fallback:first-selectable",
                )
        raise ValueError("PolicyInput contains no selectable candidate")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _case_trace(task: WorkflowTask, baseline: PublicReadyGreedy) -> dict[str, Any]:
    environment = TransactionalWorkflowEnv(task)
    mask = ActionMask(task.tool_schemas)
    decisions: list[dict[str, Any]] = []
    while not environment.done:
        observation = environment.observe()
        candidates = environment.candidate_actions()
        action_mask = mask.mask(observation, candidates)
        policy_input = PolicyInput.from_decision(
            observation,
            candidates,
            action_mask,
            task.tool_schemas,
        )
        selection = baseline.select(policy_input)
        selected_public = policy_input.candidates[selection.action_index].tool_call
        outcome = environment.step(candidates[selection.action_index])
        decisions.append(
            {
                "step_index": observation.step_index,
                "policy_input_sha256": policy_input.sha256(),
                "action_index": selection.action_index,
                "selected_tool_call": selected_public.model_dump(mode="json"),
                "selection_reason": selection.reason,
                "accepted": outcome.accepted,
                "done": outcome.done,
                "success": outcome.success,
            }
        )
    evaluation = environment.evaluate()
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "case_id": task.case_id,
        "family": task.family,
        "difficulty": task.difficulty,
        "topology": task.topology,
        "success": evaluation.success,
        "forbidden_side_effect_count": evaluation.forbidden_side_effect_count,
        "steps": environment.steps_taken,
        "optimal_steps": task.optimal_steps,
        "policy_input_sequence_sha256": sha256_json(
            [decision["policy_input_sha256"] for decision in decisions]
        ),
        "decisions": decisions,
    }


def _validate_v13_tasks(tasks: Sequence[WorkflowTask]) -> list[WorkflowTask]:
    ordered = sorted(tasks, key=lambda task: task.case_id)
    if not ordered:
        raise ValueError("audit requires at least one task")
    if len({task.case_id for task in ordered}) != len(ordered):
        raise ValueError("audit task case_id values must be unique")
    if any(task.generator_version != V13_GENERATOR_VERSION for task in ordered):
        raise ValueError(f"audit accepts only frozen {V13_GENERATOR_VERSION} tasks")
    if any(task.split != Split.TEST for task in ordered):
        raise ValueError("v1.3 public-ready audit accepts only test tasks")
    return ordered


def build_public_ready_audit(
    tasks: Sequence[WorkflowTask],
    *,
    base_seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run R1 and return deterministic per-case traces plus their manifest."""

    ordered = _validate_v13_tasks(tasks)
    regenerated = generate_tasks(Split.TEST, len(ordered), base_seed=base_seed)
    if [task.model_dump(mode="json") for task in ordered] != [
        task.model_dump(mode="json") for task in regenerated
    ]:
        raise ValueError("tasks do not match the declared frozen v1.3 base seed and size")
    baseline = PublicReadyGreedy()
    traces = [_case_trace(task, baseline) for task in ordered]
    successful = sum(bool(trace["success"]) for trace in traces)
    total_steps = sum(int(trace["steps"]) for trace in traces)
    forbidden = sum(int(trace["forbidden_side_effect_count"]) for trace in traces)
    cases = [
        {
            "case_id": trace["case_id"],
            "success": trace["success"],
            "steps": trace["steps"],
            "optimal_steps": trace["optimal_steps"],
            "policy_input_sequence_sha256": trace["policy_input_sequence_sha256"],
            "trace_row_sha256": sha256_json(trace),
        }
        for trace in traces
    ]
    trace_payload = "".join(f"{canonical_json(row)}\n" for row in traces).encode("utf-8")
    manifest: dict[str, Any] = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "baseline_id": PUBLIC_READY_GREEDY_ID,
        "benchmark_generator_version": V13_GENERATOR_VERSION,
        "split": Split.TEST.value,
        "base_seed": base_seed,
        "case_count": len(traces),
        "successful_cases": successful,
        "task_success_rate": successful / len(traces),
        "mean_steps": total_steps / len(traces),
        "forbidden_side_effect_count": forbidden,
        "trace": {
            "path": TRACE_FILENAME,
            "rows": len(traces),
            "sha256": hashlib.sha256(trace_payload).hexdigest(),
        },
        "source_case_ids_sha256": sha256_json([task.case_id for task in ordered]),
        "source_tasks_sha256": sha256_json(
            [task.model_dump(mode="json") for task in ordered]
        ),
        "cases": cases,
    }
    return traces, manifest


def write_public_ready_audit(
    output_dir: str | Path,
    tasks: Sequence[WorkflowTask],
    *,
    base_seed: int,
) -> tuple[Path, Path]:
    """Write deterministic trace and per-case manifest files atomically."""

    directory = Path(output_dir)
    traces, manifest = build_public_ready_audit(tasks, base_seed=base_seed)
    trace_path = write_jsonl_atomic(directory / TRACE_FILENAME, traces)
    if _file_sha256(trace_path) != manifest["trace"]["sha256"]:
        raise RuntimeError("written audit trace differs from the canonical payload")
    manifest_path = write_json_atomic(directory / MANIFEST_FILENAME, manifest)
    write_json_atomic(
        directory / VERIFICATION_FILENAME,
        {
            "schema_version": AUDIT_SCHEMA_VERSION,
            "passed": False,
            "status": "pending-independent-replay",
            "trace_sha256": _file_sha256(trace_path),
            "manifest_sha256": _file_sha256(manifest_path),
        },
    )
    return trace_path, manifest_path


def verify_public_ready_audit(
    output_dir: str | Path,
    tasks: Sequence[WorkflowTask],
    *,
    base_seed: int,
) -> dict[str, Any]:
    """Independently replay all cases and compare every trace and manifest byte."""

    directory = Path(output_dir)
    trace_path = directory / TRACE_FILENAME
    manifest_path = directory / MANIFEST_FILENAME
    actual_traces = read_jsonl(trace_path)
    expected_traces, expected_manifest = build_public_ready_audit(tasks, base_seed=base_seed)
    if actual_traces != expected_traces:
        raise RuntimeError("public-ready audit trace replay mismatch")
    actual_manifest = _read_mapping(manifest_path)
    if actual_manifest != expected_manifest:
        raise RuntimeError("public-ready audit manifest recomputation mismatch")
    if _file_sha256(trace_path) != actual_manifest["trace"]["sha256"]:
        raise RuntimeError("public-ready audit trace checksum mismatch")
    verification = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "passed": True,
        "replayed_case_count": len(expected_traces),
        "successful_cases": expected_manifest["successful_cases"],
        "task_success_rate": expected_manifest["task_success_rate"],
        "mean_steps": expected_manifest["mean_steps"],
        "trace_sha256": _file_sha256(trace_path),
        "manifest_sha256": _file_sha256(manifest_path),
    }
    write_json_atomic(directory / VERIFICATION_FILENAME, verification)
    return verification


def _read_mapping(path: Path) -> dict[str, Any]:
    import json

    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must contain a JSON object")
    return dict(value)


__all__ = [
    "AUDIT_SCHEMA_VERSION",
    "MANIFEST_FILENAME",
    "PUBLIC_READY_GREEDY_ID",
    "TRACE_FILENAME",
    "V13_DEFAULT_BASE_SEED",
    "V13_GENERATOR_VERSION",
    "VERIFICATION_FILENAME",
    "PublicBaselineDecision",
    "PublicReadyGreedy",
    "build_public_ready_audit",
    "verify_public_ready_audit",
    "write_public_ready_audit",
]
