"""Registered public baselines and quality gates for benchmark-v1.4."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from agentic_tool_rl.contracts import CounterfactualWorkflowTask
from agentic_tool_rl.envs.benchmark import _FAMILY_TEMPLATES
from agentic_tool_rl.envs.oracle import verify_task_solvable
from agentic_tool_rl.envs.workflow import TransactionalWorkflowEnv
from agentic_tool_rl.evaluation.io import (
    canonical_json,
    read_jsonl,
    sha256_json,
    write_json_atomic,
    write_jsonl_atomic,
)
from agentic_tool_rl.evaluation.public_baselines import (
    PublicBaselineDecision,
    PublicReadyGreedy,
)
from agentic_tool_rl.grounding.action_mask import ActionMask
from agentic_tool_rl.policy_input import PolicyInput

BASELINE_REPORT_SCHEMA = "benchmark-v1.4-baselines-v1"
PUBLIC_BASELINE_IDS = (
    "R0-masked-random-v1",
    PublicReadyGreedy.baseline_id,
    "R2-public-dag-greedy-v1",
    "R3-goal-schema-overlap-v1",
    "R4-reviewer-composite-shortcut-v1",
    "R5-source-aware-family-stage-v1",
    "R6-public-downstream-centrality-v1",
)
_TOKEN = re.compile(r"[a-z0-9]+")
_R5_FAMILY_STAGES = {
    template.key: frozenset(stage.replace("_", " ") for stage in template.stages)
    for template in _FAMILY_TEMPLATES
}
_R5_TRANSITION_GOALS = {
    "0": (
        "Introduce the update through successive limited waves.",
        "Phase adoption across contained groups, checking each before continuing.",
        "Move through a sequence of small cohorts instead of changing everything together.",
        "Spread the rollout over several controlled stages.",
    ),
    "1": (
        "Activate the whole scope together in one coordinated event.",
        "Move every affected unit in a single organization-wide switchover.",
        "Complete one unified change across the entire scope.",
        "Use a single cutover for all affected groups.",
    ),
}
_R5_DECISION_GOALS = {
    "0": (
        "Keep an accountable human owner at the final decision.",
        "A designated individual must remain responsible for authorization.",
        "Retain a human sign-off before completion.",
        "Place the concluding judgment with a named reviewer.",
    ),
    "1": (
        "Let automated validation authorize completion without a manual handoff.",
        "Proceed on system verification alone at the concluding checkpoint.",
        "Allow machine validation to carry the workflow through completion.",
        "Use automated controls, without human sign-off, for the closing decision.",
    ),
}
_R5_GATE_DESCRIPTIONS = {
    "a0": (
        "Apply the change to bounded batches with checkpoints.",
        "Advance through staged waves and verify each checkpoint.",
        "Roll out to one contained cohort at a time.",
        "Use a phased deployment over multiple limited groups.",
    ),
    "a1": (
        "Move the full scope through one coordinated cutover.",
        "Switch every affected unit during the same deployment event.",
        "Apply the change everywhere in one operation.",
        "Use a unified all-scope transition.",
    ),
    "b0": (
        "Require a named reviewer before completion.",
        "Wait for authorization from a designated human owner.",
        "Route the concluding judgment to an accountable individual.",
        "Hold completion until a human sign-off is recorded.",
    ),
    "b1": (
        "Continue after machine checks without a manual handoff.",
        "Let automated controls authorize the closing step.",
        "Proceed when system validation passes, with no reviewer required.",
        "Use automatic verification as the final authority.",
    ),
}


class PublicBaseline(Protocol):
    baseline_id: str

    def select(self, policy_input: PolicyInput) -> PublicBaselineDecision: ...


def _require_policy_input(value: PolicyInput) -> None:
    if not isinstance(value, PolicyInput):
        raise TypeError("public baselines accept only PolicyInput")


def _number(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    return float(value)


def _fallback(policy_input: PolicyInput) -> PublicBaselineDecision:
    for index, candidate in enumerate(policy_input.candidates):
        if candidate.action_mask:
            return PublicBaselineDecision(index, "fallback:first-selectable")
    raise ValueError("PolicyInput contains no selectable candidate")


class MaskedRandom:
    """Deterministic masked-random floor that deliberately ignores the goal."""

    baseline_id = PUBLIC_BASELINE_IDS[0]

    def __init__(self, seed: int = 20_260_810) -> None:
        self._seed = seed

    def select(self, policy_input: PolicyInput) -> PublicBaselineDecision:
        _require_policy_input(policy_input)
        allowed = [
            index
            for index, candidate in enumerate(policy_input.candidates)
            if candidate.action_mask
        ]
        if not allowed:
            raise ValueError("PolicyInput contains no selectable candidate")
        payload = policy_input.model_dump(mode="json")
        observation = payload["observation"]
        if not isinstance(observation, dict):  # pragma: no cover - Pydantic invariant
            raise TypeError("PolicyInput observation must serialize to an object")
        observation = dict(observation)
        observation["user_goal"] = "<goal-withheld>"
        payload["observation"] = observation
        digest = hashlib.sha256(f"{self._seed}:{canonical_json(payload)}".encode()).digest()
        action_index = allowed[int.from_bytes(digest[:8], "big") % len(allowed)]
        return PublicBaselineDecision(action_index, "masked-random:goal-withheld")


class PublicDagGreedy:
    """Goal-blind heuristic preferring the deepest ready public DAG node."""

    baseline_id = PUBLIC_BASELINE_IDS[2]

    def select(self, policy_input: PolicyInput) -> PublicBaselineDecision:
        _require_policy_input(policy_input)
        completed = set(policy_input.observation.visible_state.get("completed_nodes", []))
        ranked: list[tuple[int, int]] = []
        for index, candidate in enumerate(policy_input.candidates):
            schema = candidate.tool_schema
            if (
                candidate.action_mask
                and schema.mutating
                and schema.policy_allowed
                and schema.operation_id is not None
                and schema.operation_id not in completed
            ):
                ranked.append((len(schema.required_completed_operations), -index))
        if not ranked:
            return _fallback(policy_input)
        _, negative_index = max(ranked)
        return PublicBaselineDecision(-negative_index, "deepest-ready-public-dag-node")


class GoalSchemaOverlap:
    """Shallow lexical goal/schema heuristic registered before dev evaluation."""

    baseline_id = PUBLIC_BASELINE_IDS[3]

    def select(self, policy_input: PolicyInput) -> PublicBaselineDecision:
        _require_policy_input(policy_input)
        goal_tokens = set(_TOKEN.findall(policy_input.observation.user_goal.casefold()))
        completed = set(policy_input.observation.visible_state.get("completed_nodes", []))
        ranked: list[tuple[float, int]] = []
        for index, candidate in enumerate(policy_input.candidates):
            schema = candidate.tool_schema
            if not (
                candidate.action_mask
                and schema.mutating
                and schema.operation_id is not None
                and schema.operation_id not in completed
            ):
                continue
            schema_tokens = set(_TOKEN.findall(f"{schema.name} {schema.description}".casefold()))
            union = goal_tokens | schema_tokens
            score = len(goal_tokens & schema_tokens) / len(union) if union else 0.0
            ranked.append((score, -index))
        if not ranked:
            return _fallback(policy_input)
        score, negative_index = max(ranked)
        return PublicBaselineDecision(-negative_index, f"goal-schema-jaccard:{score:.8f}")


class ReviewerCompositeShortcut:
    """Frozen public-only rule that exposed the first v1.4 generator shortcut.

    Before the red-team fix, composing an explicit ``optional advisory``
    filter, two global goal-to-tool-name mappings, and the ``finalize_`` name
    marker solved all 400 canonical development cases.  Keeping the exact rule
    registered prevents that failure mode from silently returning.
    """

    baseline_id = PUBLIC_BASELINE_IDS[4]

    def select(self, policy_input: PolicyInput) -> PublicBaselineDecision:
        _require_policy_input(policy_input)
        goal = policy_input.observation.user_goal.casefold()
        state = policy_input.observation.visible_state
        completed = set(state.get("completed_nodes", []))
        ready = [
            (index, candidate)
            for index, candidate in enumerate(policy_input.candidates)
            if candidate.action_mask and candidate.tool_schema.operation_id not in completed
        ]
        if not ready:
            return _fallback(policy_input)

        for index, candidate in ready:
            if ".finalize_" in candidate.tool_schema.name:
                return PublicBaselineDecision(index, "reviewer:finalize-ready")
        if state.get("transition_mode") is None:
            transition = "stage_in_bounded_batches" if "gradual" in goal else "switch_all_at_once"
            for index, candidate in ready:
                if transition in candidate.tool_schema.name:
                    return PublicBaselineDecision(index, "reviewer:transition-name-map")
        if state.get("decision_mode") is None:
            decision = (
                "hold_for_named_reviewer" if "person" in goal else "continue_after_automatic_checks"
            )
            for index, candidate in ready:
                if decision in candidate.tool_schema.name:
                    return PublicBaselineDecision(index, "reviewer:decision-name-map")
        for index, candidate in ready:
            schema = candidate.tool_schema
            if (
                schema.mutating
                and "optional advisory" not in schema.description.casefold()
                and "force_complete" not in schema.name
            ):
                return PublicBaselineDecision(index, "reviewer:non-advisory-mutation")
        return PublicBaselineDecision(ready[0][0], "reviewer:first-ready")


def _contains_frozen_text(value: str, options: Sequence[str]) -> bool:
    normalized = value.casefold()
    return any(option.casefold() in normalized for option in options)


class SourceAwareFamilyStageShortcut:
    """Frozen source-aware rule that solved the first R4-hardened generator.

    The rule contains only repository-public, pre-fix semantic tables and
    consumes exactly one :class:`PolicyInput` per decision.  It never reads a
    task object, evaluator field, oracle plan, case ID, or goal-variant label.
    """

    baseline_id = PUBLIC_BASELINE_IDS[5]

    def select(self, policy_input: PolicyInput) -> PublicBaselineDecision:
        _require_policy_input(policy_input)
        state = policy_input.observation.visible_state
        completed = set(state.get("completed_nodes", []))
        ready = [
            (index, candidate)
            for index, candidate in enumerate(policy_input.candidates)
            if candidate.action_mask
            and candidate.tool_schema.mutating
            and candidate.tool_schema.operation_id not in completed
        ]
        if not ready:
            return _fallback(policy_input)

        goal = policy_input.observation.user_goal
        if state.get("transition_mode") is None:
            goal_bits = [
                bit
                for bit, options in _R5_TRANSITION_GOALS.items()
                if _contains_frozen_text(goal, options)
            ]
            if len(goal_bits) == 1:
                descriptions = _R5_GATE_DESCRIPTIONS[f"a{goal_bits[0]}"]
                for index, candidate in ready:
                    if _contains_frozen_text(
                        candidate.tool_schema.description,
                        descriptions,
                    ):
                        return PublicBaselineDecision(index, "source-aware:transition-text")

        if state.get("decision_mode") is None:
            goal_bits = [
                bit
                for bit, options in _R5_DECISION_GOALS.items()
                if _contains_frozen_text(goal, options)
            ]
            if len(goal_bits) == 1:
                descriptions = _R5_GATE_DESCRIPTIONS[f"b{goal_bits[0]}"]
                for index, candidate in ready:
                    if _contains_frozen_text(
                        candidate.tool_schema.description,
                        descriptions,
                    ):
                        return PublicBaselineDecision(index, "source-aware:decision-text")

        for index, candidate in ready:
            schema = candidate.tool_schema
            family = schema.name.split(".", 1)[0]
            stages = _R5_FAMILY_STAGES.get(family, frozenset())
            description = schema.description.casefold()
            if any(f"carry out {stage}" in description for stage in stages):
                return PublicBaselineDecision(index, "source-aware:family-stage")
        for index, candidate in ready:
            if candidate.tool_schema.side_effect is not None:
                return PublicBaselineDecision(index, "source-aware:non-null-side-effect")
        return PublicBaselineDecision(ready[0][0], "source-aware:first-ready")


class PublicDownstreamCentrality:
    """Prefer ready operations referenced by the visible candidate DAG frontier."""

    baseline_id = PUBLIC_BASELINE_IDS[6]

    def select(self, policy_input: PolicyInput) -> PublicBaselineDecision:
        _require_policy_input(policy_input)
        completed = set(policy_input.observation.visible_state.get("completed_nodes", []))
        public_schemas = {
            candidate.tool_schema.operation_id: candidate.tool_schema
            for candidate in policy_input.candidates
            if candidate.tool_schema.operation_id is not None
        }
        downstream_counts: Counter[str] = Counter(
            predecessor
            for schema in public_schemas.values()
            for predecessor in schema.required_completed_operations
        )
        ranked: list[tuple[int, int, int, int]] = []
        for index, candidate in enumerate(policy_input.candidates):
            schema = candidate.tool_schema
            operation = schema.operation_id
            if (
                candidate.action_mask
                and schema.mutating
                and operation is not None
                and operation not in completed
            ):
                ranked.append(
                    (
                        downstream_counts[operation],
                        len(schema.required_completed_operations),
                        -index,
                        index,
                    )
                )
        if not ranked:
            return _fallback(policy_input)
        downstream, depth, _, index = max(ranked)
        return PublicBaselineDecision(
            index,
            f"downstream-centrality:{downstream}:public-depth:{depth}",
        )


@dataclass(frozen=True, slots=True)
class BaselineSuiteResult:
    report: dict[str, object]
    traces: dict[str, list[dict[str, object]]]


def registered_public_baselines(*, random_seed: int = 20_260_810) -> tuple[PublicBaseline, ...]:
    return (
        MaskedRandom(random_seed),
        PublicReadyGreedy(),
        PublicDagGreedy(),
        GoalSchemaOverlap(),
        ReviewerCompositeShortcut(),
        SourceAwareFamilyStageShortcut(),
        PublicDownstreamCentrality(),
    )


def _run_public_baseline(
    baseline: PublicBaseline,
    tasks: Sequence[CounterfactualWorkflowTask],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    traces: list[dict[str, object]] = []
    for task in sorted(tasks, key=lambda item: item.case_id):
        environment = TransactionalWorkflowEnv(task)
        action_mask = ActionMask(task.tool_schemas)
        decisions: list[dict[str, object]] = []
        while not environment.done:
            observation = environment.observe()
            candidates = environment.candidate_actions(include_invalid=True)
            mask = action_mask.mask(observation, candidates)
            policy_input = PolicyInput.from_decision(
                observation,
                candidates,
                mask,
                task.tool_schemas,
            )
            selection = baseline.select(policy_input)
            if not mask[selection.action_index]:
                raise RuntimeError(f"{baseline.baseline_id} selected a masked action")
            selected = policy_input.candidates[selection.action_index].tool_call
            outcome = environment.step(candidates[selection.action_index])
            decisions.append(
                {
                    "step_index": observation.step_index,
                    "policy_input_sha256": policy_input.sha256(),
                    "action_index": selection.action_index,
                    "selected_tool_call": selected.model_dump(mode="json"),
                    "reason": selection.reason,
                    "accepted": outcome.accepted,
                    "done": outcome.done,
                }
            )
        evaluation = environment.evaluate()
        traces.append(
            {
                "schema_version": BASELINE_REPORT_SCHEMA,
                "baseline_id": baseline.baseline_id,
                "case_id": task.case_id,
                "group_id": task.group_id,
                "goal_variant": task.goal_variant,
                "success": evaluation.success,
                "forbidden_side_effect_count": evaluation.forbidden_side_effect_count,
                "steps": environment.steps_taken,
                "decisions": decisions,
                "trace_sha256": sha256_json(decisions),
            }
        )
    successes = sum(bool(trace["success"]) for trace in traces)
    summary: dict[str, object] = {
        "baseline_id": baseline.baseline_id,
        "cases": len(traces),
        "successful_cases": successes,
        "tsr": successes / len(traces),
        "forbidden_side_effect_count": sum(
            int(_number(trace["forbidden_side_effect_count"], name="forbidden count"))
            for trace in traces
        ),
        "mean_steps": sum(int(_number(trace["steps"], name="trace steps")) for trace in traces)
        / len(traces),
        "traces_sha256": sha256_json(traces),
    }
    return traces, summary


def run_baseline_suite(
    tasks: Sequence[CounterfactualWorkflowTask],
    *,
    random_seed: int = 20_260_810,
) -> BaselineSuiteResult:
    if not tasks:
        raise ValueError("baseline suite requires at least one task")
    if len({task.case_id for task in tasks}) != len(tasks):
        raise ValueError("baseline suite case IDs must be unique")
    if any(not task.generator_version.startswith("benchmark-v1.4") for task in tasks):
        raise ValueError("baseline suite accepts only benchmark-v1.4 tasks")

    oracle_reports = [
        [verify_task_solvable(task, plan_index) for plan_index in range(len(task.oracle_plans))]
        for task in tasks
    ]
    oracle_successes = sum(
        len(reports) >= 4
        and all(
            report.solvable and report.executed_steps == task.optimal_steps for report in reports
        )
        for reports, task in zip(oracle_reports, tasks, strict=True)
    )
    oracle_plans_replayed = sum(len(reports) for reports in oracle_reports)
    traces: dict[str, list[dict[str, object]]] = {}
    summaries: dict[str, dict[str, object]] = {}
    for baseline in registered_public_baselines(random_seed=random_seed):
        baseline_traces, summary = _run_public_baseline(baseline, tasks)
        traces[baseline.baseline_id] = baseline_traces
        summaries[baseline.baseline_id] = summary

    oracle_tsr = oracle_successes / len(tasks)
    r1_tsr = _number(
        summaries[PublicReadyGreedy.baseline_id]["tsr"],
        name="R1 TSR",
    )
    strongest_rule_tsr = max(
        _number(summary["tsr"], name=f"{baseline_id} TSR")
        for baseline_id, summary in summaries.items()
        if baseline_id != MaskedRandom.baseline_id
    )
    checks = {
        "oracle_tsr_eq_1": oracle_tsr == 1.0,
        "oracle_steps_exact": oracle_successes == len(tasks),
        "r1_tsr_lte_0_25": r1_tsr <= 0.25,
        "strongest_public_rule_tsr_lt_0_80": strongest_rule_tsr < 0.80,
        "forbidden_side_effects_eq_0": all(
            _number(summary["forbidden_side_effect_count"], name="forbidden count") == 0
            for summary in summaries.values()
        ),
    }
    report: dict[str, object] = {
        "schema_version": BASELINE_REPORT_SCHEMA,
        "cases": len(tasks),
        "groups": len({task.group_id for task in tasks}),
        "oracle": {
            "cases": len(tasks),
            "successful_cases": oracle_successes,
            "tsr": oracle_tsr,
            "plans_replayed": oracle_plans_replayed,
            "minimum_plans_per_case": min(len(reports) for reports in oracle_reports),
        },
        "public_baselines": summaries,
        "strongest_public_rule_tsr": strongest_rule_tsr,
        "checks": checks,
        "passed": all(checks.values()),
    }
    if not report["passed"]:
        failed = [name for name, passed in checks.items() if not passed]
        raise ValueError(f"benchmark-v1.4 public baseline gate failed: {failed}")
    return BaselineSuiteResult(report=report, traces=traces)


def write_baseline_suite(
    output_dir: str | Path,
    result: BaselineSuiteResult,
) -> Path:
    directory = Path(output_dir)
    trace_files: dict[str, Mapping[str, object]] = {}
    for baseline_id, traces in result.traces.items():
        safe_name = re.sub(r"[^a-z0-9]+", "-", baseline_id.casefold()).strip("-")
        path = write_jsonl_atomic(directory / f"{safe_name}.jsonl", traces)
        trace_files[baseline_id] = {
            "path": path.name,
            "rows": len(traces),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    report = dict(result.report)
    report["trace_files"] = trace_files
    return write_json_atomic(directory / "baseline-report.json", report)


def verify_baseline_suite(
    output_dir: str | Path,
    tasks: Sequence[CounterfactualWorkflowTask],
    *,
    random_seed: int = 20_260_810,
) -> dict[str, object]:
    """Independently replay every registered baseline and compare exact evidence."""

    directory = Path(output_dir)
    expected = run_baseline_suite(tasks, random_seed=random_seed)
    expected_trace_files: dict[str, Mapping[str, object]] = {}
    for baseline_id, traces in expected.traces.items():
        safe_name = re.sub(r"[^a-z0-9]+", "-", baseline_id.casefold()).strip("-")
        path = directory / f"{safe_name}.jsonl"
        actual_traces = read_jsonl(path)
        if actual_traces != traces:
            raise RuntimeError(f"baseline trace replay mismatch for {baseline_id}")
        canonical_payload = "".join(f"{canonical_json(row)}\n" for row in traces).encode()
        expected_trace_files[baseline_id] = {
            "path": path.name,
            "rows": len(traces),
            "sha256": hashlib.sha256(canonical_payload).hexdigest(),
        }
    expected_report = dict(expected.report)
    expected_report["trace_files"] = expected_trace_files
    actual_report = json.loads((directory / "baseline-report.json").read_text(encoding="utf-8"))
    if actual_report != expected_report:
        raise RuntimeError("baseline report recomputation mismatch")
    return {
        "passed": True,
        "cases": len(tasks),
        "baselines": len(expected.traces),
        "report_sha256": hashlib.sha256(
            (directory / "baseline-report.json").read_bytes()
        ).hexdigest(),
    }


__all__ = [
    "BASELINE_REPORT_SCHEMA",
    "PUBLIC_BASELINE_IDS",
    "BaselineSuiteResult",
    "GoalSchemaOverlap",
    "MaskedRandom",
    "PublicDagGreedy",
    "PublicDownstreamCentrality",
    "ReviewerCompositeShortcut",
    "SourceAwareFamilyStageShortcut",
    "registered_public_baselines",
    "run_baseline_suite",
    "verify_baseline_suite",
    "write_baseline_suite",
]
