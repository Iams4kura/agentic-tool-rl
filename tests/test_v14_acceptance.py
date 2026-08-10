from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from agentic_tool_rl.cli import app
from agentic_tool_rl.contracts import CounterfactualWorkflowTask, Split
from agentic_tool_rl.envs.benchmark_v14 import generate_counterfactual_tasks
from agentic_tool_rl.envs.persistence_v14 import (
    _development_protocol_identity,
    load_counterfactual_tasks_jsonl,
)
from agentic_tool_rl.envs.workflow import TransactionalWorkflowEnv
from agentic_tool_rl.evaluation.benchmark_v14 import (
    GoalSchemaOverlap,
    MaskedRandom,
    PublicDagGreedy,
    PublicDownstreamCentrality,
    ReviewerCompositeShortcut,
    SourceAwareFamilyStageShortcut,
    run_baseline_suite,
    verify_baseline_suite,
)
from agentic_tool_rl.evaluation.public_baselines import PublicBaselineDecision
from agentic_tool_rl.grounding.action_mask import ActionMask
from agentic_tool_rl.policy_input import PolicyInput
from agentic_tool_rl.v14_acceptance import run_v14_development_acceptance

RUNNER = CliRunner()


class _FrozenReviewerCompositeRule:
    """The public-only shortcut that achieved 400/400 before the fix."""

    baseline_id = "test-frozen-reviewer-composite-v1"

    def select(self, policy_input: PolicyInput) -> PublicBaselineDecision:
        goal = policy_input.observation.user_goal.casefold()
        state = policy_input.observation.visible_state
        completed = set(state.get("completed_nodes", []))
        ready = [
            (index, candidate)
            for index, candidate in enumerate(policy_input.candidates)
            if candidate.action_mask and candidate.tool_schema.operation_id not in completed
        ]
        for index, candidate in ready:
            if ".finalize_" in candidate.tool_schema.name:
                return PublicBaselineDecision(index, "legacy:finalize-ready")
        if state.get("transition_mode") is None:
            transition = "stage_in_bounded_batches" if "gradual" in goal else "switch_all_at_once"
            for index, candidate in ready:
                if transition in candidate.tool_schema.name:
                    return PublicBaselineDecision(index, "legacy:transition-map")
        if state.get("decision_mode") is None:
            decision = (
                "hold_for_named_reviewer" if "person" in goal else "continue_after_automatic_checks"
            )
            for index, candidate in ready:
                if decision in candidate.tool_schema.name:
                    return PublicBaselineDecision(index, "legacy:decision-map")
        for index, candidate in ready:
            schema = candidate.tool_schema
            if (
                schema.mutating
                and "optional advisory" not in schema.description.casefold()
                and "force_complete" not in schema.name
            ):
                return PublicBaselineDecision(index, "legacy:non-advisory")
        return PublicBaselineDecision(ready[0][0], "legacy:first-ready")


def _public_rule_tsr(tasks: list[CounterfactualWorkflowTask], baseline: object) -> float:
    successes = 0
    for task in tasks:
        environment = TransactionalWorkflowEnv(task)
        action_mask = ActionMask(task.tool_schemas)
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
            selection = baseline.select(policy_input)  # type: ignore[attr-defined]
            environment.step(candidates[selection.action_index])
        successes += environment.evaluate().success
    return successes / len(tasks)


@pytest.mark.parametrize(
    "baseline",
    [
        MaskedRandom(),
        PublicDagGreedy(),
        GoalSchemaOverlap(),
        ReviewerCompositeShortcut(),
        SourceAwareFamilyStageShortcut(),
        PublicDownstreamCentrality(),
    ],
)
def test_registered_baselines_reject_non_policy_input(baseline: object) -> None:
    with pytest.raises(TypeError, match="only PolicyInput"):
        baseline.select({})  # type: ignore[attr-defined]


def test_v14_dev_baseline_gate_blocks_public_shortcuts() -> None:
    tasks = generate_counterfactual_tasks(Split.DEV, 100)

    result = run_baseline_suite(tasks)

    assert result.report["passed"] is True
    oracle = result.report["oracle"]
    assert isinstance(oracle, dict)
    assert oracle["tsr"] == 1.0
    assert oracle["plans_replayed"] == 1_600
    assert oracle["minimum_plans_per_case"] >= 4
    summaries = result.report["public_baselines"]
    assert isinstance(summaries, dict)
    r1 = summaries["R1-public-ready-greedy-v1"]
    assert isinstance(r1, dict)
    assert float(r1["tsr"]) <= 0.25
    r4 = summaries[ReviewerCompositeShortcut.baseline_id]
    assert isinstance(r4, dict)
    assert float(r4["tsr"]) < 0.80
    r5 = summaries[SourceAwareFamilyStageShortcut.baseline_id]
    assert isinstance(r5, dict)
    assert float(r5["tsr"]) < 0.80
    r6 = summaries[PublicDownstreamCentrality.baseline_id]
    assert isinstance(r6, dict)
    assert float(r6["tsr"]) < 0.80
    strongest = result.report["strongest_public_rule_tsr"]
    assert isinstance(strongest, (int, float))
    assert float(strongest) < 0.80


def test_v14_canonical_dev_blocks_frozen_reviewer_composite_rule() -> None:
    tasks = generate_counterfactual_tasks(Split.DEV, 100)

    assert _public_rule_tsr(tasks, _FrozenReviewerCompositeRule()) < 0.80
    assert _public_rule_tsr(tasks, SourceAwareFamilyStageShortcut()) < 0.80
    assert _public_rule_tsr(tasks, PublicDownstreamCentrality()) < 0.80


def test_v14_real_acceptance_writes_replayable_artifacts(tmp_path: Path) -> None:
    output = tmp_path / "benchmark-v14"

    result = run_v14_development_acceptance(
        output,
        train_groups=2,
        dev_groups=2,
        test_groups=2,
        base_seed=19,
    )

    assert result["passed"] is True
    assert result["canonical"] is False
    assert result["canonical_final_executed"] is False
    assert all(result["deterministic_replay"].values())
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["canonical_final_permitted"] is False
    assert manifest["protocol"]["external_time_anchor_locked"] is False
    assert manifest["protocol_identity"]["exact_counts_match"] is False
    assert manifest["protocol_identity"]["exact_development_match"] is False
    assert manifest["protocol_identity"]["checks"]["base_seed"] is False
    assert manifest["protocol_identity"]["checks"]["quality_gates"] is True
    assert manifest["protocol_identity"]["checks"]["baseline_registry"] is True
    assert manifest["implementation_identity"]["schema"] == "implementation-fingerprint-v2"
    tasks = load_counterfactual_tasks_jsonl(output / "test.jsonl")
    assert len(tasks) == 8
    assert all(isinstance(task, CounterfactualWorkflowTask) for task in tasks)
    assert (output / "dev-baselines" / "baseline-report.json").is_file()
    baseline_report = json.loads(
        (output / "dev-baselines" / "baseline-report.json").read_text(encoding="utf-8")
    )
    r4_trace = baseline_report["trace_files"][ReviewerCompositeShortcut.baseline_id]
    assert r4_trace["rows"] == 8
    assert (output / "dev-baselines" / r4_trace["path"]).is_file()
    r5_trace = baseline_report["trace_files"][SourceAwareFamilyStageShortcut.baseline_id]
    assert r5_trace["rows"] == 8
    assert (output / "dev-baselines" / r5_trace["path"]).is_file()
    r6_trace = baseline_report["trace_files"][PublicDownstreamCentrality.baseline_id]
    assert r6_trace["rows"] == 8
    assert (output / "dev-baselines" / r6_trace["path"]).is_file()
    assert (output / "verification.json").is_file()
    verification = verify_baseline_suite(
        output / "dev-baselines",
        [task for task in load_counterfactual_tasks_jsonl(output / "dev.jsonl")],
        random_seed=19,
    )
    assert verification["passed"] is True


def test_exact_development_protocol_identity_binds_seed_and_registry() -> None:
    counts = {
        "train": {"groups": 500, "cases": 2000},
        "dev": {"groups": 100, "cases": 400},
        "test": {"groups": 250, "cases": 1000},
    }

    exact = _development_protocol_identity(counts, base_seed=20_260_810)
    changed_seed = _development_protocol_identity(counts, base_seed=20_260_811)

    assert exact["exact_counts_match"] is True
    assert exact["exact_development_match"] is True
    assert all(exact["checks"].values())
    assert changed_seed["exact_counts_match"] is True
    assert changed_seed["checks"]["base_seed"] is False
    assert changed_seed["exact_development_match"] is False


def test_installed_cli_surface_runs_v14_development_acceptance(tmp_path: Path) -> None:
    output = tmp_path / "cli-v14"

    result = RUNNER.invoke(
        app,
        [
            "benchmark-v14",
            "--output",
            str(output),
            "--train-groups",
            "1",
            "--dev-groups",
            "1",
            "--test-groups",
            "1",
            "--base-seed",
            "31",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["passed"] is True
    assert payload["canonical_final_executed"] is False
    assert (output / "verification.json").is_file()
