from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import replace

import pytest
import torch

import agentic_tool_rl.training as training
from agentic_tool_rl.contracts import Split
from agentic_tool_rl.envs import generate_tasks
from agentic_tool_rl.envs.benchmark_v14 import (
    GOAL_VARIANTS,
    generate_counterfactual_tasks,
)
from agentic_tool_rl.features import FeatureEncoder
from agentic_tool_rl.models.progress_estimator import ProgressMetrics


def _stratum(task: object) -> tuple[str, str, str]:
    return (task.family, task.difficulty, task.topology)  # type: ignore[attr-defined]


def test_v14_train_selection_is_360_cases_across_90_complete_goal_strata() -> None:
    tasks = generate_counterfactual_tasks(Split.TRAIN, 500)

    selected = training._select_progress_tasks(tasks, max_tasks=360, seed=41)
    reordered = training._select_progress_tasks(
        list(reversed(tasks)), max_tasks=360, seed=41
    )

    assert [task.case_id for task in selected] == [task.case_id for task in reordered]
    assert len(selected) == 360
    assert len({_stratum(task) for task in tasks}) == 90
    stratum_counts = Counter(_stratum(task) for task in selected)
    assert len(stratum_counts) == 90
    assert set(stratum_counts.values()) == {4}

    variants_by_stratum: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    groups_by_stratum: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    for task in selected:
        variants_by_stratum[_stratum(task)].add(task.goal_variant)
        groups_by_stratum[_stratum(task)].add(task.group_id)
    assert all(variants == set(GOAL_VARIANTS) for variants in variants_by_stratum.values())
    assert all(len(groups) == 1 for groups in groups_by_stratum.values())


def test_v14_dev_selection_uses_all_400_cases() -> None:
    tasks = generate_counterfactual_tasks(Split.DEV, 100)

    selected = training._select_progress_tasks(tasks, max_tasks=None, seed=42)

    assert len(selected) == 400
    assert {task.case_id for task in selected} == {task.case_id for task in tasks}
    stratum_counts = Counter(_stratum(task) for task in selected)
    assert len(stratum_counts) == 90
    assert set(stratum_counts.values()) == {4, 8}
    assert Counter(task.goal_variant for task in selected) == {
        variant: 100 for variant in GOAL_VARIANTS
    }


def test_progress_sampler_uses_one_fixed_probability_and_eight_trials_per_prefix() -> None:
    tasks = generate_tasks(Split.TRAIN, 20, base_seed=920)
    encoder = FeatureEncoder(16, 16)

    sampled = training._sample_progress_examples(
        tasks,
        encoder,
        seed=43,
        max_tasks=20,
    )

    audit = sampled.audit
    assert audit.policy_id == training.FIXED_CONTINUATION_POLICY.policy_id
    assert audit.policy_sha256 == training.FIXED_CONTINUATION_POLICY.sha256
    assert (
        audit.policy_implementation_sha256
        == training.FIXED_CONTINUATION_POLICY.implementation_sha256
    )
    assert (
        training.FIXED_CONTINUATION_POLICY.canonical_payload()[
            "implementation_sha256"
        ]
        == audit.policy_implementation_sha256
    )
    assert audit.seed == 43
    assert audit.selected_tasks == 20
    assert audit.prefixes_per_task == 4
    assert audit.trials_per_prefix == 8
    assert audit.continuation_runs == 20 * 4 * 8
    assert audit.examples == audit.continuation_runs * 2
    assert set(audit.prefix_trials) == {"0", "33", "67", "last_nonterminal"}
    assert set(audit.prefix_trials.values()) == {20 * 8}
    assert set(audit.follow_probability_by_prefix.values()) == {
        training.FIXED_CONTINUATION_POLICY.oracle_positive_probability
    }
    assert 0.20 <= audit.success_rate <= 0.80
    assert all(
        counts["success"] > 0 and counts["failure"] > 0
        for counts in audit.family_outcomes.values()
    )
    assert sampled.features.shape[0] == sampled.labels.shape[0] == audit.examples
    assert sampled.done_mask.dtype == torch.bool
    assert audit.done_counts == {
        "0": audit.continuation_runs,
        "1": audit.continuation_runs,
    }
    assert int((~sampled.done_mask).sum()) == audit.continuation_runs
    assert int(sampled.done_mask.sum()) == audit.continuation_runs
    sampled.validate()


def test_progress_gate_view_excludes_terminal_outcome_examples() -> None:
    tasks = generate_tasks(Split.DEV, 20, base_seed=921)
    sampled = training._sample_progress_examples(
        tasks,
        FeatureEncoder(16, 16),
        seed=44,
        max_tasks=20,
    )

    gate_features, gate_labels = training._nonterminal_progress_gate_view(sampled)
    changed_labels = sampled.labels.clone()
    changed_labels[sampled.done_mask] = 1.0 - changed_labels[sampled.done_mask]
    changed = replace(sampled, labels=changed_labels)
    changed_features, changed_gate_labels = training._nonterminal_progress_gate_view(
        changed
    )

    assert gate_features.shape[0] == sampled.audit.continuation_runs
    assert torch.all(gate_features[:, 2] == 0.0)
    assert torch.equal(gate_features, changed_features)
    assert torch.equal(gate_labels, changed_gate_labels)


def test_progress_label_audit_fails_closed_on_global_or_family_imbalance() -> None:
    with pytest.raises(RuntimeError, match="global success fraction"):
        training._validate_progress_label_balance(
            successes=90,
            failures=10,
            family_outcomes={"family-a": {"success": 90, "failure": 10}},
        )

    with pytest.raises(RuntimeError, match="family-b"):
        training._validate_progress_label_balance(
            successes=50,
            failures=50,
            family_outcomes={
                "family-a": {"success": 25, "failure": 50},
                "family-b": {"success": 25, "failure": 0},
            },
        )


def test_progress_quality_status_requires_every_dev_gate() -> None:
    labels = torch.tensor([0.0, 0.0, 1.0, 1.0])
    passing = training._progress_quality_gate(
        ProgressMetrics(loss=0.4, brier=0.20, ece=0.10, auroc=0.70),
        labels,
    )

    assert passing.constant_prevalence_brier == pytest.approx(0.25)
    assert passing.brier_threshold == pytest.approx(0.225)
    assert passing.auroc_pass is True
    assert passing.ece_pass is True
    assert passing.brier_pass is True
    assert passing.passed is True

    failing = training._progress_quality_gate(
        ProgressMetrics(loss=0.4, brier=0.20, ece=0.100001, auroc=0.70),
        labels,
    )
    assert failing.ece_pass is False
    assert failing.passed is False
