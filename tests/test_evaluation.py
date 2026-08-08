from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from agentic_tool_rl.config import BenchmarkConfig
from agentic_tool_rl.evaluation import (
    CANONICAL_VARIANT_DEFINITIONS,
    CanonicalClaimEvidence,
    TraceConflictError,
    TraceSchemaError,
    TraceStore,
    cluster_bootstrap,
    compute_metrics,
    paired_seed_case_bootstrap_tsr_difference,
    recompute_metrics,
    verify_canonical_claims,
    write_json_atomic,
)
from agentic_tool_rl.evaluation.io import canonical_json


def _traces() -> list[dict[str, object]]:
    return [
        {
            "case_id": "calendar-001",
            "family": "calendar",
            "success": True,
            "optimal_steps": 2,
            "simulated_latency_s": 10.0,
            "forbidden_side_effect": False,
            "steps": [
                {"executed_valid": True},
                {"executed_valid": True},
            ],
            "action_evaluations": [
                {"valid_label": True, "predicted_valid": True},
                {"valid_label": False, "predicted_valid": False},
            ],
        },
        {
            "case_id": "travel-001",
            "family": "travel",
            "success": False,
            "optimal_steps": 2,
            "simulated_latency_s": 5.0,
            "forbidden_side_effects_count": 1,
            "steps": [
                {"executed_valid": True},
                {"executed_valid": False},
                {"executed_valid": True},
                {},
            ],
            "action_evaluations": [
                {"valid_label": True, "predicted_valid": False},
                {"valid_label": False, "predicted_valid": True},
            ],
        },
    ]


@pytest.mark.parametrize(
    ("field", "value"),
    (("families", 9), ("min_steps", 5), ("max_steps", 16)),
)
def test_benchmark_config_rejects_nonfunctional_generator_knobs(
    field: str, value: int
) -> None:
    config = {
        "generator_version": "benchmark-v1.3.0",
        "train_cases": 10,
        "dev_cases": 5,
        "test_cases": 5,
        "families": 10,
        "min_steps": 6,
        "max_steps": 15,
        "seed": 17,
    }
    config[field] = value

    with pytest.raises(ValueError, match="generator is frozen"):
        BenchmarkConfig.model_validate(config)


def test_policy_metrics_are_derived_only_from_case_traces() -> None:
    metrics = compute_metrics(_traces(), timeout_s=30.0)

    assert metrics.cases == 2
    assert metrics.successful_cases == 1
    assert metrics.tsr == pytest.approx(0.5)
    assert metrics.macro_tsr == pytest.approx(0.5)
    assert metrics.invalid_action_rate == pytest.approx(0.2)
    assert metrics.forbidden_side_effect_rate == pytest.approx(0.5)
    assert metrics.steps_efficiency == pytest.approx(0.5)
    assert metrics.mean_steps == pytest.approx(3.0)
    assert metrics.successful_conditional_simulated_service_time_s == pytest.approx(10.0)
    # Failure is charged the 30 second timeout only in the separately named cost.
    assert metrics.timeout_penalized_simulated_cost_s == pytest.approx(20.0)
    assert "simulated_latency_s" not in metrics.to_dict()
    assert metrics.schema_version == "3.0"
    assert {
        "valid_recall",
        "invalid_recall",
        "balanced_accuracy",
        "macro_f1",
        "action_predictions",
        "confusion_matrix",
    }.isdisjoint(metrics.to_dict())


def test_metrics_reject_missing_latency_evidence() -> None:
    trace = _traces()[0]
    trace.pop("simulated_latency_s")
    with pytest.raises(TraceSchemaError, match="latency"):
        compute_metrics([trace])


def test_case_clustered_bootstrap_is_seeded_and_keeps_clusters() -> None:
    traces = [
        *_traces(),
        {**_traces()[0], "case_id": "calendar-001", "run_seed": 2},
        {**_traces()[1], "case_id": "travel-001", "run_seed": 2},
    ]
    first = cluster_bootstrap(traces, timeout_s=30.0, samples=40, seed=123)
    second = cluster_bootstrap(traces, timeout_s=30.0, samples=40, seed=123)

    assert first == second
    assert first["tsr"]["method"] == "case_cluster_percentile"
    assert first["tsr"]["samples"] == 40
    assert "timeout_penalized_simulated_cost_s" in first
    assert 0.0 <= float(first["tsr"]["low"]) <= float(first["tsr"]["high"]) <= 1.0


def test_metrics_json_can_be_independently_recomputed(tmp_path: Path) -> None:
    trace_path = tmp_path / "traces.jsonl"
    trace_path.write_text(
        "".join(f"{canonical_json(trace)}\n" for trace in _traces()), encoding="utf-8"
    )
    metrics = compute_metrics(
        _traces(),
        timeout_s=30.0,
        bootstrap_samples=20,
        bootstrap_seed=91,
        confidence=0.9,
    )
    metrics_path = write_json_atomic(tmp_path / "metrics.json", metrics.to_dict())

    result = recompute_metrics(
        trace_path,
        published_metrics_path=metrics_path,
        output_path=tmp_path / "recompute.json",
    )

    assert result.matches
    assert result.differences == ()
    assert json.loads((tmp_path / "recompute.json").read_text())["matches"] is True

    tampered = metrics.to_dict()
    tampered["tsr"] = 0.99
    write_json_atomic(metrics_path, tampered)
    mismatch = recompute_metrics(trace_path, published_metrics_path=metrics_path)
    assert not mismatch.matches
    assert [item.path for item in mismatch.differences] == ["tsr"]


def test_paired_seed_case_bootstrap_is_deterministic_and_rejects_unpaired_data() -> None:
    baseline = {
        17: {"a": False, "b": True, "c": False, "d": True},
        29: {"a": False, "b": True, "c": False, "d": True},
    }
    treatment = {
        17: {"a": True, "b": True, "c": False, "d": True},
        29: {"a": True, "b": True, "c": True, "d": True},
    }

    first = paired_seed_case_bootstrap_tsr_difference(
        treatment, baseline, samples=100, seed=7
    )
    second = paired_seed_case_bootstrap_tsr_difference(
        treatment, baseline, samples=100, seed=7
    )

    assert first == second
    assert first["estimate"] == pytest.approx(0.375)
    assert first["method"] == "paired_seed_case_cluster_percentile"
    assert first["paired_observations"] == 8
    with pytest.raises(ValueError, match="case ids differ"):
        paired_seed_case_bootstrap_tsr_difference(
            treatment,
            {**baseline, 17: {"a": False}},
            samples=10,
        )


def _canonical_claim_evidence() -> CanonicalClaimEvidence:
    seeds = (17, 29, 43, 71, 101)
    case_ids = tuple(f"case-{index:04d}" for index in range(1_000))
    variants = tuple(definition[0] for definition in CANONICAL_VARIANT_DEFINITIONS)
    success_counts = {
        "A-BC-Unmasked": 700,
        "B-BC-Mask": 800,
        "C-PPO-Sparse-Unmasked": 750,
        "D-PPO-Sparse-Mask": 820,
        "E-PPO-Progress-Mask": 900,
        "F-Sequence-PPO-Progress-Mask": 850,
    }
    return CanonicalClaimEvidence(
        seeds=seeds,
        test_case_ids=case_ids,
        variant_definitions=CANONICAL_VARIANT_DEFINITIONS,
        case_ids_by_variant_seed={
            variant: {seed: case_ids for seed in seeds} for variant in variants
        },
        success_by_variant_seed={
            variant: {
                seed: {
                    case_id: index < success_counts[variant]
                    for index, case_id in enumerate(case_ids)
                }
                for seed in seeds
            }
            for variant in variants
        },
        action_validity_count=20_000,
        action_validity_valid_count=10_000,
        action_validity_invalid_count=10_000,
        bootstrap_samples=100,
        bootstrap_seed=20260808,
        confidence=0.95,
    )


def _metric_summary(
    tsr: float, service_time: float, penalized_cost: float
) -> dict[str, float]:
    return {
        "tsr": tsr,
        "successful_conditional_simulated_service_time_s": service_time,
        "timeout_penalized_simulated_cost_s": penalized_cost,
    }


def test_canonical_claim_gate_proves_matrix_and_case_pairing() -> None:
    evidence = _canonical_claim_evidence()
    result = verify_canonical_claims(
        _metric_summary(0.9, 12.0, 18.0),
        _metric_summary(0.8, 13.0, 25.0),
        _metric_summary(0.7, 14.0, 32.0),
        evidence,
        action_validity_balanced_accuracy=0.975,
    )

    assert result.passed
    assert result.canonical
    assert result.hypothesis_passed
    assert result.primary_comparison.treatment_variant == "E-PPO-Progress-Mask"
    assert result.primary_comparison.baseline_variant == "B-BC-Mask"
    assert result.primary_comparison.tsr_difference == pytest.approx(0.1)
    assert result.total_system_comparison.baseline_variant == "A-BC-Unmasked"
    assert result.total_system_comparison.tsr_difference == pytest.approx(0.2)
    assert result.primary_comparison.tsr_difference_ci is not None
    assert float(result.primary_comparison.tsr_difference_ci["low"]) > 0.0
    assert all(item.passed for item in result.checks)

    wrong_by_variant = {
        variant: dict(by_seed)
        for variant, by_seed in evidence.case_ids_by_variant_seed.items()
    }
    wrong_by_variant["E-PPO-Progress-Mask"][17] = wrong_by_variant[
        "E-PPO-Progress-Mask"
    ][17][1:]
    rejected = verify_canonical_claims(
        _metric_summary(0.9, 12.0, 18.0),
        _metric_summary(0.8, 13.0, 25.0),
        _metric_summary(0.7, 14.0, 32.0),
        replace(evidence, case_ids_by_variant_seed=wrong_by_variant),
        action_validity_balanced_accuracy=0.975,
    )
    failures = {item.name for item in rejected.checks if not item.passed}
    assert not rejected.passed
    assert not rejected.canonical
    assert "canonical_e_ppo_progress_mask_case_coverage" in failures


def test_canonical_claim_gate_rejects_reduced_or_misdefined_evidence() -> None:
    evidence = _canonical_claim_evidence()
    missing_variant_cases = dict(evidence.case_ids_by_variant_seed)
    missing_variant_cases.pop("C-PPO-Sparse-Unmasked")
    malformed = (
        (
            replace(evidence, seeds=(17, 29, 43, 71, 71)),
            "canonical_unique_seeds",
        ),
        (
            replace(evidence, test_case_ids=evidence.test_case_ids[:-1]),
            "canonical_test_case_set",
        ),
        (
            replace(evidence, action_validity_count=19_999),
            "canonical_action_validity_samples",
        ),
        (
            replace(evidence, confidence=0.9),
            "canonical_95_percent_confidence",
        ),
        (
            replace(evidence, case_ids_by_variant_seed=missing_variant_cases),
            "canonical_c_ppo_sparse_unmasked_case_coverage",
        ),
        (
            replace(
                evidence,
                variant_definitions=(
                    *CANONICAL_VARIANT_DEFINITIONS[:-1],
                    (
                        "F-Sequence-PPO-Progress-Mask",
                        "action_ppo",
                        True,
                        True,
                    ),
                ),
            ),
            "canonical_variant_definitions",
        ),
    )

    for candidate, expected_failure in malformed:
        result = verify_canonical_claims(
            _metric_summary(0.9, 12.0, 18.0),
            _metric_summary(0.8, 13.0, 25.0),
            _metric_summary(0.7, 14.0, 32.0),
            candidate,
            action_validity_balanced_accuracy=0.975,
        )
        assert not result.canonical
        assert not result.passed
        assert expected_failure in {
            item.name for item in result.checks if not item.passed
        }


def test_canonical_report_has_no_fixed_target_and_preserves_null_result() -> None:
    evidence = _canonical_claim_evidence()
    equal_outcomes = {
        variant: by_seed
        for variant, by_seed in evidence.success_by_variant_seed.items()
    }
    equal_outcomes["E-PPO-Progress-Mask"] = equal_outcomes["B-BC-Mask"]
    null_result = verify_canonical_claims(
        _metric_summary(0.8, 12.0, 20.0),
        _metric_summary(0.8, 13.0, 21.0),
        _metric_summary(0.7, 14.0, 22.0),
        replace(evidence, success_by_variant_seed=equal_outcomes),
        action_validity_balanced_accuracy=0.61,
    )

    assert null_result.canonical
    assert not null_result.passed
    assert not null_result.hypothesis_passed
    assert null_result.action_validity_balanced_accuracy == pytest.approx(0.61)
    assert {
        item.name for item in null_result.checks if not item.passed
    } == {"preregistered_e_vs_b_tsr_ci_lower_gt_zero"}


def test_trace_store_resume_is_idempotent_and_preserves_record_checksums(
    tmp_path: Path,
) -> None:
    path = tmp_path / "run.jsonl"
    first = _traces()[0]
    path.write_bytes(f"{canonical_json(first)}\n".encode() + b'{"case_id":"partial')

    store = TraceStore(path)
    checksum_before = store.record_checksum("calendar-001")
    assert len(store) == 1
    assert store.append(first) is False
    assert store.record_checksum("calendar-001") == checksum_before
    assert store.append(_traces()[1]) is True
    assert store.record_checksum("calendar-001") == checksum_before
    assert store.missing_case_ids(["calendar-001", "travel-001", "it-001"]) == [
        "it-001"
    ]

    reopened = TraceStore(path)
    assert reopened.completed_case_ids() == frozenset({"calendar-001", "travel-001"})
    with pytest.raises(TraceConflictError):
        reopened.append({**first, "success": False})


def test_trace_store_treats_newline_as_commit_marker(tmp_path: Path) -> None:
    path = tmp_path / "crash.jsonl"
    path.write_text(canonical_json(_traces()[0]), encoding="utf-8")

    store = TraceStore(path)

    assert len(store) == 0
    assert path.read_bytes() == b""
