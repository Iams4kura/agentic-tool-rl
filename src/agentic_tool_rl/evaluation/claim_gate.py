"""Canonical, evidence-backed statistical report for the frozen ablation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from math import fsum, isclose, isfinite
from pathlib import Path
from typing import Any

from agentic_tool_rl.evaluation.action_validity import CANONICAL_ACTION_VALIDITY_COUNT
from agentic_tool_rl.evaluation.bootstrap import (
    paired_seed_case_bootstrap_tsr_difference,
)
from agentic_tool_rl.evaluation.io import write_json_atomic
from agentic_tool_rl.evaluation.metrics import EvaluationMetrics

CANONICAL_SEED_COUNT = 5
CANONICAL_TEST_CASE_COUNT = 1_000
CANONICAL_VARIANT_DEFINITIONS: tuple[tuple[str, str, bool, bool], ...] = (
    ("A-BC-Unmasked", "bc", False, False),
    ("B-BC-Mask", "bc", False, True),
    ("C-PPO-Sparse-Unmasked", "action_ppo", False, False),
    ("D-PPO-Sparse-Mask", "action_ppo", False, True),
    ("E-PPO-Progress-Mask", "action_ppo", True, True),
    ("F-Sequence-PPO-Progress-Mask", "sequence_ppo", True, True),
)
TREATMENT_VARIANT = "E-PPO-Progress-Mask"
MATCHED_BASELINE_VARIANT = "B-BC-Mask"
TOTAL_SYSTEM_BASELINE_VARIANT = "A-BC-Unmasked"
PREREGISTERED_HYPOTHESIS = "paired_tsr_difference_ci_lower_gt_zero"


@dataclass(frozen=True)
class CanonicalClaimEvidence:
    """Trace-derived evidence required for the canonical statistical report."""

    seeds: tuple[int, ...]
    test_case_ids: tuple[str, ...]
    variant_definitions: tuple[tuple[str, str, bool, bool], ...]
    case_ids_by_variant_seed: Mapping[str, Mapping[int, Sequence[str]]]
    success_by_variant_seed: Mapping[str, Mapping[int, Mapping[str, bool]]]
    action_validity_count: int
    action_validity_valid_count: int
    action_validity_invalid_count: int
    bootstrap_samples: int
    bootstrap_seed: int
    confidence: float


@dataclass(frozen=True)
class ClaimCheckItem:
    name: str
    passed: bool
    actual: float
    comparator: str
    threshold: float
    detail: str


@dataclass(frozen=True)
class VariantOutcomeSummary:
    variant: str
    tsr: float
    successful_conditional_simulated_service_time_s: float | None
    timeout_penalized_simulated_cost_s: float


@dataclass(frozen=True)
class ComparisonSummary:
    treatment_variant: str
    baseline_variant: str
    tsr_difference: float
    successful_conditional_simulated_service_time_difference_s: float | None
    timeout_penalized_simulated_cost_difference_s: float
    tsr_difference_ci: Mapping[str, float | int | str] | None


@dataclass(frozen=True)
class ClaimGateResult:
    """A report, not a target-score gate.

    ``passed`` records only whether the pre-registered E-vs-B hypothesis is
    supported. A false hypothesis result remains a valid canonical experiment.
    """

    schema_version: str
    passed: bool
    canonical: bool
    hypothesis_passed: bool
    preregistered_hypothesis: str
    treatment: VariantOutcomeSummary
    matched_baseline: VariantOutcomeSummary
    total_system_baseline: VariantOutcomeSummary
    primary_comparison: ComparisonSummary
    total_system_comparison: ComparisonSummary
    action_validity_balanced_accuracy: float
    checks: tuple[ClaimCheckItem, ...]

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["checks"] = [asdict(item) for item in self.checks]
        return value


def _required_metric(
    source: EvaluationMetrics | Mapping[str, Any], name: str
) -> float:
    value = getattr(source, name) if isinstance(source, EvaluationMetrics) else source.get(name)
    if value is None:
        raise ValueError(f"missing required metric {name!r}")
    if isinstance(value, bool):
        raise ValueError(f"metric {name!r} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"metric {name!r} must be numeric") from exc
    if not isfinite(result):
        raise ValueError(f"metric {name!r} must be finite")
    return result


def _optional_metric(
    source: EvaluationMetrics | Mapping[str, Any], name: str
) -> float | None:
    value = getattr(source, name) if isinstance(source, EvaluationMetrics) else source.get(name)
    if value is None:
        return None
    return _required_metric(source, name)


def _variant_summary(
    variant: str, source: EvaluationMetrics | Mapping[str, Any]
) -> VariantOutcomeSummary:
    return VariantOutcomeSummary(
        variant=variant,
        tsr=_required_metric(source, "tsr"),
        successful_conditional_simulated_service_time_s=_optional_metric(
            source, "successful_conditional_simulated_service_time_s"
        ),
        timeout_penalized_simulated_cost_s=_required_metric(
            source, "timeout_penalized_simulated_cost_s"
        ),
    )


def _difference(
    treatment: VariantOutcomeSummary,
    baseline: VariantOutcomeSummary,
    *,
    ci: Mapping[str, float | int | str] | None,
) -> ComparisonSummary:
    treatment_service_time = treatment.successful_conditional_simulated_service_time_s
    baseline_service_time = baseline.successful_conditional_simulated_service_time_s
    service_time_difference = (
        None
        if treatment_service_time is None or baseline_service_time is None
        else treatment_service_time - baseline_service_time
    )
    return ComparisonSummary(
        treatment_variant=treatment.variant,
        baseline_variant=baseline.variant,
        tsr_difference=treatment.tsr - baseline.tsr,
        successful_conditional_simulated_service_time_difference_s=(
            service_time_difference
        ),
        timeout_penalized_simulated_cost_difference_s=(
            treatment.timeout_penalized_simulated_cost_s
            - baseline.timeout_penalized_simulated_cost_s
        ),
        tsr_difference_ci=ci,
    )


def _mean_outcome(values: Mapping[int, Mapping[str, bool]]) -> float | None:
    outcomes: list[bool] = []
    for per_seed in values.values():
        for success in per_seed.values():
            if not isinstance(success, bool):
                return None
            outcomes.append(success)
    if not outcomes:
        return None
    return fsum(float(value) for value in outcomes) / len(outcomes)


def verify_canonical_claims(
    treatment_metrics: EvaluationMetrics | Mapping[str, Any],
    matched_baseline_metrics: EvaluationMetrics | Mapping[str, Any],
    total_system_baseline_metrics: EvaluationMetrics | Mapping[str, Any],
    evidence: CanonicalClaimEvidence,
    *,
    action_validity_balanced_accuracy: float,
) -> ClaimGateResult:
    """Build the canonical E-vs-B report and evaluate its pre-registered CI test."""

    treatment = _variant_summary(TREATMENT_VARIANT, treatment_metrics)
    matched_baseline = _variant_summary(
        MATCHED_BASELINE_VARIANT, matched_baseline_metrics
    )
    total_system_baseline = _variant_summary(
        TOTAL_SYSTEM_BASELINE_VARIANT, total_system_baseline_metrics
    )
    action_bacc = float(action_validity_balanced_accuracy)
    if not isfinite(action_bacc) or not 0.0 <= action_bacc <= 1.0:
        raise ValueError("action_validity_balanced_accuracy must be within [0, 1]")

    structural_checks: list[ClaimCheckItem] = []
    seeds = evidence.seeds
    seed_set = set(seeds)
    expected_case_ids = evidence.test_case_ids
    expected_case_set = frozenset(expected_case_ids)
    definitions = evidence.variant_definitions
    variants_match = (
        len(definitions) == len(CANONICAL_VARIANT_DEFINITIONS)
        and set(definitions) == set(CANONICAL_VARIANT_DEFINITIONS)
    )
    structural_checks.extend(
        [
            ClaimCheckItem(
                "canonical_seed_count",
                len(seeds) == CANONICAL_SEED_COUNT,
                float(len(seeds)),
                "==",
                float(CANONICAL_SEED_COUNT),
                "canonical report requires exactly five seeds",
            ),
            ClaimCheckItem(
                "canonical_unique_seeds",
                len(seed_set) == CANONICAL_SEED_COUNT,
                float(len(seed_set)),
                "==",
                float(CANONICAL_SEED_COUNT),
                "canonical report requires five unique seeds",
            ),
            ClaimCheckItem(
                "canonical_variant_definitions",
                variants_match,
                float(sum(item in CANONICAL_VARIANT_DEFINITIONS for item in definitions)),
                "==",
                float(len(CANONICAL_VARIANT_DEFINITIONS)),
                "all six variants must match the frozen fair-comparison matrix",
            ),
            ClaimCheckItem(
                "canonical_test_case_set",
                len(expected_case_ids) == CANONICAL_TEST_CASE_COUNT
                and len(expected_case_set) == CANONICAL_TEST_CASE_COUNT,
                float(len(expected_case_set)),
                "==",
                float(CANONICAL_TEST_CASE_COUNT),
                "canonical test split requires 1,000 unique cases",
            ),
        ]
    )

    required_variants = tuple(
        definition[0] for definition in CANONICAL_VARIANT_DEFINITIONS
    )
    evidence_variant_sets_match = (
        set(evidence.case_ids_by_variant_seed) == set(required_variants)
        and set(evidence.success_by_variant_seed) == set(required_variants)
    )
    structural_checks.append(
        ClaimCheckItem(
            "canonical_variant_evidence_sets",
            evidence_variant_sets_match,
            float(
                len(
                    set(evidence.case_ids_by_variant_seed)
                    & set(evidence.success_by_variant_seed)
                    & set(required_variants)
                )
            ),
            "==",
            float(len(required_variants)),
            "case and outcome evidence must cover exactly all six canonical variants",
        )
    )
    coverage_ok: dict[str, bool] = {}
    outcome_ok: dict[str, bool] = {}
    for variant in required_variants:
        case_runs = evidence.case_ids_by_variant_seed.get(variant, {})
        outcome_runs = evidence.success_by_variant_seed.get(variant, {})
        matching_case_runs = sum(
            1
            for run_seed in seed_set
            if run_seed in case_runs
            and len(case_runs[run_seed]) == CANONICAL_TEST_CASE_COUNT
            and len(set(case_runs[run_seed])) == CANONICAL_TEST_CASE_COUNT
            and frozenset(case_runs[run_seed]) == expected_case_set
        )
        matching_outcome_runs = sum(
            1
            for run_seed in seed_set
            if run_seed in outcome_runs
            and set(outcome_runs[run_seed]) == expected_case_set
            and all(isinstance(value, bool) for value in outcome_runs[run_seed].values())
        )
        coverage_ok[variant] = (
            set(case_runs) == seed_set and matching_case_runs == CANONICAL_SEED_COUNT
        )
        outcome_ok[variant] = (
            set(outcome_runs) == seed_set
            and matching_outcome_runs == CANONICAL_SEED_COUNT
        )
        slug = variant.lower().replace("-", "_")
        structural_checks.extend(
            [
                ClaimCheckItem(
                    f"canonical_{slug}_case_coverage",
                    coverage_ok[variant],
                    float(matching_case_runs),
                    "==",
                    float(CANONICAL_SEED_COUNT),
                    f"{variant} must evaluate the exact test case set for every seed",
                ),
                ClaimCheckItem(
                    f"canonical_{slug}_outcome_coverage",
                    outcome_ok[variant],
                    float(matching_outcome_runs),
                    "==",
                    float(CANONICAL_SEED_COUNT),
                    f"{variant} must provide one boolean outcome per seed and case",
                ),
            ]
        )

    summary_sources = {
        TOTAL_SYSTEM_BASELINE_VARIANT: total_system_baseline,
        MATCHED_BASELINE_VARIANT: matched_baseline,
        TREATMENT_VARIANT: treatment,
    }
    for variant, summary in summary_sources.items():
        outcome_mean = _mean_outcome(
            evidence.success_by_variant_seed.get(variant, {})
        )
        matches = outcome_mean is not None and isclose(
            summary.tsr, outcome_mean, rel_tol=0.0, abs_tol=1e-12
        )
        structural_checks.append(
            ClaimCheckItem(
                f"canonical_{variant.lower().replace('-', '_')}_tsr_recomputed",
                matches,
                -1.0 if outcome_mean is None else outcome_mean,
                "==",
                summary.tsr,
                f"{variant} aggregate TSR must recompute from paired case outcomes",
            )
        )

    structural_checks.extend(
        [
            ClaimCheckItem(
                "canonical_action_validity_samples",
                evidence.action_validity_count == CANONICAL_ACTION_VALIDITY_COUNT,
                float(evidence.action_validity_count),
                "==",
                float(CANONICAL_ACTION_VALIDITY_COUNT),
                "canonical action-validity evaluation requires 20,000 samples",
            ),
            ClaimCheckItem(
                "canonical_action_validity_balance",
                evidence.action_validity_valid_count
                == CANONICAL_ACTION_VALIDITY_COUNT // 2
                and evidence.action_validity_invalid_count
                == CANONICAL_ACTION_VALIDITY_COUNT // 2,
                float(
                    min(
                        evidence.action_validity_valid_count,
                        evidence.action_validity_invalid_count,
                    )
                ),
                "==",
                float(CANONICAL_ACTION_VALIDITY_COUNT // 2),
                "canonical action labels require 10,000 examples per class",
            ),
            ClaimCheckItem(
                "canonical_paired_bootstrap_samples",
                evidence.bootstrap_samples >= 2,
                float(evidence.bootstrap_samples),
                ">=",
                2.0,
                "canonical report requires a paired bootstrap confidence interval",
            ),
            ClaimCheckItem(
                "canonical_95_percent_confidence",
                isclose(evidence.confidence, 0.95, rel_tol=0.0, abs_tol=1e-12),
                evidence.confidence,
                "==",
                0.95,
                "the pre-registered primary interval uses 95% confidence",
            ),
        ]
    )

    canonical = all(item.passed for item in structural_checks)
    primary_ci: Mapping[str, float | int | str] | None = None
    if canonical:
        primary_ci = paired_seed_case_bootstrap_tsr_difference(
            evidence.success_by_variant_seed[TREATMENT_VARIANT],
            evidence.success_by_variant_seed[MATCHED_BASELINE_VARIANT],
            samples=evidence.bootstrap_samples,
            seed=evidence.bootstrap_seed,
            confidence=evidence.confidence,
        )
    hypothesis_passed = primary_ci is not None and float(primary_ci["low"]) > 0.0
    hypothesis_check = ClaimCheckItem(
        "preregistered_e_vs_b_tsr_ci_lower_gt_zero",
        hypothesis_passed,
        -1.0 if primary_ci is None else float(primary_ci["low"]),
        ">",
        0.0,
        "the paired E-vs-B TSR-difference 95% CI lower bound must exceed zero",
    )
    return ClaimGateResult(
        schema_version="3.0",
        passed=hypothesis_passed,
        canonical=canonical,
        hypothesis_passed=hypothesis_passed,
        preregistered_hypothesis=PREREGISTERED_HYPOTHESIS,
        treatment=treatment,
        matched_baseline=matched_baseline,
        total_system_baseline=total_system_baseline,
        primary_comparison=_difference(treatment, matched_baseline, ci=primary_ci),
        total_system_comparison=_difference(
            treatment, total_system_baseline, ci=None
        ),
        action_validity_balanced_accuracy=action_bacc,
        checks=(*structural_checks, hypothesis_check),
    )


def write_claim_check(path: str | Path, result: ClaimGateResult) -> Path:
    return write_json_atomic(path, result.to_dict())
