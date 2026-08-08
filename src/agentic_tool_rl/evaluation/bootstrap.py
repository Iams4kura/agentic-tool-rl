"""Deterministic case-clustered bootstrap confidence intervals."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from math import fsum
from typing import Any

import numpy as np


def paired_seed_case_bootstrap_tsr_difference(
    treatment: Mapping[int, Mapping[str, bool]],
    baseline: Mapping[int, Mapping[str, bool]],
    *,
    samples: int = 1000,
    seed: int = 20260808,
    confidence: float = 0.95,
) -> dict[str, float | int | str]:
    """Estimate a paired TSR difference with seed and case-cluster resampling.

    Every observation is paired by the exact ``(seed, case_id)`` key before any
    resampling. Replicates independently resample seeds and whole case-id
    clusters; a selected case therefore stays paired across treatment/baseline
    and across every selected seed. This is the canonical uncertainty estimate
    for the E-vs-B comparison.
    """

    if samples < 2:
        raise ValueError("samples must be at least two")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between zero and one")
    if not treatment or set(treatment) != set(baseline):
        raise ValueError("treatment and baseline must contain the same non-empty seed set")

    seed_ids = sorted(treatment)
    case_ids: list[str] | None = None
    differences: dict[int, dict[str, float]] = {}
    for run_seed in seed_ids:
        treatment_rows = treatment[run_seed]
        baseline_rows = baseline[run_seed]
        if not treatment_rows or set(treatment_rows) != set(baseline_rows):
            raise ValueError(
                f"treatment and baseline case ids differ for seed {run_seed}"
            )
        current_case_ids = sorted(treatment_rows)
        if case_ids is None:
            case_ids = current_case_ids
        elif current_case_ids != case_ids:
            raise ValueError("every seed must evaluate the same case-id clusters")
        seed_differences: dict[str, float] = {}
        for case_id in current_case_ids:
            treatment_success = treatment_rows[case_id]
            baseline_success = baseline_rows[case_id]
            if not isinstance(treatment_success, bool) or not isinstance(
                baseline_success, bool
            ):
                raise ValueError("paired TSR outcomes must be booleans")
            seed_differences[case_id] = float(treatment_success) - float(
                baseline_success
            )
        differences[run_seed] = seed_differences

    assert case_ids is not None
    pair_count = len(seed_ids) * len(case_ids)
    estimate = fsum(
        differences[run_seed][case_id]
        for run_seed in seed_ids
        for case_id in case_ids
    ) / pair_count

    rng = np.random.default_rng(seed)
    replicates: list[float] = []
    for _ in range(samples):
        selected_seeds = rng.choice(seed_ids, size=len(seed_ids), replace=True)
        selected_cases = rng.choice(case_ids, size=len(case_ids), replace=True)
        replicate_sum = fsum(
            differences[int(run_seed)][str(case_id)]
            for run_seed in selected_seeds
            for case_id in selected_cases
        )
        replicates.append(replicate_sum / pair_count)

    alpha = (1.0 - confidence) / 2.0
    return {
        "estimate": float(estimate),
        "low": float(np.quantile(replicates, alpha)),
        "high": float(np.quantile(replicates, 1.0 - alpha)),
        "confidence": confidence,
        "method": "paired_seed_case_cluster_percentile",
        "samples": samples,
        "seed": seed,
        "seed_count": len(seed_ids),
        "case_cluster_count": len(case_ids),
        "paired_observations": pair_count,
    }


def cluster_bootstrap(
    traces: Iterable[Mapping[str, Any]],
    *,
    timeout_s: float,
    samples: int = 1000,
    seed: int = 20260808,
    confidence: float = 0.95,
) -> dict[str, dict[str, float | int | str]]:
    """Resample whole ``case_id`` clusters and return percentile intervals.

    If an evaluation contains several seeds for the same case, all seed rows
    stay in the same resampled cluster. This avoids treating correlated actions
    or repeated seeds as independent examples.
    """

    if samples < 2:
        raise ValueError("samples must be at least two")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between zero and one")
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for trace in traces:
        case_id = str(trace.get("case_id", ""))
        if not case_id:
            raise ValueError("every trace must have a non-empty case_id")
        grouped[case_id].append(trace)
    if not grouped:
        raise ValueError("at least one trace is required")

    # Local import prevents the metrics -> bootstrap optional call from cycling
    # during module initialization.
    from agentic_tool_rl.evaluation.metrics import compute_metrics

    cluster_ids = sorted(grouped)
    rng = np.random.default_rng(seed)
    values: dict[str, list[float]] = defaultdict(list)
    for _ in range(samples):
        selected = rng.choice(cluster_ids, size=len(cluster_ids), replace=True)
        replicate = [trace for cluster_id in selected for trace in grouped[str(cluster_id)]]
        metrics = compute_metrics(replicate, timeout_s=timeout_s)
        for name, value in metrics.scalar_values().items():
            values[name].append(value)

    alpha = (1.0 - confidence) / 2.0
    return {
        name: {
            "low": float(np.quantile(sample_values, alpha)),
            "high": float(np.quantile(sample_values, 1.0 - alpha)),
            "confidence": confidence,
            "method": "case_cluster_percentile",
            "samples": samples,
            "seed": seed,
        }
        for name, sample_values in sorted(values.items())
    }
