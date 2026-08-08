"""Evidence-based metric consistency recomputation and structural diffing."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from math import isclose
from pathlib import Path
from typing import Any

from agentic_tool_rl.evaluation.io import read_jsonl, write_json_atomic
from agentic_tool_rl.evaluation.metrics import EvaluationMetrics, compute_metrics


@dataclass(frozen=True)
class MetricDifference:
    path: str
    expected: Any
    actual: Any
    absolute_difference: float | None = None


@dataclass(frozen=True)
class RecomputeResult:
    matches: bool
    tolerance: float
    differences: tuple[MetricDifference, ...]
    metrics: EvaluationMetrics

    def to_dict(self) -> dict[str, Any]:
        return {
            "matches": self.matches,
            "tolerance": self.tolerance,
            "differences": [asdict(item) for item in self.differences],
            "metrics": self.metrics.to_dict(),
        }


def _diff(
    expected: Any,
    actual: Any,
    *,
    path: str,
    tolerance: float,
    output: list[MetricDifference],
) -> None:
    if isinstance(expected, Mapping) and isinstance(actual, Mapping):
        keys = sorted(set(expected) | set(actual))
        for key in keys:
            child = f"{path}.{key}" if path else str(key)
            if key not in expected:
                output.append(MetricDifference(child, "<missing>", actual[key]))
            elif key not in actual:
                output.append(MetricDifference(child, expected[key], "<missing>"))
            else:
                _diff(expected[key], actual[key], path=child, tolerance=tolerance, output=output)
        return
    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            output.append(MetricDifference(path, expected, actual))
            return
        for index, (expected_item, actual_item) in enumerate(zip(expected, actual, strict=True)):
            _diff(
                expected_item,
                actual_item,
                path=f"{path}[{index}]",
                tolerance=tolerance,
                output=output,
            )
        return
    numeric = (
        isinstance(expected, (int, float))
        and not isinstance(expected, bool)
        and isinstance(actual, (int, float))
        and not isinstance(actual, bool)
    )
    if numeric:
        difference = abs(float(expected) - float(actual))
        if not isclose(float(expected), float(actual), rel_tol=0.0, abs_tol=tolerance):
            output.append(MetricDifference(path, expected, actual, difference))
        return
    if expected != actual:
        output.append(MetricDifference(path, expected, actual))


def diff_metrics(
    expected: Mapping[str, Any],
    actual: Mapping[str, Any],
    *,
    tolerance: float = 1e-9,
) -> tuple[MetricDifference, ...]:
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")
    output: list[MetricDifference] = []
    _diff(expected, actual, path="", tolerance=tolerance, output=output)
    return tuple(output)


def recompute_metrics(
    trace_path: str | Path,
    *,
    published_metrics_path: str | Path | None = None,
    output_path: str | Path | None = None,
    timeout_s: float | None = None,
    bootstrap_samples: int | None = None,
    bootstrap_seed: int | None = None,
    confidence: float | None = None,
    tolerance: float = 1e-9,
) -> RecomputeResult:
    """Recompute from JSONL only and optionally compare with published JSON."""

    published: dict[str, Any] | None = None
    if published_metrics_path is not None:
        with Path(published_metrics_path).open("r", encoding="utf-8") as handle:
            loaded = json.load(handle)
        if not isinstance(loaded, dict):
            raise ValueError("published metrics must be a JSON object")
        published = loaded

    effective_timeout = timeout_s
    effective_samples = bootstrap_samples
    effective_seed = bootstrap_seed
    effective_confidence = confidence
    if published is not None:
        if effective_timeout is None:
            effective_timeout = float(published.get("timeout_s", 30.0))
        if effective_samples is None:
            effective_samples = int(published.get("bootstrap_samples", 0))
        if effective_seed is None:
            effective_seed = int(published.get("bootstrap_seed", 20260808))
        if effective_confidence is None:
            effective_confidence = float(published.get("confidence", 0.95))

    metrics = compute_metrics(
        read_jsonl(trace_path),
        timeout_s=effective_timeout if effective_timeout is not None else 30.0,
        bootstrap_samples=effective_samples if effective_samples is not None else 0,
        bootstrap_seed=effective_seed if effective_seed is not None else 20260808,
        confidence=effective_confidence if effective_confidence is not None else 0.95,
    )
    actual = metrics.to_dict()
    differences = diff_metrics(published, actual, tolerance=tolerance) if published else ()
    result = RecomputeResult(
        matches=not differences,
        tolerance=tolerance,
        differences=differences,
        metrics=metrics,
    )
    if output_path is not None:
        write_json_atomic(output_path, result.to_dict())
    return result
