"""Metrics computed exclusively from per-case evaluation traces.

The evaluator deliberately accepts JSON-like mappings rather than environment
objects.  This keeps metric recomputation independent from the policy and the
environment implementation that produced the traces.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from math import isfinite
from pathlib import Path
from statistics import fmean
from typing import Any

from agentic_tool_rl.evaluation.io import write_json_atomic


class TraceSchemaError(ValueError):
    """Raised when a trace does not contain enough evidence for evaluation."""


@dataclass(frozen=True)
class EvaluationMetrics:
    """Auditable aggregate metrics for one evaluation run."""

    schema_version: str
    cases: int
    successful_cases: int
    tsr: float
    macro_tsr: float
    invalid_action_rate: float
    forbidden_side_effect_rate: float
    steps_efficiency: float
    mean_steps: float
    successful_conditional_simulated_service_time_s: float | None
    timeout_penalized_simulated_cost_s: float
    timeout_s: float
    executed_actions: int
    families: dict[str, dict[str, float | int]]
    bootstrap_samples: int = 0
    bootstrap_seed: int = 0
    confidence: float = 0.95
    confidence_intervals: dict[str, dict[str, float | int | str]] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def scalar_values(self) -> dict[str, float]:
        """Return metrics eligible for case-clustered confidence intervals."""

        return {
            "tsr": self.tsr,
            "macro_tsr": self.macro_tsr,
            "invalid_action_rate": self.invalid_action_rate,
            "forbidden_side_effect_rate": self.forbidden_side_effect_rate,
            "steps_efficiency": self.steps_efficiency,
            "mean_steps": self.mean_steps,
            "timeout_penalized_simulated_cost_s": self.timeout_penalized_simulated_cost_s,
        }


def _require(trace: Mapping[str, Any], key: str) -> Any:
    if key not in trace:
        case_id = trace.get("case_id", "<unknown>")
        raise TraceSchemaError(f"trace {case_id!r} is missing required field {key!r}")
    return trace[key]


def _as_bool(value: Any, *, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    if value in (0, 1):
        return bool(value)
    raise TraceSchemaError(f"{field_name} must be a boolean, got {value!r}")


def _as_non_negative_float(value: Any, *, field_name: str) -> float:
    if isinstance(value, bool):
        raise TraceSchemaError(f"{field_name} must be numeric, got {value!r}")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TraceSchemaError(f"{field_name} must be numeric, got {value!r}") from exc
    if not isfinite(result) or result < 0:
        raise TraceSchemaError(f"{field_name} must be finite and non-negative")
    return result


def _step_records(trace: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    raw = trace.get("steps", trace.get("records", []))
    if isinstance(raw, int):
        return []
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
        raise TraceSchemaError("steps must be a list of JSON objects or an integer count")
    records: list[Mapping[str, Any]] = []
    for index, item in enumerate(raw):
        if not isinstance(item, Mapping):
            raise TraceSchemaError(f"steps[{index}] must be a JSON object")
        records.append(item)
    return records


def _step_count(trace: Mapping[str, Any], records: Sequence[Mapping[str, Any]]) -> int:
    raw_field = "steps" if "steps" in trace else "records" if "records" in trace else None
    raw_evidence = trace[raw_field] if raw_field is not None else None
    explicit = trace.get("step_count")
    if explicit is None and isinstance(raw_evidence, int):
        explicit = raw_evidence
    if explicit is None:
        explicit = len(records)
    count = _as_non_negative_float(explicit, field_name="step_count")
    if not count.is_integer():
        raise TraceSchemaError("step_count must be an integer")
    resolved = int(count)
    if isinstance(raw_evidence, int):
        evidence_count = _as_non_negative_float(raw_evidence, field_name="step evidence")
        if resolved != int(evidence_count):
            raise TraceSchemaError(
                f"step_count {resolved} does not match {int(evidence_count)} "
                f"from {raw_field} step evidence"
            )
    elif (
        isinstance(raw_evidence, Sequence)
        and not isinstance(raw_evidence, (str, bytes, bytearray))
        and resolved != len(records)
    ):
        raise TraceSchemaError(
            f"step_count {resolved} does not match {len(records)} step records"
        )
    return resolved


def _latency(trace: Mapping[str, Any], records: Sequence[Mapping[str, Any]]) -> float:
    for key in ("simulated_latency_s", "latency_s"):
        if key in trace:
            return _as_non_negative_float(trace[key], field_name=key)

    latency_trace = trace.get("latency_trace")
    if latency_trace is not None:
        if not isinstance(latency_trace, Sequence) or isinstance(
            latency_trace, (str, bytes, bytearray)
        ):
            raise TraceSchemaError("latency_trace must be a sequence")
        return sum(
            _as_non_negative_float(item, field_name="latency_trace item")
            for item in latency_trace
        )

    values: list[float] = []
    for record in records:
        for key in ("simulated_latency_s", "latency_s", "tool_latency_s"):
            if key in record:
                values.append(_as_non_negative_float(record[key], field_name=key))
                break
    if values:
        return sum(values)
    raise TraceSchemaError(
        "trace must contain simulated_latency_s, latency_s, latency_trace, "
        "or per-step latency"
    )


def _side_effect(trace: Mapping[str, Any]) -> bool:
    for key in ("forbidden_side_effect", "has_forbidden_side_effect"):
        if key in trace:
            return _as_bool(trace[key], field_name=key)
    for key in (
        "forbidden_side_effect_count",
        "forbidden_side_effects_count",
        "side_effect_count",
    ):
        if key in trace:
            return _as_non_negative_float(trace[key], field_name=key) > 0
    final = trace.get("final_state")
    if isinstance(final, Mapping):
        for key in ("forbidden_side_effect", "has_forbidden_side_effect"):
            if key in final:
                return _as_bool(final[key], field_name=f"final_state.{key}")
    return False


def _lookup_bool(record: Mapping[str, Any], aliases: Sequence[str]) -> bool | None:
    for key in aliases:
        if key in record:
            return _as_bool(record[key], field_name=key)
    return None


def _executed_validity(
    trace: Mapping[str, Any], records: Sequence[Mapping[str, Any]]
) -> Iterable[bool]:
    raw = trace.get("executed_actions")
    items: Sequence[Any]
    if raw is not None:
        if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
            raise TraceSchemaError("executed_actions must be a list")
        items = raw
    else:
        items = records

    for index, item in enumerate(items):
        if not isinstance(item, Mapping):
            raise TraceSchemaError(f"executed action {index} must be an object")
        valid = _lookup_bool(
            item,
            (
                "executed_valid",
                "action_valid",
                "selected_action_valid",
                "accepted",
                "valid_label",
            ),
        )
        if valid is not None:
            yield valid
            continue
        invalid = _lookup_bool(item, ("executed_invalid", "action_invalid"))
        if invalid is not None:
            yield not invalid


def compute_metrics(
    traces: Iterable[Mapping[str, Any]],
    *,
    timeout_s: float = 30.0,
    bootstrap_samples: int = 0,
    bootstrap_seed: int = 20260808,
    confidence: float = 0.95,
) -> EvaluationMetrics:
    """Compute all published metrics from raw case traces.

    ``successful_conditional_simulated_service_time_s`` is the uncapped mean
    simulated service execution time among successful cases only.  The
    separately named ``timeout_penalized_simulated_cost_s`` charges failed
    tasks ``timeout_s`` and caps successful cases at the same value.  Neither
    quantity is wall-clock completion latency. Step efficiency is zero for a
    failed task and otherwise ``optimal_steps / max(actual_steps, optimal_steps)``.
    """

    timeout = _as_non_negative_float(timeout_s, field_name="timeout_s")
    if timeout == 0:
        raise ValueError("timeout_s must be greater than zero")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between zero and one")
    rows = list(traces)
    if not rows:
        raise ValueError("at least one trace is required")

    successes: list[bool] = []
    families: dict[str, list[bool]] = {}
    successful_service_times: list[float] = []
    timeout_penalized_costs: list[float] = []
    step_counts: list[int] = []
    efficiencies: list[float] = []
    side_effects: list[bool] = []
    executed_valid: list[bool] = []

    for row in rows:
        case_id = str(_require(row, "case_id"))
        if not case_id:
            raise TraceSchemaError("case_id must not be empty")
        family = str(_require(row, "family"))
        if not family:
            raise TraceSchemaError(f"trace {case_id!r} has an empty family")
        success = _as_bool(_require(row, "success"), field_name="success")
        records = _step_records(row)
        steps = _step_count(row, records)
        raw_optimal = _as_non_negative_float(
            _require(row, "optimal_steps"), field_name="optimal_steps"
        )
        if not raw_optimal.is_integer():
            raise TraceSchemaError(f"trace {case_id!r} must have integer optimal_steps")
        optimal = int(raw_optimal)
        if optimal <= 0:
            raise TraceSchemaError(f"trace {case_id!r} must have optimal_steps > 0")

        raw_latency = _latency(row, records)
        timeout_penalized_cost = min(raw_latency, timeout) if success else timeout
        efficiency = min(1.0, optimal / max(steps, optimal)) if success else 0.0

        successes.append(success)
        families.setdefault(family, []).append(success)
        if success:
            successful_service_times.append(raw_latency)
        timeout_penalized_costs.append(timeout_penalized_cost)
        step_counts.append(steps)
        efficiencies.append(efficiency)
        side_effects.append(_side_effect(row))
        executed_valid.extend(_executed_validity(row, records))

    family_summary = {
        name: {"cases": len(values), "successful_cases": sum(values), "tsr": fmean(values)}
        for name, values in sorted(families.items())
    }
    metric = EvaluationMetrics(
        schema_version="3.0",
        cases=len(rows),
        successful_cases=sum(successes),
        tsr=fmean(successes),
        macro_tsr=fmean(float(summary["tsr"]) for summary in family_summary.values()),
        invalid_action_rate=(
            1.0 - fmean(executed_valid) if executed_valid else 0.0
        ),
        forbidden_side_effect_rate=fmean(side_effects),
        steps_efficiency=fmean(efficiencies),
        mean_steps=fmean(step_counts),
        successful_conditional_simulated_service_time_s=(
            fmean(successful_service_times) if successful_service_times else None
        ),
        timeout_penalized_simulated_cost_s=fmean(timeout_penalized_costs),
        timeout_s=timeout,
        executed_actions=len(executed_valid),
        families=family_summary,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
        confidence=confidence,
    )
    if bootstrap_samples:
        if bootstrap_samples < 2:
            raise ValueError("bootstrap_samples must be zero or at least two")
        from agentic_tool_rl.evaluation.bootstrap import cluster_bootstrap

        intervals = cluster_bootstrap(
            rows,
            timeout_s=timeout,
            samples=bootstrap_samples,
            seed=bootstrap_seed,
            confidence=confidence,
        )
        metric = EvaluationMetrics(**{**metric.to_dict(), "confidence_intervals": intervals})
    return metric


def write_metrics(path: str | Path, metrics: EvaluationMetrics) -> Path:
    """Persist the exact recomputable metric document in canonical JSON."""

    return write_json_atomic(path, metrics.to_dict())
