"""Independent evaluation of the frozen action-validity benchmark.

The benchmark labels are produced by the transactional environment, while this
module deliberately imports only the policy-side :class:`ActionMask`.  Labels
are immutable evaluation evidence: they are read for scoring and are never
recomputed or replaced by environment validation.
"""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from agentic_tool_rl.contracts import (
    ActionValidityExample,
    InvalidActionKind,
    WorkflowTask,
)
from agentic_tool_rl.evaluation.io import sha256_json
from agentic_tool_rl.grounding.action_mask import ActionMask

CANONICAL_ACTION_VALIDITY_COUNT = 20_000
_INVALID_KIND_NAMES = tuple(kind.value for kind in InvalidActionKind)


@dataclass(frozen=True, slots=True)
class ActionValidityPrediction:
    """One policy-side prediction joined to its untouched frozen label."""

    sample_id: str
    case_id: str
    family: str
    state_index: int
    valid_label: bool
    predicted_valid: bool
    invalid_kind: str | None
    predicted_invalid_kind: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "case_id": self.case_id,
            "family": self.family,
            "state_index": self.state_index,
            "valid_label": self.valid_label,
            "predicted_valid": self.predicted_valid,
            "invalid_kind": self.invalid_kind,
            "predicted_invalid_kind": self.predicted_invalid_kind,
        }


@dataclass(frozen=True, slots=True)
class ActionValidityEvaluation:
    """Binary and invalid-kind metrics derived from frozen examples."""

    count: int
    cases: int
    valid_count: int
    invalid_count: int
    valid_recall: float
    invalid_recall: float
    balanced_accuracy: float
    macro_f1: float
    invalid_kind_recall: dict[str, float | None]
    confusion_matrix: dict[str, int]
    predictions: tuple[ActionValidityPrediction, ...]
    bootstrap: dict[str, dict[str, float | int | str]] = field(default_factory=dict)

    @property
    def is_canonical_20k(self) -> bool:
        return self.count == CANONICAL_ACTION_VALIDITY_COUNT

    def predictions_by_case(self) -> dict[str, list[dict[str, Any]]]:
        """Return trace-ready prediction rows grouped by ``case_id``."""

        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for prediction in self.predictions:
            grouped[prediction.case_id].append(prediction.to_dict())
        return {
            case_id: sorted(rows, key=lambda row: str(row["sample_id"]))
            for case_id, rows in sorted(grouped.items())
        }

    def to_dict(self, *, include_predictions: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "count": self.count,
            "cases": self.cases,
            "valid_count": self.valid_count,
            "invalid_count": self.invalid_count,
            "valid_recall": self.valid_recall,
            "invalid_recall": self.invalid_recall,
            "balanced_accuracy": self.balanced_accuracy,
            "macro_f1": self.macro_f1,
            "invalid_kind_recall": dict(self.invalid_kind_recall),
            "confusion_matrix": dict(self.confusion_matrix),
            "is_canonical_20k": self.is_canonical_20k,
            "bootstrap": self.bootstrap,
        }
        if include_predictions:
            result["predictions"] = [prediction.to_dict() for prediction in self.predictions]
        return result


@dataclass(slots=True)
class _MetricCounts:
    tp: int = 0
    tn: int = 0
    fp: int = 0
    fn: int = 0
    kind_total: dict[str, int] = field(
        default_factory=lambda: dict.fromkeys(_INVALID_KIND_NAMES, 0)
    )
    kind_correct: dict[str, int] = field(
        default_factory=lambda: dict.fromkeys(_INVALID_KIND_NAMES, 0)
    )

    def add_prediction(self, prediction: ActionValidityPrediction) -> None:
        if prediction.valid_label:
            if prediction.predicted_valid:
                self.tp += 1
            else:
                self.fn += 1
            return
        if prediction.predicted_valid:
            self.fp += 1
        else:
            self.tn += 1
        if prediction.invalid_kind is None:  # prevented by the contract
            raise ValueError(f"invalid example {prediction.sample_id!r} has no invalid_kind")
        self.kind_total[prediction.invalid_kind] += 1
        if prediction.predicted_invalid_kind == prediction.invalid_kind:
            self.kind_correct[prediction.invalid_kind] += 1

    def add_counts(self, other: _MetricCounts) -> None:
        self.tp += other.tp
        self.tn += other.tn
        self.fp += other.fp
        self.fn += other.fn
        for name in _INVALID_KIND_NAMES:
            self.kind_total[name] += other.kind_total[name]
            self.kind_correct[name] += other.kind_correct[name]


def _divide(numerator: int, denominator: int, *, metric: str) -> float:
    if denominator == 0:
        raise ValueError(f"cannot compute {metric}: its label class is absent")
    return numerator / denominator


def _metrics_from_counts(counts: _MetricCounts) -> dict[str, float | None]:
    valid_recall = _divide(counts.tp, counts.tp + counts.fn, metric="valid recall")
    invalid_recall = _divide(counts.tn, counts.tn + counts.fp, metric="invalid recall")
    valid_f1 = _divide(
        2 * counts.tp,
        2 * counts.tp + counts.fp + counts.fn,
        metric="valid F1",
    )
    invalid_f1 = _divide(
        2 * counts.tn,
        2 * counts.tn + counts.fp + counts.fn,
        metric="invalid F1",
    )
    metrics: dict[str, float | None] = {
        "valid_recall": valid_recall,
        "invalid_recall": invalid_recall,
        "balanced_accuracy": (valid_recall + invalid_recall) / 2.0,
        "macro_f1": (valid_f1 + invalid_f1) / 2.0,
    }
    for name in _INVALID_KIND_NAMES:
        total = counts.kind_total[name]
        metrics[f"invalid_kind_recall.{name}"] = (
            None if total == 0 else counts.kind_correct[name] / total
        )
    return metrics


def _required_metric(metrics: Mapping[str, float | None], name: str) -> float:
    value = metrics[name]
    if value is None:  # binary metrics are guaranteed above; narrow for mypy
        raise RuntimeError(f"required action-validity metric {name!r} is undefined")
    return value


def _cluster_bootstrap(
    predictions: Iterable[ActionValidityPrediction],
    *,
    samples: int,
    seed: int,
    confidence: float,
) -> dict[str, dict[str, float | int | str]]:
    if samples == 0:
        return {}
    if samples < 2:
        raise ValueError("bootstrap_samples must be zero or at least two")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between zero and one")

    by_case: dict[str, _MetricCounts] = {}
    for prediction in predictions:
        counts = by_case.setdefault(prediction.case_id, _MetricCounts())
        counts.add_prediction(prediction)
    case_ids = sorted(by_case)
    if not case_ids:
        raise ValueError("at least one case cluster is required")

    rng = random.Random(seed)
    samples_by_metric: dict[str, list[float]] = defaultdict(list)
    for _ in range(samples):
        replicate = _MetricCounts()
        for case_id in rng.choices(case_ids, k=len(case_ids)):
            replicate.add_counts(by_case[case_id])
        try:
            metrics = _metrics_from_counts(replicate)
        except ValueError:
            # A small, class-segregated dataset can lose a binary class in one
            # replicate. Such a replicate does not define balanced metrics.
            continue
        for name, value in metrics.items():
            if value is not None:
                samples_by_metric[name].append(value)

    alpha = (1.0 - confidence) / 2.0
    intervals: dict[str, dict[str, float | int | str]] = {}
    for name, values in sorted(samples_by_metric.items()):
        if not values:
            continue
        intervals[name] = {
            "low": float(np.quantile(values, alpha)),
            "high": float(np.quantile(values, 1.0 - alpha)),
            "confidence": confidence,
            "method": "case_cluster_percentile",
            "samples": len(values),
            "requested_samples": samples,
            "seed": seed,
        }
    return intervals


def _coerce_example(
    example: ActionValidityExample | Mapping[str, Any],
) -> ActionValidityExample:
    if isinstance(example, ActionValidityExample):
        return example
    return ActionValidityExample.model_validate(example)


def evaluate_action_validity(
    examples: Iterable[ActionValidityExample | Mapping[str, Any]],
    tasks_by_id: Mapping[str, WorkflowTask],
    *,
    expected_count: int | None = None,
    bootstrap_samples: int = 0,
    bootstrap_seed: int = 20_260_808,
    confidence: float = 0.95,
) -> ActionValidityEvaluation:
    """Score frozen labels with the policy-side mask, without environment calls.

    Set ``expected_count=20_000`` (or use
    :data:`CANONICAL_ACTION_VALIDITY_COUNT`) to make the canonical benchmark
    cardinality an enforced gate rather than a descriptive metric.
    """

    frozen = [_coerce_example(example) for example in examples]
    if not frozen:
        raise ValueError("at least one action-validity example is required")
    if expected_count is not None:
        if expected_count < 1:
            raise ValueError("expected_count must be positive")
        if len(frozen) != expected_count:
            raise ValueError(
                f"expected {expected_count} action-validity examples, observed {len(frozen)}"
            )

    sample_ids = [example.sample_id for example in frozen]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("action-validity examples contain duplicate sample_id values")

    predictions: list[ActionValidityPrediction] = []
    counts = _MetricCounts()
    for example in sorted(frozen, key=lambda item: item.sample_id):
        task = tasks_by_id.get(example.case_id)
        if task is None:
            raise KeyError(f"no task found for case_id {example.case_id!r}")
        if task.case_id != example.case_id:
            raise ValueError(
                f"task mapping key {example.case_id!r} points to {task.case_id!r}"
            )
        if example.observation.task_id != example.case_id:
            raise ValueError(
                f"example {example.sample_id!r} observation belongs to "
                f"{example.observation.task_id!r}"
            )
        observed_state_digest = sha256_json(example.observation.model_dump(mode="json"))
        if observed_state_digest != example.state_sha256:
            raise ValueError(f"state digest mismatch in example {example.sample_id!r}")

        decision = ActionMask(task).validate(example.observation, example.tool_call)
        prediction = ActionValidityPrediction(
            sample_id=example.sample_id,
            case_id=example.case_id,
            family=example.family,
            state_index=example.state_index,
            valid_label=example.valid_label,
            predicted_valid=decision.valid,
            invalid_kind=(
                None if example.invalid_kind is None else example.invalid_kind.value
            ),
            predicted_invalid_kind=(
                None if decision.invalid_kind is None else decision.invalid_kind.value
            ),
        )
        counts.add_prediction(prediction)
        predictions.append(prediction)

    metrics = _metrics_from_counts(counts)
    invalid_kind_recall = {
        name: metrics[f"invalid_kind_recall.{name}"] for name in _INVALID_KIND_NAMES
    }
    bootstrap = _cluster_bootstrap(
        predictions,
        samples=bootstrap_samples,
        seed=bootstrap_seed,
        confidence=confidence,
    )
    return ActionValidityEvaluation(
        count=len(predictions),
        cases=len({prediction.case_id for prediction in predictions}),
        valid_count=counts.tp + counts.fn,
        invalid_count=counts.tn + counts.fp,
        valid_recall=_required_metric(metrics, "valid_recall"),
        invalid_recall=_required_metric(metrics, "invalid_recall"),
        balanced_accuracy=_required_metric(metrics, "balanced_accuracy"),
        macro_f1=_required_metric(metrics, "macro_f1"),
        invalid_kind_recall=invalid_kind_recall,
        confusion_matrix={
            "tp": counts.tp,
            "tn": counts.tn,
            "fp": counts.fp,
            "fn": counts.fn,
        },
        predictions=tuple(predictions),
        bootstrap=bootstrap,
    )


__all__ = [
    "CANONICAL_ACTION_VALIDITY_COUNT",
    "ActionValidityEvaluation",
    "ActionValidityPrediction",
    "evaluate_action_validity",
]
