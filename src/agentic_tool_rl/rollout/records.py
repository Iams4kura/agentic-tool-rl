"""Auditable per-tool-call rollout records and trajectory grouping."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any

import torch
from torch import Tensor


def _snapshot_tensor(value: Tensor | None) -> Tensor | None:
    return None if value is None else value.detach().clone().cpu()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Tensor):
        return value.detach().cpu().tolist()
    if hasattr(value, "model_dump"):
        return _jsonable(value.model_dump(mode="json"))
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_jsonable(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def _read_field(record: object, name: str, default: Any = None) -> Any:
    return getattr(record, name, default)


@dataclass(frozen=True)
class StepRecord:
    """One structured tool invocation, which is exactly one RL step."""

    trajectory_id: str
    step_index: int
    observation: Any
    candidates: tuple[Any, ...]
    action_index: int
    action_mask: tuple[bool, ...]
    log_prob: float
    value: float
    reward: float
    done: bool
    next_observation: Any | None = None
    state_features: Tensor | None = None
    action_features: Tensor | None = None
    task_reward: float = 0.0
    progress_reward: float = 0.0
    step_reward: float = 0.0
    invalid_reward: float = 0.0
    valid_label: bool | None = None
    predicted_valid: bool | None = None
    invalid_kind: str | None = None
    info: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidates", tuple(self.candidates))
        object.__setattr__(self, "action_mask", tuple(bool(item) for item in self.action_mask))
        object.__setattr__(self, "state_features", _snapshot_tensor(self.state_features))
        object.__setattr__(self, "action_features", _snapshot_tensor(self.action_features))
        object.__setattr__(self, "info", dict(self.info))
        if not self.trajectory_id:
            raise ValueError("trajectory_id must not be empty")
        if self.step_index < 0:
            raise ValueError("step_index must be non-negative")
        if not self.candidates or len(self.candidates) != len(self.action_mask):
            raise ValueError("candidates and action_mask must be non-empty and aligned")
        if not 0 <= self.action_index < len(self.candidates):
            raise ValueError("action_index is outside the candidate set")
        if not self.action_mask[self.action_index]:
            raise ValueError("selected action is masked out")
        scalar_values = (self.log_prob, self.value, self.reward)
        if not all(torch.isfinite(torch.tensor(item)) for item in scalar_values):
            raise ValueError("log_prob, value and reward must be finite")
        if self.state_features is not None and self.state_features.ndim != 1:
            raise ValueError("state_features must have shape [S]")
        if self.action_features is not None:
            if self.action_features.ndim != 2:
                raise ValueError("action_features must have shape [A, D]")
            if self.action_features.shape[0] != len(self.candidates):
                raise ValueError("candidate and action feature counts differ")

    @property
    def action(self) -> Any:
        return self.candidates[self.action_index]

    @property
    def reward_components(self) -> dict[str, float]:
        return {
            "task": self.task_reward,
            "progress": self.progress_reward,
            "step": self.step_reward,
            "invalid": self.invalid_reward,
        }

    def to_trace_dict(self) -> dict[str, Any]:
        """Return a JSON-safe trace row; features stay present for reproducibility."""

        return {
            "trajectory_id": self.trajectory_id,
            "step_index": self.step_index,
            "observation": _jsonable(self.observation),
            "candidates": _jsonable(self.candidates),
            "action_index": self.action_index,
            "action": _jsonable(self.action),
            "action_mask": list(self.action_mask),
            "log_prob": self.log_prob,
            "value": self.value,
            "reward": self.reward,
            "reward_components": self.reward_components,
            "done": self.done,
            "valid_label": self.valid_label,
            "predicted_valid": self.predicted_valid,
            "invalid_kind": self.invalid_kind,
            "next_observation": _jsonable(self.next_observation),
            "state_features": _jsonable(self.state_features),
            "action_features": _jsonable(self.action_features),
            "info": _jsonable(self.info),
        }

    @classmethod
    def from_contract(
        cls,
        record: Any,
        *,
        trajectory_id: str | None = None,
        state_features: Tensor | None = None,
        action_features: Tensor | None = None,
    ) -> StepRecord:
        """Adapt a Pydantic/dataclass contract without importing it eagerly."""

        raw_candidates = _read_field(
            record, "candidates", _read_field(record, "candidate_actions", ())
        )
        if isinstance(raw_candidates, (str, bytes)) or not isinstance(
            raw_candidates, Sequence
        ):
            raise TypeError("contract candidates must be a sequence")
        candidates = tuple(raw_candidates)
        raw_action_index = _read_field(record, "action_index")
        if raw_action_index is None:
            selected = _read_field(record, "action")
            action_index = candidates.index(selected)
        else:
            action_index = int(str(raw_action_index))
        raw_mask = _read_field(record, "action_mask", [True] * len(candidates))
        if isinstance(raw_mask, (str, bytes)) or not isinstance(raw_mask, Sequence):
            raise TypeError("contract action_mask must be a sequence")
        raw_info = _read_field(record, "info", {})
        if not isinstance(raw_info, Mapping):
            raise TypeError("contract info must be a mapping")
        raw_valid_label = _read_field(record, "valid_label")
        raw_predicted_valid = _read_field(record, "predicted_valid")
        raw_invalid_kind = _read_field(record, "invalid_kind")
        if hasattr(raw_invalid_kind, "value"):
            raw_invalid_kind = raw_invalid_kind.value
        return cls(
            trajectory_id=trajectory_id
            or str(
                _read_field(
                    record, "trajectory_id", _read_field(record, "task_id", "")
                )
            ),
            step_index=int(str(_read_field(record, "step_index", 0))),
            observation=_read_field(record, "observation"),
            candidates=candidates,
            action_index=action_index,
            action_mask=tuple(bool(item) for item in raw_mask),
            log_prob=float(str(_read_field(record, "log_prob", 0.0))),
            value=float(str(_read_field(record, "value", 0.0))),
            reward=float(str(_read_field(record, "reward", 0.0))),
            done=bool(_read_field(record, "done", False)),
            next_observation=_read_field(record, "next_observation"),
            state_features=state_features,
            action_features=action_features,
            task_reward=float(str(_read_field(record, "task_reward", 0.0))),
            progress_reward=float(str(_read_field(record, "progress_reward", 0.0))),
            step_reward=float(str(_read_field(record, "step_reward", 0.0))),
            invalid_reward=float(str(_read_field(record, "invalid_reward", 0.0))),
            valid_label=(
                None if raw_valid_label is None else bool(raw_valid_label)
            ),
            predicted_valid=(
                None if raw_predicted_valid is None else bool(raw_predicted_valid)
            ),
            invalid_kind=(None if raw_invalid_kind is None else str(raw_invalid_kind)),
            info=raw_info,
        )


@dataclass
class Trajectory:
    trajectory_id: str
    steps: list[StepRecord] = field(default_factory=list)

    def append(self, record: StepRecord) -> None:
        if record.trajectory_id != self.trajectory_id:
            raise ValueError("record belongs to another trajectory")
        if self.steps and self.steps[-1].done:
            raise ValueError("cannot append after a terminal step")
        expected_index = self.steps[-1].step_index + 1 if self.steps else 0
        if record.step_index != expected_index:
            raise ValueError(
                f"expected step_index={expected_index}, got {record.step_index}"
            )
        self.steps.append(record)

    @property
    def done(self) -> bool:
        return bool(self.steps and self.steps[-1].done)

    @property
    def total_reward(self) -> float:
        return sum(record.reward for record in self.steps)

    def __len__(self) -> int:
        return len(self.steps)
