"""Backend-neutral structured-action policy contract."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
from math import isfinite
from typing import Any


class BackendContractError(ValueError):
    """A policy backend returned an unsafe or incomplete structured action."""


def to_jsonable(value: Any) -> Any:
    """Convert Pydantic/dataclass/domain values to a JSON-compatible shape."""

    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if isinstance(value, Mapping):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [to_jsonable(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"value of type {type(value).__name__} is not JSON serializable")


@dataclass(frozen=True)
class EncodedPolicyInput:
    """Stable representation passed from an adapter to its runtime."""

    prompt: str
    observation: Mapping[str, Any]
    candidates: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True)
class PolicyOutput:
    """One action-level RL decision."""

    action_index: int
    tool_call: Mapping[str, Any]
    log_prob: float
    value: float
    raw_output: str | None = None

    def __post_init__(self) -> None:
        if self.action_index < 0:
            raise BackendContractError("action_index must be non-negative")
        if not isfinite(self.log_prob):
            raise BackendContractError("log_prob must be finite")
        if not isfinite(self.value):
            raise BackendContractError("value must be finite")
        if "tool_name" not in self.tool_call or "arguments" not in self.tool_call:
            raise BackendContractError("tool_call must contain tool_name and arguments")
        if not isinstance(self.tool_call["arguments"], Mapping):
            raise BackendContractError("tool_call.arguments must be an object")

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_index": self.action_index,
            "tool_call": to_jsonable(self.tool_call),
            "log_prob": self.log_prob,
            "value": self.value,
            "raw_output": self.raw_output,
        }


class PolicyBackend(ABC):
    """Common contract shared by lightweight and optional LLM policies."""

    @abstractmethod
    def encode(
        self, observation: Mapping[str, Any] | Any, candidates: Sequence[Mapping[str, Any] | Any]
    ) -> EncodedPolicyInput:
        """Encode visible state and executable candidates without hidden labels."""

    @abstractmethod
    def act(
        self,
        observation: Mapping[str, Any] | Any,
        candidates: Sequence[Mapping[str, Any] | Any],
        *,
        deterministic: bool = False,
    ) -> PolicyOutput:
        """Choose one candidate and provide action log-probability and value."""


def stable_mapping(value: Mapping[str, Any] | Any, *, name: str) -> dict[str, Any]:
    converted = to_jsonable(value)
    if not isinstance(converted, dict):
        raise BackendContractError(f"{name} must encode to a JSON object")
    # Round-trip also rejects NaN/Infinity and removes custom mapping types.
    try:
        encoded = json.dumps(converted, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise BackendContractError(f"{name} is not valid JSON: {exc}") from exc
    decoded = json.loads(encoded)
    if not isinstance(decoded, dict):  # defensive, guaranteed by the input check
        raise BackendContractError(f"{name} must encode to a JSON object")
    return decoded
