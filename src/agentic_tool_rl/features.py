"""Leakage-resistant feature extraction for the lightweight policy.

The encoder consumes only the observation and grounded candidate calls exposed
to a policy.  It never reads oracle plans, hidden goal predicates, validity
labels, or candidate ``call_id`` values (which are evidence identifiers rather
than policy inputs).
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from agentic_tool_rl.contracts import Observation, ToolCall

_TOKEN_PATTERN = re.compile(r"[^\W_]+", re.UNICODE)
_OPAQUE_OPERATION_PATTERN = re.compile(r"\bop_[0-9a-f]{20}\b", re.IGNORECASE)
_INTERNAL_POLICY_FIELDS = frozenset(
    {
        "approval_token",
        "audit_log",
        "call_id",
        "candidate_valid",
        "entity_id",
        "idempotency_results",
        "invalid_kind",
        "label_source",
        "predicted_valid",
        "processed_call_ids",
        "schema_label",
        "state_sha256",
        "valid_label",
    }
)


def _stable_bucket(token: str, buckets: int) -> tuple[int, float]:
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    number = int.from_bytes(digest, "big")
    return number % buckets, 1.0 if (number >> 1) & 1 else -1.0


def _tokens(value: Any) -> list[str]:
    if value is None:
        return ["<none>"]
    if isinstance(value, bool):
        return [f"bool:{value}"]
    if isinstance(value, (int, float)):
        return [f"number:{value}"]
    if isinstance(value, str):
        return [token.lower() for token in _TOKEN_PATTERN.findall(value)]
    if isinstance(value, Mapping):
        result: list[str] = []
        for key in sorted(value, key=str):
            result.append(f"key:{key}")
            result.extend(_tokens(value[key]))
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        result = [f"length:{len(value)}"]
        for item in value:
            result.extend(_tokens(item))
        return result
    return _tokens(str(value))


def _hashed_add(
    vector: Tensor,
    tokens: Sequence[str],
    *,
    start: int,
    end: int | None = None,
) -> None:
    resolved_end = vector.numel() if end is None else end
    buckets = resolved_end - start
    if buckets <= 0:
        return
    for token in tokens:
        bucket, sign = _stable_bucket(token, buckets)
        vector[start + bucket] += sign
    scale = max(1.0, float(len(tokens)) ** 0.5)
    vector[start:resolved_end] /= scale


def _hashed_operations_add(
    vector: Tensor,
    operation_ids: Sequence[str],
    *,
    buckets: int,
) -> None:
    opaque = [
        value for value in operation_ids if _OPAQUE_OPERATION_PATTERN.fullmatch(value) is not None
    ]
    if not opaque:
        return
    _hashed_add(
        vector,
        [f"operation-id:{value.lower()}" for value in opaque],
        start=vector.numel() - buckets,
    )


def _visible_observation(observation: Observation | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(observation, Observation):
        return observation.model_dump(mode="json")
    return dict(observation)


def _deidentified_goal(data: Mapping[str, Any], visible: Mapping[str, Any]) -> str:
    goal = data.get("user_goal", "")
    if not isinstance(goal, str):
        return ""
    identifiers: set[str] = set()
    for value in (data.get("task_id"), visible.get("entity_id"), visible.get("approval_token")):
        if isinstance(value, str) and value:
            identifiers.add(value)
    available_entities = data.get("available_entities", [])
    if isinstance(available_entities, Sequence) and not isinstance(
        available_entities, (str, bytes, bytearray)
    ):
        identifiers.update(item for item in available_entities if isinstance(item, str) and item)
    result = goal
    for identifier in sorted(identifiers, key=len, reverse=True):
        result = re.sub(re.escape(identifier), "<entity>", result, flags=re.IGNORECASE)
    return result


def _deidentified_message(data: Mapping[str, Any]) -> str:
    message = data.get("message", "")
    if not isinstance(message, str):
        return ""
    return _OPAQUE_OPERATION_PATTERN.sub("<operation>", message)


def _tool_call(call: ToolCall | Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    if isinstance(call, ToolCall):
        return call.tool_name, dict(call.arguments)
    nested = call.get("tool_call")
    source = nested if isinstance(nested, Mapping) else call
    name = source.get("tool_name", source.get("name"))
    arguments = source.get("arguments", source.get("parameters", {}))
    if not isinstance(name, str) or not isinstance(arguments, Mapping):
        raise ValueError("candidate must contain a tool name and argument object")
    return name, dict(arguments)


@dataclass(frozen=True)
class DecisionFeatures:
    state: Tensor
    actions: Tensor
    mask: Tensor

    def as_batch(self) -> tuple[Tensor, Tensor, Tensor]:
        return self.state.unsqueeze(0), self.actions.unsqueeze(0), self.mask.unsqueeze(0)


class FeatureEncoder:
    """Fixed-width deterministic encoder used by the CPU policy."""

    def __init__(self, state_dim: int = 64, action_dim: int = 64) -> None:
        if state_dim < 16 or action_dim < 16:
            raise ValueError("feature dimensions must be at least 16")
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.operation_buckets = min(
            16,
            (min(state_dim, action_dim) - 8) // 2,
        )

    def encode_state(self, observation: Observation | Mapping[str, Any]) -> Tensor:
        data = _visible_observation(observation)
        visible = data.get("visible_state", data)
        if not isinstance(visible, Mapping):
            raise ValueError("observation.visible_state must be an object")
        version = float(visible.get("version", data.get("step_index", 0)))
        completed = visible.get("completed_nodes", [])
        completed_count = len(completed) if isinstance(completed, Sequence) else 0
        vector = torch.zeros(self.state_dim, dtype=torch.float32)
        vector[0] = version / 15.0
        vector[1] = completed_count / 15.0
        vector[2] = float(bool(data.get("done", False)))
        vector[3] = float(bool(visible.get("approval_token")))
        step_index = float(data.get("step_index", version))
        max_steps = max(1.0, float(data.get("max_steps", 19)))
        remaining_steps = float(data.get("remaining_steps", max_steps - step_index))
        vector[4] = step_index / max_steps
        vector[5] = max(0.0, remaining_steps) / max_steps
        vector[6] = min(max_steps, 20.0) / 20.0
        # Entity IDs, task IDs and approval tokens are deliberately excluded:
        # they are per-case noise and encourage memorisation.  The user goal is
        # retained after exact identifiers are redacted, so semantic intent can
        # affect both the actor and the progress estimator without enabling
        # per-case lookup.
        safe_visible = {
            key: value
            for key, value in visible.items()
            if key not in _INTERNAL_POLICY_FIELDS and key != "completed_nodes"
        }
        safe = {
            "user_goal": _deidentified_goal(data, visible),
            "visible_state": safe_visible,
            "message": _deidentified_message(data),
            "step_index": data.get("step_index"),
            "max_steps": data.get("max_steps"),
            "remaining_steps": data.get("remaining_steps"),
        }
        _hashed_add(
            vector,
            _tokens(safe),
            start=8,
            end=vector.numel() - self.operation_buckets,
        )
        if isinstance(completed, Sequence) and not isinstance(completed, (str, bytes, bytearray)):
            _hashed_operations_add(
                vector,
                [value for value in completed if isinstance(value, str)],
                buckets=self.operation_buckets,
            )
        return vector

    def encode_action(self, call: ToolCall | Mapping[str, Any]) -> Tensor:
        tool_name, arguments = _tool_call(call)
        vector = torch.zeros(self.action_dim, dtype=torch.float32)
        expected_version = arguments.get("expected_version")
        if isinstance(expected_version, int) and not isinstance(expected_version, bool):
            vector[0] = expected_version / 15.0
        vector[2] = float("entity_id" in arguments)
        vector[3] = float("operation_id" in arguments)
        vector[4] = float("expected_version" in arguments)
        vector[5] = float(bool(arguments.get("approval_token")))
        vector[6] = float(bool(arguments.get("idempotency_key")))
        safe = {
            "tool_name": tool_name,
            "argument_keys": sorted(key for key in arguments if key not in _INTERNAL_POLICY_FIELDS),
        }
        _hashed_add(
            vector,
            _tokens(safe),
            start=8,
            end=vector.numel() - self.operation_buckets,
        )
        operation_id = arguments.get("operation_id")
        if isinstance(operation_id, str):
            _hashed_operations_add(
                vector,
                [operation_id],
                buckets=self.operation_buckets,
            )
        return vector

    def encode_decision(
        self,
        observation: Observation | Mapping[str, Any],
        candidates: Sequence[ToolCall | Mapping[str, Any]],
        mask: Sequence[bool] | None = None,
    ) -> DecisionFeatures:
        if not candidates:
            raise ValueError("a decision must contain at least one candidate")
        resolved_mask = list(mask) if mask is not None else [True] * len(candidates)
        if len(resolved_mask) != len(candidates):
            raise ValueError("mask and candidates must have equal length")
        if not any(resolved_mask):
            raise ValueError("at least one candidate must be selectable")
        actions = torch.stack([self.encode_action(candidate) for candidate in candidates])
        # Entity grounding is a public relation, not an instance identity. The
        # standalone action encoder deliberately omits raw entity IDs; at the
        # decision boundary we can still expose whether each candidate refers
        # to one of the entities visible in the current observation. This
        # prevents the unmasked baseline from facing an information-theoretic
        # impossible grounding classification while retaining de-identification.
        data = _visible_observation(observation)
        visible = data.get("visible_state", data)
        available = data.get("available_entities", [])
        available_entities = {
            value
            for value in available
            if isinstance(value, str)
        } if isinstance(available, Sequence) and not isinstance(
            available, (str, bytes, bytearray)
        ) else set()
        if isinstance(visible, Mapping):
            visible_entity = visible.get("entity_id")
            if isinstance(visible_entity, str):
                available_entities.add(visible_entity)
        for index, candidate in enumerate(candidates):
            _, arguments = _tool_call(candidate)
            entity_id = arguments.get("entity_id")
            actions[index, 1] = float(
                isinstance(entity_id, str) and entity_id in available_entities
            )
        return DecisionFeatures(
            state=self.encode_state(observation),
            actions=actions,
            mask=torch.tensor(resolved_mask, dtype=torch.bool),
        )

    def fingerprint(self) -> str:
        payload = json.dumps(
            {
                "version": 4,
                "state_dim": self.state_dim,
                "action_dim": self.action_dim,
                "operation_buckets": self.operation_buckets,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


__all__ = ["DecisionFeatures", "FeatureEncoder"]
