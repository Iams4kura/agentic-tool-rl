"""Leakage-resistant feature extraction for the lightweight policy.

The canonical path consumes one :class:`PolicyInput`, including the same
public observation, candidate calls, action mask, and schemas available to
registered baselines and Qwen.  It never receives oracle plans, hidden goal
predicates, validity labels, or candidate ``call_id`` evidence identifiers.
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

from agentic_tool_rl.contracts import Observation, ToolCall, ToolSchema
from agentic_tool_rl.policy_input import PolicyInput, PublicToolCall, PublicToolSchema

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


def _tool_call(
    call: ToolCall | PublicToolCall | Mapping[str, Any],
) -> tuple[str, dict[str, Any]]:
    if isinstance(call, (ToolCall, PublicToolCall)):
        return call.tool_name, dict(call.arguments)
    nested = call.get("tool_call")
    source = nested if isinstance(nested, Mapping) else call
    name = source.get("tool_name", source.get("name"))
    arguments = source.get("arguments", source.get("parameters", {}))
    if not isinstance(name, str) or not isinstance(arguments, Mapping):
        raise ValueError("candidate must contain a tool name and argument object")
    return name, dict(arguments)


def _schema_mapping(
    schema: ToolSchema | PublicToolSchema | Mapping[str, Any],
) -> dict[str, Any]:
    if isinstance(schema, (ToolSchema, PublicToolSchema)):
        return schema.model_dump(mode="json")
    return dict(schema)


@dataclass(frozen=True)
class DecisionFeatures:
    state: Tensor
    actions: Tensor
    mask: Tensor
    policy_input_sha256: str | None = None

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

    def encode_action(
        self,
        call: ToolCall | PublicToolCall | Mapping[str, Any],
        *,
        schema: ToolSchema | PublicToolSchema | Mapping[str, Any] | None = None,
    ) -> Tensor:
        """Encode an action, including its public schema when one is supplied.

        The schema-less form is retained as an explicitly lower-information
        compatibility path.  Policy execution uses :meth:`encode_policy_input`
        so descriptions, safety flags, and public DAG prerequisites are not
        available only to heuristic baselines.
        """

        tool_name, arguments = _tool_call(call)
        vector = torch.zeros(self.action_dim, dtype=torch.float32)
        expected_version = arguments.get("expected_version")
        if isinstance(expected_version, int) and not isinstance(expected_version, bool):
            vector[0] = expected_version / 15.0
        if schema is None:
            vector[2] = float("entity_id" in arguments)
            vector[3] = float("operation_id" in arguments)
            vector[4] = float("expected_version" in arguments)
            vector[5] = float(bool(arguments.get("approval_token")))
            vector[6] = float(bool(arguments.get("idempotency_key")))
            safe: dict[str, Any] = {
                "tool_name": tool_name,
                "argument_keys": sorted(
                    key for key in arguments if key not in _INTERNAL_POLICY_FIELDS
                ),
            }
            operation_ids = [arguments.get("operation_id")]
        else:
            schema_data = _schema_mapping(schema)
            required_predecessors = schema_data.get("required_completed_operations", [])
            if not isinstance(required_predecessors, Sequence) or isinstance(
                required_predecessors, (str, bytes, bytearray)
            ):
                raise ValueError("schema.required_completed_operations must be a sequence")
            predecessor_ids = [
                value for value in required_predecessors if isinstance(value, str)
            ]
            vector[2] = float(bool(schema_data.get("mutating", True)))
            vector[3] = float(bool(schema_data.get("idempotent", True)))
            vector[4] = float(bool(schema_data.get("policy_allowed", True)))
            vector[5] = float(bool(schema_data.get("additional_properties", False)))
            vector[6] = min(len(predecessor_ids), 15) / 15.0
            vector[7] = float(schema_data.get("side_effect") is not None)
            safe = {
                "tool_name": tool_name,
                "argument_keys": sorted(
                    key for key in arguments if key not in _INTERNAL_POLICY_FIELDS
                ),
                "schema_name": schema_data.get("name"),
                "schema_description": schema_data.get("description"),
                "required_arguments": schema_data.get("required_arguments", {}),
                "optional_arguments": schema_data.get("optional_arguments", {}),
                "additional_properties": schema_data.get("additional_properties", False),
                "mutating": schema_data.get("mutating", True),
                "idempotent": schema_data.get("idempotent", True),
                "policy_allowed": schema_data.get("policy_allowed", True),
                "side_effect": schema_data.get("side_effect"),
                "required_predecessor_count": len(predecessor_ids),
            }
            operation_ids = [schema_data.get("operation_id"), *predecessor_ids]
        _hashed_add(
            vector,
            _tokens(safe),
            start=8,
            end=vector.numel() - self.operation_buckets,
        )
        public_operation_ids = [value for value in operation_ids if isinstance(value, str)]
        if public_operation_ids:
            _hashed_operations_add(
                vector,
                public_operation_ids,
                buckets=self.operation_buckets,
            )
        return vector

    def encode_policy_input(
        self,
        policy_input: PolicyInput,
        *,
        selection_mask: Sequence[bool] | None = None,
    ) -> DecisionFeatures:
        """Encode the canonical public DTO used by every policy decision."""

        if not isinstance(policy_input, PolicyInput):
            raise TypeError("encode_policy_input accepts only PolicyInput")
        public_mask = [candidate.action_mask for candidate in policy_input.candidates]
        resolved_mask = list(selection_mask) if selection_mask is not None else public_mask
        if len(resolved_mask) != len(policy_input.candidates):
            raise ValueError("selection_mask and PolicyInput candidates must have equal length")
        if any(not isinstance(value, bool) for value in resolved_mask):
            raise TypeError("selection_mask entries must be bool")
        if not any(resolved_mask):
            raise ValueError("at least one candidate must be selectable")

        observation = policy_input.observation.model_dump(mode="json")
        calls = [candidate.tool_call for candidate in policy_input.candidates]
        actions = torch.stack(
            [
                self.encode_action(candidate.tool_call, schema=candidate.tool_schema)
                for candidate in policy_input.candidates
            ]
        )
        self._encode_grounding_relation(observation, calls, actions)
        return DecisionFeatures(
            state=self.encode_state(observation),
            actions=actions,
            mask=torch.tensor(resolved_mask, dtype=torch.bool),
            policy_input_sha256=policy_input.sha256(),
        )

    def encode_decision(
        self,
        observation: Observation | Mapping[str, Any],
        candidates: Sequence[ToolCall | PublicToolCall | Mapping[str, Any]],
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
        self._encode_grounding_relation(observation, candidates, actions)
        return DecisionFeatures(
            state=self.encode_state(observation),
            actions=actions,
            mask=torch.tensor(resolved_mask, dtype=torch.bool),
        )

    def _encode_grounding_relation(
        self,
        observation: Observation | Mapping[str, Any],
        candidates: Sequence[ToolCall | PublicToolCall | Mapping[str, Any]],
        actions: Tensor,
    ) -> None:
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

    def fingerprint(self) -> str:
        payload = json.dumps(
            {
                "version": 5,
                "state_dim": self.state_dim,
                "action_dim": self.action_dim,
                "operation_buckets": self.operation_buckets,
                "policy_input_schema": "policy-input-v1",
                "schema_action_features": 1,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


__all__ = ["DecisionFeatures", "FeatureEncoder"]
