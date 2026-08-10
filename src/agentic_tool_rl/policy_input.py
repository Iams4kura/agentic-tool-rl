"""Canonical public input boundary shared by policies and public baselines.

``PolicyInput`` is the only DTO that should cross the environment-to-policy
boundary in benchmark v1.4.  Its factory deliberately accepts the public
pieces of a decision separately instead of accepting ``WorkflowTask``; hidden
task fields therefore cannot be consulted accidentally.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from agentic_tool_rl.contracts import Observation, ToolCall, ToolSchema

_ENTITY_ALIAS = re.compile(r"^<entity:[0-9]+>$")
_FORBIDDEN_KEYS = frozenset(
    {
        "audit_log",
        "call_id",
        "candidate_valid",
        "case_id",
        "evaluator_result",
        "forbidden_side_effects",
        "goal_predicates",
        "hidden_goal_predicates",
        "idempotency_results",
        "invalid_kind",
        "label_source",
        "oracle_distance",
        "oracle_plan",
        "oracle_plans",
        "positive_actions",
        "positive_label",
        "predicted_invalid_kind",
        "predicted_valid",
        "processed_call_ids",
        "safety_token",
        "schema_label",
        "state_sha256",
        "target_node_id",
        "target_node_ids",
        "task_id",
        "valid_label",
        "validity_label",
        "workflow_nodes",
    }
)
_OBSERVATION_FIELDS = (
    "step_index",
    "max_steps",
    "remaining_steps",
    "user_goal",
    "visible_state",
    "available_entities",
    "available_tools",
    "message",
    "done",
)


class _PolicyModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class PublicObservation(_PolicyModel):
    """De-identified, policy-visible subset of an environment observation."""

    step_index: int = Field(ge=0)
    max_steps: int = Field(ge=1)
    remaining_steps: int = Field(ge=0)
    user_goal: str
    visible_state: dict[str, Any]
    available_entities: tuple[str, ...]
    available_tools: tuple[str, ...]
    message: str = ""
    done: bool = False

    @model_validator(mode="after")
    def public_state_is_consistent(self) -> PublicObservation:
        if self.step_index + self.remaining_steps != self.max_steps:
            raise ValueError("step_index + remaining_steps must equal max_steps")
        if any(_ENTITY_ALIAS.fullmatch(value) is None for value in self.available_entities):
            raise ValueError("available_entities must contain only canonical aliases")
        _reject_forbidden_nested_keys(self.visible_state)
        return self


class PublicToolCall(_PolicyModel):
    """Candidate action without its evidence-only ``call_id``."""

    tool_name: str = Field(min_length=1)
    arguments: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def arguments_are_public(self) -> PublicToolCall:
        _reject_forbidden_nested_keys(self.arguments)
        return self


class PublicToolSchema(_PolicyModel):
    """Policy-visible schema fields associated with one candidate."""

    name: str = Field(min_length=1)
    description: str
    required_arguments: dict[str, str]
    optional_arguments: dict[str, str] = Field(default_factory=dict)
    additional_properties: bool = False
    mutating: bool = True
    idempotent: bool = True
    side_effect: str | None = None
    operation_id: str | None = None
    required_completed_operations: tuple[str, ...] = ()
    policy_allowed: bool = True


class PolicyCandidate(_PolicyModel):
    """One grounded call, its mask bit, and the corresponding public schema."""

    tool_call: PublicToolCall
    action_mask: bool
    tool_schema: PublicToolSchema

    @model_validator(mode="after")
    def call_matches_schema(self) -> PolicyCandidate:
        if self.tool_call.tool_name != self.tool_schema.name:
            raise ValueError("candidate tool_name must match its public schema")
        return self


class PolicyInput(_PolicyModel):
    """Stable and serializable complete public view of one policy decision."""

    schema_version: Literal["policy-input-v1"] = "policy-input-v1"
    observation: PublicObservation
    candidates: tuple[PolicyCandidate, ...]

    @model_validator(mode="after")
    def decision_is_selectable(self) -> PolicyInput:
        if not self.candidates:
            raise ValueError("PolicyInput requires at least one candidate")
        if not any(candidate.action_mask for candidate in self.candidates):
            raise ValueError("PolicyInput requires at least one selectable candidate")
        return self

    @classmethod
    def from_decision(
        cls,
        observation: Observation | Mapping[str, Any],
        candidates: Sequence[ToolCall | Mapping[str, Any]],
        action_mask: Sequence[bool],
        tool_schemas: Sequence[ToolSchema | Mapping[str, Any]],
    ) -> PolicyInput:
        """Build a public snapshot without accepting any hidden task object."""

        if not candidates:
            raise ValueError("PolicyInput requires at least one candidate")
        if len(candidates) != len(action_mask):
            raise ValueError("action_mask and candidates must have equal length")
        if any(not isinstance(value, bool) for value in action_mask):
            raise TypeError("action_mask entries must be bool")

        raw_observation = _as_mapping(observation, name="observation")
        raw_candidates = [
            _as_mapping(candidate, name=f"candidates[{index}]")
            for index, candidate in enumerate(candidates)
        ]
        aliases = _entity_aliases(raw_observation, raw_candidates)
        public_observation = _build_observation(raw_observation, aliases)

        schemas_by_name: dict[str, PublicToolSchema] = {}
        for index, schema_like in enumerate(tool_schemas):
            raw_schema = _as_mapping(schema_like, name=f"tool_schemas[{index}]")
            name = raw_schema.get("name")
            if not isinstance(name, str) or not name:
                raise ValueError(f"tool_schemas[{index}] must contain a name")
            if name in schemas_by_name:
                raise ValueError(f"duplicate public tool schema: {name}")
            schemas_by_name[name] = _build_schema(raw_schema, aliases)

        public_candidates: list[PolicyCandidate] = []
        for index, (raw_candidate, allowed) in enumerate(
            zip(raw_candidates, action_mask, strict=True)
        ):
            nested = raw_candidate.get("tool_call")
            source = nested if isinstance(nested, Mapping) else raw_candidate
            tool_name = source.get("tool_name", source.get("name"))
            arguments = source.get("arguments", source.get("parameters", {}))
            if not isinstance(tool_name, str) or not tool_name:
                raise ValueError(f"candidates[{index}] must contain a tool name")
            if not isinstance(arguments, Mapping):
                raise ValueError(f"candidates[{index}].arguments must be an object")
            try:
                schema = schemas_by_name[tool_name]
            except KeyError as exc:
                raise ValueError(
                    f"candidate references unknown public schema: {tool_name}"
                ) from exc
            public_candidates.append(
                PolicyCandidate(
                    tool_call=PublicToolCall(
                        tool_name=tool_name,
                        arguments=_sanitize(arguments, aliases),
                    ),
                    action_mask=allowed,
                    tool_schema=schema,
                )
            )
        return cls(observation=public_observation, candidates=tuple(public_candidates))

    def canonical_bytes(self) -> bytes:
        """Return deterministic UTF-8 JSON bytes for hashing and evidence."""

        payload = self.model_dump(mode="json")
        return json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")

    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()


def _as_mapping(value: Any, *, name: str) -> dict[str, Any]:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping or Pydantic model")
    try:
        encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain only finite JSON values") from exc
    decoded = json.loads(encoded)
    if not isinstance(decoded, dict):
        raise TypeError(f"{name} must encode to a JSON object")
    return decoded


def _walk_entity_ids(value: Any) -> list[str]:
    result: list[str] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            if str(key) == "entity_id" and isinstance(item, str) and item:
                result.append(item)
            result.extend(_walk_entity_ids(item))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for item in value:
            result.extend(_walk_entity_ids(item))
    return result


def _entity_aliases(
    observation: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
) -> dict[str, str]:
    identities: list[str] = []
    available = observation.get("available_entities", [])
    if isinstance(available, Sequence) and not isinstance(available, (str, bytes, bytearray)):
        identities.extend(value for value in available if isinstance(value, str) and value)
    identities.extend(_walk_entity_ids(observation.get("visible_state", {})))
    for candidate in candidates:
        identities.extend(_walk_entity_ids(candidate))
    unique = list(dict.fromkeys(identities))
    return {value: f"<entity:{index}>" for index, value in enumerate(unique)}


def _sanitize(value: Any, aliases: Mapping[str, str]) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _sanitize(item, aliases)
            for key, item in value.items()
            if str(key) not in _FORBIDDEN_KEYS
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_sanitize(item, aliases) for item in value]
    if isinstance(value, str):
        result = value
        for identity, alias in sorted(aliases.items(), key=lambda item: len(item[0]), reverse=True):
            result = result.replace(identity, alias)
        return result
    return value


def _build_observation(
    raw: Mapping[str, Any],
    aliases: Mapping[str, str],
) -> PublicObservation:
    missing = [field for field in _OBSERVATION_FIELDS[:7] if field not in raw]
    if missing:
        raise ValueError(f"observation is missing public fields: {missing}")
    public = {
        field: _sanitize(raw[field], aliases)
        for field in _OBSERVATION_FIELDS
        if field in raw
    }
    public.setdefault("message", "")
    public.setdefault("done", False)
    return PublicObservation.model_validate(public)


def _build_schema(
    raw: Mapping[str, Any],
    aliases: Mapping[str, str],
) -> PublicToolSchema:
    public = {
        "name": raw.get("name"),
        "description": raw.get("description", ""),
        "required_arguments": raw.get("required_arguments", {}),
        "optional_arguments": raw.get("optional_arguments", {}),
        "additional_properties": raw.get("additional_properties", False),
        "mutating": raw.get("mutating", True),
        "idempotent": raw.get("idempotent", True),
        "side_effect": raw.get("side_effect"),
        "operation_id": raw.get("operation_id"),
        "required_completed_operations": raw.get("required_completed_operations", []),
        "policy_allowed": raw.get("policy_allowed", True),
    }
    return PublicToolSchema.model_validate(_sanitize(public, aliases))


def _reject_forbidden_nested_keys(value: Any) -> None:
    if isinstance(value, Mapping):
        forbidden = set(value) & _FORBIDDEN_KEYS
        if forbidden:
            raise ValueError(f"private policy fields are forbidden: {sorted(forbidden)}")
        for item in value.values():
            _reject_forbidden_nested_keys(item)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for item in value:
            _reject_forbidden_nested_keys(item)


__all__ = [
    "PolicyCandidate",
    "PolicyInput",
    "PublicObservation",
    "PublicToolCall",
    "PublicToolSchema",
]
