"""Policy-side executable action mask over public inputs only.

The environment validator remains the independent ground-truth judge.  This
module intentionally receives a :class:`WorkflowTask` for call-site
compatibility, but copies only its public tool schemas; workflow nodes, oracle
plans, hidden safety tokens, and forbidden-side-effect labels are never read.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from agentic_tool_rl.contracts import (
    InvalidActionKind,
    Observation,
    ToolCall,
    ToolSchema,
    ValidationResult,
    WorkflowTask,
)


def _mask_type_matches(value: object, expected: str) -> bool:
    checks = {
        "string": lambda item: isinstance(item, str),
        "integer": lambda item: isinstance(item, int) and not isinstance(item, bool),
        "number": lambda item: isinstance(item, (int, float)) and not isinstance(item, bool),
        "boolean": lambda item: isinstance(item, bool),
        "array": lambda item: isinstance(item, list),
        "object": lambda item: isinstance(item, dict),
        "null": lambda item: item is None,
    }
    predicate = checks.get(expected)
    return predicate(value) if predicate is not None else False


def _public_observation(
    observation: Observation | Mapping[str, Any],
) -> tuple[Mapping[str, Any], frozenset[str], frozenset[str]]:
    state: Mapping[str, Any]
    if isinstance(observation, Observation):
        state = observation.visible_state
        entities = observation.available_entities
        tools = observation.available_tools
    else:
        nested = observation.get("visible_state")
        state = (
            {str(key): value for key, value in nested.items()}
            if isinstance(nested, Mapping)
            else observation
        )
        raw_entities = observation.get("available_entities", [])
        raw_tools = observation.get("available_tools", [])
        entities = (
            [item for item in raw_entities if isinstance(item, str)]
            if isinstance(raw_entities, Sequence)
            and not isinstance(raw_entities, (str, bytes, bytearray))
            else []
        )
        tools = (
            [item for item in raw_tools if isinstance(item, str)]
            if isinstance(raw_tools, Sequence)
            and not isinstance(raw_tools, (str, bytes, bytearray))
            else []
        )
    state_entity = state.get("entity_id")
    grounded_entities = set(entities)
    if isinstance(state_entity, str):
        grounded_entities.add(state_entity)
    return state, frozenset(grounded_entities), frozenset(tools)


class ActionMask:
    """Deterministic constraint mask derived from public schemas and observation."""

    def __init__(self, source: WorkflowTask | Sequence[ToolSchema]) -> None:
        schemas = source.tool_schemas if isinstance(source, WorkflowTask) else list(source)
        self._schemas = {schema.name: schema.model_copy(deep=True) for schema in schemas}
        if len(self._schemas) != len(schemas):
            raise ValueError("tool schema names must be unique")

    @staticmethod
    def _schema_error(schema: ToolSchema | None, call: ToolCall) -> str | None:
        if schema is None:
            return f"unknown tool: {call.tool_name}"
        names = set(call.arguments)
        missing = set(schema.required_arguments) - names
        if missing:
            return f"missing required arguments: {sorted(missing)}"
        allowed = set(schema.required_arguments) | set(schema.optional_arguments)
        extra = names - allowed
        if extra and not schema.additional_properties:
            return f"unexpected arguments: {sorted(extra)}"
        declared = {**schema.required_arguments, **schema.optional_arguments}
        for name, value in call.arguments.items():
            expected = declared.get(name)
            if expected is not None and not _mask_type_matches(value, expected):
                return f"argument {name!r} must be {expected}"
        return None

    def validate(
        self,
        observation: Observation | Mapping[str, Any],
        call: ToolCall,
    ) -> ValidationResult:
        state, available_entities, available_tools = _public_observation(observation)
        schema = self._schemas.get(call.tool_name)
        schema_error = self._schema_error(schema, call)
        if schema_error is not None:
            return ValidationResult(
                valid=False,
                invalid_kind=InvalidActionKind.SCHEMA,
                reason=schema_error,
            )
        if schema is None:  # narrowed above; retained for static type checking
            raise RuntimeError("schema disappeared during action-mask validation")

        if available_tools and call.tool_name not in available_tools:
            return ValidationResult(
                valid=False,
                invalid_kind=InvalidActionKind.GROUNDING,
                reason="tool is not available in the current observation",
            )
        if not schema.policy_allowed:
            return ValidationResult(
                valid=False,
                invalid_kind=InvalidActionKind.SAFETY,
                reason="public tool policy forbids this operation",
            )

        entity_id = call.arguments["entity_id"]
        operation_id = str(call.arguments["operation_id"])
        if not isinstance(entity_id, str) or entity_id not in available_entities:
            return ValidationResult(
                valid=False,
                invalid_kind=InvalidActionKind.GROUNDING,
                reason="entity is not grounded in the current observation",
            )

        expected_operation = schema.operation_id
        if expected_operation is not None and operation_id != expected_operation:
            return ValidationResult(
                valid=False,
                invalid_kind=InvalidActionKind.GROUNDING,
                reason="operation does not match the public tool schema",
            )
        if expected_operation is None and not schema.mutating and operation_id != "inspect-status":
            return ValidationResult(
                valid=False,
                invalid_kind=InvalidActionKind.GROUNDING,
                reason="read-only operation does not ground to inspect-status",
            )

        completed = set(state.get("completed_nodes", []))
        resolved_operation = expected_operation or operation_id
        if schema.mutating and resolved_operation in completed:
            if schema.idempotent:
                return ValidationResult(
                    valid=True,
                    node_id=resolved_operation,
                    reason="idempotent replay",
                    idempotent_replay=True,
                )
            return ValidationResult(
                valid=False,
                invalid_kind=InvalidActionKind.SAFETY,
                reason="non-idempotent operation cannot be replayed",
                node_id=resolved_operation,
            )

        if call.arguments["expected_version"] != state.get("version"):
            return ValidationResult(
                valid=False,
                invalid_kind=InvalidActionKind.PRECONDITION,
                reason="stale expected_version",
                node_id=resolved_operation if schema.mutating else None,
            )

        missing = set(schema.required_completed_operations) - completed
        if missing:
            return ValidationResult(
                valid=False,
                invalid_kind=InvalidActionKind.PRECONDITION,
                reason=f"unfinished public prerequisites: {sorted(missing)}",
                node_id=resolved_operation,
            )

        if "approval_token" in schema.required_arguments:
            observed_approval = state.get("approval_token")
            provided_approval = call.arguments.get("approval_token")
            if not isinstance(observed_approval, str) or not observed_approval:
                return ValidationResult(
                    valid=False,
                    invalid_kind=InvalidActionKind.PRECONDITION,
                    reason="approval is not visible in current state",
                    node_id=resolved_operation,
                )
            if provided_approval != observed_approval:
                return ValidationResult(
                    valid=False,
                    invalid_kind=InvalidActionKind.SAFETY,
                    reason="approval token does not match visible state",
                    node_id=resolved_operation,
                )
            idempotency_key = call.arguments.get("idempotency_key")
            if not isinstance(idempotency_key, str) or not idempotency_key.strip():
                return ValidationResult(
                    valid=False,
                    invalid_kind=InvalidActionKind.SAFETY,
                    reason="non-empty idempotency key required",
                    node_id=resolved_operation,
                )

        return ValidationResult(
            valid=True,
            node_id=resolved_operation if schema.mutating else None,
            reason="read-only status query" if not schema.mutating else "",
        )

    def evaluate(
        self,
        observation: Observation | Mapping[str, Any],
        candidates: Sequence[ToolCall],
    ) -> list[ValidationResult]:
        return [self.validate(observation, candidate) for candidate in candidates]

    def mask(
        self,
        observation: Observation | Mapping[str, Any],
        candidates: Sequence[ToolCall],
    ) -> list[bool]:
        return [decision.valid for decision in self.evaluate(observation, candidates)]


__all__ = ["ActionMask"]
