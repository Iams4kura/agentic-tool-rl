"""Independent environment-side validation and transactional tool execution.

This module intentionally does not import the policy-side action mask.  Its
``validate`` method is the dry-run oracle that creates action-validity labels;
sharing the implementation with the learned/heuristic mask would make the
benchmark circular.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from agentic_tool_rl.contracts import (
    InvalidActionKind,
    ToolCall,
    ToolSchema,
    ValidationResult,
    WorkflowTask,
)
from agentic_tool_rl.latency_profile import (
    IDEMPOTENT_REPLAY_S,
    INVALID_VALIDATION_S,
    READ_ONLY_QUERY_S,
)


@dataclass(frozen=True, slots=True)
class ToolExecution:
    state: dict[str, Any]
    validation: ValidationResult
    simulated_latency_s: float


def _matches_json_type(value: object, expected: str) -> bool:
    if expected == "string":
        return isinstance(value, str)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "array":
        return isinstance(value, list)
    if expected == "object":
        return isinstance(value, dict)
    if expected == "null":
        return value is None
    return False


class ToolRegistry:
    """Task-scoped registry with an all-or-nothing mutation boundary."""

    def __init__(self, task: WorkflowTask) -> None:
        self.task = task
        self._schemas = {schema.name: schema for schema in task.tool_schemas}
        self._nodes = {node.node_id: node for node in task.workflow_nodes}
        if len(self._schemas) != len(task.tool_schemas):
            raise ValueError("tool schema names must be unique")
        if len(self._nodes) != len(task.workflow_nodes):
            raise ValueError("workflow node ids must be unique")

    def _validate_schema(self, call: ToolCall) -> tuple[ToolSchema | None, str]:
        schema = self._schemas.get(call.tool_name)
        if schema is None:
            return None, f"unknown tool: {call.tool_name}"
        arguments = call.arguments
        missing = set(schema.required_arguments) - set(arguments)
        if missing:
            return None, f"missing required arguments: {sorted(missing)}"
        allowed = set(schema.required_arguments) | set(schema.optional_arguments)
        unexpected = set(arguments) - allowed
        if unexpected and not schema.additional_properties:
            return None, f"unexpected arguments: {sorted(unexpected)}"
        expected_types = {**schema.required_arguments, **schema.optional_arguments}
        for name, value in arguments.items():
            expected = expected_types.get(name)
            if expected is not None and not _matches_json_type(value, expected):
                return None, f"argument {name!r} must be {expected}"
        return schema, ""

    def validate(self, state: dict[str, Any], call: ToolCall) -> ValidationResult:
        """Label a call without changing state (the benchmark dry-run oracle)."""

        schema, schema_error = self._validate_schema(call)
        if schema is None:
            return ValidationResult(
                valid=False,
                invalid_kind=InvalidActionKind.SCHEMA,
                reason=schema_error,
            )

        if schema.side_effect in self.task.forbidden_side_effects:
            return ValidationResult(
                valid=False,
                invalid_kind=InvalidActionKind.SAFETY,
                reason=f"forbidden side effect: {schema.side_effect}",
            )

        entity_id = call.arguments["entity_id"]
        operation_id = call.arguments["operation_id"]
        if entity_id != state.get("entity_id") or entity_id != self.task.entity_id:
            return ValidationResult(
                valid=False,
                invalid_kind=InvalidActionKind.GROUNDING,
                reason=f"entity {entity_id!r} is not present in the current observation",
            )
        if not schema.mutating:
            if operation_id != "inspect-status":
                return ValidationResult(
                    valid=False,
                    invalid_kind=InvalidActionKind.GROUNDING,
                    reason="read-only operation does not ground to inspect-status",
                )
            if call.arguments["expected_version"] != state.get("version"):
                return ValidationResult(
                    valid=False,
                    invalid_kind=InvalidActionKind.PRECONDITION,
                    reason="read-only query uses a stale version",
                )
            return ValidationResult(valid=True, reason="read-only status query")
        node = self._nodes.get(str(operation_id))
        if node is None or node.tool_name != call.tool_name:
            return ValidationResult(
                valid=False,
                invalid_kind=InvalidActionKind.GROUNDING,
                reason="operation_id does not ground to the selected tool",
            )

        processed_call_ids = set(state.get("processed_call_ids", []))
        if call.call_id and call.call_id in processed_call_ids:
            return ValidationResult(
                valid=True,
                reason="exact call replay resolved from the idempotency ledger",
                node_id=node.node_id,
                idempotent_replay=True,
            )

        completed = set(state.get("completed_nodes", []))
        if node.node_id in completed:
            if schema.idempotent:
                return ValidationResult(
                    valid=True,
                    reason="operation already completed; returning idempotent result",
                    node_id=node.node_id,
                    idempotent_replay=True,
                )
            return ValidationResult(
                valid=False,
                invalid_kind=InvalidActionKind.SAFETY,
                reason="non-idempotent operation cannot be repeated",
                node_id=node.node_id,
            )

        if call.arguments["expected_version"] != state.get("version"):
            return ValidationResult(
                valid=False,
                invalid_kind=InvalidActionKind.PRECONDITION,
                reason="optimistic concurrency version is stale",
                node_id=node.node_id,
            )
        missing_predecessors = set(node.required_predecessors) - completed
        if missing_predecessors:
            return ValidationResult(
                valid=False,
                invalid_kind=InvalidActionKind.PRECONDITION,
                reason=f"unfinished predecessors: {sorted(missing_predecessors)}",
                node_id=node.node_id,
            )

        if node.safety_token is not None:
            if call.arguments.get("approval_token") != node.safety_token:
                return ValidationResult(
                    valid=False,
                    invalid_kind=InvalidActionKind.SAFETY,
                    reason="approval token is missing or forged",
                    node_id=node.node_id,
                )
            if state.get("approval_token") != node.safety_token:
                return ValidationResult(
                    valid=False,
                    invalid_kind=InvalidActionKind.PRECONDITION,
                    reason="approval has not been recorded in business state",
                    node_id=node.node_id,
                )
            idempotency_key = call.arguments.get("idempotency_key")
            if not isinstance(idempotency_key, str) or not idempotency_key.strip():
                return ValidationResult(
                    valid=False,
                    invalid_kind=InvalidActionKind.SAFETY,
                    reason="a non-empty idempotency key is required",
                    node_id=node.node_id,
                )
            ledger = state.get("idempotency_results", {})
            if idempotency_key in ledger and ledger[idempotency_key] != node.node_id:
                return ValidationResult(
                    valid=False,
                    invalid_kind=InvalidActionKind.SAFETY,
                    reason="idempotency key was already used for another operation",
                    node_id=node.node_id,
                )

        return ValidationResult(valid=True, node_id=node.node_id)

    def execute(self, state: dict[str, Any], call: ToolCall) -> ToolExecution:
        """Validate then atomically apply a call to a deep-copied state."""

        validation = self.validate(state, call)
        if not validation.valid:
            return ToolExecution(
                state=deepcopy(state),
                validation=validation,
                simulated_latency_s=INVALID_VALIDATION_S,
            )
        schema = self._schemas[call.tool_name]
        if not schema.mutating:
            return ToolExecution(
                state=deepcopy(state),
                validation=validation,
                simulated_latency_s=READ_ONLY_QUERY_S,
            )
        if validation.node_id is None:
            raise RuntimeError("valid tool call did not resolve to a workflow node")
        node = self._nodes[validation.node_id]
        if validation.idempotent_replay:
            return ToolExecution(
                state=deepcopy(state),
                validation=validation,
                simulated_latency_s=IDEMPOTENT_REPLAY_S,
            )

        next_state = deepcopy(state)
        completed_nodes = list(next_state.get("completed_nodes", []))
        completed_nodes.append(node.node_id)
        next_state["completed_nodes"] = completed_nodes
        next_state["version"] = int(next_state.get("version", 0)) + 1
        next_state.update(deepcopy(node.effects))

        if node.side_effect is not None:
            side_effects = list(next_state.get("side_effects", []))
            side_effects.append(node.side_effect)
            next_state["side_effects"] = side_effects
        if call.call_id:
            processed = list(next_state.get("processed_call_ids", []))
            processed.append(call.call_id)
            next_state["processed_call_ids"] = processed
        idempotency_key = call.arguments.get("idempotency_key")
        if isinstance(idempotency_key, str):
            ledger = dict(next_state.get("idempotency_results", {}))
            ledger[idempotency_key] = node.node_id
            next_state["idempotency_results"] = ledger
        audit_log = list(next_state.get("audit_log", []))
        audit_log.append(
            {
                "call_id": call.call_id,
                "tool_name": call.tool_name,
                "node_id": node.node_id,
                "version": next_state["version"],
            }
        )
        next_state["audit_log"] = audit_log
        return ToolExecution(
            state=next_state,
            validation=validation,
            simulated_latency_s=node.simulated_latency_s,
        )


__all__ = ["ToolExecution", "ToolRegistry"]
