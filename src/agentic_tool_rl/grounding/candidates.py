"""Candidate construction from public schemas and visible workflow state."""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Mapping
from typing import Any

from agentic_tool_rl.contracts import ToolCall, ToolSchema, WorkflowTask

_PUBLIC_STATE_SEED_FIELDS = (
    "entity_id",
    "version",
    "status",
    "priority",
    "region",
    "channel",
)


def _public_seed(
    task: WorkflowTask,
    state: Mapping[str, Any],
    *,
    namespace: str,
) -> int:
    """Derive a local seed from public schemas and visible state only."""

    completed = state.get("completed_nodes", [])
    payload = {
        "namespace": namespace,
        "state": {field: state.get(field) for field in _PUBLIC_STATE_SEED_FIELDS},
        "approval_visible": bool(state.get("approval_token")),
        "completed_operations": sorted(str(item) for item in completed),
        "schemas": sorted(
            (
                schema.name,
                schema.operation_id,
                tuple(sorted(schema.required_completed_operations)),
                schema.mutating,
                schema.policy_allowed,
            )
            for schema in task.tool_schemas
        ),
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big")


def _choose(
    values: list[ToolSchema],
    task: WorkflowTask,
    state: Mapping[str, Any],
    *,
    namespace: str,
) -> ToolSchema:
    if not values:
        raise ValueError(f"cannot choose from an empty {namespace} candidate set")
    return random.Random(_public_seed(task, state, namespace=namespace)).choice(values)


def _permuted(
    task: WorkflowTask,
    state: Mapping[str, Any],
    candidates: list[ToolCall],
) -> list[ToolCall]:
    result = list(candidates)
    random.Random(_public_seed(task, state, namespace="candidate-order")).shuffle(result)
    return result


def _arguments(state: Mapping[str, Any], schema: ToolSchema) -> dict[str, Any]:
    operation_id = schema.operation_id
    if operation_id is None:
        raise ValueError(f"tool schema {schema.name!r} does not declare an operation_id")
    entity_id = state.get("entity_id")
    result: dict[str, Any] = {
        "entity_id": entity_id,
        "operation_id": operation_id,
        "expected_version": state.get("version", 0),
    }
    if "approval_token" in schema.required_arguments:
        # Approval becomes a policy input only after it is present in the
        # visible business state.  Never copy the hidden workflow token into a
        # candidate for an earlier state.
        approval_token = state.get("approval_token")
        if isinstance(approval_token, str) and approval_token:
            result["approval_token"] = approval_token
        result["idempotency_key"] = f"idem-{entity_id}-{operation_id}"
    return result


def _progress_schemas(task: WorkflowTask) -> list[ToolSchema]:
    return sorted(
        (
            schema
            for schema in task.tool_schemas
            if schema.mutating and schema.policy_allowed and schema.operation_id is not None
        ),
        key=lambda schema: schema.name,
    )


def build_candidate_actions(
    task: WorkflowTask,
    state: Mapping[str, Any],
    *,
    include_invalid: bool = True,
) -> list[ToolCall]:
    """Build actions using only policy-visible schema constraints and state."""

    completed = set(state.get("completed_nodes", []))
    progress = _progress_schemas(task)
    ready = [
        schema
        for schema in progress
        if schema.operation_id not in completed
        and set(schema.required_completed_operations).issubset(completed)
    ]
    candidates = [
        ToolCall(
            tool_name=schema.name,
            arguments=_arguments(state, schema),
            call_id=(
                f"candidate-valid-{task.case_id}-{state.get('version', 0)}-{schema.operation_id}"
            ),
        )
        for schema in ready
    ]

    # These calls are executable but do not advance the hidden goal.  They are
    # intentionally *not* masked: learning relevance among legal actions is a
    # policy problem, whereas ActionMask is only an executability constraint.
    query_schema = next(
        schema
        for schema in task.tool_schemas
        if not schema.mutating and schema.policy_allowed and schema.name.endswith(".inspect_status")
    )
    candidates.append(
        ToolCall(
            tool_name=query_schema.name,
            arguments=_arguments(state, query_schema),
            call_id=f"candidate-query-{task.case_id}-{state.get('version', 0)}",
        )
    )
    contextual_queries = sorted(
        (
            schema
            for schema in task.tool_schemas
            if not schema.mutating
            and schema.policy_allowed
            and schema.operation_id is not None
            and schema.name != query_schema.name
        ),
        key=lambda schema: schema.name,
    )
    contextual_query = _choose(
        contextual_queries,
        task,
        state,
        namespace="context-query",
    )
    candidates.append(
        ToolCall(
            tool_name=contextual_query.name,
            arguments=_arguments(state, contextual_query),
            call_id=f"candidate-context-{task.case_id}-{state.get('version', 0)}",
        )
    )
    if completed:
        replay_operation = random.Random(
            _public_seed(task, state, namespace="replay-operation")
        ).choice(sorted(completed))
        replay_schema = next(
            schema for schema in progress if schema.operation_id == replay_operation
        )
        candidates.append(
            ToolCall(
                tool_name=replay_schema.name,
                arguments=_arguments(state, replay_schema),
                call_id=f"candidate-replay-{task.case_id}-{state.get('version', 0)}",
            )
        )
    if not include_invalid or not progress:
        return _permuted(task, state, candidates)

    anchor = _choose(
        ready if ready else progress,
        task,
        state,
        namespace="negative-anchor",
    )
    valid_arguments = _arguments(state, anchor)

    schema_arguments = dict(valid_arguments)
    schema_arguments.pop("operation_id", None)
    candidates.append(
        ToolCall(
            tool_name=anchor.name,
            arguments=schema_arguments,
            call_id=f"candidate-schema-{task.case_id}-{state.get('version', 0)}",
        )
    )

    grounding_arguments = dict(valid_arguments)
    grounding_arguments["entity_id"] = f"missing-{state.get('entity_id')}"
    candidates.append(
        ToolCall(
            tool_name=anchor.name,
            arguments=grounding_arguments,
            call_id=f"candidate-grounding-{task.case_id}-{state.get('version', 0)}",
        )
    )

    # Prefer a semantically plausible future operation whose *public* schema
    # prerequisites are not satisfied yet. This makes the unmasked policy
    # learn state/action compatibility instead of separating negatives through
    # an obvious stale-version shortcut. At the terminal frontier there may be
    # no future operation left, so a stale version is the deterministic fallback.
    blocked = [
        schema
        for schema in progress
        if schema.operation_id not in completed
        and not set(schema.required_completed_operations).issubset(completed)
        and ("approval_token" not in schema.required_arguments or bool(state.get("approval_token")))
    ]
    precondition_schema = (
        _choose(blocked, task, state, namespace="blocked-precondition") if blocked else anchor
    )
    precondition_arguments = _arguments(state, precondition_schema)
    if not blocked:
        precondition_arguments["expected_version"] = int(state.get("version", 0)) + 1
    candidates.append(
        ToolCall(
            tool_name=precondition_schema.name,
            arguments=precondition_arguments,
            call_id=f"candidate-precondition-{task.case_id}-{state.get('version', 0)}",
        )
    )

    unsafe_schema = next(schema for schema in task.tool_schemas if not schema.policy_allowed)
    candidates.append(
        ToolCall(
            tool_name=unsafe_schema.name,
            arguments=_arguments(state, unsafe_schema),
            call_id=f"candidate-safety-{task.case_id}-{state.get('version', 0)}",
        )
    )
    return _permuted(task, state, candidates)


__all__ = ["build_candidate_actions"]
