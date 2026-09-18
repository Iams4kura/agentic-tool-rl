"""Transactional state-machine environment for structured tool calls."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from agentic_tool_rl.contracts import (
    InvalidActionKind,
    Observation,
    StatePredicate,
    StepOutcome,
    TaskEvaluation,
    ToolCall,
    ValidationResult,
    WorkflowTask,
)
from agentic_tool_rl.grounding.candidates import build_candidate_actions
from agentic_tool_rl.tools.registry import ToolRegistry

_INTERNAL_STATE_KEYS = {"processed_call_ids", "idempotency_results", "audit_log"}
_MISSING = object()


def _read_path(state: Mapping[str, Any], path: str) -> Any:
    value: Any = state
    for component in path.split("."):
        if not isinstance(value, Mapping) or component not in value:
            return _MISSING
        value = value[component]
    return value


def _json_equal(actual: Any, expected: Any) -> bool:
    """Compare JSON values without Python's bool/int equivalence."""

    if isinstance(actual, bool) or isinstance(expected, bool):
        return isinstance(actual, bool) and isinstance(expected, bool) and actual == expected
    if isinstance(actual, Mapping) and isinstance(expected, Mapping):
        return actual.keys() == expected.keys() and all(
            _json_equal(value, expected[key]) for key, value in actual.items()
        )
    if isinstance(actual, (list, tuple)) and type(actual) is type(expected):
        return len(actual) == len(expected) and all(
            _json_equal(left, right) for left, right in zip(actual, expected, strict=True)
        )
    return bool(actual == expected)


def predicate_holds(state: Mapping[str, Any], predicate: StatePredicate) -> bool:
    actual = _read_path(state, predicate.path)
    if actual is _MISSING:
        return False
    expected = predicate.value
    if predicate.operator == "eq":
        return _json_equal(actual, expected)
    if predicate.operator == "ne":
        return not _json_equal(actual, expected)
    if predicate.operator == "contains":
        if isinstance(actual, (list, tuple, set)):
            return any(_json_equal(item, expected) for item in actual)
        return isinstance(actual, (str, dict)) and expected in actual
    if predicate.operator == "contains_all":
        return (
            isinstance(actual, (list, tuple, set))
            and isinstance(expected, list)
            and all(any(_json_equal(item, required) for item in actual) for required in expected)
        )
    if predicate.operator == "gte":
        return (
            isinstance(actual, (int, float))
            and not isinstance(actual, bool)
            and isinstance(expected, (int, float))
            and not isinstance(expected, bool)
            and (actual >= expected)
        )
    if predicate.operator == "lte":
        return (
            isinstance(actual, (int, float))
            and not isinstance(actual, bool)
            and isinstance(expected, (int, float))
            and not isinstance(expected, bool)
            and (actual <= expected)
        )
    return False


class TransactionalWorkflowEnv:
    """A task-scoped environment with a strict business-state transaction boundary."""

    def __init__(self, task: WorkflowTask) -> None:
        self.task = task
        self.registry = ToolRegistry(task)
        self._state: dict[str, Any] = {}
        self.steps_taken = 0
        self.simulated_latency_s = 0.0
        self._done = False
        self._last_message = ""
        self.reset()

    @property
    def done(self) -> bool:
        return self._done

    @property
    def state(self) -> dict[str, Any]:
        """Return a copy so policy code cannot bypass the tool transaction boundary."""

        return deepcopy(self._state)

    def state_snapshot(self) -> dict[str, Any]:
        return deepcopy(self._state)

    def reset(self) -> Observation:
        self._state = deepcopy(self.task.initial_state)
        self.steps_taken = 0
        self.simulated_latency_s = 0.0
        self._done = False
        self._last_message = "environment reset"
        return self.observe()

    def _visible_state(self) -> dict[str, Any]:
        return {
            key: deepcopy(value)
            for key, value in self._state.items()
            if key not in _INTERNAL_STATE_KEYS
        }

    def observe(self) -> Observation:
        return Observation(
            task_id=self.task.case_id,
            step_index=self.steps_taken,
            max_steps=self.task.max_steps,
            remaining_steps=max(0, self.task.max_steps - self.steps_taken),
            user_goal=self.task.user_goal,
            visible_state=self._visible_state(),
            available_entities=[self.task.entity_id],
            available_tools=[schema.name for schema in self.task.tool_schemas],
            message=self._last_message,
            done=self._done,
        )

    def candidate_actions(self, *, include_invalid: bool = True) -> list[ToolCall]:
        return build_candidate_actions(self.task, self._state, include_invalid=include_invalid)

    def dry_run(self, call: ToolCall) -> ValidationResult:
        """Independent ground-truth label; never delegates to the ActionMask."""

        return self.registry.validate(self._state, call)

    def evaluate(self) -> TaskEvaluation:
        goals_satisfied = all(
            predicate_holds(self._state, predicate)
            for predicate in self.task.hidden_goal_predicates
        )
        side_effects = list(self._state.get("side_effects", []))
        forbidden_count = sum(
            side_effect in self.task.forbidden_side_effects for side_effect in side_effects
        )
        return TaskEvaluation(
            case_id=self.task.case_id,
            success=goals_satisfied and forbidden_count == 0,
            goal_predicates_satisfied=goals_satisfied,
            forbidden_side_effect_count=forbidden_count,
            completed_steps=len(self._state.get("completed_nodes", [])),
            optimal_steps=self.task.optimal_steps,
            max_steps=self.task.max_steps,
        )

    def step(self, call: ToolCall) -> StepOutcome:
        if self._done:
            raise RuntimeError("cannot step a terminated environment; call reset() first")

        state_before = deepcopy(self._state)
        execution = self.registry.execute(self._state, call)
        self.steps_taken += 1
        self.simulated_latency_s += execution.simulated_latency_s
        if execution.validation.valid:
            self._state = execution.state
        elif execution.state != state_before:
            raise RuntimeError("rejected tool call violated the transaction boundary")

        evaluation = self.evaluate()
        timed_out = self.steps_taken >= self.task.max_steps and not evaluation.success
        self._done = evaluation.success or timed_out
        if evaluation.success:
            reward = 1.0
        elif timed_out:
            reward = -1.0
        elif execution.validation.valid:
            reward = -0.01
        else:
            reward = -0.15
        self._last_message = execution.validation.reason or (
            "tool call accepted" if execution.validation.valid else "tool call rejected"
        )
        invalid_kind: InvalidActionKind | None = execution.validation.invalid_kind
        return StepOutcome(
            observation=self.observe(),
            reward=reward,
            done=self._done,
            success=evaluation.success,
            accepted=execution.validation.valid,
            invalid_kind=invalid_kind,
            reason=self._last_message,
            simulated_latency_s=execution.simulated_latency_s,
            info={
                "case_id": self.task.case_id,
                "node_id": execution.validation.node_id,
                "idempotent_replay": execution.validation.idempotent_replay,
                "timed_out": timed_out,
                "state_version": self._state.get("version", 0),
                "total_simulated_latency_s": round(self.simulated_latency_s, 6),
            },
        )


__all__ = ["TransactionalWorkflowEnv", "predicate_holds"]
