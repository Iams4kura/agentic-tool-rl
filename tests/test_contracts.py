from __future__ import annotations

from math import inf, nan

import pytest
from pydantic import ValidationError

from agentic_tool_rl.contracts import (
    Observation,
    Split,
    StepOutcome,
    WorkflowNode,
    WorkflowTask,
)


def _observation() -> Observation:
    return Observation(
        task_id="task",
        step_index=0,
        max_steps=1,
        remaining_steps=1,
        user_goal="complete the workflow",
        visible_state={},
        available_entities=[],
        available_tools=[],
    )


def _step_outcome(*, reward: float = 0.0) -> StepOutcome:
    return StepOutcome(
        observation=_observation(),
        reward=reward,
        done=False,
        success=False,
        accepted=False,
        simulated_latency_s=0.0,
    )


def _workflow_task(*, latency: float) -> WorkflowTask:
    return WorkflowTask(
        case_id="case",
        split=Split.TRAIN,
        family="family",
        topology="diamond_tail",
        difficulty="short",
        seed=1,
        entity_id="entity",
        user_goal="complete the workflow",
        initial_state={},
        visible_observation={},
        tool_schemas=[],
        workflow_nodes=[
            WorkflowNode(node_id="node", tool_name="tool", simulated_latency_s=0.0)
        ],
        hidden_goal_predicates=[],
        forbidden_side_effects=[],
        oracle_plans=[],
        optimal_steps=1,
        max_steps=1,
        latency_trace={"node": latency},
        generator_version="benchmark-v1.3",
    )


@pytest.mark.parametrize(
    "value",
    [nan, inf, -inf],
    ids=["nan", "positive-infinity", "negative-infinity"],
)
def test_contract_rejects_non_finite_scalar(value: float) -> None:
    with pytest.raises(ValidationError) as exc_info:
        _step_outcome(reward=value)
    assert [error["type"] for error in exc_info.value.errors()] == ["finite_number"]


def test_contract_rejects_positive_infinity_for_bounded_float() -> None:
    with pytest.raises(ValidationError) as exc_info:
        WorkflowNode(node_id="node", tool_name="tool", simulated_latency_s=inf)
    assert [error["type"] for error in exc_info.value.errors()] == ["finite_number"]


@pytest.mark.parametrize(
    "value",
    [nan, inf, -inf],
    ids=["nan", "positive-infinity", "negative-infinity"],
)
def test_contract_rejects_non_finite_typed_mapping_value(value: float) -> None:
    with pytest.raises(ValidationError) as exc_info:
        _workflow_task(latency=value)
    assert [error["type"] for error in exc_info.value.errors()] == ["finite_number"]


@pytest.mark.parametrize(
    "value",
    [nan, inf, -inf],
    ids=["nan", "positive-infinity", "negative-infinity"],
)
def test_contract_rejects_non_finite_assignment(value: float) -> None:
    outcome = _step_outcome()
    with pytest.raises(ValidationError) as exc_info:
        outcome.reward = value
    assert [error["type"] for error in exc_info.value.errors()] == ["finite_number"]
    assert outcome.reward == 0.0
