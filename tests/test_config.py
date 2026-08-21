from __future__ import annotations

from math import inf, nan

import pytest
from pydantic import ValidationError

from agentic_tool_rl.config import BehaviorCloningConfig, RewardConfig


@pytest.mark.parametrize(
    "value",
    [nan, inf, -inf],
    ids=["nan", "positive-infinity", "negative-infinity"],
)
def test_reward_config_rejects_non_finite_values(value: float) -> None:
    with pytest.raises(ValidationError) as exc_info:
        RewardConfig(
            task_success=value,
            task_failure=-1.0,
            invalid_action=-0.1,
            forbidden_side_effect=-1.0,
            step_cost=-0.01,
            progress_beta=1.0,
        )
    assert [error["type"] for error in exc_info.value.errors()] == ["finite_number"]


def test_positive_float_config_rejects_positive_infinity() -> None:
    with pytest.raises(ValidationError) as exc_info:
        BehaviorCloningConfig(epochs=1, batch_size=1, learning_rate=inf)
    assert [error["type"] for error in exc_info.value.errors()] == ["finite_number"]
