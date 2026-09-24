from __future__ import annotations

from math import inf, nan

import pytest
from pydantic import ValidationError

from agentic_tool_rl.config import (
    AblationConfig,
    AblationVariant,
    BehaviorCloningConfig,
    BenchmarkConfig,
    RewardConfig,
    load_packaged_config,
)


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


@pytest.mark.parametrize(
    ("field", "value", "error_type"),
    [
        ("epochs", True, "int_type"),
        ("epochs", "2", "int_type"),
        ("batch_size", False, "int_type"),
        ("batch_size", "4", "int_type"),
        ("learning_rate", True, "float_type"),
        ("learning_rate", "0.1", "float_type"),
    ],
)
def test_training_config_rejects_coerced_scalar_types(
    field: str, value: object, error_type: str
) -> None:
    values: dict[str, object] = {
        "epochs": 2,
        "batch_size": 4,
        "learning_rate": 0.1,
    }
    values[field] = value

    with pytest.raises(ValidationError) as exc_info:
        BehaviorCloningConfig.model_validate(values)

    assert [error["type"] for error in exc_info.value.errors()] == [error_type]


def test_seed_rejects_boolean_coercion() -> None:
    with pytest.raises(ValidationError) as exc_info:
        BenchmarkConfig(
            train_cases=1,
            dev_cases=1,
            test_cases=1,
            seed=True,
        )

    assert [error["type"] for error in exc_info.value.errors()] == ["int_type"]


def test_boolean_flags_reject_integer_coercion() -> None:
    with pytest.raises(ValidationError) as exc_info:
        AblationVariant(
            name="B-BC-Mask",
            algorithm="bc",
            progress_reward=0,
            action_mask=1,
        )

    assert [error["type"] for error in exc_info.value.errors()] == [
        "bool_type",
        "bool_type",
    ]


def test_yaml_sequence_normalization_keeps_elements_strict() -> None:
    config = load_packaged_config("ablation.yaml")
    raw = config.model_dump()
    raw["seeds"] = ["17", *config.seeds[1:]]

    with pytest.raises(ValidationError) as exc_info:
        AblationConfig.model_validate(raw)

    assert [error["type"] for error in exc_info.value.errors()] == ["int_type"]
