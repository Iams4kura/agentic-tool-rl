from __future__ import annotations

import pytest
import torch

from agentic_tool_rl.rewards import potential_shaping, shape_trajectory


@pytest.mark.parametrize(
    ("current", "following", "expected"),
    [
        (0, 0.5, 0.45),
        (torch.tensor(0), torch.tensor(0.5), 0.45),
        (0, 1, 0.9),
    ],
)
def test_integer_potential_inputs_preserve_fractional_shaping(
    current: torch.Tensor | int, following: torch.Tensor | float | int, expected: float
) -> None:
    result = potential_shaping(current, following, gamma=0.9)
    assert result.is_floating_point()
    assert result.item() == pytest.approx(expected)


def test_integer_base_rewards_preserve_fractional_shaping() -> None:
    base = torch.tensor([1, 1])
    potentials = torch.tensor([0.0, 0.5, 1.0])

    shaped, terms = shape_trajectory(base, potentials, gamma=0.9)

    assert shaped.is_floating_point()
    torch.testing.assert_close(terms, torch.tensor([0.45, 0.4]))
    torch.testing.assert_close(shaped, torch.tensor([1.45, 1.4]))
    torch.testing.assert_close(base, torch.tensor([1, 1]))


def test_existing_floating_reward_dtype_and_values_are_unchanged() -> None:
    base = torch.tensor([1.0, 1.0], dtype=torch.float32)
    potentials = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64)

    shaped, terms = shape_trajectory(base, potentials, gamma=0.9)

    assert shaped.dtype == terms.dtype == base.dtype
    expected_terms = potential_shaping(potentials[:-1], potentials[1:], gamma=0.9)
    torch.testing.assert_close(terms, expected_terms.to(dtype=base.dtype))
    torch.testing.assert_close(shaped, base + terms)
