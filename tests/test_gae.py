from __future__ import annotations

import pytest
import torch

from agentic_tool_rl.algorithms import compute_gae


@pytest.mark.parametrize("dtype", [torch.int64, torch.int32, torch.bool, torch.complex64])
def test_gae_rejects_non_floating_rewards(dtype: torch.dtype) -> None:
    # Integer rewards used to truncate both fractional values and advantages.
    with pytest.raises(ValueError, match="rewards must use a floating dtype"):
        compute_gae(
            torch.tensor([1, 1], dtype=dtype),
            torch.tensor([0.25, 0.5]),
            torch.tensor([False, True]),
            gamma=0.9,
            gae_lambda=0.95,
        )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_gae_preserves_fractional_advantages_and_reward_dtype(dtype: torch.dtype) -> None:
    advantages, returns = compute_gae(
        torch.tensor([1, 1], dtype=dtype),
        torch.tensor([0.25, 0.5]),
        torch.tensor([False, True]),
        gamma=0.9,
        gae_lambda=0.95,
    )
    # A1 = 1 - .5; A0 = 1 + .9 * .5 - .25 + .9 * .95 * A1.
    torch.testing.assert_close(advantages, torch.tensor([1.6275, 0.5], dtype=dtype))
    torch.testing.assert_close(returns, torch.tensor([1.8775, 1.0], dtype=dtype))
