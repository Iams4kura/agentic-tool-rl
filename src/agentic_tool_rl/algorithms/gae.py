"""Generalised advantage estimation with explicit trajectory boundaries."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor


def compute_gae(
    rewards: Tensor,
    values: Tensor,
    dones: Tensor,
    *,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    next_values: Tensor | None = None,
    trajectory_ids: Sequence[str] | None = None,
) -> tuple[Tensor, Tensor]:
    """Return ``(advantages, returns)`` for flattened, grouped trajectories.

    ``trajectory_ids`` makes episode boundaries explicit even when a truncated
    trajectory has ``done=False``.  When ``next_values`` is omitted, values are
    shifted only inside the same trajectory and terminal/truncated tails use
    zero bootstrap.  Supplying ``next_values`` allows a caller to bootstrap a
    truncated tail without ever borrowing the following episode's value.

    ``rewards`` must use a floating dtype, which also determines the output
    dtype. Integer storage would silently truncate values and advantages.
    """

    if not 0 <= gamma <= 1 or not 0 <= gae_lambda <= 1:
        raise ValueError("gamma and gae_lambda must lie in [0, 1]")
    if not rewards.is_floating_point():
        raise ValueError("rewards must use a floating dtype")
    rewards = rewards.flatten()
    values = values.flatten().to(device=rewards.device, dtype=rewards.dtype)
    dones = dones.flatten().to(device=rewards.device, dtype=torch.bool)
    size = rewards.numel()
    if size == 0 or values.numel() != size or dones.numel() != size:
        raise ValueError("rewards, values and dones must be non-empty and aligned")
    if trajectory_ids is None:
        ids: Sequence[str] = ["trajectory"] * size
    else:
        if len(trajectory_ids) != size:
            raise ValueError("trajectory_ids must align with rewards")
        ids = trajectory_ids

    if next_values is None:
        bootstraps = torch.zeros_like(values)
        for index in range(size - 1):
            if ids[index] == ids[index + 1] and not bool(dones[index]):
                bootstraps[index] = values[index + 1]
    else:
        bootstraps = next_values.flatten().to(device=values.device, dtype=values.dtype)
        if bootstraps.numel() != size:
            raise ValueError("next_values must align with rewards")

    advantages = torch.zeros_like(rewards)
    running = torch.zeros((), device=rewards.device, dtype=rewards.dtype)
    for index in range(size - 1, -1, -1):
        terminal = bool(dones[index])
        same_next_trajectory = index + 1 < size and ids[index] == ids[index + 1]
        continuation = 0.0 if terminal or not same_next_trajectory else 1.0
        bootstrap = 0.0 if terminal else bootstraps[index]
        delta = rewards[index] + gamma * bootstrap - values[index]
        running = delta + gamma * gae_lambda * continuation * running
        advantages[index] = running
    returns = advantages + values
    if not bool(torch.isfinite(advantages).all() and torch.isfinite(returns).all()):
        raise FloatingPointError("GAE produced non-finite values")
    return advantages, returns
