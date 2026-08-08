"""Potential-based dense reward shaping for long-horizon tasks."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class RewardBreakdown:
    """Auditable components saved alongside every tool-call step."""

    task: float = 0.0
    progress: float = 0.0
    step: float = 0.0
    invalid: float = 0.0
    forbidden: float = 0.0

    @property
    def total(self) -> float:
        return self.task + self.progress + self.step + self.invalid + self.forbidden


def potential_shaping(
    phi: Tensor | float,
    phi_next: Tensor | float,
    *,
    gamma: float = 0.99,
    beta: float = 1.0,
) -> Tensor:
    """Return ``beta * (gamma * Phi(s_next) - Phi(s))``."""

    if not 0 <= gamma <= 1:
        raise ValueError("gamma must lie in [0, 1]")
    if beta < 0:
        raise ValueError("beta must be non-negative")
    current = torch.as_tensor(phi)
    following = torch.as_tensor(phi_next, device=current.device, dtype=current.dtype)
    if current.shape != following.shape:
        raise ValueError("phi and phi_next must have equal shapes")
    return beta * (gamma * following - current)


def shape_trajectory(
    base_rewards: Tensor,
    potentials: Tensor,
    *,
    gamma: float = 0.99,
    beta: float = 1.0,
) -> tuple[Tensor, Tensor]:
    """Apply shaping to a trajectory.

    ``potentials`` contains ``T + 1`` values for ``T`` base rewards.  Returns
    ``(shaped_rewards, shaping_terms)`` so raw and dense contributions remain
    independently auditable.
    """

    if base_rewards.ndim != 1 or potentials.ndim != 1:
        raise ValueError("base_rewards and potentials must be vectors")
    if potentials.numel() != base_rewards.numel() + 1:
        raise ValueError("potentials must contain one more item than base_rewards")
    terms = potential_shaping(
        potentials[:-1], potentials[1:], gamma=gamma, beta=beta
    ).to(device=base_rewards.device, dtype=base_rewards.dtype)
    return base_rewards + terms, terms


def discounted_shaping_sum(terms: Tensor, *, gamma: float) -> Tensor:
    """Return ``sum_t gamma**t * F_t`` for a one-dimensional trajectory."""

    if terms.ndim != 1:
        raise ValueError("terms must be a vector")
    if not 0 <= gamma <= 1:
        raise ValueError("gamma must lie in [0, 1]")
    powers = gamma ** torch.arange(
        terms.numel(), device=terms.device, dtype=terms.dtype
    )
    return (powers * terms).sum()
