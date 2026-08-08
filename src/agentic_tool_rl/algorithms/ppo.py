"""Clipped action-level PPO for dynamic structured action spaces."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from agentic_tool_rl.algorithms.types import PPOBatch
from agentic_tool_rl.models.actor_critic import ActorCritic


@dataclass(frozen=True)
class PPOConfig:
    learning_rate: float = 3e-4
    clip_coefficient: float = 0.2
    value_clip_coefficient: float = 0.2
    value_loss_coefficient: float = 0.5
    entropy_coefficient: float = 0.01
    epochs: int = 4
    batch_size: int = 64
    max_grad_norm: float = 0.5
    normalize_advantages: bool = True
    target_kl: float | None = None
    seed: int = 0


@dataclass(frozen=True)
class PPOMetrics:
    policy_loss: float
    value_loss: float
    entropy: float
    approximate_kl: float
    clip_fraction: float
    grad_norm: float
    total_loss: float
    updates: int
    stopped_early: bool


class PPOTrainer:
    """Optimise a policy from action-level rollout records."""

    def __init__(self, model: ActorCritic, config: PPOConfig | None = None) -> None:
        self.model = model
        self.config = config or PPOConfig()
        if min(
            self.config.learning_rate,
            self.config.clip_coefficient,
            self.config.value_clip_coefficient,
            self.config.value_loss_coefficient,
            self.config.epochs,
            self.config.batch_size,
            self.config.max_grad_norm,
        ) <= 0:
            raise ValueError("PPO optimisation settings must be positive")
        if self.config.entropy_coefficient < 0:
            raise ValueError("entropy_coefficient must be non-negative")
        if self.config.target_kl is not None and self.config.target_kl <= 0:
            raise ValueError("target_kl must be positive when provided")
        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.config.learning_rate)

    @staticmethod
    def _mean(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    def update(self, batch: PPOBatch) -> PPOMetrics:
        batch.validate()
        advantages = batch.advantages.detach()
        if self.config.normalize_advantages and advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (
                advantages.std(unbiased=False) + 1e-8
            )
        size = batch.states.shape[0]
        generator = torch.Generator(device="cpu").manual_seed(self.config.seed)
        logged: dict[str, list[float]] = {
            name: []
            for name in (
                "policy",
                "value",
                "entropy",
                "kl",
                "clip_fraction",
                "grad_norm",
                "total",
            )
        }
        stopped_early = False
        self.model.train()
        for _ in range(self.config.epochs):
            permutation = torch.randperm(size, generator=generator)
            for start in range(0, size, self.config.batch_size):
                indices = permutation[start : start + self.config.batch_size].to(
                    batch.states.device
                )
                new_log_probs, entropy, new_values, _ = self.model.evaluate_actions(
                    batch.states[indices],
                    batch.action_features[indices],
                    batch.action_masks[indices],
                    batch.actions[indices],
                )
                old_log_probs = batch.old_log_probs[indices].detach()
                log_ratio = new_log_probs - old_log_probs
                ratio = log_ratio.exp()
                selected_advantages = advantages[indices]
                unclipped = ratio * selected_advantages
                clipped = ratio.clamp(
                    1 - self.config.clip_coefficient,
                    1 + self.config.clip_coefficient,
                ) * selected_advantages
                policy_loss = -torch.minimum(unclipped, clipped).mean()

                old_values = batch.old_values[indices].detach()
                returns = batch.returns[indices].detach()
                value_delta = new_values - old_values
                clipped_values = old_values + value_delta.clamp(
                    -self.config.value_clip_coefficient,
                    self.config.value_clip_coefficient,
                )
                value_loss = 0.5 * torch.maximum(
                    (new_values - returns).pow(2),
                    (clipped_values - returns).pow(2),
                ).mean()
                entropy_mean = entropy.mean()
                total_loss = (
                    policy_loss
                    + self.config.value_loss_coefficient * value_loss
                    - self.config.entropy_coefficient * entropy_mean
                )

                self.optimizer.zero_grad(set_to_none=True)
                total_loss.backward()  # type: ignore[no-untyped-call]
                grad_norm: Tensor = nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.max_grad_norm
                )
                if not bool(torch.isfinite(grad_norm)):
                    raise FloatingPointError("PPO gradient norm is non-finite")
                self.optimizer.step()

                with torch.no_grad():
                    approximate_kl = ((ratio - 1) - log_ratio).mean()
                    clip_fraction = (
                        (ratio - 1).abs() > self.config.clip_coefficient
                    ).float().mean()
                logged["policy"].append(float(policy_loss.detach()))
                logged["value"].append(float(value_loss.detach()))
                logged["entropy"].append(float(entropy_mean.detach()))
                logged["kl"].append(float(approximate_kl.detach()))
                logged["clip_fraction"].append(float(clip_fraction.detach()))
                logged["grad_norm"].append(float(grad_norm.detach()))
                logged["total"].append(float(total_loss.detach()))

            if (
                self.config.target_kl is not None
                and logged["kl"]
                and logged["kl"][-1] > self.config.target_kl
            ):
                stopped_early = True
                break

        if not logged["total"]:
            raise RuntimeError("PPO did not execute an update")
        return PPOMetrics(
            policy_loss=self._mean(logged["policy"]),
            value_loss=self._mean(logged["value"]),
            entropy=self._mean(logged["entropy"]),
            approximate_kl=self._mean(logged["kl"]),
            clip_fraction=self._mean(logged["clip_fraction"]),
            grad_norm=self._mean(logged["grad_norm"]),
            total_loss=self._mean(logged["total"]),
            updates=len(logged["total"]),
            stopped_early=stopped_early,
        )
