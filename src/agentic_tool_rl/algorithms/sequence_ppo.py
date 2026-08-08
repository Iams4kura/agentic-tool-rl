"""Clipped sequence-level PPO for complete structured-action episodes."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from agentic_tool_rl.algorithms.ppo import PPOConfig, PPOMetrics
from agentic_tool_rl.algorithms.types import SequencePPOBatch
from agentic_tool_rl.models.actor_critic import ActorCritic


@dataclass(frozen=True)
class _EpisodeEvaluation:
    log_probs: Tensor
    entropy: Tensor
    initial_values: Tensor


class SequencePPOTrainer:
    """Optimise one clipped PPO objective per complete episode.

    All action log-probabilities in an episode are summed before a single PPO
    ratio is calculated.  The critic predicts the discounted episode return
    only from the initial state.  This deliberately does not run GAE and never
    turns one terminal reward into repeated action-level labels.
    """

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

    def _evaluate_episodes(
        self,
        batch: SequencePPOBatch,
        selected_episodes: Tensor,
    ) -> _EpisodeEvaluation:
        """Evaluate every step, then reduce exactly once at episode scope."""

        selected_steps = (
            batch.episode_indices.unsqueeze(1) == selected_episodes.unsqueeze(0)
        ).any(dim=1)
        step_indices = torch.nonzero(selected_steps, as_tuple=False).flatten()
        step_log_probs, step_entropy, step_values, _ = self.model.evaluate_actions(
            batch.states[step_indices],
            batch.action_features[step_indices],
            batch.action_masks[step_indices],
            batch.actions[step_indices],
        )
        step_episode_indices = batch.episode_indices[step_indices]

        sequence_log_probs = step_log_probs.new_zeros(batch.num_episodes).index_add(
            0, step_episode_indices, step_log_probs
        )
        entropy_sums = step_entropy.new_zeros(batch.num_episodes).index_add(
            0, step_episode_indices, step_entropy
        )
        episode_counts = step_entropy.new_zeros(batch.num_episodes).index_add(
            0, step_episode_indices, torch.ones_like(step_entropy)
        )

        initial_global_indices = batch.initial_state_indices[selected_episodes]
        initial_local_indices = torch.searchsorted(step_indices, initial_global_indices)
        if not torch.equal(step_indices[initial_local_indices], initial_global_indices):
            raise RuntimeError("selected episode batch lost an initial state")

        return _EpisodeEvaluation(
            log_probs=sequence_log_probs[selected_episodes],
            # Mean action entropy inside each episode prevents long episodes
            # from receiving a larger regularisation coefficient by accident.
            entropy=(entropy_sums / episode_counts.clamp_min(1.0))[selected_episodes],
            initial_values=step_values[initial_local_indices],
        )

    def update(self, batch: SequencePPOBatch) -> PPOMetrics:
        batch.validate()
        advantages = batch.advantages.detach()
        if self.config.normalize_advantages and advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (
                advantages.std(unbiased=False) + 1e-8
            )

        old_sequence_log_probs = batch.old_sequence_log_probs.detach()
        old_initial_values = batch.old_initial_values.detach()
        returns = batch.discounted_returns.detach()
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
            permutation = torch.randperm(batch.num_episodes, generator=generator)
            for start in range(0, batch.num_episodes, self.config.batch_size):
                episode_indices = permutation[
                    start : start + self.config.batch_size
                ].to(batch.states.device)
                evaluated = self._evaluate_episodes(batch, episode_indices)

                old_log_probs = old_sequence_log_probs[episode_indices]
                log_ratio = evaluated.log_probs - old_log_probs
                ratio = log_ratio.exp()
                selected_advantages = advantages[episode_indices]
                unclipped = ratio * selected_advantages
                clipped = ratio.clamp(
                    1 - self.config.clip_coefficient,
                    1 + self.config.clip_coefficient,
                ) * selected_advantages
                policy_loss = -torch.minimum(unclipped, clipped).mean()

                selected_old_values = old_initial_values[episode_indices]
                selected_returns = returns[episode_indices]
                value_delta = evaluated.initial_values - selected_old_values
                clipped_values = selected_old_values + value_delta.clamp(
                    -self.config.value_clip_coefficient,
                    self.config.value_clip_coefficient,
                )
                value_loss = 0.5 * torch.maximum(
                    (evaluated.initial_values - selected_returns).pow(2),
                    (clipped_values - selected_returns).pow(2),
                ).mean()
                entropy_mean = evaluated.entropy.mean()
                total_loss = (
                    policy_loss
                    + self.config.value_loss_coefficient * value_loss
                    - self.config.entropy_coefficient * entropy_mean
                )
                if not bool(torch.isfinite(total_loss)):
                    raise FloatingPointError("sequence PPO loss is non-finite")

                self.optimizer.zero_grad(set_to_none=True)
                total_loss.backward()  # type: ignore[no-untyped-call]
                grad_norm: Tensor = nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.max_grad_norm
                )
                if not bool(torch.isfinite(grad_norm)):
                    raise FloatingPointError("sequence PPO gradient norm is non-finite")
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
            raise RuntimeError("sequence PPO did not execute an update")
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
