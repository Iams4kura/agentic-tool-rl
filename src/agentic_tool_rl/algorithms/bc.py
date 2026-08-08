"""Supervised behaviour cloning for structured candidate selection."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from agentic_tool_rl.algorithms.types import PolicyBatch
from agentic_tool_rl.models.actor_critic import ActorCritic


@dataclass(frozen=True)
class BCConfig:
    learning_rate: float = 3e-3
    epochs: int = 20
    batch_size: int = 64
    max_grad_norm: float = 1.0
    seed: int = 0


@dataclass(frozen=True)
class BCMetrics:
    loss: float
    accuracy: float
    entropy: float
    grad_norm: float
    updates: int


class BehaviorCloningTrainer:
    """Run genuine cross-entropy updates from oracle demonstrations."""

    def __init__(self, model: ActorCritic, config: BCConfig | None = None) -> None:
        self.model = model
        self.config = config or BCConfig()
        if min(
            self.config.learning_rate,
            self.config.epochs,
            self.config.batch_size,
            self.config.max_grad_norm,
        ) <= 0:
            raise ValueError("BC optimisation settings must be positive")
        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.config.learning_rate)

    def update(self, batch: PolicyBatch) -> BCMetrics:
        batch.validate()
        size = batch.states.shape[0]
        generator = torch.Generator(device="cpu").manual_seed(self.config.seed)
        final_loss = final_entropy = final_grad_norm = 0.0
        updates = 0
        self.model.train()
        for _ in range(self.config.epochs):
            permutation = torch.randperm(size, generator=generator)
            for start in range(0, size, self.config.batch_size):
                indices = permutation[start : start + self.config.batch_size].to(
                    batch.states.device
                )
                log_probs, entropy, _, _ = self.model.evaluate_actions(
                    batch.states[indices],
                    batch.action_features[indices],
                    batch.action_masks[indices],
                    batch.actions[indices],
                )
                loss = -log_probs.mean()
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()  # type: ignore[no-untyped-call]
                grad_norm = nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.max_grad_norm
                )
                self.optimizer.step()
                final_loss = float(loss.detach())
                final_entropy = float(entropy.detach().mean())
                final_grad_norm = float(grad_norm.detach())
                updates += 1

        self.model.eval()
        with torch.no_grad():
            _, _, _, logits = self.model.evaluate_actions(
                batch.states,
                batch.action_features,
                batch.action_masks,
                batch.actions,
            )
            accuracy = (logits.argmax(dim=-1) == batch.actions).float().mean()
        return BCMetrics(
            loss=final_loss,
            accuracy=float(accuracy),
            entropy=final_entropy,
            grad_norm=final_grad_norm,
            updates=updates,
        )
