"""Supervised behaviour cloning for structured candidate selection."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from agentic_tool_rl.algorithms.types import ExpertBatch, PolicyBatch
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
    # v1.4 candidate-set metrics.  ``loss`` and ``accuracy`` retain their
    # frozen v1.3 meanings; legacy batches therefore leave these fields unset.
    set_nll: float | None = None
    positive_probability_mass: float | None = None
    top1_in_positive_set_accuracy: float | None = None
    mean_positive_count: float | None = None


def multi_positive_bc_loss(logits: Tensor, positive_mask: Tensor) -> Tensor:
    """Return ``-log(sum(pi(a) for a in positives))`` over a batch.

    ``logits`` must already carry the candidate mask (masked candidates use
    ``-inf``).  ``ExpertBatch.validate`` additionally proves that every
    positive is an exposed candidate before this loss is evaluated.
    """

    if logits.ndim != 2:
        raise ValueError("logits must have shape [N, A]")
    if positive_mask.ndim != 2 or positive_mask.shape != logits.shape:
        raise ValueError("positive_mask must match logits shape [N, A]")
    if positive_mask.dtype != torch.bool:
        raise ValueError("positive_mask must use bool dtype")
    if positive_mask.device != logits.device:
        raise ValueError("positive_mask and logits must share a device")
    if logits.shape[0] == 0 or logits.shape[1] == 0:
        raise ValueError("multi-positive BC requires a non-empty candidate batch")
    if not logits.is_floating_point():
        raise ValueError("logits must use a floating dtype")
    if not bool(positive_mask.any(dim=1).all()):
        raise ValueError("every state must expose at least one positive candidate")
    if not bool(torch.isfinite(logits[positive_mask]).all()):
        raise ValueError("positive candidates must have finite logits")
    if bool(torch.isnan(logits).any()) or bool(torch.isposinf(logits).any()):
        raise ValueError("logits must not contain NaN or positive infinity")

    log_partition = torch.logsumexp(logits, dim=1)
    positive_logits = logits.masked_fill(~positive_mask, -torch.inf)
    positive_log_partition = torch.logsumexp(positive_logits, dim=1)
    return (log_partition - positive_log_partition).mean()


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

    def update(self, batch: PolicyBatch | ExpertBatch) -> BCMetrics:
        if isinstance(batch, ExpertBatch):
            return self._update_expert(batch)

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
                if not bool(torch.isfinite(grad_norm)):
                    raise FloatingPointError("BC gradient norm is non-finite")
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

    def _update_expert(self, batch: ExpertBatch) -> BCMetrics:
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
                positive_mask = batch.positive_mask[indices]
                representative_actions = positive_mask.to(torch.long).argmax(dim=1)
                _, entropy, _, logits = self.model.evaluate_actions(
                    batch.states[indices],
                    batch.action_features[indices],
                    batch.candidate_mask[indices],
                    representative_actions,
                )
                loss = multi_positive_bc_loss(logits, positive_mask)
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()  # type: ignore[no-untyped-call]
                grad_norm = nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.max_grad_norm
                )
                if not bool(torch.isfinite(grad_norm)):
                    raise FloatingPointError("BC gradient norm is non-finite")
                self.optimizer.step()
                final_loss = float(loss.detach())
                final_entropy = float(entropy.detach().mean())
                final_grad_norm = float(grad_norm.detach())
                updates += 1

        self.model.eval()
        with torch.no_grad():
            representative_actions = batch.positive_mask.to(torch.long).argmax(dim=1)
            _, _, _, logits = self.model.evaluate_actions(
                batch.states,
                batch.action_features,
                batch.candidate_mask,
                representative_actions,
            )
            predictions = logits.argmax(dim=1, keepdim=True)
            accuracy = batch.positive_mask.gather(1, predictions).float().mean()
            set_nll = multi_positive_bc_loss(logits, batch.positive_mask)
            probabilities = torch.softmax(logits, dim=1)
            positive_probability_mass = (
                probabilities.masked_fill(~batch.positive_mask, 0.0).sum(dim=1).mean()
            )
            mean_positive_count = batch.positive_mask.sum(dim=1).float().mean()
        return BCMetrics(
            loss=final_loss,
            accuracy=float(accuracy),
            entropy=final_entropy,
            grad_norm=final_grad_norm,
            updates=updates,
            set_nll=float(set_nll),
            positive_probability_mass=float(positive_probability_mass),
            top1_in_positive_set_accuracy=float(accuracy),
            mean_positive_count=float(mean_positive_count),
        )
