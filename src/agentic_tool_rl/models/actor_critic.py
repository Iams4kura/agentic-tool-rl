"""A small actor-critic that scores a different candidate set at every step.

The policy deliberately does not assign fixed output neurons to tool names.  It
encodes the current state and each structured candidate independently, then
scores their pair.  This keeps the CPU implementation faithful to tool-calling
where available actions change with the page or workflow state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch
from torch import Tensor, nn
from torch.distributions import Categorical


@dataclass(frozen=True)
class ActionSample:
    """One batched policy decision."""

    action_index: Tensor
    log_prob: Tensor
    value: Tensor
    entropy: Tensor
    logits: Tensor


class ActorCritic(nn.Module):
    """Score structured action candidates and estimate the current state value.

    Args:
        state_dim: Width of the already featurised state/goal vector.
        action_dim: Width of each candidate tool-call feature vector.
        hidden_dim: Shared latent width.

    Input shapes are ``state_features=[B, S]``,
    ``action_features=[B, A, D]`` and ``action_mask=[B, A]``.  Candidate count
    ``A`` may differ between calls; padding is allowed when its mask is false.
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        if min(state_dim, action_dim, hidden_dim) <= 0:
            raise ValueError("state_dim, action_dim and hidden_dim must be positive")
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
        )
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
        )
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=2**0.5)
                nn.init.zeros_(module.bias)
        final_action = self.action_head[-1]
        final_value = self.value_head[-1]
        assert isinstance(final_action, nn.Linear)
        assert isinstance(final_value, nn.Linear)
        nn.init.orthogonal_(final_action.weight, gain=0.01)
        nn.init.orthogonal_(final_value.weight, gain=1.0)

    def _validate_inputs(
        self,
        state_features: Tensor,
        action_features: Tensor,
        action_mask: Tensor,
    ) -> Tensor:
        if state_features.ndim != 2 or state_features.shape[-1] != self.state_dim:
            raise ValueError(f"state_features must have shape [B, {self.state_dim}]")
        if action_features.ndim != 3 or action_features.shape[-1] != self.action_dim:
            raise ValueError(f"action_features must have shape [B, A, {self.action_dim}]")
        if action_mask.ndim != 2:
            raise ValueError("action_mask must have shape [B, A]")
        if state_features.shape[0] != action_features.shape[0]:
            raise ValueError("state and action batch sizes differ")
        if action_features.shape[:2] != action_mask.shape:
            raise ValueError("candidate and action-mask shapes differ")
        mask = action_mask.to(device=action_features.device, dtype=torch.bool)
        if not bool(mask.any(dim=-1).all()):
            raise ValueError("every state must expose at least one valid action")
        return mask

    def forward(
        self,
        state_features: Tensor,
        action_features: Tensor,
        action_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Return masked candidate logits and state values."""

        mask = self._validate_inputs(state_features, action_features, action_mask)
        state_latent = self.state_encoder(state_features)
        action_latent = self.action_encoder(action_features)
        expanded_state = state_latent.unsqueeze(1).expand_as(action_latent)
        joint = torch.cat(
            (expanded_state, action_latent, expanded_state * action_latent), dim=-1
        )
        logits = self.action_head(joint).squeeze(-1)
        logits = logits.masked_fill(~mask, -torch.inf)
        values = self.value_head(state_latent).squeeze(-1)
        return logits, values

    def distribution(
        self,
        state_features: Tensor,
        action_features: Tensor,
        action_mask: Tensor,
    ) -> tuple[Categorical, Tensor, Tensor]:
        logits, values = self(state_features, action_features, action_mask)
        return Categorical(logits=logits), values, logits

    @torch.no_grad()
    def act(
        self,
        state_features: Tensor,
        action_features: Tensor,
        action_mask: Tensor,
        *,
        deterministic: bool = False,
    ) -> ActionSample:
        """Sample (or greedily select) an action without retaining a graph."""

        distribution, values, logits = self.distribution(
            state_features, action_features, action_mask
        )
        actions = (
            logits.argmax(dim=-1)
            if deterministic
            else cast(Tensor, distribution.sample())  # type: ignore[no-untyped-call]
        )
        return ActionSample(
            action_index=actions,
            log_prob=cast(
                Tensor, distribution.log_prob(actions)  # type: ignore[no-untyped-call]
            ),
            value=values,
            entropy=cast(
                Tensor, distribution.entropy()  # type: ignore[no-untyped-call]
            ),
            logits=logits,
        )

    def evaluate_actions(
        self,
        state_features: Tensor,
        action_features: Tensor,
        action_mask: Tensor,
        actions: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Evaluate recorded actions for BC or PPO.

        Returns ``(log_prob, entropy, value, logits)``.
        """

        distribution, values, logits = self.distribution(
            state_features, action_features, action_mask
        )
        actions = actions.to(device=logits.device, dtype=torch.long)
        if actions.ndim != 1 or actions.shape[0] != logits.shape[0]:
            raise ValueError("actions must have shape [B]")
        selected_is_valid = action_mask.to(device=logits.device, dtype=torch.bool).gather(
            1, actions.unsqueeze(1)
        )
        if not bool(selected_is_valid.all()):
            raise ValueError("recorded action is masked out")
        log_prob = cast(
            Tensor, distribution.log_prob(actions)  # type: ignore[no-untyped-call]
        )
        entropy = cast(Tensor, distribution.entropy())  # type: ignore[no-untyped-call]
        return log_prob, entropy, values, logits
