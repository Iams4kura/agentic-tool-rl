"""Tensor batches shared by behaviour cloning and PPO."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class PolicyBatch:
    states: Tensor
    action_features: Tensor
    action_masks: Tensor
    actions: Tensor

    def validate(self) -> None:
        if self.states.ndim != 2:
            raise ValueError("states must have shape [N, S]")
        if self.action_features.ndim != 3:
            raise ValueError("action_features must have shape [N, A, D]")
        if self.action_masks.ndim != 2:
            raise ValueError("action_masks must have shape [N, A]")
        if self.actions.ndim != 1:
            raise ValueError("actions must have shape [N]")
        size = self.states.shape[0]
        if any(
            tensor.shape[0] != size
            for tensor in (self.action_features, self.action_masks, self.actions)
        ):
            raise ValueError("policy batch tensors are not aligned")
        if self.action_features.shape[:2] != self.action_masks.shape:
            raise ValueError("candidate and mask shapes differ")


@dataclass(frozen=True)
class ExpertBatch:
    """Candidate-set supervision with one or more valid expert actions per state."""

    states: Tensor
    action_features: Tensor
    candidate_mask: Tensor
    positive_mask: Tensor
    visible_state_sha256s: tuple[str, ...] = ()
    source_case_ids: tuple[str, ...] = ()
    source_path_counts: tuple[tuple[str, int], ...] = ()

    def validate(self) -> None:
        if self.states.ndim != 2:
            raise ValueError("states must have shape [N, S]")
        if self.action_features.ndim != 3:
            raise ValueError("action_features must have shape [N, A, D]")
        if self.candidate_mask.ndim != 2:
            raise ValueError("candidate_mask must have shape [N, A]")
        if self.positive_mask.ndim != 2:
            raise ValueError("positive_mask must have shape [N, A]")
        size = self.states.shape[0]
        if size == 0:
            raise ValueError("expert batch must contain at least one state")
        if any(
            tensor.shape[0] != size
            for tensor in (
                self.action_features,
                self.candidate_mask,
                self.positive_mask,
            )
        ):
            raise ValueError("expert batch tensors are not aligned")
        if self.action_features.shape[:2] != self.candidate_mask.shape:
            raise ValueError("candidate features and candidate_mask shapes differ")
        if self.candidate_mask.shape != self.positive_mask.shape:
            raise ValueError("candidate_mask and positive_mask shapes differ")
        if self.action_features.shape[1] == 0:
            raise ValueError("every state must expose at least one candidate slot")

        device = self.states.device
        if not self.states.is_floating_point():
            raise ValueError("states must use a floating dtype")
        if (
            not self.action_features.is_floating_point()
            or self.action_features.dtype != self.states.dtype
        ):
            raise ValueError("action_features must use the states floating dtype")
        for name, mask in (
            ("candidate_mask", self.candidate_mask),
            ("positive_mask", self.positive_mask),
        ):
            if mask.dtype != torch.bool:
                raise ValueError(f"{name} must use bool dtype")
            if mask.device != device:
                raise ValueError(f"{name} must share the policy tensor device")
        if self.action_features.device != device:
            raise ValueError("action_features must share the states device")
        if not bool(torch.isfinite(self.states).all()) or not bool(
            torch.isfinite(self.action_features).all()
        ):
            raise ValueError("expert features must be finite")
        if not bool(self.candidate_mask.any(dim=1).all()):
            raise ValueError("every state must expose at least one candidate")
        if not bool(self.positive_mask.any(dim=1).all()):
            raise ValueError("every state must expose at least one positive candidate")
        if bool((self.positive_mask & ~self.candidate_mask).any()):
            raise ValueError("positive_mask must be a subset of candidate_mask")
        if self.visible_state_sha256s:
            if len(self.visible_state_sha256s) != size:
                raise ValueError("visible_state_sha256s must align with expert states")
            if any(
                len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
                for value in self.visible_state_sha256s
            ):
                raise ValueError("visible_state_sha256s must contain lowercase SHA-256 values")
            if len(set(self.visible_state_sha256s)) != size:
                raise ValueError("expert states must be unique by visible-state SHA-256")
        if self.source_case_ids and (
            len(self.source_case_ids) != size
            or any(not case_id for case_id in self.source_case_ids)
        ):
            raise ValueError("source_case_ids must align with expert states")
        if bool(self.visible_state_sha256s) != bool(self.source_case_ids):
            raise ValueError(
                "visible-state hashes and source case ids must be audited together"
            )
        if self.source_path_counts:
            case_ids = [case_id for case_id, _count in self.source_path_counts]
            if any(not case_id for case_id in case_ids) or len(set(case_ids)) != len(case_ids):
                raise ValueError("source_path_counts must contain unique non-empty case ids")
            if any(count < 4 for _case_id, count in self.source_path_counts):
                raise ValueError("every v1.4 task must contribute at least four source paths")
            if set(case_ids) != set(self.source_case_ids):
                raise ValueError(
                    "source_path_counts must cover every audited expert-state task"
                )


@dataclass(frozen=True)
class PPOBatch(PolicyBatch):
    old_log_probs: Tensor
    old_values: Tensor
    returns: Tensor
    advantages: Tensor

    def validate(self) -> None:
        super().validate()
        size = self.states.shape[0]
        for name, tensor in (
            ("old_log_probs", self.old_log_probs),
            ("old_values", self.old_values),
            ("returns", self.returns),
            ("advantages", self.advantages),
        ):
            if tensor.ndim != 1 or tensor.shape[0] != size:
                raise ValueError(f"{name} must have shape [N]")


@dataclass(frozen=True)
class SequencePPOBatch(PolicyBatch):
    """Auditable episode-level PPO input backed by per-action observations.

    Policy tensors remain step-shaped because the current policy must evaluate
    every selected action.  Credit-assignment tensors are deliberately episode
    shaped: one discounted return, one initial-state value and one summed
    sequence log-probability per episode.
    """

    episode_ids: tuple[str, ...]
    episode_indices: Tensor
    initial_state_indices: Tensor
    old_step_log_probs: Tensor
    old_step_values: Tensor
    step_rewards: Tensor
    dones: Tensor
    discounted_returns: Tensor
    gamma: float

    @property
    def num_steps(self) -> int:
        return int(self.states.shape[0])

    @property
    def num_episodes(self) -> int:
        return len(self.episode_ids)

    @property
    def episode_lengths(self) -> Tensor:
        if self.initial_state_indices.numel() == 0:
            return self.initial_state_indices.clone()
        final_index = self.initial_state_indices.new_tensor([self.num_steps])
        episode_ends = torch.cat((self.initial_state_indices[1:], final_index))
        return episode_ends - self.initial_state_indices

    def sum_by_episode(self, step_values: Tensor) -> Tensor:
        """Sum a scalar step tensor into exactly one scalar per episode."""

        if step_values.ndim != 1 or step_values.shape[0] != self.num_steps:
            raise ValueError("step_values must have shape [N]")
        if step_values.device != self.episode_indices.device:
            raise ValueError("step_values and episode_indices must share a device")
        return step_values.new_zeros(self.num_episodes).index_add(
            0, self.episode_indices, step_values
        )

    @property
    def old_sequence_log_probs(self) -> Tensor:
        """Recorded sequence log-probabilities, summed over all actions."""

        return self.sum_by_episode(self.old_step_log_probs)

    @property
    def old_initial_values(self) -> Tensor:
        """The sole old value baseline used for each complete episode."""

        return self.old_step_values[self.initial_state_indices]

    @property
    def advantages(self) -> Tensor:
        """Monte-Carlo sequence advantages; action-level GAE is not involved."""

        return self.discounted_returns - self.old_initial_values

    def validate(self) -> None:
        super().validate()
        if self.num_steps == 0:
            raise ValueError("sequence PPO batch must contain at least one step")
        if self.num_episodes == 0:
            raise ValueError("sequence PPO batch must contain at least one episode")
        if any(not episode_id for episode_id in self.episode_ids):
            raise ValueError("episode_ids must not contain empty identifiers")
        if len(set(self.episode_ids)) != self.num_episodes:
            raise ValueError("episode_ids must be unique")
        if not math.isfinite(self.gamma) or not 0.0 <= self.gamma <= 1.0:
            raise ValueError("gamma must be finite and in [0, 1]")

        device = self.states.device
        if not self.states.is_floating_point():
            raise ValueError("states must use a floating dtype")
        if (
            not self.action_features.is_floating_point()
            or self.action_features.dtype != self.states.dtype
        ):
            raise ValueError("action_features must use the states floating dtype")
        if self.action_masks.dtype != torch.bool:
            raise ValueError("action_masks must use bool dtype")
        if self.actions.dtype != torch.long:
            raise ValueError("actions must use torch.long dtype")
        if self.action_features.shape[1] == 0:
            raise ValueError("every step must expose at least one candidate slot")
        if not bool(self.action_masks.any(dim=1).all()):
            raise ValueError("every step must expose at least one valid action")
        if bool((self.actions < 0).any()) or bool(
            (self.actions >= self.action_masks.shape[1]).any()
        ):
            raise ValueError("actions contain an out-of-range candidate index")
        if not bool(
            self.action_masks.gather(1, self.actions.unsqueeze(1)).squeeze(1).all()
        ):
            raise ValueError("a recorded action is masked out")
        if not bool(torch.isfinite(self.states).all()) or not bool(
            torch.isfinite(self.action_features).all()
        ):
            raise ValueError("policy features must be finite")

        for name, tensor, expected_dtype in (
            ("episode_indices", self.episode_indices, torch.long),
            ("initial_state_indices", self.initial_state_indices, torch.long),
            ("dones", self.dones, torch.bool),
        ):
            if tensor.device != device:
                raise ValueError(f"{name} must be on the policy tensor device")
            if tensor.dtype != expected_dtype:
                raise ValueError(f"{name} has the wrong dtype")
        if self.episode_indices.ndim != 1 or self.episode_indices.shape[0] != self.num_steps:
            raise ValueError("episode_indices must have shape [N]")
        if (
            self.initial_state_indices.ndim != 1
            or self.initial_state_indices.shape[0] != self.num_episodes
        ):
            raise ValueError("initial_state_indices must have shape [E]")
        if self.dones.ndim != 1 or self.dones.shape[0] != self.num_steps:
            raise ValueError("dones must have shape [N]")

        if int(self.episode_indices[0]) != 0 or int(self.episode_indices[-1]) != (
            self.num_episodes - 1
        ):
            raise ValueError("episode_indices must cover every episode exactly once")
        transitions = self.episode_indices[1:] - self.episode_indices[:-1]
        if bool(((transitions < 0) | (transitions > 1)).any()):
            raise ValueError("episode_indices must describe contiguous episode groups")
        observed_starts = torch.cat(
            (
                self.episode_indices.new_tensor([0]),
                torch.nonzero(transitions == 1, as_tuple=False).flatten() + 1,
            )
        )
        if not torch.equal(observed_starts, self.initial_state_indices):
            raise ValueError(
                "initial_state_indices must exactly identify each episode group's first step"
            )
        if bool((self.episode_lengths <= 0).any()):
            raise ValueError("every episode must contain at least one step")

        expected_dones = torch.zeros_like(self.dones)
        terminal_indices = self.initial_state_indices + self.episode_lengths - 1
        expected_dones[terminal_indices] = True
        if not torch.equal(self.dones, expected_dones):
            raise ValueError("dones must be true only at each episode's terminal step")

        for name, tensor, expected_size in (
            ("old_step_log_probs", self.old_step_log_probs, self.num_steps),
            ("old_step_values", self.old_step_values, self.num_steps),
            ("step_rewards", self.step_rewards, self.num_steps),
            ("discounted_returns", self.discounted_returns, self.num_episodes),
        ):
            if tensor.ndim != 1 or tensor.shape[0] != expected_size:
                dimension = "N" if expected_size == self.num_steps else "E"
                raise ValueError(f"{name} must have shape [{dimension}]")
            if tensor.device != device or tensor.dtype != self.states.dtype:
                raise ValueError(f"{name} must share the policy tensor device and dtype")
            if not bool(torch.isfinite(tensor).all()):
                raise ValueError(f"{name} must contain only finite values")

        recomputed_returns = self.discounted_returns.new_zeros(self.num_episodes)
        for episode_index, start_tensor in enumerate(self.initial_state_indices):
            start = int(start_tensor)
            length = int(self.episode_lengths[episode_index])
            rewards = self.step_rewards[start : start + length]
            exponents = torch.arange(length, device=device, dtype=self.states.dtype)
            discounts = torch.full_like(exponents, self.gamma).pow(exponents)
            recomputed_returns[episode_index] = (discounts * rewards).sum()
        if not torch.allclose(
            self.discounted_returns, recomputed_returns, rtol=1e-5, atol=1e-6
        ):
            raise ValueError(
                "discounted_returns must contain one discounted reward sum per episode"
            )
