"""Trajectory-aware rollout storage and PPO tensor collation."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterable

import torch

from agentic_tool_rl.algorithms.gae import compute_gae
from agentic_tool_rl.algorithms.types import PolicyBatch, PPOBatch, SequencePPOBatch
from agentic_tool_rl.rollout.records import StepRecord, Trajectory


class RolloutBuffer:
    """Keep tool-call records grouped by trajectory, never by flat batch alone."""

    def __init__(self) -> None:
        self._trajectories: OrderedDict[str, Trajectory] = OrderedDict()

    @property
    def trajectories(self) -> tuple[Trajectory, ...]:
        return tuple(self._trajectories.values())

    @property
    def records(self) -> tuple[StepRecord, ...]:
        return tuple(step for trajectory in self.trajectories for step in trajectory.steps)

    def add(self, record: StepRecord) -> None:
        """Append one step without registering a trajectory if validation fails."""

        trajectory = self._trajectories.get(record.trajectory_id)
        if trajectory is None:
            trajectory = Trajectory(record.trajectory_id)
            trajectory.append(record)
            self._trajectories[record.trajectory_id] = trajectory
            return
        trajectory.append(record)

    def add_contract(
        self,
        record: object,
        *,
        state_features: torch.Tensor,
        action_features: torch.Tensor,
        trajectory_id: str | None = None,
    ) -> StepRecord:
        """Convert and add a portable ``contracts.StepRecord`` lazily."""

        converted = StepRecord.from_contract(
            record,
            trajectory_id=trajectory_id,
            state_features=state_features,
            action_features=action_features,
        )
        self.add(converted)
        return converted

    def extend(self, records: Iterable[StepRecord]) -> None:
        for record in records:
            self.add(record)

    def add_trajectory(self, trajectory: Trajectory) -> None:
        if trajectory.trajectory_id in self._trajectories:
            raise ValueError(f"duplicate trajectory: {trajectory.trajectory_id}")
        copied = Trajectory(trajectory.trajectory_id)
        for record in trajectory.steps:
            copied.append(record)
        self._trajectories[trajectory.trajectory_id] = copied

    def clear(self) -> None:
        self._trajectories.clear()

    def __len__(self) -> int:
        return sum(len(trajectory) for trajectory in self._trajectories.values())

    def _policy_tensors(self, *, device: torch.device | str | None = None) -> PolicyBatch:
        records = self.records
        if not records:
            raise ValueError("cannot collate an empty rollout buffer")
        if any(
            record.state_features is None or record.action_features is None
            for record in records
        ):
            raise ValueError("every record needs state_features and action_features")
        states = [record.state_features for record in records]
        actions_features = [record.action_features for record in records]
        assert all(item is not None for item in states)
        assert all(item is not None for item in actions_features)
        state_dim = states[0].shape[0]  # type: ignore[union-attr]
        action_dim = actions_features[0].shape[1]  # type: ignore[union-attr]
        if any(item.shape[0] != state_dim for item in states if item is not None):
            raise ValueError("state feature dimensions differ")
        if any(item.shape[1] != action_dim for item in actions_features if item is not None):
            raise ValueError("action feature dimensions differ")

        maximum_actions = max(len(record.candidates) for record in records)
        dtype = states[0].dtype  # type: ignore[union-attr]
        padded_actions = torch.zeros(
            (len(records), maximum_actions, action_dim), dtype=dtype
        )
        masks = torch.zeros((len(records), maximum_actions), dtype=torch.bool)
        for index, record in enumerate(records):
            assert record.action_features is not None
            count = record.action_features.shape[0]
            padded_actions[index, :count] = record.action_features.to(dtype=dtype)
            masks[index, :count] = torch.tensor(record.action_mask, dtype=torch.bool)

        return PolicyBatch(
            states=torch.stack([item for item in states if item is not None]).to(device),
            action_features=padded_actions.to(device),
            action_masks=masks.to(device),
            actions=torch.tensor(
                [record.action_index for record in records], dtype=torch.long, device=device
            ),
        )

    def as_policy_batch(self, *, device: torch.device | str | None = None) -> PolicyBatch:
        return self._policy_tensors(device=device)

    def as_ppo_batch(
        self,
        *,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        device: torch.device | str | None = None,
        next_values: torch.Tensor | None = None,
    ) -> PPOBatch:
        """Collate dynamic candidates and calculate trajectory-safe GAE."""

        policy = self._policy_tensors(device=device)
        records = self.records
        rewards = torch.tensor(
            [record.reward for record in records], dtype=policy.states.dtype, device=device
        )
        values = torch.tensor(
            [record.value for record in records], dtype=policy.states.dtype, device=device
        )
        dones = torch.tensor(
            [record.done for record in records], dtype=torch.bool, device=device
        )
        trajectory_ids = [record.trajectory_id for record in records]
        if next_values is not None:
            next_values = next_values.to(device=device, dtype=values.dtype)
        advantages, returns = compute_gae(
            rewards,
            values,
            dones,
            gamma=gamma,
            gae_lambda=gae_lambda,
            next_values=next_values,
            trajectory_ids=trajectory_ids,
        )
        return PPOBatch(
            states=policy.states,
            action_features=policy.action_features,
            action_masks=policy.action_masks,
            actions=policy.actions,
            old_log_probs=torch.tensor(
                [record.log_prob for record in records],
                dtype=policy.states.dtype,
                device=device,
            ),
            old_values=values,
            returns=returns,
            advantages=advantages,
        )

    def as_sequence_ppo_batch(
        self,
        *,
        gamma: float = 0.99,
        device: torch.device | str | None = None,
    ) -> SequencePPOBatch:
        """Collate complete trajectories for true sequence-level PPO.

        Per-step rewards are preserved exactly as recorded.  They are reduced
        once into one discounted return per episode; no terminal reward is
        copied and no action-level GAE calculation occurs on this path.
        """

        trajectories = self.trajectories
        if not trajectories:
            raise ValueError("cannot collate an empty rollout buffer")
        if any(not trajectory.done for trajectory in trajectories):
            raise ValueError("sequence PPO requires complete terminal episodes")
        if any(
            record.task_reward != 0.0
            for trajectory in trajectories
            for record in trajectory.steps[:-1]
        ):
            raise ValueError(
                "task_reward is terminal-only and must appear only on an episode's "
                "terminal step"
            )

        policy = self._policy_tensors(device=device)
        records = self.records
        dtype = policy.states.dtype
        initial_state_indices: list[int] = []
        episode_indices: list[torch.Tensor] = []
        discounted_returns: list[float] = []
        offset = 0
        for episode_index, trajectory in enumerate(trajectories):
            initial_state_indices.append(offset)
            episode_indices.append(
                torch.full(
                    (len(trajectory),),
                    episode_index,
                    dtype=torch.long,
                    device=device,
                )
            )
            discounted_returns.append(
                sum(
                    (gamma**step_index) * record.reward
                    for step_index, record in enumerate(trajectory.steps)
                )
            )
            offset += len(trajectory)

        batch = SequencePPOBatch(
            states=policy.states,
            action_features=policy.action_features,
            action_masks=policy.action_masks,
            actions=policy.actions,
            episode_ids=tuple(trajectory.trajectory_id for trajectory in trajectories),
            episode_indices=torch.cat(episode_indices),
            initial_state_indices=torch.tensor(
                initial_state_indices, dtype=torch.long, device=device
            ),
            old_step_log_probs=torch.tensor(
                [record.log_prob for record in records], dtype=dtype, device=device
            ),
            old_step_values=torch.tensor(
                [record.value for record in records], dtype=dtype, device=device
            ),
            step_rewards=torch.tensor(
                [record.reward for record in records], dtype=dtype, device=device
            ),
            dones=torch.tensor(
                [record.done for record in records], dtype=torch.bool, device=device
            ),
            discounted_returns=torch.tensor(
                discounted_returns, dtype=dtype, device=device
            ),
            gamma=gamma,
        )
        batch.validate()
        return batch
