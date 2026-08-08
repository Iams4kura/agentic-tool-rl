from __future__ import annotations

import math
from dataclasses import replace

import pytest
import torch

from agentic_tool_rl.algorithms import PPOConfig, SequencePPOTrainer
from agentic_tool_rl.models import ActorCritic
from agentic_tool_rl.rollout import RolloutBuffer, StepRecord


def _record(
    trajectory_id: str,
    step_index: int,
    *,
    candidate_count: int,
    log_prob: float,
    value: float,
    reward: float,
    done: bool,
) -> StepRecord:
    return StepRecord(
        trajectory_id=trajectory_id,
        step_index=step_index,
        observation={"step": step_index},
        candidates=tuple({"tool": f"tool_{index}"} for index in range(candidate_count)),
        action_index=step_index % candidate_count,
        action_mask=tuple(True for _ in range(candidate_count)),
        log_prob=log_prob,
        value=value,
        reward=reward,
        done=done,
        next_observation={"step": step_index + 1},
        state_features=torch.tensor(
            [float(step_index), float(len(trajectory_id)), 1.0], dtype=torch.float32
        ),
        action_features=torch.arange(
            candidate_count * 2, dtype=torch.float32
        ).reshape(candidate_count, 2)
        / 10.0,
        task_reward=reward if done else 0.0,
    )


def _two_episode_buffer() -> RolloutBuffer:
    buffer = RolloutBuffer()
    buffer.extend(
        [
            _record(
                "short",
                0,
                candidate_count=2,
                log_prob=-0.2,
                value=0.4,
                reward=0.0,
                done=False,
            ),
            _record(
                "short",
                1,
                candidate_count=4,
                log_prob=-0.3,
                value=99.0,
                reward=2.0,
                done=True,
            ),
            _record(
                "long",
                0,
                candidate_count=3,
                log_prob=-0.1,
                value=-0.5,
                reward=0.0,
                done=False,
            ),
            _record(
                "long",
                1,
                candidate_count=1,
                log_prob=-0.4,
                value=199.0,
                reward=0.0,
                done=False,
            ),
            _record(
                "long",
                2,
                candidate_count=2,
                log_prob=-0.2,
                value=299.0,
                reward=2.0,
                done=True,
            ),
        ]
    )
    return buffer


def test_sequence_batch_aggregates_log_probs_and_one_discounted_return_per_episode() -> None:
    batch = _two_episode_buffer().as_sequence_ppo_batch(gamma=0.5)

    assert batch.states.shape == (5, 3)
    assert batch.action_features.shape == (5, 4, 2)
    assert batch.action_masks.tolist()[3] == [True, False, False, False]
    assert batch.episode_ids == ("short", "long")
    assert batch.episode_indices.tolist() == [0, 0, 1, 1, 1]
    assert batch.initial_state_indices.tolist() == [0, 2]
    assert batch.episode_lengths.tolist() == [2, 3]

    # Both old and newly evaluated sequence log-probabilities use a sum over
    # every selected action in that episode, never a mean or the terminal step.
    assert torch.allclose(batch.old_sequence_log_probs, torch.tensor([-0.5, -0.7]))
    new_step_log_probs = torch.tensor([-0.4, -0.1, -0.2, -0.3, -0.5])
    assert torch.allclose(
        batch.sum_by_episode(new_step_log_probs), torch.tensor([-0.5, -1.0])
    )

    # The terminal reward exists once in each trajectory.  A longer trajectory
    # discounts it further; it never receives reward * trajectory_length.
    assert batch.step_rewards.tolist() == [0.0, 2.0, 0.0, 0.0, 2.0]
    assert batch.discounted_returns.shape == (2,)
    assert torch.allclose(batch.discounted_returns, torch.tensor([1.0, 0.5]))
    assert torch.allclose(batch.old_initial_values, torch.tensor([0.4, -0.5]))
    assert torch.allclose(batch.advantages, torch.tensor([0.6, 1.0]))


def test_sequence_batch_rejects_broken_episode_grouping_and_initial_indices() -> None:
    batch = _two_episode_buffer().as_sequence_ppo_batch()

    with pytest.raises(ValueError, match="contiguous episode groups"):
        replace(
            batch,
            episode_indices=torch.tensor([0, 1, 0, 1, 1]),
        ).validate()
    with pytest.raises(ValueError, match="initial_state_indices"):
        replace(batch, initial_state_indices=torch.tensor([0, 1])).validate()
    with pytest.raises(ValueError, match="terminal step"):
        replace(
            batch,
            dones=torch.tensor([False, True, False, True, False]),
        ).validate()


def test_sequence_batch_requires_complete_episodes() -> None:
    buffer = RolloutBuffer()
    buffer.add(
        _record(
            "truncated",
            0,
            candidate_count=2,
            log_prob=-0.2,
            value=0.0,
            reward=0.0,
            done=False,
        )
    )
    with pytest.raises(ValueError, match="complete terminal episodes"):
        buffer.as_sequence_ppo_batch()


def test_sequence_batch_rejects_terminal_task_reward_copied_to_each_step() -> None:
    buffer = RolloutBuffer()
    first = _record(
        "duplicated",
        0,
        candidate_count=2,
        log_prob=-0.2,
        value=0.0,
        reward=2.0,
        done=False,
    )
    buffer.extend(
        [
            replace(first, task_reward=2.0),
            _record(
                "duplicated",
                1,
                candidate_count=2,
                log_prob=-0.3,
                value=0.1,
                reward=2.0,
                done=True,
            ),
        ]
    )
    with pytest.raises(ValueError, match=r"task_reward.*terminal step"):
        buffer.as_sequence_ppo_batch()


def _batch_with_current_policy_baseline(
    model: ActorCritic,
) -> tuple[RolloutBuffer, torch.Tensor]:
    source = _two_episode_buffer()
    policy = source.as_policy_batch()
    with torch.no_grad():
        current_log_probs, _, current_values, _ = model.evaluate_actions(
            policy.states,
            policy.action_features,
            policy.action_masks,
            policy.actions,
        )

    # new - old at the action level; expected sequence deltas are [0.30, 0.20].
    deltas = torch.tensor([0.10, 0.20, -0.05, 0.15, 0.10])
    rebuilt = RolloutBuffer()
    for index, record in enumerate(source.records):
        # Later-step old values are deliberately enormous.  A valid
        # sequence-level value loss must ignore them and use only indices 0, 2.
        old_value = (
            float(current_values[index])
            if record.step_index == 0
            else float(100_000 + index)
        )
        rebuilt.add(
            replace(
                record,
                log_prob=float(current_log_probs[index] - deltas[index]),
                value=old_value,
            )
        )
    return rebuilt, current_values


def test_sequence_ppo_uses_one_ratio_and_only_initial_value_per_episode() -> None:
    torch.manual_seed(21)
    model = ActorCritic(state_dim=3, action_dim=2, hidden_dim=16)
    buffer, current_values = _batch_with_current_policy_baseline(model)
    batch = buffer.as_sequence_ppo_batch(gamma=0.5)

    expected_log_ratios = torch.tensor([0.30, 0.20])
    expected_ratios = expected_log_ratios.exp()
    expected_kl = ((expected_ratios - 1.0) - expected_log_ratios).mean()
    expected_clip_fraction = (
        (expected_ratios - 1.0).abs() > 0.2
    ).float().mean()
    expected_value_loss = 0.5 * torch.mean(
        (current_values[batch.initial_state_indices] - batch.discounted_returns).pow(2)
    )

    metrics = SequencePPOTrainer(
        model,
        PPOConfig(
            learning_rate=1e-8,
            epochs=1,
            batch_size=2,
            entropy_coefficient=0.0,
            normalize_advantages=False,
            seed=4,
        ),
    ).update(batch)

    assert metrics.updates == 1
    assert metrics.approximate_kl == pytest.approx(float(expected_kl), abs=1e-6)
    assert metrics.clip_fraction == pytest.approx(float(expected_clip_fraction))
    assert metrics.value_loss == pytest.approx(float(expected_value_loss), abs=1e-5)
    assert metrics.value_loss < 10.0  # later-state values around 100,000 were ignored


def test_sequence_ppo_really_backpropagates_with_mixed_lengths() -> None:
    torch.manual_seed(9)
    model = ActorCritic(state_dim=3, action_dim=2, hidden_dim=24)
    batch = _two_episode_buffer().as_sequence_ppo_batch(gamma=0.9)
    before = torch.cat([parameter.detach().flatten() for parameter in model.parameters()])

    metrics = SequencePPOTrainer(
        model,
        PPOConfig(
            learning_rate=3e-3,
            epochs=3,
            batch_size=2,
            entropy_coefficient=0.01,
            normalize_advantages=False,
            seed=17,
        ),
    ).update(batch)

    after = torch.cat([parameter.detach().flatten() for parameter in model.parameters()])
    assert not torch.equal(before, after)
    assert metrics.updates == 3
    for value in (
        metrics.policy_loss,
        metrics.value_loss,
        metrics.entropy,
        metrics.approximate_kl,
        metrics.clip_fraction,
        metrics.grad_norm,
        metrics.total_loss,
    ):
        assert math.isfinite(value)
    assert metrics.value_loss >= 0
    assert metrics.entropy > 0
    assert metrics.grad_norm > 0
