from __future__ import annotations

import pytest
import torch

from agentic_tool_rl.rollout import RolloutBuffer, StepRecord


def _record(trajectory_id: str, step_index: int, *, done: bool = True) -> StepRecord:
    return StepRecord(
        trajectory_id=trajectory_id,
        step_index=step_index,
        observation={},
        candidates=("lookup",),
        action_index=0,
        action_mask=(True,),
        log_prob=0.0,
        value=0.0,
        reward=1.0,
        done=done,
        state_features=torch.tensor([1.0]),
        action_features=torch.tensor([[1.0]]),
    )


@pytest.mark.parametrize("populated", [False, True])
def test_rejected_first_step_preserves_buffer_and_allows_retry(populated: bool) -> None:
    buffer = RolloutBuffer()
    if populated:
        buffer.add(_record("healthy", 0))
    before_trajectories = buffer.trajectories
    before_records = buffer.records

    with pytest.raises(ValueError, match="expected step_index=0, got 1"):
        buffer.add(_record("retry", 1))

    assert buffer.trajectories == before_trajectories
    assert buffer.records == before_records
    assert len(buffer) == int(populated)
    if populated:
        batch = buffer.as_sequence_ppo_batch()
        assert batch.episode_ids == ("healthy",)
        assert batch.discounted_returns.tolist() == [1.0]

    buffer.add(_record("later", 0))
    buffer.add(_record("retry", 0))
    batch = buffer.as_sequence_ppo_batch()
    assert batch.episode_ids == (("healthy",) if populated else ()) + ("later", "retry")
    assert batch.discounted_returns.tolist() == [1.0] * (2 + int(populated))


def test_rejected_existing_step_preserves_trajectory_and_allows_retry() -> None:
    buffer = RolloutBuffer()
    first = _record("episode", 0, done=False)
    buffer.add(first)
    trajectory = buffer.trajectories[0]

    with pytest.raises(ValueError, match="expected step_index=1, got 2"):
        buffer.add(_record("episode", 2))

    assert buffer.trajectories == (trajectory,)
    assert buffer.records == (first,)
    buffer.add(_record("episode", 1))
    batch = buffer.as_sequence_ppo_batch(gamma=0.5)
    assert batch.episode_ids == ("episode",)
    assert batch.discounted_returns.tolist() == [1.5]

    with pytest.raises(ValueError, match="cannot append after a terminal step"):
        buffer.add(_record("episode", 2))
    assert len(buffer) == 2
    assert buffer.as_sequence_ppo_batch(gamma=0.5).discounted_returns.tolist() == [1.5]
