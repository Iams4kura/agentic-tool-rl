from __future__ import annotations

import math

import torch

from agentic_tool_rl.algorithms import (
    BCConfig,
    BehaviorCloningTrainer,
    PolicyBatch,
    PPOBatch,
    PPOConfig,
    PPOTrainer,
    compute_gae,
)
from agentic_tool_rl.models import ActorCritic
from agentic_tool_rl.rewards import discounted_shaping_sum, shape_trajectory
from agentic_tool_rl.rollout import RolloutBuffer, StepRecord


def _candidate_batch(size: int = 64) -> PolicyBatch:
    generator = torch.Generator().manual_seed(7)
    states = torch.randn(size, 3, generator=generator)
    action_features = torch.randn(size, 4, 2, generator=generator) * 0.05
    scores = torch.tensor([-1.0, 0.25, 1.5, 0.75])
    action_features[:, :, 0] += scores
    masks = torch.ones(size, 4, dtype=torch.bool)
    masks[::3, 2] = False
    masked_scores = scores.unsqueeze(0).expand(size, -1).masked_fill(~masks, -torch.inf)
    actions = masked_scores.argmax(dim=-1)
    return PolicyBatch(states, action_features, masks, actions)


def test_actor_critic_masks_dynamic_candidates() -> None:
    torch.manual_seed(0)
    model = ActorCritic(state_dim=3, action_dim=2, hidden_dim=16)
    batch = _candidate_batch(8)
    logits, values = model(batch.states, batch.action_features, batch.action_masks)
    assert logits.shape == (8, 4)
    assert values.shape == (8,)
    assert torch.isneginf(logits[~batch.action_masks]).all()
    sample = model.act(
        batch.states, batch.action_features, batch.action_masks, deterministic=True
    )
    assert batch.action_masks.gather(1, sample.action_index[:, None]).all()


def test_behavior_cloning_really_updates_and_learns() -> None:
    torch.manual_seed(3)
    model = ActorCritic(state_dim=3, action_dim=2, hidden_dim=24)
    batch = _candidate_batch(96)
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    trainer = BehaviorCloningTrainer(
        model,
        BCConfig(learning_rate=5e-3, epochs=18, batch_size=24, seed=11),
    )
    metrics = trainer.update(batch)
    assert metrics.updates == 72
    assert metrics.accuracy >= 0.95
    assert metrics.loss >= 0
    assert metrics.grad_norm > 0
    assert any(
        not torch.equal(before[name], parameter.detach())
        for name, parameter in model.named_parameters()
    )


def test_gae_respects_trajectory_boundary_without_done_flag() -> None:
    rewards = torch.tensor([1.0, 1.0, 100.0])
    values = torch.zeros(3)
    dones = torch.tensor([False, False, True])
    advantages, returns = compute_gae(
        rewards,
        values,
        dones,
        gamma=1.0,
        gae_lambda=1.0,
        trajectory_ids=["first", "first", "second"],
    )
    assert torch.equal(advantages, torch.tensor([2.0, 1.0, 100.0]))
    assert torch.equal(returns, advantages)


def test_gae_can_bootstrap_truncated_tail_explicitly() -> None:
    advantages, _ = compute_gae(
        torch.tensor([1.0]),
        torch.tensor([0.5]),
        torch.tensor([False]),
        gamma=0.9,
        gae_lambda=0.95,
        next_values=torch.tensor([2.0]),
        trajectory_ids=["truncated"],
    )
    assert torch.allclose(advantages, torch.tensor([2.3]))


def test_clipped_action_level_ppo_updates_parameters_and_logs_metrics() -> None:
    torch.manual_seed(13)
    model = ActorCritic(state_dim=3, action_dim=2, hidden_dim=24)
    policy_batch = _candidate_batch(48)
    with torch.no_grad():
        old_log_probs, _, old_values, _ = model.evaluate_actions(
            policy_batch.states,
            policy_batch.action_features,
            policy_batch.action_masks,
            policy_batch.actions,
        )
    advantages = torch.linspace(-1.5, 2.0, 48)
    batch = PPOBatch(
        states=policy_batch.states,
        action_features=policy_batch.action_features,
        action_masks=policy_batch.action_masks,
        actions=policy_batch.actions,
        old_log_probs=old_log_probs,
        old_values=old_values,
        returns=old_values + advantages,
        advantages=advantages,
    )
    before = torch.cat([parameter.detach().flatten() for parameter in model.parameters()])
    metrics = PPOTrainer(
        model,
        PPOConfig(
            learning_rate=3e-3,
            epochs=3,
            batch_size=16,
            entropy_coefficient=0.01,
            seed=17,
        ),
    ).update(batch)
    after = torch.cat([parameter.detach().flatten() for parameter in model.parameters()])
    assert not torch.equal(before, after)
    assert metrics.updates == 9
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
    assert metrics.approximate_kl >= -1e-6
    assert 0 <= metrics.clip_fraction <= 1
    assert metrics.grad_norm > 0


def _record(
    trajectory_id: str,
    step_index: int,
    *,
    candidate_count: int,
    reward: float,
    done: bool,
) -> StepRecord:
    return StepRecord(
        trajectory_id=trajectory_id,
        step_index=step_index,
        observation={"visible": step_index},
        candidates=tuple({"tool": f"tool_{item}"} for item in range(candidate_count)),
        action_index=0,
        action_mask=tuple(True for _ in range(candidate_count)),
        log_prob=-0.5,
        value=0.1,
        reward=reward,
        done=done,
        next_observation={"visible": step_index + 1},
        state_features=torch.tensor([step_index, 1.0, 0.0]),
        action_features=torch.arange(candidate_count * 2, dtype=torch.float32).reshape(
            candidate_count, 2
        ),
        task_reward=reward,
        info={"tool_latency": 0.2},
    )


def test_rollout_buffer_preserves_tool_steps_and_pads_candidates() -> None:
    buffer = RolloutBuffer()
    buffer.extend(
        [
            _record("a", 0, candidate_count=2, reward=0.1, done=False),
            _record("a", 1, candidate_count=3, reward=1.0, done=True),
            _record("b", 0, candidate_count=1, reward=-1.0, done=True),
        ]
    )
    assert len(buffer) == 3
    assert [len(item) for item in buffer.trajectories] == [2, 1]
    batch = buffer.as_ppo_batch(gamma=0.9, gae_lambda=0.95)
    assert batch.states.shape == (3, 3)
    assert batch.action_features.shape == (3, 3, 2)
    assert batch.action_masks.tolist()[-1] == [True, False, False]
    assert batch.advantages.shape == (3,)
    trace = buffer.records[0].to_trace_dict()
    assert trace["action"] == {"tool": "tool_0"}
    assert trace["reward_components"]["task"] == 0.1


def test_rollout_buffer_adapts_portable_contract_record() -> None:
    from agentic_tool_rl.contracts import InvalidActionKind, ToolCall
    from agentic_tool_rl.contracts import StepRecord as ContractRecord

    portable = ContractRecord(
        trajectory_id="portable",
        step_index=0,
        observation={"visible": "only public state"},
        candidates=[ToolCall(tool_name="lookup", arguments={"id": "x"})],
        action_index=0,
        action_mask=[True],
        log_prob=-0.1,
        value=0.2,
        reward=-1.0,
        done=True,
        valid_label=False,
        predicted_valid=False,
        invalid_kind=InvalidActionKind.PRECONDITION,
    )
    buffer = RolloutBuffer()
    converted = buffer.add_contract(
        portable,
        state_features=torch.tensor([1.0, 0.0, 0.5]),
        action_features=torch.tensor([[0.2, 0.8]]),
    )
    trace = converted.to_trace_dict()
    assert trace["candidates"][0]["tool_name"] == "lookup"
    assert trace["valid_label"] is False
    assert trace["invalid_kind"] == "precondition"
    assert buffer.as_ppo_batch().states.shape == (1, 3)


def test_potential_shaping_telescopes_under_discounting() -> None:
    gamma = 0.9
    beta = 0.7
    potentials = torch.tensor([0.2, 0.5, 0.4, 0.9], dtype=torch.float64)
    base = torch.zeros(3, dtype=torch.float64)
    shaped, terms = shape_trajectory(base, potentials, gamma=gamma, beta=beta)
    expected_boundary = beta * (
        -(potentials[0]) + gamma ** (len(potentials) - 1) * potentials[-1]
    )
    assert torch.allclose(discounted_shaping_sum(terms, gamma=gamma), expected_boundary)
    assert torch.equal(shaped, terms)
