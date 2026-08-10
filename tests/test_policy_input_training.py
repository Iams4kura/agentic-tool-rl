from __future__ import annotations

from collections.abc import Sequence

import pytest

from agentic_tool_rl.config import ExperimentConfig, load_config
from agentic_tool_rl.contracts import Split
from agentic_tool_rl.envs.benchmark import generate_tasks
from agentic_tool_rl.envs.benchmark_v14 import generate_counterfactual_tasks
from agentic_tool_rl.features import DecisionFeatures, FeatureEncoder
from agentic_tool_rl.models import ActorCritic
from agentic_tool_rl.policy_input import PolicyInput
from agentic_tool_rl.training import (
    collect_counterfactual_expert_batch,
    collect_expert_demonstrations,
    run_episode,
)


def test_bc_and_episode_policy_paths_never_call_legacy_decision_encoder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_config("configs/smoke.yaml")
    assert isinstance(config, ExperimentConfig)
    encoder = FeatureEncoder(
        config.policy.observation_dim,
        config.policy.action_dim,
    )
    observed: list[tuple[PolicyInput, Sequence[bool] | None]] = []
    original = FeatureEncoder.encode_policy_input

    def record_policy_input(
        self: FeatureEncoder,
        policy_input: PolicyInput,
        *,
        selection_mask: Sequence[bool] | None = None,
    ) -> DecisionFeatures:
        observed.append((policy_input, selection_mask))
        return original(self, policy_input, selection_mask=selection_mask)

    def reject_legacy(*_args: object, **_kwargs: object) -> DecisionFeatures:
        raise AssertionError("policy execution must not call encode_decision")

    monkeypatch.setattr(FeatureEncoder, "encode_policy_input", record_policy_input)
    monkeypatch.setattr(FeatureEncoder, "encode_decision", reject_legacy)

    v13_tasks = generate_tasks(Split.TRAIN, 1, base_seed=1701)
    demonstrations = collect_expert_demonstrations(v13_tasks, encoder)
    v14_tasks = generate_counterfactual_tasks(Split.TRAIN, 1, base_seed=1702)
    expert_batch = collect_counterfactual_expert_batch(v14_tasks, encoder, max_tasks=1)
    model = ActorCritic(
        state_dim=encoder.state_dim,
        action_dim=encoder.action_dim,
        hidden_dim=config.policy.hidden_dim,
    )
    episode = run_episode(
        model,
        None,
        v13_tasks[0],
        encoder,
        config.reward,
        trajectory_id="policy-input-main-path",
        use_action_mask=True,
        use_progress_reward=False,
        deterministic=True,
    )

    assert observed
    assert expert_batch.states.shape[0] > 0
    assert all(len(policy_input.sha256()) == 64 for policy_input, _ in observed)
    assert any(selection_mask is None for _, selection_mask in observed)
    assert any(
        selection_mask is not None and all(selection_mask)
        for _, selection_mask in observed
    )
    assert all(
        len(str(record.info["policy_input_sha256"])) == 64
        for record in (*demonstrations.records, *episode.records)
    )
