from __future__ import annotations

import importlib
from dataclasses import replace

import pytest


def _semantic_canary_module() -> object:
    return importlib.import_module("agentic_tool_rl.semantic_canary")


def test_semantic_gate_rejects_double_zero_without_requiring_e_to_beat_b() -> None:
    semantic = _semantic_canary_module()

    with pytest.raises(semantic.SemanticCanaryError, match=r"both.*zero"):
        semantic.validate_semantic_canary(
            {
                "B-BC-Mask": {
                    "successes": 0,
                    "cases": 8,
                    "independent_training_repetitions": 20,
                },
                "E-PPO-Progress-Mask": {
                    "successes": 0,
                    "cases": 8,
                    "independent_training_repetitions": 20,
                },
            },
            bc_action_set_accuracy=1.0,
            oracle_tsr=1.0,
            trace_consistency={
                "reward": True,
                "done": True,
                "metrics": True,
                "replay": True,
            },
            repetitions=20,
        )

    equal_positive = semantic.validate_semantic_canary(
        {
            "B-BC-Mask": {
                "successes": 6,
                "cases": 8,
                "independent_training_repetitions": 20,
            },
            "E-PPO-Progress-Mask": {
                "successes": 6,
                "cases": 8,
                "independent_training_repetitions": 20,
            },
        },
        bc_action_set_accuracy=0.95,
        oracle_tsr=1.0,
        trace_consistency={
            "reward": True,
            "done": True,
            "metrics": True,
            "replay": True,
        },
        repetitions=20,
    )
    assert equal_positive["passed"] is True
    assert equal_positive["criterion"] == "deterministic_engineering_canary_v2"
    assert equal_positive["requires_e_greater_than_b"] is False

    with pytest.raises(semantic.SemanticCanaryError, match="metrics_consistency"):
        semantic.validate_semantic_canary(
            {
                "B-BC-Mask": {
                    "successes": 6,
                    "cases": 8,
                    "independent_training_repetitions": 20,
                },
                "E-PPO-Progress-Mask": {
                    "successes": 6,
                    "cases": 8,
                    "independent_training_repetitions": 20,
                },
            },
            bc_action_set_accuracy=0.95,
            oracle_tsr=1.0,
            trace_consistency={
                "reward": True,
                "done": True,
                "metrics": False,
                "replay": True,
            },
            repetitions=20,
        )


def test_deterministic_overfit_canary_exercises_b_and_e_training_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    semantic = _semantic_canary_module()
    original_bc = semantic.train_behavior_policy
    original_ppo = semantic.train_ppo_variant
    training_calls = {"bc": 0, "ppo": 0}

    def counted_bc(*args: object, **kwargs: object) -> object:
        training_calls["bc"] += 1
        return original_bc(*args, **kwargs)

    def counted_ppo(*args: object, **kwargs: object) -> object:
        training_calls["ppo"] += 1
        return original_ppo(*args, **kwargs)

    monkeypatch.setattr(semantic, "train_behavior_policy", counted_bc)
    monkeypatch.setattr(semantic, "train_ppo_variant", counted_ppo)
    result = semantic.run_semantic_canary()

    assert training_calls == {"bc": 20, "ppo": 20}
    assert result["gate"]["passed"] is True
    assert result["fixed_cases"] == 8
    assert result["deterministic_repetitions"] == 20
    assert result["independent_training_repetitions"] == 20
    assert result["unique_task_hashes"] == 1
    assert result["oracle"]["tsr"] == 1.0
    assert set(result["runs"]) == {"B-BC-Mask", "E-PPO-Progress-Mask"}
    assert result["runs"]["B-BC-Mask"]["tsr"] >= 0.75
    assert result["training"]["bc_action_set_accuracy"] >= 0.95
    assert set(result["trace_consistency"]) == {"reward", "done", "metrics", "replay"}
    assert all(result["trace_consistency"].values())
    assert all(run["unique_trace_hashes"] == 1 for run in result["runs"].values())
    assert all(run["unique_model_hashes"] == 1 for run in result["runs"].values())
    assert all(
        run["unique_serialized_tool_call_hashes"] == 1
        for run in result["runs"].values()
    )
    assert all(
        run["independent_training_repetitions"] == 20
        for run in result["runs"].values()
    )
    assert result["training"]["bc_updates"] > 0
    assert result["training"]["ppo_updates"] > 0
    assert result["training"]["rollout_steps"] > 0


def test_serialized_tool_call_replay_rejects_intermediate_observation_drift() -> None:
    semantic = _semantic_canary_module()
    config = semantic._canary_config(semantic.load_packaged_config("smoke.yaml"))
    task = semantic.generate_tasks("train", 1, base_seed=424_242)[0]
    encoder = semantic.FeatureEncoder(
        config.policy.observation_dim,
        config.policy.action_dim,
    )
    model = semantic.ActorCritic(
        state_dim=encoder.state_dim,
        action_dim=encoder.action_dim,
        hidden_dim=config.policy.hidden_dim,
    )
    estimator = semantic.ProgressEstimator(encoder.state_dim, hidden_dim=64).freeze()
    episode = semantic.run_episode(
        model,
        estimator,
        task,
        encoder,
        config.reward,
        trajectory_id="semantic-canary-corruption-test",
        use_action_mask=True,
        use_progress_reward=False,
        deterministic=True,
        gamma=config.training.ppo.gamma,
        trace_mode="compact",
    )
    assert len(episode.records) >= 2

    wrong_current = list(episode.records)
    wrong_current[0] = replace(
        wrong_current[0],
        observation=episode.records[1].observation,
    )
    _, current_consistency, _ = semantic._replay_serialized_tool_calls(
        wrong_current,
        task,
        estimator,
        encoder,
        config,
        use_progress_reward=False,
    )
    assert current_consistency["replay"] is False

    wrong_next = list(episode.records)
    wrong_next[0] = replace(
        wrong_next[0],
        next_observation=episode.records[0].observation,
    )
    _, next_consistency, _ = semantic._replay_serialized_tool_calls(
        wrong_next,
        task,
        estimator,
        encoder,
        config,
        use_progress_reward=False,
    )
    assert next_consistency["replay"] is False
