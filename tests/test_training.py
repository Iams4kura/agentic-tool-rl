from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch

import agentic_tool_rl.training as training
from agentic_tool_rl.config import ExperimentConfig, load_config
from agentic_tool_rl.envs import generate_tasks
from agentic_tool_rl.features import FeatureEncoder
from agentic_tool_rl.models import ActorCritic, ProgressEstimator
from agentic_tool_rl.training import (
    collect_expert_demonstrations,
    load_checkpoint,
    run_episode,
    save_checkpoint,
    stratified_task_schedule,
    train_behavior_policy,
    train_progress_estimator,
)


def _smoke_config() -> ExperimentConfig:
    config = load_config("configs/smoke.yaml")
    assert isinstance(config, ExperimentConfig)
    return config


def test_expert_demonstrations_are_real_aligned_tool_steps() -> None:
    tasks = generate_tasks("train", 3, base_seed=123)
    encoder = FeatureEncoder(32, 32)

    buffer = collect_expert_demonstrations(tasks, encoder)
    batch = buffer.as_policy_batch()

    assert len(buffer.trajectories) == 3
    assert all(trajectory.done for trajectory in buffer.trajectories)
    assert batch.states.shape[0] == sum(task.optimal_steps for task in tasks)
    assert batch.action_masks.gather(1, batch.actions.unsqueeze(1)).all()
    assert all(all(record.action_mask) for record in buffer.records)


def test_progress_training_and_masked_episode_is_auditable(tmp_path: Path) -> None:
    config = _smoke_config()
    train_tasks = generate_tasks("train", 8, base_seed=321)
    dev_tasks = generate_tasks("dev", 8, base_seed=321)
    encoder = FeatureEncoder(config.policy.observation_dim, config.policy.action_dim)
    model, bc_metrics = train_behavior_policy(train_tasks, encoder, config, seed=7)
    estimator, progress = train_progress_estimator(
        train_tasks, dev_tasks, encoder, seed=7, epochs=2
    )

    result = run_episode(
        model,
        estimator,
        dev_tasks[0],
        encoder,
        config.reward,
        trajectory_id="integration-episode",
        use_action_mask=True,
        use_progress_reward=True,
        deterministic=True,
    )

    assert bc_metrics.updates > 0
    assert progress.examples > 0
    assert progress.quality_status == (
        "confirmatory" if progress.dev_quality_gate.passed else "exploratory"
    )
    assert progress.dev_gate_scope == "nonterminal_prefix_only"
    assert progress.dev_gate_examples == progress.dev_sampling.done_counts["0"]
    assert progress.dev_sampling.done_counts["0"] > 0
    assert progress.dev_sampling.done_counts["1"] > 0
    assert (
        progress.continuation_implementation_sha256
        == progress.dev_sampling.policy_implementation_sha256
    )
    assert estimator.frozen
    assert result.trace["step_count"] == len(result.records)
    assert result.records[-1].done
    assert all(item["executed_valid"] for item in result.trace["executed_actions"])
    assert all(record.trajectory_id == "integration-episode" for record in result.records)
    assert any(record.progress_reward != 0.0 for record in result.records)

    checkpoint = save_checkpoint(
        tmp_path / "policy.pt",
        model,
        estimator,
        encoder,
        seed=7,
        variant="E-PPO-Progress-Mask",
        metadata={"bc_loss": bc_metrics.loss},
    )
    restored, restored_progress, restored_encoder, metadata = load_checkpoint(checkpoint)
    assert restored_encoder.fingerprint() == encoder.fingerprint()
    assert restored_progress.frozen
    assert metadata["bc_loss"] == bc_metrics.loss
    for expected, actual in zip(model.parameters(), restored.parameters(), strict=True):
        assert torch.equal(expected, actual)

    with pytest.raises(ValueError, match="seed identity mismatch"):
        load_checkpoint(checkpoint, expected_seed=8)
    with pytest.raises(ValueError, match="variant identity mismatch"):
        load_checkpoint(checkpoint, expected_variant="B-BC-Mask")


def test_checkpoint_save_is_atomic_on_serialization_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    encoder = FeatureEncoder(16, 16)
    model = ActorCritic(16, 16, 16)
    estimator = ProgressEstimator(16)
    checkpoint = save_checkpoint(
        tmp_path / "policy.pt",
        model,
        estimator,
        encoder,
        seed=7,
        variant="test",
        metadata={},
    )
    original = checkpoint.read_bytes()

    def fail_after_partial_write(_payload: object, handle: Any) -> None:
        handle.write(b"partial")
        raise RuntimeError("simulated serialization failure")

    monkeypatch.setattr(torch, "save", fail_after_partial_write)
    with pytest.raises(RuntimeError, match="simulated serialization failure"):
        save_checkpoint(
            checkpoint,
            model,
            estimator,
            encoder,
            seed=7,
            variant="test",
            metadata={},
        )

    assert checkpoint.read_bytes() == original
    assert not list(tmp_path.glob(".policy.pt.*.tmp"))


def test_checkpoint_save_does_not_unlink_reused_temporary_path_after_replace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "policy.pt"
    encoder = FeatureEncoder(16, 16)
    model = ActorCritic(16, 16, 16)
    estimator = ProgressEstimator(16)
    reused_paths: list[Path] = []
    real_replace = training.os.replace

    def replace_and_reuse(source: str | Path, target: str | Path) -> None:
        real_replace(source, target)
        reused = Path(source)
        reused.write_bytes(b"owned by another writer\n")
        reused_paths.append(reused)

    monkeypatch.setattr(training.os, "replace", replace_and_reuse)

    save_checkpoint(
        destination,
        model,
        estimator,
        encoder,
        seed=7,
        variant="test",
        metadata={},
    )

    assert len(reused_paths) == 1
    assert reused_paths[0].read_bytes() == b"owned by another writer\n"


def test_checkpoint_round_trip_preserves_progress_hidden_dim(tmp_path: Path) -> None:
    encoder = FeatureEncoder(16, 16)
    model = ActorCritic(16, 16, 16)
    estimator = ProgressEstimator(16, hidden_dim=8).freeze()

    checkpoint = save_checkpoint(
        tmp_path / "policy.pt",
        model,
        estimator,
        encoder,
        seed=7,
        variant="test",
        metadata={},
    )
    _, restored, _, _ = load_checkpoint(checkpoint)

    assert restored.hidden_dim == 8
    for expected, actual in zip(estimator.parameters(), restored.parameters(), strict=True):
        assert torch.equal(expected, actual)


def test_random_progress_estimator_is_not_required_for_sparse_episode() -> None:
    config = _smoke_config()
    tasks = generate_tasks("test", 1, base_seed=99)
    encoder = FeatureEncoder(16, 16)
    model, _ = train_behavior_policy(tasks, encoder, config, seed=3)
    estimator = ProgressEstimator(16).freeze()

    result = run_episode(
        model,
        estimator,
        tasks[0],
        encoder,
        config.reward,
        trajectory_id="sparse",
        use_action_mask=True,
        use_progress_reward=False,
        deterministic=True,
    )

    assert result.records[-1].done
    assert all(item["executed_valid"] for item in result.trace["executed_actions"])
    assert all(record.progress_reward == 0.0 for record in result.records)


def test_rollout_schedule_is_seeded_and_covers_all_families() -> None:
    tasks = generate_tasks("train", 100, base_seed=777)

    first = stratified_task_schedule(tasks, episodes=25, seed=9)
    second = stratified_task_schedule(tasks, episodes=25, seed=9)

    assert [task.case_id for task in first] == [task.case_id for task in second]
    assert {task.family for task in first} == {task.family for task in tasks}
