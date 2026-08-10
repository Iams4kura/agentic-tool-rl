from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

import pytest
import torch
from torch import Tensor

import agentic_tool_rl.training as training
from agentic_tool_rl.config import ExperimentConfig, load_config
from agentic_tool_rl.contracts import Observation, ToolCall, WorkflowTask
from agentic_tool_rl.envs import TransactionalWorkflowEnv, generate_tasks
from agentic_tool_rl.features import FeatureEncoder
from agentic_tool_rl.models import ActionSample, ActorCritic, ProgressEstimator
from agentic_tool_rl.rewards import discounted_shaping_sum


class _ScriptedActor(ActorCritic):
    def __init__(self, state_dim: int, action_dim: int, action_indices: Sequence[int]) -> None:
        super().__init__(state_dim, action_dim, hidden_dim=16)
        self._action_indices = iter(action_indices)

    @torch.no_grad()
    def act(
        self,
        state_features: Tensor,
        action_features: Tensor,
        action_mask: Tensor,
        *,
        deterministic: bool = False,
    ) -> ActionSample:
        del deterministic
        action_index = next(self._action_indices)
        assert bool(action_mask[0, action_index])
        batch_size, candidate_count = action_mask.shape
        assert batch_size == 1
        device = state_features.device
        return ActionSample(
            action_index=torch.tensor([action_index], dtype=torch.long, device=device),
            log_prob=torch.zeros(batch_size, device=device),
            value=torch.zeros(batch_size, device=device),
            entropy=torch.zeros(batch_size, device=device),
            logits=torch.zeros((batch_size, candidate_count), device=device),
        )


def _calls_match(left: ToolCall, right: ToolCall) -> bool:
    return left.tool_name == right.tool_name and left.arguments == right.arguments


def _oracle_action_indices(task: WorkflowTask) -> list[int]:
    environment = TransactionalWorkflowEnv(task)
    result: list[int] = []
    for oracle_call in task.oracle_plans[0]:
        candidates = environment.candidate_actions(include_invalid=True)
        action_index = next(
            index
            for index, candidate in enumerate(candidates)
            if _calls_match(candidate, oracle_call)
        )
        result.append(action_index)
        outcome = environment.step(candidates[action_index])
        assert outcome.accepted
    assert environment.evaluate().success
    return result


def _timeout_action_indices(task: WorkflowTask) -> list[int]:
    """Build a deterministic rejected-action script that reaches timeout."""

    environment = TransactionalWorkflowEnv(task)
    result: list[int] = []
    while not environment.done:
        candidates = environment.candidate_actions(include_invalid=True)
        action_index = next(
            index
            for index, candidate in enumerate(candidates)
            if not environment.dry_run(candidate).valid
        )
        result.append(action_index)
        outcome = environment.step(candidates[action_index])
        assert not outcome.accepted
    assert environment.evaluate().success is False
    return result


def _smoke_config() -> ExperimentConfig:
    config = load_config("configs/smoke.yaml")
    assert isinstance(config, ExperimentConfig)
    return config


def test_terminal_progress_boundary_skips_estimator_and_full_episode_telescopes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _smoke_config()
    task = generate_tasks("test", 1, base_seed=916)[0]
    encoder = FeatureEncoder(16, 16)
    model = _ScriptedActor(16, 16, _oracle_action_indices(task))
    estimator = ProgressEstimator(16).freeze()
    estimator_calls: list[int] = []

    def constant_nonterminal_probability(
        _estimator: ProgressEstimator | None,
        _encoder: FeatureEncoder,
        observation: Observation,
    ) -> float:
        assert not observation.done
        estimator_calls.append(observation.step_index)
        return 0.5

    monkeypatch.setattr(training, "_progress_probability", constant_nonterminal_probability)
    gamma = 0.91
    result = training.run_episode(
        model,
        estimator,
        task,
        encoder,
        config.reward,
        trajectory_id="terminal-potential-boundary",
        use_action_mask=True,
        use_progress_reward=True,
        deterministic=True,
        gamma=gamma,
    )

    assert result.trace["success"] is True
    assert len(estimator_calls) == len(result.records)
    assert result.records[-1].done
    for record in result.records[:-1]:
        potential = record.info["progress_potential"]
        assert potential == {
            "phi_raw": 0.5,
            "phi_next_raw": 0.5,
            "phi_next_used": 0.5,
            "terminal_zeroed": False,
            "phi_current": 0.5,
            "terminal_boundary_applied": False,
        }
    assert result.records[-1].info["progress_potential"] == {
        "phi_raw": 0.5,
        "phi_next_raw": None,
        "phi_next_used": 0.0,
        "terminal_zeroed": True,
        "phi_current": 0.5,
        "terminal_boundary_applied": True,
    }

    shaping_terms = torch.tensor(
        [record.progress_reward for record in result.records], dtype=torch.float64
    )
    expected = -config.reward.progress_beta * 0.5
    observed = float(discounted_shaping_sum(shaping_terms, gamma=gamma).item())
    assert observed == pytest.approx(expected, abs=1e-7)


def test_timeout_progress_boundary_skips_terminal_estimator_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _smoke_config()
    task = generate_tasks("test", 1, base_seed=919)[0]
    encoder = FeatureEncoder(16, 16)
    model = _ScriptedActor(16, 16, _timeout_action_indices(task))
    estimator = ProgressEstimator(16).freeze()
    estimator_calls: list[int] = []

    def constant_nonterminal_probability(
        _estimator: ProgressEstimator | None,
        _encoder: FeatureEncoder,
        observation: Observation,
    ) -> float:
        assert not observation.done
        estimator_calls.append(observation.step_index)
        return 0.4

    monkeypatch.setattr(training, "_progress_probability", constant_nonterminal_probability)
    result = training.run_episode(
        model,
        estimator,
        task,
        encoder,
        config.reward,
        trajectory_id="timeout-potential-boundary",
        use_action_mask=False,
        use_progress_reward=True,
        deterministic=True,
    )

    assert result.trace["success"] is False
    assert len(estimator_calls) == len(result.records)
    assert result.records[-1].info["timed_out"] is True
    assert result.records[-1].info["progress_potential"] == {
        "phi_raw": 0.4,
        "phi_next_raw": None,
        "phi_next_used": 0.0,
        "terminal_zeroed": True,
        "phi_current": 0.4,
        "terminal_boundary_applied": True,
    }


def test_progress_task_subset_is_deterministic_and_balanced_across_all_strata() -> None:
    tasks = generate_tasks("train", 1000, base_seed=917)

    selected = training._balanced_progress_tasks(tasks, max_tasks=256, seed=23)
    reordered = training._balanced_progress_tasks(
        list(reversed(tasks)), max_tasks=256, seed=23
    )

    assert [task.case_id for task in selected] == [task.case_id for task in reordered]
    assert len(selected) == 256
    counts = Counter(
        (task.family, task.difficulty, task.topology) for task in selected
    )
    all_strata = {
        (task.family, task.difficulty, task.topology) for task in tasks
    }
    assert set(counts) == all_strata
    assert max(counts.values()) - min(counts.values()) <= 1


def test_progress_examples_include_nonterminal_and_terminal_done_bits() -> None:
    tasks = generate_tasks("train", 12, base_seed=918)
    encoder = FeatureEncoder(16, 16)

    features, labels = training._monte_carlo_progress_examples(
        tasks,
        encoder,
        seed=29,
        max_tasks=12,
    )

    done_bits = features[:, 2]
    assert set(done_bits.tolist()) == {0.0, 1.0}
    assert int((done_bits == 0).sum()) == int((done_bits == 1).sum())
    assert features.shape[0] == labels.shape[0]
    assert set(labels.tolist()) == {0.0, 1.0}
