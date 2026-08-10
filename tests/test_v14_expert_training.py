from __future__ import annotations

import json
from collections.abc import Callable
from copy import deepcopy
from dataclasses import replace
from functools import cache
from math import inf
from typing import Any

import pytest
import torch

import agentic_tool_rl.training as training
from agentic_tool_rl.algorithms import BCMetrics, ExpertBatch, PolicyBatch
from agentic_tool_rl.config import ExperimentConfig, load_config
from agentic_tool_rl.contracts import CounterfactualWorkflowTask, Split, ToolCall
from agentic_tool_rl.envs import TransactionalWorkflowEnv, generate_tasks
from agentic_tool_rl.envs.benchmark_v14 import (
    generate_counterfactual_tasks,
    optimal_candidate_indices,
)
from agentic_tool_rl.envs.workflow import predicate_holds
from agentic_tool_rl.features import FeatureEncoder
from agentic_tool_rl.grounding import ActionMask


def _smoke_config() -> ExperimentConfig:
    config = load_config("configs/smoke.yaml")
    assert isinstance(config, ExperimentConfig)
    return config


def _matching_index(candidates: list[ToolCall], expected: ToolCall) -> int:
    return next(
        index
        for index, candidate in enumerate(candidates)
        if candidate.tool_name == expected.tool_name
        and candidate.arguments == expected.arguments
    )


def test_v14_expert_batch_contains_every_optimal_candidate_per_state() -> None:
    tasks = generate_counterfactual_tasks(Split.TRAIN, 1, base_seed=20260810)[:1]
    encoder = FeatureEncoder(32, 32)

    batch = training.collect_counterfactual_expert_batch(tasks, encoder)

    batch.validate()
    expected_rows: list[tuple[str, list[bool], set[int], int]] = []
    multiple_positive_rows = 0
    for task in tasks:
        paths = training._deterministic_shortest_oracle_paths(task)
        operation_paths = [
            tuple(str(call.arguments["operation_id"]) for call in path)
            for path in paths
        ]
        assert len(paths) == training.V14_EXPERT_PATHS_PER_TASK == 4
        assert len(set(operation_paths)) == len(paths)
        assert all(len(path) == task.optimal_steps for path in paths)
        public_mask = ActionMask(task.tool_schemas)
        seen_hashes: set[str] = set()
        for path in paths:
            environment = TransactionalWorkflowEnv(task)
            for oracle_call in path:
                observation = environment.observe()
                candidates = environment.candidate_actions(include_invalid=True)
                selection_mask = public_mask.mask(observation, candidates)
                expected = optimal_candidate_indices(
                    task,
                    environment.state.get("completed_nodes", []),
                    candidates,
                )
                policy_input = training.PolicyInput.from_decision(
                    observation,
                    candidates,
                    selection_mask,
                    task.tool_schemas,
                )
                state_sha256 = training._canonical_visible_state_sha256(policy_input)
                if state_sha256 not in seen_hashes:
                    seen_hashes.add(state_sha256)
                    expected_rows.append(
                        (state_sha256, selection_mask, expected, len(candidates))
                    )
                    multiple_positive_rows += len(expected) >= 2

                outcome = environment.step(
                    candidates[_matching_index(candidates, oracle_call)]
                )
                assert outcome.accepted
            assert environment.evaluate().success

    assert batch.source_path_counts == ((tasks[0].case_id, 4),)
    assert list(batch.visible_state_sha256s) == [row[0] for row in expected_rows]
    assert set(batch.source_case_ids) == {tasks[0].case_id}
    assert len(set(batch.visible_state_sha256s)) == len(batch.visible_state_sha256s)
    assert len(expected_rows) < 4 * tasks[0].optimal_steps
    assert batch.states.shape[0] == len(expected_rows)
    for row_index, (_digest, expected_mask, expected, candidate_count) in enumerate(
        expected_rows
    ):
        observed = set(
            torch.nonzero(batch.positive_mask[row_index], as_tuple=False)
            .flatten()
            .tolist()
        )
        assert observed == expected
        assert batch.candidate_mask[row_index, :candidate_count].tolist() == expected_mask
        assert not bool(batch.candidate_mask[row_index, candidate_count:].any())
        assert not bool(batch.positive_mask[row_index, candidate_count:].any())
    assert multiple_positive_rows / len(expected_rows) >= 0.20


def test_expert_batch_rejects_duplicate_visible_state_hashes() -> None:
    task = generate_counterfactual_tasks(
        Split.TRAIN, 1, base_seed=20260810
    )[0]
    batch = training.collect_counterfactual_expert_batch(
        [task], FeatureEncoder(16, 16)
    )
    duplicate_hashes = (
        batch.visible_state_sha256s[0],
    ) * len(batch.visible_state_sha256s)

    with pytest.raises(ValueError, match="unique by visible-state"):
        replace(batch, visible_state_sha256s=duplicate_hashes).validate()


def _exhaustive_goal_distance(
    task: CounterfactualWorkflowTask,
    state: dict[str, Any],
    *,
    remaining_steps: int,
) -> int | float:
    """Exhaust legal mutations within the real step budget, without an oracle."""

    if remaining_steps < 0:
        raise ValueError("remaining_steps must be non-negative")

    nodes = {node.node_id: node for node in task.workflow_nodes}

    @cache
    def visit(encoded_state: str, budget: int) -> int | float:
        current = json.loads(encoded_state)
        if all(
            predicate_holds(current, predicate)
            for predicate in task.hidden_goal_predicates
        ) and not any(
            effect in task.forbidden_side_effects
            for effect in current.get("side_effects", [])
        ):
            return 0
        if budget == 0:
            return inf
        completed = set(current.get("completed_nodes", []))
        best: int | float = inf
        for node in nodes.values():
            if node.node_id in completed or not set(
                node.required_predecessors
            ).issubset(completed):
                continue
            next_state = deepcopy(current)
            next_state["completed_nodes"] = sorted((*completed, node.node_id))
            next_state.update(deepcopy(node.effects))
            if node.side_effect is not None:
                next_state["side_effects"] = sorted(
                    (*next_state.get("side_effects", []), node.side_effect)
                )
            encoded_next = json.dumps(next_state, sort_keys=True, separators=(",", ":"))
            distance = visit(encoded_next, budget - 1)
            if distance != inf:
                best = min(best, 1 + distance)
        return best

    canonical_state = deepcopy(state)
    canonical_state["completed_nodes"] = sorted(canonical_state.get("completed_nodes", []))
    canonical_state["side_effects"] = sorted(canonical_state.get("side_effects", []))
    return visit(
        json.dumps(canonical_state, sort_keys=True, separators=(",", ":")),
        remaining_steps,
    )


def test_v14_positive_set_matches_exhaustive_goal_distance_on_small_dag() -> None:
    tasks = generate_counterfactual_tasks(Split.TRAIN, 30, base_seed=20260812)
    task = next(task for task in tasks if task.optimal_steps == 6)
    environment = TransactionalWorkflowEnv(task)
    first_path = training._deterministic_shortest_oracle_paths(task)[0]
    first_candidates = environment.candidate_actions(include_invalid=True)
    first_index = _matching_index(first_candidates, first_path[0])
    assert environment.step(first_candidates[first_index]).accepted

    state = environment.state
    candidates = environment.candidate_actions(include_invalid=True)
    remaining_steps = task.max_steps - environment.steps_taken
    current_distance = _exhaustive_goal_distance(
        task,
        state,
        remaining_steps=remaining_steps,
    )
    assert current_distance != inf
    expected: set[int] = set()
    node_by_id = {node.node_id: node for node in task.workflow_nodes}
    for index, candidate in enumerate(candidates):
        if not environment.dry_run(candidate).valid:
            continue
        operation = candidate.arguments.get("operation_id")
        node = node_by_id.get(str(operation))
        if node is None or node.node_id in state.get("completed_nodes", []):
            next_distance = _exhaustive_goal_distance(
                task,
                state,
                remaining_steps=remaining_steps - 1,
            )
        else:
            next_state = deepcopy(state)
            next_state["completed_nodes"] = sorted(
                (*next_state.get("completed_nodes", []), node.node_id)
            )
            next_state.update(deepcopy(node.effects))
            if node.side_effect is not None:
                next_state["side_effects"] = sorted(
                    (*next_state.get("side_effects", []), node.side_effect)
                )
            next_distance = _exhaustive_goal_distance(
                task,
                next_state,
                remaining_steps=remaining_steps - 1,
            )
        if next_distance == current_distance - 1:
            expected.add(index)

    observed = optimal_candidate_indices(
        task,
        state.get("completed_nodes", []),
        candidates,
    )
    assert observed == expected


def test_behavior_training_routes_v14_and_v13_to_explicit_batch_types(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _smoke_config()
    encoder = FeatureEncoder(config.policy.observation_dim, config.policy.action_dim)
    observed: list[PolicyBatch | ExpertBatch] = []
    original_update: Callable[..., BCMetrics] = training.BehaviorCloningTrainer.update

    def record_update(
        trainer: training.BehaviorCloningTrainer,
        batch: PolicyBatch | ExpertBatch,
    ) -> BCMetrics:
        observed.append(batch)
        return original_update(trainer, batch)

    monkeypatch.setattr(training.BehaviorCloningTrainer, "update", record_update)

    v14_tasks = generate_counterfactual_tasks(Split.TRAIN, 1, base_seed=20260811)
    training.train_behavior_policy(v14_tasks, encoder, config, seed=31)
    v13_tasks = generate_tasks(Split.TRAIN, 2, base_seed=20260811)
    training.train_behavior_policy(v13_tasks, encoder, config, seed=31)

    assert len(observed) == 2
    assert isinstance(observed[0], ExpertBatch)
    assert isinstance(observed[1], PolicyBatch)
    expected_v13 = training.collect_expert_demonstrations(
        v13_tasks, encoder
    ).as_policy_batch()
    assert torch.equal(observed[1].states, expected_v13.states)
    assert torch.equal(observed[1].action_features, expected_v13.action_features)
    assert torch.equal(observed[1].action_masks, expected_v13.action_masks)
    assert torch.equal(observed[1].actions, expected_v13.actions)
