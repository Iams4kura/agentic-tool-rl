"""Deterministic learner canary kept separate from structural smoke verification."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

import torch

from agentic_tool_rl.config import ExperimentConfig, load_packaged_config
from agentic_tool_rl.contracts import Observation, ToolCall, WorkflowTask
from agentic_tool_rl.envs import (
    TransactionalWorkflowEnv,
    generate_tasks,
    verify_task_solvable,
)
from agentic_tool_rl.evaluation import compute_metrics
from agentic_tool_rl.features import FeatureEncoder
from agentic_tool_rl.grounding import ActionMask
from agentic_tool_rl.models import ActorCritic, ProgressEstimator
from agentic_tool_rl.policy_input import PolicyInput
from agentic_tool_rl.rewards import RewardBreakdown, potential_shaping
from agentic_tool_rl.rollout import StepRecord
from agentic_tool_rl.training import (
    run_episode,
    train_behavior_policy,
    train_ppo_variant,
)

_B_VARIANT = "B-BC-Mask"
_E_VARIANT = "E-PPO-Progress-Mask"
_CANARY_CASES = 8
_CANARY_REPETITIONS = 20
_CANARY_SEED = 123
_CANARY_TASK_SEED = 424_242
_MINIMUM_ACTION_SET_ACCURACY = 0.95
_MINIMUM_BC_TSR = 0.75


class SemanticCanaryError(RuntimeError):
    """Raised when a deterministic learner acceptance condition fails."""


def validate_semantic_canary(
    runs: Mapping[str, Mapping[str, object]],
    *,
    bc_action_set_accuracy: float,
    oracle_tsr: float,
    trace_consistency: Mapping[str, bool],
    repetitions: int,
) -> dict[str, Any]:
    """Validate fixed engineering semantics without encoding an E-vs-B claim."""

    normalized: dict[str, dict[str, int]] = {}
    for variant in (_B_VARIANT, _E_VARIANT):
        row = runs.get(variant)
        if row is None:
            raise SemanticCanaryError(f"semantic canary is missing required variant {variant}")
        cases = row.get("cases")
        successes = row.get("successes")
        if (
            isinstance(cases, bool)
            or not isinstance(cases, int)
            or cases != _CANARY_CASES
            or isinstance(successes, bool)
            or not isinstance(successes, int)
            or not 0 <= successes <= cases
        ):
            raise SemanticCanaryError(
                f"semantic canary requires {_CANARY_CASES} valid cases for {variant}"
            )
        independent_training_repetitions = row.get("independent_training_repetitions")
        if (
            isinstance(independent_training_repetitions, bool)
            or not isinstance(independent_training_repetitions, int)
            or independent_training_repetitions != repetitions
        ):
            raise SemanticCanaryError(
                f"semantic canary requires {repetitions} independent training "
                f"repetitions for {variant}"
            )
        normalized[variant] = {"cases": cases, "successes": successes}

    total_successes = sum(row["successes"] for row in normalized.values())
    if total_successes == 0:
        raise SemanticCanaryError("B/E semantic canary both have zero successes")
    bc_tsr = normalized[_B_VARIANT]["successes"] / _CANARY_CASES
    checks = {
        "fixed_eight_tasks": all(
            row["cases"] == _CANARY_CASES for row in normalized.values()
        ),
        "oracle_tsr_eq_1": oracle_tsr == 1.0,
        "bc_action_set_accuracy_gte_0_95": (
            bc_action_set_accuracy >= _MINIMUM_ACTION_SET_ACCURACY
        ),
        "bc_tsr_gte_0_75": bc_tsr >= _MINIMUM_BC_TSR,
        "reward_consistency": trace_consistency.get("reward") is True,
        "done_consistency": trace_consistency.get("done") is True,
        "metrics_consistency": trace_consistency.get("metrics") is True,
        "replay_consistency": trace_consistency.get("replay") is True,
        "twenty_independent_training_repetitions": repetitions
        >= _CANARY_REPETITIONS,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise SemanticCanaryError(
            f"deterministic semantic canary failed checks: {', '.join(failed)}"
        )
    return {
        "passed": True,
        "criterion": "deterministic_engineering_canary_v2",
        "checks": checks,
        "thresholds": {
            "bc_action_set_accuracy": _MINIMUM_ACTION_SET_ACCURACY,
            "bc_tsr": _MINIMUM_BC_TSR,
            "oracle_tsr": 1.0,
            "deterministic_repetitions": _CANARY_REPETITIONS,
        },
        "requires_e_greater_than_b": False,
    }


def _canary_config(source: ExperimentConfig) -> ExperimentConfig:
    behavior_cloning = source.training.behavior_cloning.model_copy(
        update={"epochs": 25, "batch_size": 128, "learning_rate": 0.008}
    )
    ppo = source.training.ppo.model_copy(
        update={
            "iterations": 1,
            "epochs": 1,
            "rollout_episodes": 8,
            "minibatch_size": 128,
        }
    )
    return source.model_copy(
        update={
            "policy": source.policy.model_copy(
                update={"observation_dim": 64, "action_dim": 64, "hidden_dim": 64}
            ),
            "training": source.training.model_copy(
                update={"behavior_cloning": behavior_cloning, "ppo": ppo}
            ),
        }
    )


def _calls_match(left: ToolCall, right: ToolCall) -> bool:
    return left.tool_name == right.tool_name and left.arguments == right.arguments


def _bc_action_set_accuracy(
    model: ActorCritic,
    tasks: Sequence[WorkflowTask],
    encoder: FeatureEncoder,
) -> float:
    """Measure whether greedy BC actions belong to the full optimal ready set."""

    correct = 0
    total = 0
    for task in tasks:
        environment = TransactionalWorkflowEnv(task)
        observation = environment.reset()
        action_mask = ActionMask(task)
        for oracle_call in task.oracle_plans[0]:
            candidates = environment.candidate_actions(include_invalid=True)
            selection_mask = action_mask.mask(observation, candidates)
            policy_input = PolicyInput.from_decision(
                observation,
                candidates,
                selection_mask,
                task.tool_schemas,
            )
            features = encoder.encode_policy_input(policy_input)
            states, actions, masks = features.as_batch()
            prediction = int(
                model.act(states, actions, masks, deterministic=True).action_index.item()
            )
            completed = set(environment.state.get("completed_nodes", []))
            positives = {
                index
                for index, candidate in enumerate(candidates)
                if (
                    (decision := environment.dry_run(candidate)).valid
                    and decision.node_id is not None
                    and decision.node_id not in completed
                )
            }
            if not positives:
                raise SemanticCanaryError(
                    f"canary state has no optimal action set for {task.case_id}"
                )
            correct += prediction in positives
            total += 1
            oracle_index = next(
                index
                for index, candidate in enumerate(candidates)
                if _calls_match(candidate, oracle_call)
            )
            outcome = environment.step(candidates[oracle_index])
            if not outcome.accepted:
                raise SemanticCanaryError(f"canary oracle failed for {task.case_id}")
            observation = outcome.observation
    if total == 0:
        raise SemanticCanaryError("canary action-set evaluation is empty")
    return correct / total


def _model_sha256(model: ActorCritic) -> str:
    """Hash tensor values without relying on nondeterministic checkpoint bytes."""

    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
        digest.update(b"\0")
        digest.update(value.numpy().tobytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _progress_probability(
    estimator: ProgressEstimator,
    encoder: FeatureEncoder,
    observation: Observation,
) -> float:
    with torch.no_grad():
        encoded = encoder.encode_state(observation)
        return float(estimator.probabilities(encoded.unsqueeze(0)).item())


def _canonical_tool_calls(records: Sequence[StepRecord]) -> bytes:
    return json.dumps(
        [ToolCall.model_validate(record.action).model_dump(mode="json") for record in records],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _replay_serialized_tool_calls(
    records: Sequence[StepRecord],
    task: WorkflowTask,
    estimator: ProgressEstimator,
    encoder: FeatureEncoder,
    config: ExperimentConfig,
    *,
    use_progress_reward: bool,
) -> tuple[dict[str, Any], dict[str, bool], str]:
    """Decode JSON ToolCalls and replay them in a fresh environment.

    Reward components are recomputed from the replayed transaction outcomes and
    freshly evaluated progress potentials.  No rollout environment or ToolCall
    object is reused.
    """

    encoded_calls = _canonical_tool_calls(records)
    raw_calls = json.loads(encoded_calls)
    if not isinstance(raw_calls, list):
        raise SemanticCanaryError("serialized ToolCall payload must be a list")
    calls = [ToolCall.model_validate(item) for item in raw_calls]
    if len(calls) != len(records):
        raise SemanticCanaryError("serialized ToolCall cardinality changed")

    environment = TransactionalWorkflowEnv(task)
    observation = environment.reset()
    phi = (
        _progress_probability(estimator, encoder, observation)
        if use_progress_reward
        else 0.0
    )
    reward_consistent = True
    done_consistent = True
    replay_consistent = True
    replay_steps: list[dict[str, bool]] = []
    for index, (record, call) in enumerate(zip(records, calls, strict=True)):
        replay_consistent &= observation == record.observation
        candidates = environment.candidate_actions(include_invalid=True)
        if (
            record.step_index != index
            or record.action_index >= len(candidates)
            or call != candidates[record.action_index]
        ):
            replay_consistent = False
        validation = environment.dry_run(call)
        outcome = environment.step(call)
        replay_consistent &= outcome.observation == record.next_observation
        phi_next = 0.0
        if use_progress_reward and not outcome.done:
            phi_next = _progress_probability(
                estimator,
                encoder,
                outcome.observation,
            )
        progress_reward = 0.0
        if use_progress_reward:
            progress_reward = float(
                potential_shaping(
                    phi,
                    phi_next,
                    gamma=config.training.ppo.gamma,
                    beta=config.reward.progress_beta,
                ).item()
            )
        breakdown = RewardBreakdown(
            task=(
                config.reward.task_success
                if outcome.success
                else config.reward.task_failure
                if outcome.done
                else 0.0
            ),
            progress=progress_reward,
            step=config.reward.step_cost,
            invalid=0.0 if outcome.accepted else config.reward.invalid_action,
            forbidden=(
                config.reward.forbidden_side_effect
                if environment.evaluate().forbidden_side_effect_count
                else 0.0
            ),
        )
        recorded_components = record.info.get("reward_components")
        if not isinstance(recorded_components, Mapping):
            reward_consistent = False
        else:
            expected_components = {
                "task": breakdown.task,
                "progress": breakdown.progress,
                "step": breakdown.step,
                "invalid": breakdown.invalid,
                "forbidden": breakdown.forbidden,
            }
            reward_consistent &= set(recorded_components) == set(expected_components)
            reward_consistent &= all(
                abs(float(recorded_components.get(name, float("inf"))) - expected) <= 1e-9
                for name, expected in expected_components.items()
            )
        reward_consistent &= abs(float(record.reward) - breakdown.total) <= 1e-9
        done_consistent &= record.done == outcome.done
        replay_consistent &= record.valid_label == validation.valid
        replay_steps.append({"valid_label": validation.valid})
        observation = outcome.observation
        phi = phi_next

    evaluation = environment.evaluate()
    done_consistent &= bool(records) and all(not record.done for record in records[:-1])
    done_consistent &= bool(records) and records[-1].done and environment.done
    replay_consistent &= environment.done
    replay_trace: dict[str, Any] = {
        "schema_version": "semantic-canary-replay-v1",
        "case_id": task.case_id,
        "family": task.family,
        "success": evaluation.success,
        "goal_predicates_satisfied": evaluation.goal_predicates_satisfied,
        "forbidden_side_effect_count": evaluation.forbidden_side_effect_count,
        "optimal_steps": task.optimal_steps,
        "max_steps": task.max_steps,
        "step_count": len(records),
        "simulated_latency_s": environment.simulated_latency_s,
        "steps": replay_steps,
        "final_state": environment.state_snapshot(),
    }
    return (
        replay_trace,
        {
            "reward": reward_consistent,
            "done": done_consistent,
            "replay": replay_consistent,
        },
        hashlib.sha256(encoded_calls).hexdigest(),
    )


def _evaluate_canary_once(
    model: ActorCritic,
    estimator: ProgressEstimator,
    *,
    config: ExperimentConfig,
    encoder: FeatureEncoder,
    tasks: Sequence[WorkflowTask],
    variant: str,
    use_progress_reward: bool,
    model_sha256: str,
) -> tuple[dict[str, int | float | bool | str], dict[str, bool]]:
    traces: list[dict[str, Any]] = []
    replay_traces: list[dict[str, Any]] = []
    call_hashes: list[str] = []
    successes = 0
    reward_consistent = True
    done_consistent = True
    replay_consistent = True
    for task in tasks:
        result = run_episode(
            model,
            estimator,
            task,
            encoder,
            config.reward,
            trajectory_id=f"semantic-canary-{variant}-{task.case_id}",
            use_action_mask=True,
            use_progress_reward=use_progress_reward,
            deterministic=True,
            gamma=config.training.ppo.gamma,
            trace_mode="compact",
        )
        traces.append(result.trace)
        successes += int(bool(result.trace["success"]))
        replay_trace, consistency, call_hash = _replay_serialized_tool_calls(
            result.records,
            task,
            estimator,
            encoder,
            config,
            use_progress_reward=use_progress_reward,
        )
        replay_traces.append(replay_trace)
        call_hashes.append(call_hash)
        reward_consistent &= consistency["reward"]
        done_consistent &= consistency["done"]
        replay_consistent &= consistency["replay"]
        replay_consistent &= replay_trace["success"] == result.trace["success"]
        replay_consistent &= replay_trace["final_state"] == result.trace["final_state"]
    metrics = compute_metrics(
        traces,
        timeout_s=config.evaluation.timeout_s,
        bootstrap_samples=0,
        bootstrap_seed=config.evaluation.bootstrap_seed,
        confidence=config.evaluation.confidence,
    )
    replay_metrics = compute_metrics(
        replay_traces,
        timeout_s=config.evaluation.timeout_s,
        bootstrap_samples=0,
        bootstrap_seed=config.evaluation.bootstrap_seed,
        confidence=config.evaluation.confidence,
    )
    metrics_consistent = metrics.to_dict() == replay_metrics.to_dict()
    encoded_trace = json.dumps(
        traces,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    encoded_calls = json.dumps(call_hashes, separators=(",", ":")).encode("ascii")
    return (
        {
            "cases": len(tasks),
            "successes": successes,
            "tsr": successes / len(tasks),
            "trace_sha256": hashlib.sha256(encoded_trace).hexdigest(),
            "serialized_tool_calls_sha256": hashlib.sha256(encoded_calls).hexdigest(),
            "model_sha256": model_sha256,
        },
        {
            "reward": reward_consistent,
            "done": done_consistent,
            "metrics": metrics_consistent,
            "replay": replay_consistent,
        },
    )


def run_semantic_canary(
    config: ExperimentConfig | None = None,
    *,
    repetitions: int = _CANARY_REPETITIONS,
) -> dict[str, Any]:
    """Overfit eight fixed tasks and enforce deterministic engineering semantics."""

    if repetitions < _CANARY_REPETITIONS:
        raise ValueError(f"semantic canary requires at least {_CANARY_REPETITIONS} repetitions")
    if config is None:
        packaged = load_packaged_config("smoke.yaml")
        if not isinstance(packaged, ExperimentConfig):
            raise SemanticCanaryError("packaged smoke config is not an experiment config")
        config = packaged
    resolved = _canary_config(config)
    repetition_runs: dict[str, list[dict[str, int | float | bool | str]]] = {
        _B_VARIANT: [],
        _E_VARIANT: [],
    }
    trace_consistency = {name: True for name in ("reward", "done", "metrics", "replay")}
    action_set_accuracies: list[float] = []
    bc_updates: list[int] = []
    bc_training_accuracies: list[float] = []
    ppo_updates: list[int] = []
    rollout_step_counts: list[int] = []
    task_hashes: set[str] = set()
    tasks: Sequence[WorkflowTask] = ()
    for _ in range(repetitions):
        tasks = generate_tasks("train", _CANARY_CASES, base_seed=_CANARY_TASK_SEED)
        encoded_tasks = json.dumps(
            [task.model_dump(mode="json") for task in tasks],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        task_hashes.add(hashlib.sha256(encoded_tasks).hexdigest())
        encoder = FeatureEncoder(
            resolved.policy.observation_dim,
            resolved.policy.action_dim,
        )
        bc_model, bc_metrics = train_behavior_policy(
            tasks,
            encoder,
            resolved,
            seed=_CANARY_SEED,
        )
        action_set_accuracies.append(_bc_action_set_accuracy(bc_model, tasks, encoder))
        bc_updates.append(bc_metrics.updates)
        bc_training_accuracies.append(bc_metrics.accuracy)
        estimator = ProgressEstimator(encoder.state_dim, hidden_dim=64).freeze()
        e_model, ppo_metrics, rollout_steps = train_ppo_variant(
            bc_model,
            estimator,
            tasks,
            encoder,
            resolved,
            seed=_CANARY_SEED,
            use_action_mask=True,
            use_progress_reward=True,
        )
        ppo_updates.append(ppo_metrics.updates)
        rollout_step_counts.append(rollout_steps)
        for variant, model, use_progress_reward in (
            (_B_VARIANT, bc_model, False),
            (_E_VARIANT, e_model, True),
        ):
            run, consistency = _evaluate_canary_once(
                model,
                estimator,
                config=resolved,
                encoder=encoder,
                tasks=tasks,
                variant=variant,
                use_progress_reward=use_progress_reward,
                model_sha256=_model_sha256(model),
            )
            repetition_runs[variant].append(run)
            for name in trace_consistency:
                trace_consistency[name] &= consistency[name]
    if len(task_hashes) != 1:
        raise SemanticCanaryError("fixed canary task generation changed across repetitions")

    runs: dict[str, dict[str, int | float | bool | str]] = {}
    for variant, rows in repetition_runs.items():
        success_counts = {int(row["successes"]) for row in rows}
        trace_hashes = {str(row["trace_sha256"]) for row in rows}
        call_hashes = {str(row["serialized_tool_calls_sha256"]) for row in rows}
        model_hashes = {str(row["model_sha256"]) for row in rows}
        if len(success_counts) != 1:
            raise SemanticCanaryError(f"{variant} success count changed across training runs")
        runs[variant] = {
            "cases": _CANARY_CASES,
            "successes": next(iter(success_counts)),
            "tsr": next(iter(success_counts)) / _CANARY_CASES,
            "independent_training_repetitions": repetitions,
            "unique_trace_hashes": len(trace_hashes),
            "unique_model_hashes": len(model_hashes),
            "unique_serialized_tool_call_hashes": len(call_hashes),
            "trace_sha256": next(iter(trace_hashes)),
            "model_sha256": next(iter(model_hashes)),
            "serialized_tool_calls_sha256": next(iter(call_hashes)),
        }
        trace_consistency["replay"] &= (
            len(trace_hashes) == len(model_hashes) == len(call_hashes) == 1
        )
    if len(set(action_set_accuracies)) != 1:
        raise SemanticCanaryError("BC action-set accuracy changed across training runs")
    action_set_accuracy = action_set_accuracies[0]
    oracle_reports = [verify_task_solvable(task) for task in tasks]
    oracle_successes = sum(
        report.solvable and report.executed_steps == task.optimal_steps
        for report, task in zip(oracle_reports, tasks, strict=True)
    )
    oracle_tsr = oracle_successes / len(tasks)
    gate = validate_semantic_canary(
        runs,
        bc_action_set_accuracy=action_set_accuracy,
        oracle_tsr=oracle_tsr,
        trace_consistency=trace_consistency,
        repetitions=repetitions,
    )
    return {
        "schema_version": "2.0",
        "deterministic": True,
        "task_seed": _CANARY_TASK_SEED,
        "training_seed": _CANARY_SEED,
        "fixed_cases": len(tasks),
        "deterministic_repetitions": repetitions,
        "independent_training_repetitions": repetitions,
        "unique_task_hashes": len(task_hashes),
        "oracle": {"cases": len(tasks), "successes": oracle_successes, "tsr": oracle_tsr},
        "runs": runs,
        "training": {
            "bc_action_set_accuracy": action_set_accuracy,
            "bc_training_accuracy": bc_training_accuracies[0],
            "bc_updates": bc_updates[0],
            "progress_estimator": "fixed_frozen_boundary_canary",
            "ppo_updates": ppo_updates[0],
            "rollout_steps": rollout_step_counts[0],
        },
        "trace_consistency": trace_consistency,
        "gate": gate,
    }


__all__ = [
    "SemanticCanaryError",
    "run_semantic_canary",
    "validate_semantic_canary",
]
