"""Integrated CPU training and rollout pipeline.

This module is the bridge between the synthetic transactional environment and
the generic PyTorch algorithms.  Every policy decision goes through the same
observation → candidates → features → policy → environment path used at
evaluation time.
"""

from __future__ import annotations

import copy
import hashlib
import json
import random
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean
from typing import Any, Literal

import torch

from agentic_tool_rl.algorithms import (
    BCConfig,
    BCMetrics,
    BehaviorCloningTrainer,
    PPOConfig,
    PPOMetrics,
    PPOTrainer,
    SequencePPOTrainer,
)
from agentic_tool_rl.config import ExperimentConfig, RewardConfig
from agentic_tool_rl.contracts import Observation, ToolCall, WorkflowTask
from agentic_tool_rl.envs import TransactionalWorkflowEnv
from agentic_tool_rl.evaluation import TraceStore
from agentic_tool_rl.features import FeatureEncoder
from agentic_tool_rl.grounding import ActionMask
from agentic_tool_rl.models import ActorCritic, ProgressEstimator
from agentic_tool_rl.models.progress_estimator import (
    ProgressMetrics,
    compute_progress_metrics,
)
from agentic_tool_rl.rewards import RewardBreakdown, potential_shaping
from agentic_tool_rl.rollout import RolloutBuffer, StepRecord

CreditAssignment = Literal["action", "sequence"]
TraceMode = Literal["full", "compact"]


@dataclass(frozen=True)
class ProgressTrainingResult:
    train: ProgressMetrics
    dev: ProgressMetrics
    examples: int


@dataclass(frozen=True)
class TrainingResult:
    variant: str
    seed: int
    bc: BCMetrics
    progress: ProgressTrainingResult
    ppo: PPOMetrics | None
    rollout_steps: int
    checkpoint: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EpisodeResult:
    trace: dict[str, Any]
    records: tuple[StepRecord, ...]


def set_reproducible_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def _calls_match(left: ToolCall, right: ToolCall) -> bool:
    return left.tool_name == right.tool_name and left.arguments == right.arguments


def _oracle_index(candidates: Sequence[ToolCall], oracle_call: ToolCall) -> int:
    for index, candidate in enumerate(candidates):
        if _calls_match(candidate, oracle_call):
            return index
    raise RuntimeError(f"oracle call {oracle_call.tool_name!r} is not in candidate set")


def collect_expert_demonstrations(
    tasks: Sequence[WorkflowTask],
    encoder: FeatureEncoder,
    *,
    max_tasks: int | None = None,
) -> RolloutBuffer:
    """Replay real oracle calls and collate dynamic-candidate BC examples."""

    selected_tasks = tasks if max_tasks is None else tasks[:max_tasks]
    if not selected_tasks:
        raise ValueError("expert demonstration tasks must not be empty")
    buffer = RolloutBuffer()
    for task in selected_tasks:
        environment = TransactionalWorkflowEnv(task)
        observation = environment.reset()
        trajectory_id = f"bc-{task.case_id}"
        for step_index, oracle_call in enumerate(task.oracle_plans[0]):
            candidates = environment.candidate_actions(include_invalid=True)
            action_index = _oracle_index(candidates, oracle_call)
            features = encoder.encode_decision(observation, candidates)
            outcome = environment.step(candidates[action_index])
            buffer.add(
                StepRecord(
                    trajectory_id=trajectory_id,
                    step_index=step_index,
                    observation=observation,
                    candidates=tuple(candidates),
                    action_index=action_index,
                    action_mask=tuple(True for _ in candidates),
                    log_prob=0.0,
                    value=0.0,
                    reward=outcome.reward,
                    done=outcome.done,
                    next_observation=outcome.observation,
                    state_features=features.state,
                    action_features=features.actions,
                    task_reward=1.0 if outcome.success else 0.0,
                    step_reward=-0.01,
                    valid_label=True,
                    predicted_valid=True,
                    info=outcome.info,
                )
            )
            observation = outcome.observation
            if outcome.done:
                break
        if not environment.evaluate().success:
            raise RuntimeError(f"expert replay failed for {task.case_id}")
    return buffer


def train_behavior_policy(
    tasks: Sequence[WorkflowTask],
    encoder: FeatureEncoder,
    config: ExperimentConfig,
    *,
    seed: int,
) -> tuple[ActorCritic, BCMetrics]:
    set_reproducible_seed(seed)
    model = ActorCritic(
        state_dim=encoder.state_dim,
        action_dim=encoder.action_dim,
        hidden_dim=config.policy.hidden_dim,
    )
    demonstrations = collect_expert_demonstrations(tasks, encoder)
    bc = config.training.behavior_cloning
    trainer = BehaviorCloningTrainer(
        model,
        BCConfig(
            learning_rate=bc.learning_rate,
            epochs=bc.epochs,
            batch_size=bc.batch_size,
            seed=seed,
        ),
    )
    metrics = trainer.update(demonstrations.as_policy_batch())
    return model, metrics


def _replay_prefix(task: WorkflowTask, prefix: int) -> TransactionalWorkflowEnv:
    environment = TransactionalWorkflowEnv(task)
    for call in task.oracle_plans[0][:prefix]:
        outcome = environment.step(call)
        if not outcome.accepted:
            raise RuntimeError(f"oracle prefix failed for {task.case_id}: {outcome.reason}")
    return environment


def _monte_carlo_progress_examples(
    tasks: Sequence[WorkflowTask],
    encoder: FeatureEncoder,
    *,
    seed: int,
    max_tasks: int = 256,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Label visible states with actual continuation outcomes.

    A deterministic noisy continuation policy produces binary success/failure
    labels.  The estimator sees only the pre-rollout visible feature vector;
    oracle metadata is used by the data generator, never as an input feature.
    """

    if not tasks:
        raise ValueError("progress training tasks must not be empty")
    rng = random.Random(seed)
    features: list[torch.Tensor] = []
    labels: list[float] = []
    for task in tasks[:max_tasks]:
        positions = sorted(
            {
                0,
                task.optimal_steps // 3,
                2 * task.optimal_steps // 3,
                task.optimal_steps - 1,
            }
        )
        for prefix in positions:
            for _trial in range(2):
                environment = _replay_prefix(task, prefix)
                observation = environment.observe()
                features.append(encoder.encode_state(observation))
                progress = prefix / task.optimal_steps
                choose_valid_probability = 0.50 + 0.45 * progress
                while not environment.done:
                    candidates = environment.candidate_actions(include_invalid=True)
                    labels_now = [environment.dry_run(candidate).valid for candidate in candidates]
                    valid = [index for index, label in enumerate(labels_now) if label]
                    invalid = [index for index, label in enumerate(labels_now) if not label]
                    if rng.random() < choose_valid_probability or not invalid:
                        action_index = rng.choice(valid)
                    else:
                        action_index = rng.choice(invalid)
                    environment.step(candidates[action_index])
                labels.append(float(environment.evaluate().success))
    label_tensor = torch.tensor(labels, dtype=torch.float32)
    if label_tensor.min() == label_tensor.max():
        raise RuntimeError("Monte Carlo progress labels need both successes and failures")
    return torch.stack(features), label_tensor


def train_progress_estimator(
    train_tasks: Sequence[WorkflowTask],
    dev_tasks: Sequence[WorkflowTask],
    encoder: FeatureEncoder,
    *,
    seed: int,
    epochs: int = 40,
) -> tuple[ProgressEstimator, ProgressTrainingResult]:
    set_reproducible_seed(seed + 1)
    train_features, train_labels = _monte_carlo_progress_examples(
        train_tasks, encoder, seed=seed + 10
    )
    dev_features, dev_labels = _monte_carlo_progress_examples(
        dev_tasks, encoder, seed=seed + 20, max_tasks=128
    )
    estimator = ProgressEstimator(encoder.state_dim, hidden_dim=64)
    train_metrics = estimator.fit(
        train_features,
        train_labels,
        epochs=epochs,
        learning_rate=3e-3,
        batch_size=128,
        seed=seed,
        freeze_after=True,
    )
    with torch.no_grad():
        dev_metrics = compute_progress_metrics(
            estimator.probabilities(dev_features), dev_labels
        )
    return estimator, ProgressTrainingResult(
        train=train_metrics,
        dev=dev_metrics,
        examples=len(train_labels),
    )


def _progress_probability(
    estimator: ProgressEstimator | None,
    encoder: FeatureEncoder,
    observation: Observation,
) -> float:
    if estimator is None:
        return 0.0
    with torch.no_grad():
        value = estimator.probabilities(encoder.encode_state(observation).unsqueeze(0))
    return float(value.item())


def _step_trace(record: StepRecord, mode: TraceMode) -> dict[str, Any]:
    payload = record.to_trace_dict()
    if mode == "full":
        return payload
    candidates = payload.pop("candidates")
    encoded = json.dumps(candidates, sort_keys=True, separators=(",", ":")).encode("utf-8")
    payload["candidate_count"] = len(record.candidates)
    payload["candidates_sha256"] = hashlib.sha256(encoded).hexdigest()
    payload.pop("state_features", None)
    payload.pop("action_features", None)
    payload.pop("observation", None)
    payload.pop("next_observation", None)
    return payload


def run_episode(
    model: ActorCritic,
    estimator: ProgressEstimator | None,
    task: WorkflowTask,
    encoder: FeatureEncoder,
    reward_config: RewardConfig,
    *,
    trajectory_id: str,
    use_action_mask: bool,
    use_progress_reward: bool,
    deterministic: bool,
    credit_assignment: CreditAssignment = "action",
    gamma: float = 0.99,
    trace_mode: TraceMode = "full",
) -> EpisodeResult:
    """Execute one policy episode through the real transaction boundary."""

    environment = TransactionalWorkflowEnv(task)
    observation = environment.reset()
    policy_mask = ActionMask(task)
    records: list[StepRecord] = []
    candidate_evaluations: list[dict[str, Any]] = []
    executed_actions: list[dict[str, Any]] = []

    while not environment.done:
        candidates = environment.candidate_actions(include_invalid=True)
        predicted_mask = policy_mask.mask(observation, candidates)
        actual_labels = [environment.dry_run(candidate) for candidate in candidates]
        if trace_mode == "full":
            candidate_evaluations.extend(
                {
                    "step_index": len(records),
                    "candidate_index": index,
                    "valid_label": label.valid,
                    "predicted_valid": predicted_mask[index],
                    "invalid_kind": (
                        None if label.invalid_kind is None else label.invalid_kind.value
                    ),
                }
                for index, label in enumerate(actual_labels)
            )
        selection_mask = predicted_mask if use_action_mask else [True] * len(candidates)
        features = encoder.encode_decision(observation, candidates, selection_mask)
        states, actions, masks = features.as_batch()
        sample = model.act(states, actions, masks, deterministic=deterministic)
        action_index = int(sample.action_index.item())
        selected_label = actual_labels[action_index]
        phi = _progress_probability(estimator, encoder, observation)
        outcome = environment.step(candidates[action_index])
        phi_next = _progress_probability(estimator, encoder, outcome.observation)

        task_reward = 0.0
        if outcome.success:
            task_reward = reward_config.task_success
        elif outcome.done:
            task_reward = reward_config.task_failure
        invalid_reward = 0.0 if outcome.accepted else reward_config.invalid_action
        progress_reward = 0.0
        if use_progress_reward:
            progress_reward = float(
                potential_shaping(
                    phi,
                    phi_next,
                    gamma=gamma,
                    beta=reward_config.progress_beta,
                ).item()
            )
        forbidden_reward = (
            reward_config.forbidden_side_effect
            if environment.evaluate().forbidden_side_effect_count
            else 0.0
        )
        breakdown = RewardBreakdown(
            task=task_reward,
            progress=progress_reward,
            step=reward_config.step_cost,
            invalid=invalid_reward,
            forbidden=forbidden_reward,
        )
        executed_actions.append(
            {
                "step_index": len(records),
                "candidate_index": action_index,
                "executed_valid": selected_label.valid,
                "accepted": outcome.accepted,
            }
        )
        records.append(
            StepRecord(
                trajectory_id=trajectory_id,
                step_index=len(records),
                observation=observation,
                candidates=tuple(candidates),
                action_index=action_index,
                action_mask=tuple(selection_mask),
                log_prob=float(sample.log_prob.item()),
                value=float(sample.value.item()),
                reward=breakdown.total,
                done=outcome.done,
                next_observation=outcome.observation,
                state_features=features.state,
                action_features=features.actions,
                task_reward=breakdown.task,
                progress_reward=breakdown.progress,
                step_reward=breakdown.step,
                invalid_reward=breakdown.invalid,
                valid_label=selected_label.valid,
                predicted_valid=predicted_mask[action_index],
                invalid_kind=(
                    None
                    if selected_label.invalid_kind is None
                    else selected_label.invalid_kind.value
                ),
                info={**outcome.info, "reward_components": asdict(breakdown)},
            )
        )
        observation = outcome.observation

    evaluation = environment.evaluate()
    trace = {
        "schema_version": "1.0",
        "credit_assignment": credit_assignment,
        "trace_mode": trace_mode,
        "case_id": task.case_id,
        "family": task.family,
        "difficulty": task.difficulty,
        "success": evaluation.success,
        "goal_predicates_satisfied": evaluation.goal_predicates_satisfied,
        "forbidden_side_effect_count": evaluation.forbidden_side_effect_count,
        "optimal_steps": task.optimal_steps,
        "max_steps": task.max_steps,
        "step_count": len(records),
        "simulated_latency_s": environment.simulated_latency_s,
        "steps": [_step_trace(record, trace_mode) for record in records],
        "action_evaluations": candidate_evaluations,
        "executed_actions": executed_actions,
        "final_state": environment.state_snapshot(),
    }
    return EpisodeResult(trace=trace, records=tuple(records))


def collect_online_rollouts(
    model: ActorCritic,
    estimator: ProgressEstimator,
    tasks: Sequence[WorkflowTask],
    encoder: FeatureEncoder,
    reward_config: RewardConfig,
    *,
    episodes: int,
    seed: int,
    use_action_mask: bool,
    use_progress_reward: bool,
    credit_assignment: CreditAssignment,
    gamma: float = 0.99,
) -> RolloutBuffer:
    if episodes <= 0 or not tasks:
        raise ValueError("episodes and tasks must be positive/non-empty")
    set_reproducible_seed(seed)
    buffer = RolloutBuffer()
    schedule = stratified_task_schedule(tasks, episodes=episodes, seed=seed)
    for index, task in enumerate(schedule):
        result = run_episode(
            model,
            estimator,
            task,
            encoder,
            reward_config,
            trajectory_id=f"train-{seed}-{index:06d}-{task.case_id}",
            use_action_mask=use_action_mask,
            use_progress_reward=use_progress_reward,
            deterministic=False,
            credit_assignment=credit_assignment,
            gamma=gamma,
            trace_mode="compact",
        )
        buffer.extend(result.records)
    return buffer


def stratified_task_schedule(
    tasks: Sequence[WorkflowTask], *, episodes: int, seed: int
) -> list[WorkflowTask]:
    """Seeded round-robin sampling that covers every workflow family."""

    if episodes <= 0 or not tasks:
        raise ValueError("episodes and tasks must be positive/non-empty")
    rng = random.Random(seed)
    by_family: dict[str, list[WorkflowTask]] = {}
    for task in tasks:
        by_family.setdefault(task.family, []).append(task)
    for family_tasks in by_family.values():
        rng.shuffle(family_tasks)
    families = sorted(by_family)
    positions = {family: 0 for family in families}
    result: list[WorkflowTask] = []
    while len(result) < episodes:
        round_families = families.copy()
        rng.shuffle(round_families)
        for family in round_families:
            candidates = by_family[family]
            position = positions[family]
            result.append(candidates[position % len(candidates)])
            positions[family] = position + 1
            if len(result) == episodes:
                break
    return result


def train_ppo_variant(
    bc_model: ActorCritic,
    estimator: ProgressEstimator,
    tasks: Sequence[WorkflowTask],
    encoder: FeatureEncoder,
    config: ExperimentConfig,
    *,
    seed: int,
    use_action_mask: bool,
    use_progress_reward: bool,
    credit_assignment: CreditAssignment = "action",
) -> tuple[ActorCritic, PPOMetrics, int]:
    model = copy.deepcopy(bc_model)
    ppo = config.training.ppo
    algorithm_config = PPOConfig(
        learning_rate=ppo.learning_rate,
        clip_coefficient=ppo.clip_ratio,
        value_loss_coefficient=ppo.value_coefficient,
        entropy_coefficient=ppo.entropy_coefficient,
        epochs=ppo.epochs,
        batch_size=ppo.minibatch_size,
        max_grad_norm=ppo.max_grad_norm,
        seed=seed,
    )
    action_trainer = (
        PPOTrainer(model, algorithm_config) if credit_assignment == "action" else None
    )
    sequence_trainer = (
        SequencePPOTrainer(model, algorithm_config)
        if credit_assignment == "sequence"
        else None
    )
    iteration_metrics: list[PPOMetrics] = []
    rollout_steps = 0
    for iteration in range(ppo.iterations):
        rollouts = collect_online_rollouts(
            model,
            estimator,
            tasks,
            encoder,
            config.reward,
            episodes=ppo.rollout_episodes,
            seed=seed + iteration * 100_003,
            use_action_mask=use_action_mask,
            use_progress_reward=use_progress_reward,
            credit_assignment=credit_assignment,
            gamma=ppo.gamma,
        )
        rollout_steps += len(rollouts)
        if credit_assignment == "sequence":
            assert sequence_trainer is not None
            iteration_metrics.append(
                sequence_trainer.update(rollouts.as_sequence_ppo_batch(gamma=ppo.gamma))
            )
        else:
            assert action_trainer is not None
            iteration_metrics.append(
                action_trainer.update(
                    rollouts.as_ppo_batch(gamma=ppo.gamma, gae_lambda=ppo.gae_lambda)
                )
            )
    metrics = PPOMetrics(
        policy_loss=fmean(item.policy_loss for item in iteration_metrics),
        value_loss=fmean(item.value_loss for item in iteration_metrics),
        entropy=fmean(item.entropy for item in iteration_metrics),
        approximate_kl=fmean(item.approximate_kl for item in iteration_metrics),
        clip_fraction=fmean(item.clip_fraction for item in iteration_metrics),
        grad_norm=fmean(item.grad_norm for item in iteration_metrics),
        total_loss=fmean(item.total_loss for item in iteration_metrics),
        updates=sum(item.updates for item in iteration_metrics),
        stopped_early=any(item.stopped_early for item in iteration_metrics),
    )
    return model, metrics, rollout_steps


def evaluate_policy(
    model: ActorCritic,
    estimator: ProgressEstimator,
    tasks: Sequence[WorkflowTask],
    encoder: FeatureEncoder,
    reward_config: RewardConfig,
    *,
    trace_path: str | Path,
    use_action_mask: bool,
    use_progress_reward: bool,
    credit_assignment: CreditAssignment = "action",
    gamma: float = 0.99,
    trace_mode: TraceMode = "compact",
) -> TraceStore:
    """Evaluate with deterministic actions and resumable append-only evidence."""

    store = TraceStore(trace_path)
    completed = store.completed_case_ids()
    model.eval()
    for task in tasks:
        if task.case_id in completed:
            continue
        result = run_episode(
            model,
            estimator,
            task,
            encoder,
            reward_config,
            trajectory_id=task.case_id,
            use_action_mask=use_action_mask,
            use_progress_reward=use_progress_reward,
            deterministic=True,
            credit_assignment=credit_assignment,
            gamma=gamma,
            trace_mode=trace_mode,
        )
        store.append(result.trace)
    return store


def save_checkpoint(
    path: str | Path,
    model: ActorCritic,
    estimator: ProgressEstimator,
    encoder: FeatureEncoder,
    *,
    seed: int,
    variant: str,
    metadata: Mapping[str, Any],
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": 1,
            "seed": seed,
            "variant": variant,
            "state_dim": encoder.state_dim,
            "action_dim": encoder.action_dim,
            "hidden_dim": model.hidden_dim,
            "feature_fingerprint": encoder.fingerprint(),
            "model_state_dict": model.state_dict(),
            "progress_state_dict": estimator.state_dict(),
            "progress_hidden_dim": 64,
            "metadata": dict(metadata),
        },
        destination,
    )
    return destination


def load_checkpoint(
    path: str | Path,
    *,
    expected_variant: str | None = None,
) -> tuple[ActorCritic, ProgressEstimator, FeatureEncoder, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("unsupported checkpoint schema")
    checkpoint_variant = payload.get("variant")
    if not isinstance(checkpoint_variant, str) or not checkpoint_variant:
        raise ValueError("checkpoint variant identity is missing")
    if expected_variant is not None and checkpoint_variant != expected_variant:
        raise ValueError(
            "checkpoint variant identity mismatch: "
            f"expected {expected_variant!r}, observed {checkpoint_variant!r}"
        )
    encoder = FeatureEncoder(
        state_dim=int(payload["state_dim"]), action_dim=int(payload["action_dim"])
    )
    if payload.get("feature_fingerprint") != encoder.fingerprint():
        raise ValueError("checkpoint feature encoder fingerprint mismatch")
    model = ActorCritic(
        state_dim=encoder.state_dim,
        action_dim=encoder.action_dim,
        hidden_dim=int(payload["hidden_dim"]),
    )
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    estimator = ProgressEstimator(
        encoder.state_dim, hidden_dim=int(payload.get("progress_hidden_dim", 64))
    )
    estimator.load_state_dict(payload["progress_state_dict"])
    estimator.freeze()
    metadata = payload.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError("checkpoint metadata must be an object")
    metadata_variant = metadata.get("variant")
    if isinstance(metadata_variant, dict):
        if metadata_variant.get("name") != checkpoint_variant:
            raise ValueError("checkpoint variant identity differs from training metadata")
        metadata_seed = metadata.get("seed")
        if (
            isinstance(metadata_seed, bool)
            or not isinstance(metadata_seed, int)
            or payload.get("seed") != metadata_seed
        ):
            raise ValueError("checkpoint seed identity differs from training metadata")
    return model, estimator, encoder, dict(metadata)


__all__ = [
    "CreditAssignment",
    "EpisodeResult",
    "ProgressTrainingResult",
    "TraceMode",
    "TrainingResult",
    "collect_expert_demonstrations",
    "collect_online_rollouts",
    "evaluate_policy",
    "load_checkpoint",
    "run_episode",
    "save_checkpoint",
    "set_reproducible_seed",
    "stratified_task_schedule",
    "train_behavior_policy",
    "train_ppo_variant",
    "train_progress_estimator",
]
