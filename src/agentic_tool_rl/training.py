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
import os
import random
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from functools import cache
from pathlib import Path
from statistics import fmean
from tempfile import NamedTemporaryFile
from typing import Any, Literal

import torch

from agentic_tool_rl.algorithms import (
    BCConfig,
    BCMetrics,
    BehaviorCloningTrainer,
    ExpertBatch,
    PolicyBatch,
    PPOConfig,
    PPOMetrics,
    PPOTrainer,
    SequencePPOTrainer,
)
from agentic_tool_rl.config import ExperimentConfig, RewardConfig
from agentic_tool_rl.contracts import (
    CounterfactualWorkflowTask,
    Observation,
    Split,
    ToolCall,
    WorkflowTask,
)
from agentic_tool_rl.envs import TransactionalWorkflowEnv
from agentic_tool_rl.envs.benchmark_v14 import GOAL_VARIANTS, optimal_candidate_indices
from agentic_tool_rl.evaluation import TraceStore
from agentic_tool_rl.features import FeatureEncoder
from agentic_tool_rl.grounding import ActionMask
from agentic_tool_rl.models import ActorCritic, ProgressEstimator
from agentic_tool_rl.models.progress_estimator import (
    ProgressMetrics,
    compute_progress_metrics,
)
from agentic_tool_rl.package_resources import (
    IMPLEMENTATION_FINGERPRINT_SCHEMA,
    implementation_fingerprint_v2,
)
from agentic_tool_rl.policy_input import PolicyInput
from agentic_tool_rl.rewards import RewardBreakdown, potential_shaping
from agentic_tool_rl.rollout import RolloutBuffer, StepRecord

CreditAssignment = Literal["action", "sequence"]
TraceMode = Literal["full", "compact"]
V14_EXPERT_PATHS_PER_TASK = 4


@cache
def _continuation_implementation_sha256() -> str:
    return implementation_fingerprint_v2()


@dataclass(frozen=True, slots=True)
class FixedContinuationPolicy:
    """Versioned continuation behavior used only to build progress labels."""

    policy_id: str
    oracle_positive_probability: float
    decision_rule: str
    positive_rule: str
    positive_selection_rule: str
    error_selection_rule: str

    @property
    def implementation_sha256(self) -> str:
        """Bind the identity to package code without including install paths."""

        return _continuation_implementation_sha256()

    def canonical_payload(self) -> dict[str, str | float]:
        return {
            **asdict(self),
            "implementation_fingerprint_schema": IMPLEMENTATION_FINGERPRINT_SCHEMA,
            "implementation_sha256": self.implementation_sha256,
        }

    @property
    def sha256(self) -> str:
        encoded = json.dumps(
            self.canonical_payload(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


FIXED_CONTINUATION_POLICY = FixedContinuationPolicy(
    policy_id="oracle-positive-p075-absorbing-sha256-error-v1",
    oracle_positive_probability=0.75,
    decision_rule="sha256-uniform-below-one-fixed-threshold",
    positive_rule="unit-goal-distance-reduction",
    positive_selection_rule="sha256-ranked-oracle-positive",
    error_selection_rule=(
        "absorbing-sha256-ranked-executable-nonpositive-then-any-nonpositive"
    ),
)


@dataclass(frozen=True, slots=True)
class ProgressSamplingAudit:
    policy_id: str
    policy_sha256: str
    policy_implementation_sha256: str
    seed: int
    selected_tasks: int
    selected_case_ids: list[str]
    strata: int
    prefixes_per_task: int
    trials_per_prefix: int
    continuation_runs: int
    examples: int
    successes: int
    failures: int
    success_rate: float
    family_outcomes: dict[str, dict[str, int]]
    goal_variant_counts: dict[str, int]
    stratum_goal_variant_counts: dict[str, dict[str, int]]
    prefix_trials: dict[str, int]
    follow_probability_by_prefix: dict[str, float]
    done_counts: dict[str, int]


@dataclass(frozen=True, slots=True)
class ProgressExampleSet:
    features: torch.Tensor
    labels: torch.Tensor
    done_mask: torch.Tensor
    audit: ProgressSamplingAudit

    def validate(self) -> None:
        if self.features.ndim != 2:
            raise ValueError("progress features must have shape [N, D]")
        if self.labels.ndim != 1 or self.done_mask.ndim != 1:
            raise ValueError("progress labels and done_mask must have shape [N]")
        if not (
            self.features.shape[0] == self.labels.shape[0] == self.done_mask.shape[0]
        ):
            raise ValueError("progress features, labels, and done_mask must align")
        if self.features.shape[0] == 0:
            raise ValueError("progress examples must not be empty")
        if self.done_mask.dtype != torch.bool:
            raise ValueError("progress done_mask must use bool dtype")
        if bool(((self.labels != 0) & (self.labels != 1)).any()):
            raise ValueError("progress labels must be binary")
        observed_done_counts = {
            "0": int((~self.done_mask).sum()),
            "1": int(self.done_mask.sum()),
        }
        if observed_done_counts != self.audit.done_counts:
            raise ValueError("progress done_mask does not match the sampling audit")
        if any(count <= 0 for count in observed_done_counts.values()):
            raise ValueError("progress examples must contain done=0 and done=1")


@dataclass(frozen=True, slots=True)
class ProgressQualityGate:
    auroc_threshold: float
    ece_threshold: float
    brier_ratio_threshold: float
    constant_prevalence_brier: float
    brier_threshold: float
    auroc_pass: bool
    ece_pass: bool
    brier_pass: bool
    passed: bool


@dataclass(frozen=True)
class ProgressTrainingResult:
    train: ProgressMetrics
    dev: ProgressMetrics
    examples: int
    quality_status: Literal["confirmatory", "exploratory"]
    dev_quality_gate: ProgressQualityGate
    continuation_policy_id: str
    continuation_policy_sha256: str
    continuation_implementation_sha256: str
    dev_gate_scope: Literal["nonterminal_prefix_only"]
    dev_gate_examples: int
    train_sampling: ProgressSamplingAudit
    dev_sampling: ProgressSamplingAudit


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


def _canonical_visible_state_sha256(policy_input: PolicyInput) -> str:
    """Hash one public state, canonicalising fields whose order is not semantic."""

    payload = policy_input.observation.model_dump(mode="json")
    for key in ("available_entities", "available_tools"):
        values = payload.get(key)
        if isinstance(values, list):
            payload[key] = sorted(values)
    visible = payload.get("visible_state")
    if isinstance(visible, dict):
        for key in ("completed_nodes", "side_effects"):
            values = visible.get(key)
            if isinstance(values, list):
                visible[key] = sorted(
                    values,
                    key=lambda value: json.dumps(
                        value,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                )
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _deterministic_shortest_oracle_paths(
    task: CounterfactualWorkflowTask,
    *,
    minimum_paths: int = V14_EXPERT_PATHS_PER_TASK,
) -> tuple[tuple[ToolCall, ...], ...]:
    """Construct distinct legal shortest paths from the goal-specific target DAG."""

    if minimum_paths < V14_EXPERT_PATHS_PER_TASK:
        raise ValueError(
            f"v1.4 expert collection requires at least {V14_EXPERT_PATHS_PER_TASK} paths"
        )
    if not task.oracle_plans or not task.oracle_plans[0]:
        raise RuntimeError(f"v1.4 task {task.case_id} has no seed oracle plan")

    prototype_by_operation: dict[str, ToolCall] = {}
    target_operations: list[str] = []
    for call in task.oracle_plans[0]:
        operation = call.arguments.get("operation_id")
        if not isinstance(operation, str) or not operation:
            raise RuntimeError(f"v1.4 oracle call is missing operation_id for {task.case_id}")
        if operation in prototype_by_operation:
            raise RuntimeError(f"v1.4 oracle path repeats {operation!r} for {task.case_id}")
        prototype_by_operation[operation] = call
        target_operations.append(operation)
    if len(target_operations) != task.optimal_steps:
        raise RuntimeError(
            f"v1.4 oracle target has {len(target_operations)} steps, "
            f"expected {task.optimal_steps} for {task.case_id}"
        )

    node_by_id = {node.node_id: node for node in task.workflow_nodes}
    target_set = set(target_operations)
    if any(operation not in node_by_id for operation in target_operations):
        raise RuntimeError(f"v1.4 oracle references an unknown node for {task.case_id}")
    for operation in target_operations:
        missing = set(node_by_id[operation].required_predecessors) - target_set
        if missing:
            raise RuntimeError(
                f"v1.4 shortest target omits predecessors {sorted(missing)} "
                f"for {task.case_id}"
            )

    operation_paths: list[tuple[str, ...]] = []

    def visit(completed: tuple[str, ...]) -> None:
        if len(operation_paths) >= minimum_paths:
            return
        if len(completed) == task.optimal_steps:
            operation_paths.append(completed)
            return
        completed_set = set(completed)
        ready = sorted(
            operation
            for operation in target_operations
            if operation not in completed_set
            and set(node_by_id[operation].required_predecessors).issubset(completed_set)
        )
        for operation in ready:
            visit((*completed, operation))

    visit(())
    if len(operation_paths) < minimum_paths:
        raise RuntimeError(
            f"v1.4 task {task.case_id} exposes only {len(operation_paths)} distinct "
            f"shortest paths; at least {minimum_paths} are required"
        )
    if len(set(operation_paths)) != len(operation_paths):
        raise RuntimeError(f"v1.4 path enumeration duplicated a path for {task.case_id}")

    result: list[tuple[ToolCall, ...]] = []
    for path_index, operations in enumerate(operation_paths):
        calls: list[ToolCall] = []
        for step_index, operation in enumerate(operations):
            prototype = prototype_by_operation[operation]
            arguments = dict(prototype.arguments)
            arguments["expected_version"] = step_index
            calls.append(
                ToolCall(
                    tool_name=prototype.tool_name,
                    arguments=arguments,
                    call_id=(
                        f"expert-{task.case_id}-path-{path_index:02d}-step-{step_index:02d}"
                    ),
                )
            )

        environment = TransactionalWorkflowEnv(task)
        for call in calls:
            outcome = environment.step(call)
            if not outcome.accepted:
                raise RuntimeError(
                    f"constructed shortest path is illegal for {task.case_id}: "
                    f"{outcome.reason}"
                )
        evaluation = environment.evaluate()
        if not evaluation.success or environment.steps_taken != task.optimal_steps:
            raise RuntimeError(
                f"constructed path is not shortest and successful for {task.case_id}"
            )
        result.append(tuple(calls))
    return tuple(result)


def collect_counterfactual_expert_batch(
    tasks: Sequence[CounterfactualWorkflowTask],
    encoder: FeatureEncoder,
    *,
    max_tasks: int | None = None,
) -> ExpertBatch:
    """Collate A+(s) over four shortest paths, deduplicated by public state."""

    selected_tasks = tasks if max_tasks is None else tasks[:max_tasks]
    if not selected_tasks:
        raise ValueError("counterfactual expert tasks must not be empty")

    states: list[torch.Tensor] = []
    action_features: list[torch.Tensor] = []
    candidate_masks: list[torch.Tensor] = []
    positive_masks: list[torch.Tensor] = []
    visible_state_sha256s: list[str] = []
    source_case_ids: list[str] = []
    source_path_counts: list[tuple[str, int]] = []
    for task in selected_tasks:
        if not isinstance(task, CounterfactualWorkflowTask):
            raise TypeError("counterfactual expert collection accepts only v1.4 tasks")
        paths = _deterministic_shortest_oracle_paths(task)
        source_path_counts.append((task.case_id, len(paths)))
        seen_visible_states: set[str] = set()
        public_mask = ActionMask(task.tool_schemas)
        for path in paths:
            environment = TransactionalWorkflowEnv(task)
            observation = environment.reset()
            for oracle_call in path:
                candidates = environment.candidate_actions(include_invalid=True)
                selection_mask = public_mask.mask(observation, candidates)
                raw_completed = environment.state.get("completed_nodes", [])
                if not isinstance(raw_completed, list) or not all(
                    isinstance(item, str) for item in raw_completed
                ):
                    raise RuntimeError(f"completed_nodes is malformed for {task.case_id}")
                positive_indices = optimal_candidate_indices(
                    task,
                    raw_completed,
                    candidates,
                )
                if not positive_indices:
                    raise RuntimeError(
                        "counterfactual expert state has no positive action for "
                        f"{task.case_id}"
                    )
                if any(
                    index < 0
                    or index >= len(selection_mask)
                    or not selection_mask[index]
                    for index in positive_indices
                ):
                    raise RuntimeError(
                        f"counterfactual expert positive is masked for {task.case_id}"
                    )
                oracle_index = _oracle_index(candidates, oracle_call)
                if oracle_index not in positive_indices:
                    raise RuntimeError(
                        "oracle action is not optimal at an expert state for "
                        f"{task.case_id}"
                    )

                policy_input = PolicyInput.from_decision(
                    observation,
                    candidates,
                    selection_mask,
                    task.tool_schemas,
                )
                visible_state_sha256 = _canonical_visible_state_sha256(policy_input)
                if visible_state_sha256 not in seen_visible_states:
                    seen_visible_states.add(visible_state_sha256)
                    decision_features = encoder.encode_policy_input(policy_input)
                    positive_mask = torch.zeros(len(candidates), dtype=torch.bool)
                    positive_mask[list(sorted(positive_indices))] = True
                    states.append(decision_features.state)
                    action_features.append(decision_features.actions)
                    candidate_masks.append(decision_features.mask)
                    positive_masks.append(positive_mask)
                    visible_state_sha256s.append(visible_state_sha256)
                    source_case_ids.append(task.case_id)

                outcome = environment.step(candidates[oracle_index])
                if not outcome.accepted:
                    raise RuntimeError(
                        f"expert replay failed for {task.case_id}: {outcome.reason}"
                    )
                observation = outcome.observation
                if outcome.done:
                    break
            if not environment.evaluate().success:
                raise RuntimeError(f"counterfactual expert replay failed for {task.case_id}")

    maximum_candidates = max(action_tensor.shape[0] for action_tensor in action_features)
    action_dim = action_features[0].shape[1]
    padded_actions = action_features[0].new_zeros(
        (len(action_features), maximum_candidates, action_dim)
    )
    padded_candidates = torch.zeros(
        (len(candidate_masks), maximum_candidates), dtype=torch.bool
    )
    padded_positives = torch.zeros_like(padded_candidates)
    for index, (action_tensor, candidate_mask, positive_mask) in enumerate(
        zip(action_features, candidate_masks, positive_masks, strict=True)
    ):
        count = action_tensor.shape[0]
        padded_actions[index, :count] = action_tensor
        padded_candidates[index, :count] = candidate_mask
        padded_positives[index, :count] = positive_mask

    batch = ExpertBatch(
        states=torch.stack(states),
        action_features=padded_actions,
        candidate_mask=padded_candidates,
        positive_mask=padded_positives,
        visible_state_sha256s=tuple(visible_state_sha256s),
        source_case_ids=tuple(source_case_ids),
        source_path_counts=tuple(source_path_counts),
    )
    batch.validate()
    return batch


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
        public_mask = ActionMask(task.tool_schemas)
        trajectory_id = f"bc-{task.case_id}"
        for step_index, oracle_call in enumerate(task.oracle_plans[0]):
            candidates = environment.candidate_actions(include_invalid=True)
            predicted_mask = public_mask.mask(observation, candidates)
            action_index = _oracle_index(candidates, oracle_call)
            if not predicted_mask[action_index]:
                raise RuntimeError(f"oracle action is publicly masked for {task.case_id}")
            policy_input = PolicyInput.from_decision(
                observation,
                candidates,
                predicted_mask,
                task.tool_schemas,
            )
            # The frozen v1.3 BC checkpoint was shared with its unmasked
            # ablation. Preserve that optimization mask explicitly while the
            # canonical DTO still records the real public ActionMask output.
            training_mask = [True] * len(candidates)
            features = encoder.encode_policy_input(
                policy_input,
                selection_mask=training_mask,
            )
            outcome = environment.step(candidates[action_index])
            buffer.add(
                StepRecord(
                    trajectory_id=trajectory_id,
                    step_index=step_index,
                    observation=observation,
                    candidates=tuple(candidates),
                    action_index=action_index,
                    action_mask=tuple(training_mask),
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
                    predicted_valid=predicted_mask[action_index],
                    info={
                        **outcome.info,
                        "policy_input_sha256": policy_input.sha256(),
                    },
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
    counterfactual_tasks = [
        task for task in tasks if isinstance(task, CounterfactualWorkflowTask)
    ]
    if counterfactual_tasks and len(counterfactual_tasks) != len(tasks):
        raise ValueError("behavior cloning cannot mix v1.3 and v1.4 tasks")
    if any(
        task.generator_version.startswith("benchmark-v1.4")
        and not isinstance(task, CounterfactualWorkflowTask)
        for task in tasks
    ):
        raise TypeError("v1.4 behavior cloning requires CounterfactualWorkflowTask")
    batch: PolicyBatch | ExpertBatch
    if counterfactual_tasks:
        batch = collect_counterfactual_expert_batch(counterfactual_tasks, encoder)
    else:
        demonstrations = collect_expert_demonstrations(tasks, encoder)
        batch = demonstrations.as_policy_batch()
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
    metrics = trainer.update(batch)
    return model, metrics


def _replay_prefix(task: WorkflowTask, prefix: int) -> TransactionalWorkflowEnv:
    environment = TransactionalWorkflowEnv(task)
    for call in task.oracle_plans[0][:prefix]:
        outcome = environment.step(call)
        if not outcome.accepted:
            raise RuntimeError(f"oracle prefix failed for {task.case_id}: {outcome.reason}")
    return environment


def _stratum_tuple(task: WorkflowTask) -> tuple[str, str, str]:
    return task.family, task.difficulty, task.topology


def _stratum_name(task: WorkflowTask) -> str:
    return "|".join(_stratum_tuple(task))


def _balanced_progress_tasks(
    tasks: Sequence[WorkflowTask],
    *,
    max_tasks: int,
    seed: int,
) -> list[WorkflowTask]:
    """Select a deterministic balanced subset for progress supervision.

    Balance is applied across every non-empty
    ``family x difficulty x topology`` stratum.  Sorting before seeded
    shuffling makes the result independent of the caller's task ordering.
    """

    if not tasks:
        raise ValueError("progress training tasks must not be empty")
    if max_tasks <= 0:
        raise ValueError("max_tasks must be positive")

    rng = random.Random(seed)
    by_stratum: dict[tuple[str, str, str], list[WorkflowTask]] = {}
    for task in tasks:
        key = (task.family, task.difficulty, task.topology)
        by_stratum.setdefault(key, []).append(task)
    strata = sorted(by_stratum)
    for stratum in strata:
        stratum_tasks = by_stratum[stratum]
        stratum_tasks.sort(key=lambda task: task.case_id)
        rng.shuffle(stratum_tasks)

    positions = {stratum: 0 for stratum in strata}
    limit = min(max_tasks, len(tasks))
    selected: list[WorkflowTask] = []
    while len(selected) < limit:
        round_strata = [
            stratum
            for stratum in strata
            if positions[stratum] < len(by_stratum[stratum])
        ]
        if not round_strata:
            raise RuntimeError("balanced progress sampler exhausted tasks early")
        rng.shuffle(round_strata)
        for stratum in round_strata:
            position = positions[stratum]
            selected.append(by_stratum[stratum][position])
            positions[stratum] = position + 1
            if len(selected) == limit:
                break
    return selected


def _selection_index(*, seed: int, stratum: tuple[str, str, str], size: int) -> int:
    payload = {
        "namespace": "progress-v14-stratum-group-selection-v1",
        "seed": seed,
        "stratum": stratum,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big") % size


def _select_v14_train_progress_tasks(
    tasks: Sequence[CounterfactualWorkflowTask],
    *,
    seed: int,
) -> list[CounterfactualWorkflowTask]:
    """Choose one complete four-goal group from each canonical train stratum."""

    by_stratum: dict[
        tuple[str, str, str],
        dict[str, dict[str, CounterfactualWorkflowTask]],
    ] = {}
    for task in tasks:
        stratum = _stratum_tuple(task)
        variants = by_stratum.setdefault(stratum, {}).setdefault(task.group_id, {})
        if task.goal_variant in variants:
            raise RuntimeError(
                f"duplicate goal variant {task.goal_variant} in {task.group_id}"
            )
        variants[task.goal_variant] = task

    if len(by_stratum) != 90:
        raise RuntimeError(
            "canonical v1.4 progress train sampling requires exactly 90 populated "
            f"family x difficulty x topology strata, found {len(by_stratum)}"
        )

    selected: list[CounterfactualWorkflowTask] = []
    expected_variants = set(GOAL_VARIANTS)
    for stratum in sorted(by_stratum):
        complete_groups = sorted(
            group_id
            for group_id, variants in by_stratum[stratum].items()
            if set(variants) == expected_variants
        )
        if not complete_groups:
            raise RuntimeError(
                f"v1.4 progress stratum {stratum!r} has no complete four-goal group"
            )
        group_id = complete_groups[
            _selection_index(seed=seed, stratum=stratum, size=len(complete_groups))
        ]
        variants = by_stratum[stratum][group_id]
        selected.extend(variants[variant] for variant in GOAL_VARIANTS)

    if len(selected) != 360:
        raise RuntimeError(f"v1.4 progress train sampler selected {len(selected)}, expected 360")
    return selected


def _select_progress_tasks(
    tasks: Sequence[WorkflowTask],
    *,
    max_tasks: int | None,
    seed: int,
) -> list[WorkflowTask]:
    """Apply the frozen v1.4 split contract or the v1.3 balanced smoke path."""

    if not tasks:
        raise ValueError("progress training tasks must not be empty")
    v14_tasks = [task for task in tasks if isinstance(task, CounterfactualWorkflowTask)]
    if v14_tasks and len(v14_tasks) != len(tasks):
        raise ValueError("progress sampling cannot mix v1.3 and v1.4 tasks")
    if v14_tasks:
        splits = {task.split for task in v14_tasks}
        if len(splits) != 1:
            raise ValueError("progress sampling requires exactly one split")
        split = next(iter(splits))
        if split == Split.TRAIN:
            if max_tasks not in (None, 360):
                raise ValueError("v1.4 progress train sampling is fixed at 360 cases")
            return list(_select_v14_train_progress_tasks(v14_tasks, seed=seed))
        if split == Split.DEV:
            if len(v14_tasks) != 400:
                raise RuntimeError(
                    f"v1.4 progress dev sampling requires all 400 cases, found {len(v14_tasks)}"
                )
            if max_tasks not in (None, 400):
                raise ValueError("v1.4 progress dev sampling must use all 400 cases")
            return sorted(v14_tasks, key=lambda task: task.case_id)
        raise ValueError("progress supervision accepts only train or dev tasks")

    if max_tasks is None:
        return sorted(tasks, key=lambda task: task.case_id)
    return _balanced_progress_tasks(tasks, max_tasks=max_tasks, seed=seed)


def _progress_prefixes(task: WorkflowTask) -> tuple[tuple[str, int], ...]:
    result = (
        ("0", 0),
        ("33", task.optimal_steps // 3),
        ("67", (2 * task.optimal_steps) // 3),
        ("last_nonterminal", task.optimal_steps - 1),
    )
    if len({position for _name, position in result}) != 4:
        raise RuntimeError(
            f"task {task.case_id} cannot provide four distinct progress prefixes"
        )
    return result


def _continuation_digest(
    *,
    seed: int,
    namespace: str,
    context: Mapping[str, Any],
) -> bytes:
    payload = {
        "policy_sha256": FIXED_CONTINUATION_POLICY.sha256,
        "seed": seed,
        "namespace": namespace,
        "context": dict(context),
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).digest()


def _should_follow_oracle_positive(
    *,
    seed: int,
    task: WorkflowTask,
    prefix_name: str,
    prefix: int,
    trial: int,
    step: int,
) -> bool:
    digest = _continuation_digest(
        seed=seed,
        namespace="follow-oracle-positive",
        context={
            "case_id": task.case_id,
            "prefix_name": prefix_name,
            "prefix": prefix,
            "trial": trial,
            "step": step,
        },
    )
    draw = int.from_bytes(digest[:8], "big") / float(1 << 64)
    return draw < FIXED_CONTINUATION_POLICY.oracle_positive_probability


def _continuation_positive_indices(
    task: WorkflowTask,
    environment: TransactionalWorkflowEnv,
    candidates: Sequence[ToolCall],
) -> set[int]:
    completed_raw = environment.state.get("completed_nodes", [])
    if not isinstance(completed_raw, list) or not all(
        isinstance(item, str) for item in completed_raw
    ):
        raise RuntimeError(f"completed_nodes is malformed for {task.case_id}")
    if isinstance(task, CounterfactualWorkflowTask):
        return optimal_candidate_indices(task, completed_raw, candidates)

    completed = set(completed_raw)
    target_operations = {
        str(call.arguments.get("operation_id")) for call in task.oracle_plans[0]
    }
    return {
        index
        for index, candidate in enumerate(candidates)
        if str(candidate.arguments.get("operation_id")) in target_operations
        and str(candidate.arguments.get("operation_id")) not in completed
        and environment.dry_run(candidate).valid
    }


def _hash_ranked_candidate(
    candidates: Sequence[ToolCall],
    indices: Sequence[int],
    *,
    seed: int,
    namespace: str,
    context: Mapping[str, Any],
) -> int:
    if not indices:
        raise ValueError("cannot rank an empty candidate index set")

    def rank(index: int) -> bytes:
        return _continuation_digest(
            seed=seed,
            namespace=namespace,
            context={
                **context,
                "candidate_index": index,
                "candidate": candidates[index].model_dump(mode="json"),
            },
        )

    return min(indices, key=lambda index: (rank(index), index))


def _run_fixed_continuation(
    task: WorkflowTask,
    environment: TransactionalWorkflowEnv,
    *,
    seed: int,
    prefix_name: str,
    prefix: int,
    trial: int,
) -> bool:
    error_branch = False
    while not environment.done:
        candidates = environment.candidate_actions(include_invalid=True)
        positives = _continuation_positive_indices(task, environment, candidates)
        nonpositives = sorted(set(range(len(candidates))) - positives)
        context = {
            "case_id": task.case_id,
            "prefix_name": prefix_name,
            "prefix": prefix,
            "trial": trial,
            "step": environment.steps_taken,
        }
        follow_positive = False
        if not error_branch:
            follow_positive = _should_follow_oracle_positive(
                seed=seed,
                task=task,
                prefix_name=prefix_name,
                prefix=prefix,
                trial=trial,
                step=environment.steps_taken,
            )
        if follow_positive and positives:
            action_index = _hash_ranked_candidate(
                candidates,
                sorted(positives),
                seed=seed,
                namespace="select-oracle-positive",
                context=context,
            )
        else:
            error_branch = True
            executable_errors = [
                index for index in nonpositives if environment.dry_run(candidates[index]).valid
            ]
            error_pool = executable_errors or nonpositives
            if error_pool:
                action_index = _hash_ranked_candidate(
                    candidates,
                    error_pool,
                    seed=seed,
                    namespace="select-fixed-error",
                    context=context,
                )
            elif positives:
                action_index = _hash_ranked_candidate(
                    candidates,
                    sorted(positives),
                    seed=seed,
                    namespace="positive-only-fallback",
                    context=context,
                )
            else:
                raise RuntimeError(f"continuation has no selectable action for {task.case_id}")
        environment.step(candidates[action_index])
    return environment.evaluate().success


def _validate_progress_label_balance(
    *,
    successes: int,
    failures: int,
    family_outcomes: Mapping[str, Mapping[str, int]],
) -> None:
    total = successes + failures
    if total <= 0:
        raise RuntimeError("progress sampling produced no continuation outcomes")
    success_rate = successes / total
    if not 0.20 <= success_rate <= 0.80:
        raise RuntimeError(
            "progress global success fraction must be within [0.20, 0.80], "
            f"observed {success_rate:.6f}"
        )
    family_successes = sum(counts.get("success", 0) for counts in family_outcomes.values())
    family_failures = sum(counts.get("failure", 0) for counts in family_outcomes.values())
    if family_successes != successes or family_failures != failures:
        raise RuntimeError("progress family label counts do not match global label counts")
    for family in sorted(family_outcomes):
        counts = family_outcomes[family]
        if counts.get("success", 0) <= 0 or counts.get("failure", 0) <= 0:
            raise RuntimeError(
                f"progress family {family!r} must contain both success and failure labels"
            )


def _progress_quality_gate(
    metrics: ProgressMetrics,
    labels: torch.Tensor,
) -> ProgressQualityGate:
    flattened = labels.detach().flatten().to(dtype=torch.float64, device="cpu")
    if flattened.numel() == 0 or bool(((flattened != 0) & (flattened != 1)).any()):
        raise ValueError("progress quality gate requires non-empty binary labels")
    prevalence = float(flattened.mean().item())
    constant_prevalence_brier = prevalence * (1.0 - prevalence)
    brier_threshold = 0.90 * constant_prevalence_brier
    auroc_pass = metrics.auroc >= 0.70
    ece_pass = metrics.ece <= 0.10
    brier_pass = metrics.brier <= brier_threshold
    return ProgressQualityGate(
        auroc_threshold=0.70,
        ece_threshold=0.10,
        brier_ratio_threshold=0.90,
        constant_prevalence_brier=constant_prevalence_brier,
        brier_threshold=brier_threshold,
        auroc_pass=auroc_pass,
        ece_pass=ece_pass,
        brier_pass=brier_pass,
        passed=auroc_pass and ece_pass and brier_pass,
    )


def _sample_progress_examples(
    tasks: Sequence[WorkflowTask],
    encoder: FeatureEncoder,
    *,
    seed: int,
    max_tasks: int | None = 360,
) -> ProgressExampleSet:
    """Label visible states with actual continuation outcomes.

    A versioned, hash-deterministic continuation policy produces binary labels.
    Its oracle-positive probability is one fixed scalar, independent of prefix.
    The estimator sees only visible state features; oracle metadata remains a
    label-generation input and never enters the estimator feature vector.
    """

    selected_tasks = _select_progress_tasks(tasks, max_tasks=max_tasks, seed=seed)
    features: list[torch.Tensor] = []
    labels: list[float] = []
    done_mask: list[bool] = []
    successes = 0
    failures = 0
    family_outcomes = {
        family: {"success": 0, "failure": 0}
        for family in sorted({task.family for task in selected_tasks})
    }
    prefix_trials = {name: 0 for name, _position in _progress_prefixes(selected_tasks[0])}
    for task in selected_tasks:
        positions = _progress_prefixes(task)
        for prefix_name, prefix in positions:
            for trial in range(8):
                environment = _replay_prefix(task, prefix)
                observation = environment.observe()
                if observation.done:
                    raise RuntimeError(
                        f"progress prefix must be nonterminal for {task.case_id}"
                    )
                starting_features = encoder.encode_state(observation)
                succeeded = _run_fixed_continuation(
                    task,
                    environment,
                    seed=seed,
                    prefix_name=prefix_name,
                    prefix=prefix,
                    trial=trial,
                )
                success = float(succeeded)
                terminal_observation = environment.observe()
                if not terminal_observation.done:
                    raise RuntimeError(
                        f"progress continuation did not terminate for {task.case_id}"
                    )
                features.extend(
                    (
                        starting_features,
                        encoder.encode_state(terminal_observation),
                    )
                )
                labels.extend((success, success))
                done_mask.extend((False, True))
                outcome_name = "success" if succeeded else "failure"
                family_outcomes[task.family][outcome_name] += 1
                successes += int(succeeded)
                failures += int(not succeeded)
                prefix_trials[prefix_name] += 1

    _validate_progress_label_balance(
        successes=successes,
        failures=failures,
        family_outcomes=family_outcomes,
    )
    goal_variant_counts: dict[str, int] = {}
    stratum_goal_variant_counts: dict[str, dict[str, int]] = {}
    for task in selected_tasks:
        if not isinstance(task, CounterfactualWorkflowTask):
            continue
        goal_variant_counts[task.goal_variant] = (
            goal_variant_counts.get(task.goal_variant, 0) + 1
        )
        stratum_counts = stratum_goal_variant_counts.setdefault(_stratum_name(task), {})
        stratum_counts[task.goal_variant] = stratum_counts.get(task.goal_variant, 0) + 1

    continuation_runs = successes + failures
    audit = ProgressSamplingAudit(
        policy_id=FIXED_CONTINUATION_POLICY.policy_id,
        policy_sha256=FIXED_CONTINUATION_POLICY.sha256,
        policy_implementation_sha256=(
            FIXED_CONTINUATION_POLICY.implementation_sha256
        ),
        seed=seed,
        selected_tasks=len(selected_tasks),
        selected_case_ids=[task.case_id for task in selected_tasks],
        strata=len({_stratum_tuple(task) for task in selected_tasks}),
        prefixes_per_task=4,
        trials_per_prefix=8,
        continuation_runs=continuation_runs,
        examples=len(labels),
        successes=successes,
        failures=failures,
        success_rate=successes / continuation_runs,
        family_outcomes={
            family: dict(counts) for family, counts in sorted(family_outcomes.items())
        },
        goal_variant_counts=dict(sorted(goal_variant_counts.items())),
        stratum_goal_variant_counts={
            stratum: dict(sorted(counts.items()))
            for stratum, counts in sorted(stratum_goal_variant_counts.items())
        },
        prefix_trials=dict(prefix_trials),
        follow_probability_by_prefix={
            name: FIXED_CONTINUATION_POLICY.oracle_positive_probability
            for name in prefix_trials
        },
        done_counts={
            "0": done_mask.count(False),
            "1": done_mask.count(True),
        },
    )
    result = ProgressExampleSet(
        features=torch.stack(features),
        labels=torch.tensor(labels, dtype=torch.float32),
        done_mask=torch.tensor(done_mask, dtype=torch.bool),
        audit=audit,
    )
    result.validate()
    return result


def _monte_carlo_progress_examples(
    tasks: Sequence[WorkflowTask],
    encoder: FeatureEncoder,
    *,
    seed: int,
    max_tasks: int | None = 360,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compatibility wrapper for callers that need only tensors."""

    sampled = _sample_progress_examples(
        tasks,
        encoder,
        seed=seed,
        max_tasks=max_tasks,
    )
    return sampled.features, sampled.labels


def _nonterminal_progress_gate_view(
    sampled: ProgressExampleSet,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return only prefix states used by the confirmatory development gate."""

    sampled.validate()
    nonterminal_mask = ~sampled.done_mask
    if not bool(nonterminal_mask.any()):
        raise RuntimeError("progress development gate requires nonterminal prefix samples")
    features = sampled.features[nonterminal_mask]
    labels = sampled.labels[nonterminal_mask]
    if labels.numel() == 0 or bool(labels.min() == labels.max()):
        raise RuntimeError(
            "progress development gate requires both labels among nonterminal prefixes"
        )
    return features, labels


def train_progress_estimator(
    train_tasks: Sequence[WorkflowTask],
    dev_tasks: Sequence[WorkflowTask],
    encoder: FeatureEncoder,
    *,
    seed: int,
    epochs: int = 40,
) -> tuple[ProgressEstimator, ProgressTrainingResult]:
    set_reproducible_seed(seed + 1)
    train_sample = _sample_progress_examples(
        train_tasks,
        encoder,
        seed=seed + 10,
        max_tasks=360,
    )
    dev_sample = _sample_progress_examples(
        dev_tasks,
        encoder,
        seed=seed + 20,
        max_tasks=None,
    )
    estimator = ProgressEstimator(encoder.state_dim, hidden_dim=64)
    train_metrics = estimator.fit(
        train_sample.features,
        train_sample.labels,
        epochs=epochs,
        learning_rate=3e-3,
        batch_size=128,
        seed=seed,
        freeze_after=True,
    )
    dev_gate_features, dev_gate_labels = _nonterminal_progress_gate_view(dev_sample)
    with torch.no_grad():
        dev_metrics = compute_progress_metrics(
            estimator.probabilities(dev_gate_features), dev_gate_labels
        )
    dev_quality_gate = _progress_quality_gate(dev_metrics, dev_gate_labels)
    return estimator, ProgressTrainingResult(
        train=train_metrics,
        dev=dev_metrics,
        examples=len(train_sample.labels),
        quality_status="confirmatory" if dev_quality_gate.passed else "exploratory",
        dev_quality_gate=dev_quality_gate,
        continuation_policy_id=FIXED_CONTINUATION_POLICY.policy_id,
        continuation_policy_sha256=FIXED_CONTINUATION_POLICY.sha256,
        continuation_implementation_sha256=(
            FIXED_CONTINUATION_POLICY.implementation_sha256
        ),
        dev_gate_scope="nonterminal_prefix_only",
        dev_gate_examples=len(dev_gate_labels),
        train_sampling=train_sample.audit,
        dev_sampling=dev_sample.audit,
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
    policy_mask = ActionMask(task.tool_schemas)
    records: list[StepRecord] = []
    candidate_evaluations: list[dict[str, Any]] = []
    executed_actions: list[dict[str, Any]] = []
    phi = (
        _progress_probability(estimator, encoder, observation)
        if use_progress_reward
        else 0.0
    )

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
        policy_input = PolicyInput.from_decision(
            observation,
            candidates,
            predicted_mask,
            task.tool_schemas,
        )
        features = (
            encoder.encode_policy_input(policy_input)
            if use_action_mask
            else encoder.encode_policy_input(
                policy_input,
                selection_mask=selection_mask,
            )
        )
        states, actions, masks = features.as_batch()
        sample = model.act(states, actions, masks, deterministic=deterministic)
        action_index = int(sample.action_index.item())
        selected_label = actual_labels[action_index]
        outcome = environment.step(candidates[action_index])
        phi_raw = phi
        phi_next_raw: float | None = None
        if use_progress_reward and not outcome.done:
            phi_next_raw = _progress_probability(estimator, encoder, outcome.observation)
        phi_next_used = 0.0 if phi_next_raw is None else phi_next_raw
        terminal_zeroed = outcome.done

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
                    phi_next_used,
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
                "policy_input_sha256": policy_input.sha256(),
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
                info={
                    **outcome.info,
                    "policy_input_sha256": policy_input.sha256(),
                    "reward_components": asdict(breakdown),
                    "progress_potential": {
                        "phi_raw": phi_raw,
                        "phi_next_raw": phi_next_raw,
                        "phi_next_used": phi_next_used,
                        "terminal_zeroed": terminal_zeroed,
                        # Compatibility aliases retained for existing traces.
                        "phi_current": phi_raw,
                        "terminal_boundary_applied": terminal_zeroed,
                    },
                },
            )
        )
        observation = outcome.observation
        phi = phi_next_used

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
    payload = {
        "schema_version": 1,
        "seed": seed,
        "variant": variant,
        "state_dim": encoder.state_dim,
        "action_dim": encoder.action_dim,
        "hidden_dim": model.hidden_dim,
        "feature_fingerprint": encoder.fingerprint(),
        "model_state_dict": model.state_dict(),
        "progress_state_dict": estimator.state_dict(),
        "progress_hidden_dim": estimator.hidden_dim,
        "metadata": dict(metadata),
    }
    temporary: Path | None = None
    try:
        with NamedTemporaryFile(
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        # The source path is no longer ours after replace and may be reused.
        temporary = None
        directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        try:
            directory_fd = os.open(destination.parent, directory_flags)
        except OSError:
            directory_fd = None
        if directory_fd is not None:
            try:
                os.fsync(directory_fd)
            except OSError:
                # Some supported filesystems do not implement directory fsync;
                # the file remains atomically replaced and fully fsync'd.
                pass
            finally:
                os.close(directory_fd)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return destination


def load_checkpoint(
    path: str | Path,
    *,
    expected_variant: str | None = None,
    expected_seed: int | None = None,
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
    checkpoint_seed = payload.get("seed")
    if isinstance(checkpoint_seed, bool) or not isinstance(checkpoint_seed, int):
        raise ValueError("checkpoint seed identity is missing")
    if expected_seed is not None and checkpoint_seed != expected_seed:
        raise ValueError(
            "checkpoint seed identity mismatch: "
            f"expected {expected_seed!r}, observed {checkpoint_seed!r}"
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
            or checkpoint_seed != metadata_seed
        ):
            raise ValueError("checkpoint seed identity differs from training metadata")
    return model, estimator, encoder, dict(metadata)


__all__ = [
    "FIXED_CONTINUATION_POLICY",
    "CreditAssignment",
    "EpisodeResult",
    "FixedContinuationPolicy",
    "ProgressExampleSet",
    "ProgressQualityGate",
    "ProgressSamplingAudit",
    "ProgressTrainingResult",
    "TraceMode",
    "TrainingResult",
    "collect_counterfactual_expert_batch",
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
