"""Serializable contracts shared by the environment, policies, and evaluators.

The benchmark deliberately keeps these objects free of PyTorch dependencies.  A
trajectory can therefore be inspected and checked for metric consistency using
only the core package.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ContractModel(BaseModel):
    """Base class with deterministic, forward-compatible JSON semantics."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class Split(StrEnum):
    TRAIN = "train"
    DEV = "dev"
    TEST = "test"


class InvalidActionKind(StrEnum):
    SCHEMA = "schema"
    GROUNDING = "grounding"
    PRECONDITION = "precondition"
    SAFETY = "safety"


ActionValidityChallenge: TypeAlias = Literal["standard_candidate", "hidden_ledger_collision"]
WorkflowTopology: TypeAlias = Literal[
    "diamond_tail",
    "fanout_gate",
    "dual_lane",
    "dual_root_mesh",
]


class ToolSchema(ContractModel):
    """Small JSON-schema subset used by the synthetic tool registry."""

    name: str
    description: str
    required_arguments: dict[str, str]
    optional_arguments: dict[str, str] = Field(default_factory=dict)
    additional_properties: bool = False
    mutating: bool = True
    idempotent: bool = True
    side_effect: str | None = None
    # Policy-visible execution constraints.  They deliberately live on the
    # public schema instead of the hidden workflow DAG so an action mask can
    # enforce grounding, ordering, and safety without consulting oracle data.
    operation_id: str | None = None
    required_completed_operations: list[str] = Field(default_factory=list)
    policy_allowed: bool = True

    @model_validator(mode="after")
    def arguments_do_not_overlap(self) -> ToolSchema:
        overlap = set(self.required_arguments) & set(self.optional_arguments)
        if overlap:
            raise ValueError(f"arguments cannot be both required and optional: {sorted(overlap)}")
        return self


class ToolCall(ContractModel):
    """One structured agent action.  Each accepted or rejected call is one RL step."""

    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    call_id: str = ""


class StatePredicate(ContractModel):
    """Declarative final-state predicate consumed by the independent evaluator."""

    path: str
    operator: Literal["eq", "ne", "contains", "contains_all", "gte", "lte"]
    value: Any


class WorkflowNode(ContractModel):
    """A node in a hidden workflow DAG.

    ``effects`` are applied transactionally only after all schema, grounding,
    precondition, and safety checks have passed.
    """

    node_id: str
    tool_name: str
    required_predecessors: list[str] = Field(default_factory=list)
    effects: dict[str, Any] = Field(default_factory=dict)
    side_effect: str | None = None
    safety_token: str | None = None
    simulated_latency_s: float = Field(ge=0.0)


class Observation(ContractModel):
    task_id: str
    step_index: int = Field(ge=0)
    max_steps: int = Field(ge=1)
    remaining_steps: int = Field(ge=0)
    user_goal: str
    visible_state: dict[str, Any]
    available_entities: list[str]
    available_tools: list[str]
    message: str = ""
    done: bool = False

    @model_validator(mode="after")
    def step_budget_is_consistent(self) -> Observation:
        if self.step_index + self.remaining_steps != self.max_steps:
            raise ValueError("step_index + remaining_steps must equal max_steps")
        return self


class WorkflowTask(ContractModel):
    """Frozen benchmark case plus hidden evaluator/oracle information."""

    case_id: str
    split: Split
    family: str
    topology: WorkflowTopology
    difficulty: Literal["short", "medium", "long"]
    seed: int
    entity_id: str
    user_goal: str
    initial_state: dict[str, Any]
    visible_observation: dict[str, Any]
    tool_schemas: list[ToolSchema]
    workflow_nodes: list[WorkflowNode]
    hidden_goal_predicates: list[StatePredicate]
    forbidden_side_effects: list[str]
    oracle_plans: list[list[ToolCall]]
    optimal_steps: int = Field(ge=1)
    max_steps: int = Field(ge=1)
    latency_trace: dict[str, float]
    generator_version: str

    @model_validator(mode="after")
    def limits_are_consistent(self) -> WorkflowTask:
        if self.max_steps < self.optimal_steps:
            raise ValueError("max_steps must be greater than or equal to optimal_steps")
        if self.generator_version.startswith("benchmark-v1.4"):
            if len(self.workflow_nodes) < self.optimal_steps:
                raise ValueError("v1.4 world must contain at least the optimal-path nodes")
        elif len(self.workflow_nodes) != self.optimal_steps:
            # v1.3's equality is a frozen historical contract.  v1.4 models a
            # complete world containing executable counterfactual branches, so
            # only the goal-specific shortest path contributes to optimal_steps.
            raise ValueError("optimal_steps must equal the number of workflow nodes")
        return self

    @property
    def task_id(self) -> str:
        """Compatibility alias used by rollout code."""

        return self.case_id

    @property
    def goal_predicates(self) -> list[StatePredicate]:
        return self.hidden_goal_predicates


class CounterfactualWorkflowTask(WorkflowTask):
    """One goal-conditioned case inside a v1.4 counterfactual world.

    Four cases share the same ``group_id``, world, schemas, initial state, and
    public candidate process.  ``goal_variant`` is evaluator metadata and must
    never enter :class:`agentic_tool_rl.policy_input.PolicyInput`.
    """

    group_id: str
    goal_variant: Literal["00", "01", "10", "11"]

    @model_validator(mode="after")
    def version_is_v14(self) -> CounterfactualWorkflowTask:
        if not self.generator_version.startswith("benchmark-v1.4"):
            raise ValueError("counterfactual tasks require a benchmark-v1.4 generator")
        if not self.case_id.startswith(f"{self.group_id}-goal-"):
            raise ValueError("counterfactual case_id must be scoped by group_id")
        return self


class ValidationResult(ContractModel):
    valid: bool
    invalid_kind: InvalidActionKind | None = None
    reason: str = ""
    node_id: str | None = None
    idempotent_replay: bool = False

    @model_validator(mode="after")
    def invalid_results_have_a_kind(self) -> ValidationResult:
        if self.valid and self.invalid_kind is not None:
            raise ValueError("a valid result cannot have invalid_kind")
        if not self.valid and self.invalid_kind is None:
            raise ValueError("an invalid result must have invalid_kind")
        return self


class TaskEvaluation(ContractModel):
    case_id: str
    success: bool
    goal_predicates_satisfied: bool
    forbidden_side_effect_count: int = Field(ge=0)
    completed_steps: int = Field(ge=0)
    optimal_steps: int = Field(ge=1)
    max_steps: int = Field(ge=1)


class StepOutcome(ContractModel):
    observation: Observation
    reward: float
    done: bool
    success: bool
    accepted: bool
    invalid_kind: InvalidActionKind | None = None
    reason: str = ""
    simulated_latency_s: float = Field(ge=0.0)
    info: dict[str, Any] = Field(default_factory=dict)


class StepRecord(ContractModel):
    """Portable action-level rollout record.

    The reward decomposition is explicit so a report can distinguish sparse
    task reward from potential shaping and invalid-action penalties.
    """

    trajectory_id: str
    step_index: int = Field(ge=0)
    observation: Observation | dict[str, Any]
    candidates: list[ToolCall]
    action_index: int = Field(ge=0)
    action_mask: list[bool]
    action: ToolCall | None = None
    log_prob: float
    value: float
    reward: float
    done: bool
    task_reward: float = 0.0
    progress_reward: float = 0.0
    step_reward: float = 0.0
    invalid_reward: float = 0.0
    next_observation: Observation | dict[str, Any] | None = None
    valid_label: bool | None = None
    predicted_valid: bool | None = None
    invalid_kind: InvalidActionKind | None = None
    info: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def action_fields_align(self) -> StepRecord:
        if self.action_index >= len(self.candidates):
            raise ValueError("action_index must refer to candidates")
        if len(self.action_mask) != len(self.candidates):
            raise ValueError("action_mask and candidates must have equal length")
        if not self.action_mask[self.action_index]:
            raise ValueError("selected action must not be masked")
        selected = self.candidates[self.action_index]
        if self.action is not None and self.action != selected:
            raise ValueError("action must equal candidates[action_index]")
        return self


class EpisodeSummary(ContractModel):
    case_id: str
    family: str
    success: bool
    forbidden_side_effect_count: int = Field(ge=0)
    optimal_steps: int = Field(ge=1)
    max_steps: int = Field(ge=1)
    steps: int = Field(ge=0)
    simulated_latency_s: float = Field(ge=0.0)
    timeout_s: float = Field(gt=0.0)


class ActionValidityExample(ContractModel):
    """One independently labelled candidate at a frozen workflow state."""

    sample_id: str
    case_id: str
    split: Split
    family: str
    state_index: int = Field(ge=0)
    workflow_step_index: int = Field(ge=0)
    observation: Observation
    tool_call: ToolCall
    valid_label: bool
    invalid_kind: InvalidActionKind | None = None
    label_source: Literal["environment_dry_run"] = "environment_dry_run"
    challenge_source: ActionValidityChallenge = "standard_candidate"
    state_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def labels_align(self) -> ActionValidityExample:
        if self.valid_label and self.invalid_kind is not None:
            raise ValueError("valid examples cannot have an invalid_kind")
        if not self.valid_label and self.invalid_kind is None:
            raise ValueError("invalid examples require an invalid_kind")
        return self


class ActionValidityManifest(ContractModel):
    dataset_name: str = "action-validity-v2"
    generator_version: str
    source_task_count: int = Field(ge=1)
    states_per_task: int = Field(ge=1)
    candidates_per_state: int = Field(ge=2)
    seed: int
    count: int = Field(ge=1)
    valid_count: int = Field(ge=0)
    invalid_count: int = Field(ge=0)
    invalid_kind_counts: dict[str, int]
    challenge_counts: dict[str, int]
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_cases_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    label_source: Literal["environment_dry_run"] = "environment_dry_run"


class BenchmarkFile(ContractModel):
    split: Split
    path: str
    count: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    family_counts: dict[str, int]
    length_counts: dict[str, int]
    seed_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    entity_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    oracle_solvable: int = Field(ge=0)


class BenchmarkManifest(ContractModel):
    benchmark_name: str = "benchmark-v1"
    generator_version: str
    base_seed: int
    files: dict[str, BenchmarkFile]


__all__ = [
    "ActionValidityChallenge",
    "ActionValidityExample",
    "ActionValidityManifest",
    "BenchmarkFile",
    "BenchmarkManifest",
    "ContractModel",
    "EpisodeSummary",
    "InvalidActionKind",
    "Observation",
    "Split",
    "StatePredicate",
    "StepOutcome",
    "StepRecord",
    "TaskEvaluation",
    "ToolCall",
    "ToolSchema",
    "ValidationResult",
    "WorkflowNode",
    "WorkflowTask",
    "WorkflowTopology",
]
