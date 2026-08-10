"""Counterfactual benchmark-v1.4 generator.

Unlike the frozen v1.3 generator, one v1.4 world contains executable branches
that are irrelevant to a particular goal.  Four cases share all public world
state and differ only in the natural-language goal and hidden evaluator target.
"""

from __future__ import annotations

import hashlib
import random
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from agentic_tool_rl.contracts import (
    CounterfactualWorkflowTask,
    Split,
    StatePredicate,
    ToolCall,
    ToolSchema,
    WorkflowNode,
    WorkflowTopology,
)
from agentic_tool_rl.envs.benchmark import (
    _FAMILY_TEMPLATES,
    WORKFLOW_TOPOLOGIES,
)
from agentic_tool_rl.envs.oracle import verify_task_solvable
from agentic_tool_rl.grounding.action_mask import ActionMask
from agentic_tool_rl.latency_profile import sample_mutating_service_time
from agentic_tool_rl.policy_input import PolicyInput

GENERATOR_VERSION_V14 = "benchmark-v1.4.0"
DEVELOPMENT_BASE_SEED_V14 = 20_260_810
GOAL_VARIANTS: tuple[Literal["00", "01", "10", "11"], ...] = (
    "00",
    "01",
    "10",
    "11",
)
CANONICAL_GROUP_COUNTS = {Split.TRAIN: 500, Split.DEV: 100, Split.TEST: 250}

_SPLIT_OFFSETS = {Split.TRAIN: 0, Split.DEV: 1_000_000_000, Split.TEST: 2_000_000_000}
_TRANSITION_GOAL_TEXT = {
    "0": (
        "Introduce the update through successive limited waves.",
        "Phase adoption across contained groups, checking each before continuing.",
        "Move through a sequence of small cohorts instead of changing everything together.",
        "Spread the rollout over several controlled stages.",
    ),
    "1": (
        "Activate the whole scope together in one coordinated event.",
        "Move every affected unit in a single organization-wide switchover.",
        "Complete one unified change across the entire scope.",
        "Use a single cutover for all affected groups.",
    ),
}
_DECISION_GOAL_TEXT = {
    "0": (
        "Keep an accountable human owner at the final decision.",
        "A designated individual must remain responsible for authorization.",
        "Retain a human sign-off before completion.",
        "Place the concluding judgment with a named reviewer.",
    ),
    "1": (
        "Let automated validation authorize completion without a manual handoff.",
        "Proceed on system verification alone at the concluding checkpoint.",
        "Allow machine validation to carry the workflow through completion.",
        "Use automated controls, without human sign-off, for the closing decision.",
    ),
}
_PREFIX_AXIS_DESCRIPTION = {
    "a0": (
        "Use incremental batches separated by validation pauses.",
        "Advance piecemeal and reassess between portions.",
        "Divide execution into bounded tranches.",
        "Process one limited subset before opening the next.",
    ),
    "a1": (
        "Execute as an indivisible deployment.",
        "Move all units through one atomic switch.",
        "Use a simultaneous organization-wide activation.",
        "Bundle the entire change into a single event.",
    ),
    "b0": (
        "Pause for an identified approver's judgment.",
        "Obtain authorization from a responsible owner.",
        "Reserve the closing choice for human oversight.",
        "Require an accountable reviewer to decide.",
    ),
    "b1": (
        "Let software controls authorize continuation.",
        "Proceed solely from machine validation.",
        "Use automatic tests as the decision authority.",
        "Continue without an individual handoff once checks pass.",
    ),
}
_CHECKPOINT_AXIS_DESCRIPTION = {
    "a0": (
        "Confirm that the segmented rollout remains inside its current tranche.",
        "Verify the next portion stays isolated from later portions.",
        "Record assurance for the present slice before expansion.",
        "Check the current segment before another segment opens.",
    ),
    "a1": (
        "Confirm that the atomic switchover remains unified.",
        "Verify all units remain in the same activation event.",
        "Record assurance for the one-piece deployment.",
        "Check that no portion is deferred to a later cutover.",
    ),
    "b0": (
        "Record that an identified approver retains authority.",
        "Confirm responsibility remains with a human owner.",
        "Verify the closing judgment still requires individual authorization.",
        "Check that reviewer accountability has not been delegated.",
    ),
    "b1": (
        "Record that software controls retain authority.",
        "Confirm system validation remains sufficient to proceed.",
        "Verify no individual authorization has been introduced.",
        "Check that automated controls still govern the decision.",
    ),
}
_COMPLETION_AXIS_DESCRIPTION = {
    "a0": (
        "Close after the bounded portions have advanced in sequence.",
        "Conclude the segmented rollout after its staged progression.",
        "Finish the divided deployment after each tranche.",
        "Complete the sliced transition after its ordered expansion.",
    ),
    "a1": (
        "Close after the indivisible deployment has moved together.",
        "Conclude the unified activation as one event.",
        "Finish the atomic switchover across all units.",
        "Complete the one-piece transition without deferred portions.",
    ),
    "b0": (
        "Conclude under the identified approver's authority.",
        "Finish with responsibility held by a human owner.",
        "Close under an individual's recorded authorization.",
        "Complete with reviewer accountability preserved.",
    ),
    "b1": (
        "Conclude under software-control authority.",
        "Finish on the strength of system validation.",
        "Close without introducing individual authorization.",
        "Complete with automated controls governing the decision.",
    ),
}


@dataclass(frozen=True, slots=True)
class CounterfactualGateReport:
    groups: int
    cases: int
    oracle_solvable: int
    minimum_oracle_plans_per_case: int
    invariant_groups: int
    multi_positive_states: int
    expert_states: int
    goal_incompatible_states: int

    @property
    def multi_positive_fraction(self) -> float:
        return self.multi_positive_states / self.expert_states if self.expert_states else 0.0

    @property
    def goal_incompatible_fraction(self) -> float:
        return self.goal_incompatible_states / self.expert_states if self.expert_states else 0.0

    def to_dict(self) -> dict[str, int | float]:
        return {
            "groups": self.groups,
            "cases": self.cases,
            "oracle_solvable": self.oracle_solvable,
            "minimum_oracle_plans_per_case": self.minimum_oracle_plans_per_case,
            "invariant_groups": self.invariant_groups,
            "multi_positive_states": self.multi_positive_states,
            "expert_states": self.expert_states,
            "multi_positive_fraction": self.multi_positive_fraction,
            "goal_incompatible_states": self.goal_incompatible_states,
            "goal_incompatible_fraction": self.goal_incompatible_fraction,
        }


def _allocate(total: int, buckets: int) -> list[int]:
    base, remainder = divmod(total, buckets)
    return [base + (1 if index < remainder else 0) for index in range(buckets)]


def _tier_counts(count: int) -> dict[str, int]:
    raw = {"short": count * 0.3, "medium": count * 0.4, "long": count * 0.3}
    result = {name: int(value) for name, value in raw.items()}
    order = sorted(raw, key=lambda name: (-(raw[name] - result[name]), name))
    for name in order[: count - sum(result.values())]:
        result[name] += 1
    return result


def _optimal_length(tier: str, ordinal: int) -> int:
    values = {
        "short": (6, 7, 8),
        "medium": (9, 10, 11),
        "long": (12, 13, 14, 15),
    }[tier]
    return values[ordinal % len(values)]


def _opaque(seed: int, label: str, *, size: int = 10) -> str:
    payload = f"{GENERATOR_VERSION_V14}:{seed}:{label}".encode()
    return hashlib.blake2s(payload, digest_size=size).hexdigest()


def _operation_id(seed: int, label: str) -> str:
    return f"op_{_opaque(seed, label)}"


def _tool_name(family: str, seed: int, label: str) -> str:
    """Return a role-neutral public name; semantics live in the description."""

    return f"{family}.execute_{_opaque(seed, f'tool:{label}', size=6)}"


def _semantic_text(options: Sequence[str], seed: int, label: str) -> str:
    digest = hashlib.sha256(f"{GENERATOR_VERSION_V14}:{seed}:{label}".encode()).digest()
    return options[int.from_bytes(digest[:4], "big") % len(options)]


def _topology(
    split: Split,
    family_index: int,
    tier_ordinal: int,
) -> WorkflowTopology:
    held_out = WORKFLOW_TOPOLOGIES[family_index % len(WORKFLOW_TOPOLOGIES)]
    if split == Split.TEST:
        return held_out
    available = tuple(value for value in WORKFLOW_TOPOLOGIES if value != held_out)
    # Cycle inside each family x difficulty cell.  Seed modulo assignment can
    # leave a nominally populated development stratum empty by chance; the
    # ordinal rotation guarantees all three non-held-out topologies whenever
    # the cell contains at least three groups (true for canonical train/dev).
    return available[tier_ordinal % len(available)]


def _prefix_predecessors(topology: WorkflowTopology, size: int) -> list[list[int]]:
    if size < 3:
        raise ValueError("v1.4 prefix must contain at least three operations")
    result: list[list[int]] = [[] for _ in range(size)]
    if topology == "diamond_tail":
        result[1] = [0]
        result[2] = [0]
        for index in range(3, size):
            result[index] = [index - 2, index - 1]
    elif topology == "fanout_gate":
        for index in range(1, size):
            result[index] = [0]
    elif topology == "dual_lane":
        for index in range(2, size):
            result[index] = [index - 2]
    elif topology == "dual_root_mesh":
        for index in range(2, size):
            result[index] = sorted({index - 1, max(0, index - 3)})
    else:  # pragma: no cover - exhaustive Literal guard
        raise ValueError(f"unsupported topology {topology!r}")
    return result


def _topological_orders(nodes: Sequence[WorkflowNode], limit: int = 4) -> list[list[WorkflowNode]]:
    results: list[list[WorkflowNode]] = []

    def visit(completed: tuple[str, ...], path: list[WorkflowNode]) -> None:
        if len(results) >= limit:
            return
        if len(path) == len(nodes):
            results.append(path.copy())
            return
        completed_set = set(completed)
        ready = sorted(
            (
                node
                for node in nodes
                if node.node_id not in completed_set
                and set(node.required_predecessors).issubset(completed_set)
            ),
            key=lambda node: node.node_id,
        )
        for node in ready:
            visit((*completed, node.node_id), [*path, node])

    visit((), [])
    return results


def _plan_calls(
    case_id: str,
    entity_id: str,
    plan_index: int,
    order: Sequence[WorkflowNode],
) -> list[ToolCall]:
    return [
        ToolCall(
            tool_name=node.tool_name,
            arguments={
                "entity_id": entity_id,
                "operation_id": node.node_id,
                "expected_version": step_index,
            },
            call_id=f"oracle-{case_id}-{plan_index}-{step_index:02d}",
        )
        for step_index, node in enumerate(order)
    ]


def _required_arguments() -> dict[str, str]:
    return {
        "entity_id": "string",
        "operation_id": "string",
        "expected_version": "integer",
    }


def _build_group(
    *,
    split: Split,
    family_index: int,
    family_ordinal: int,
    tier: str,
    tier_ordinal: int,
    base_seed: int,
) -> list[CounterfactualWorkflowTask]:
    template = _FAMILY_TEMPLATES[family_index]
    seed = _SPLIT_OFFSETS[split] + base_seed + family_index * 1_000_000 + family_ordinal
    rng = random.Random(seed)
    optimal_steps = _optimal_length(tier, tier_ordinal)
    prefix_size = optimal_steps - 3
    topology = _topology(split, family_index, tier_ordinal)
    group_id = f"{split.value}-{template.key}-group-{family_ordinal:04d}"
    entity_id = f"{split.value}-{template.key[:4]}-world-{_opaque(seed, 'entity', size=6)}"
    priority = rng.choice(("standard", "priority", "urgent"))
    region = rng.choice(("apac", "emea", "na"))
    channel = rng.choice(("web", "mobile", "api"))

    nodes: list[WorkflowNode] = []
    schemas: list[ToolSchema] = []
    latency_trace: dict[str, float] = {}
    prefix_dependencies = _prefix_predecessors(topology, prefix_size)
    workflow_side_effect = f"{template.key}.mutation_recorded"
    lane_prefix_ids: dict[str, list[str]] = {}
    lane_checkpoints: dict[str, tuple[WorkflowNode, WorkflowNode]] = {}
    lane_finals: dict[tuple[str, str], WorkflowNode] = {}

    for lane in GOAL_VARIANTS:
        prefix_ids = [
            _operation_id(seed, f"lane:{lane}:prefix:{index}") for index in range(prefix_size)
        ]
        lane_prefix_ids[lane] = prefix_ids
        for index, stage in enumerate(template.stages[:prefix_size]):
            predecessors = [prefix_ids[value] for value in prefix_dependencies[index]]
            node_id = prefix_ids[index]
            tool_name = _tool_name(template.key, seed, f"lane:{lane}:prefix:{index}")
            transition_semantics = _semantic_text(
                _PREFIX_AXIS_DESCRIPTION[f"a{lane[0]}"],
                seed,
                f"lane-prefix-a:{lane}:{index}",
            )
            decision_semantics = _semantic_text(
                _PREFIX_AXIS_DESCRIPTION[f"b{lane[1]}"],
                seed,
                f"lane-prefix-b:{lane}:{index}",
            )
            latency = sample_mutating_service_time(rng)
            marker = f"workflow_marker_{_opaque(seed, f'lane:{lane}:prefix:{index}', size=5)}"
            node = WorkflowNode(
                node_id=node_id,
                tool_name=tool_name,
                required_predecessors=predecessors,
                effects={marker: True},
                side_effect=workflow_side_effect,
                simulated_latency_s=latency,
            )
            nodes.append(node)
            schemas.append(
                ToolSchema(
                    name=tool_name,
                    description=(
                        f"Within the {template.title} workflow, carry out "
                        f"{stage.replace('_', ' ')}. {transition_semantics} "
                        f"{decision_semantics}"
                    ),
                    required_arguments=_required_arguments(),
                    mutating=True,
                    idempotent=True,
                    side_effect=workflow_side_effect,
                    operation_id=node_id,
                    required_completed_operations=predecessors,
                )
            )
            latency_trace[node_id] = latency

        checkpoints: list[WorkflowNode] = []
        for checkpoint_index in range(2):
            node_id = _operation_id(seed, f"lane:{lane}:checkpoint:{checkpoint_index}")
            tool_name = _tool_name(
                template.key,
                seed,
                f"lane:{lane}:checkpoint:{checkpoint_index}",
            )
            transition_semantics = _semantic_text(
                _CHECKPOINT_AXIS_DESCRIPTION[f"a{lane[0]}"],
                seed,
                f"lane-checkpoint-a:{lane}:{checkpoint_index}",
            )
            decision_semantics = _semantic_text(
                _CHECKPOINT_AXIS_DESCRIPTION[f"b{lane[1]}"],
                seed,
                f"lane-checkpoint-b:{lane}:{checkpoint_index}",
            )
            latency = sample_mutating_service_time(rng)
            marker_label = f"lane:{lane}:checkpoint:{checkpoint_index}"
            marker = f"workflow_marker_{_opaque(seed, marker_label, size=5)}"
            node = WorkflowNode(
                node_id=node_id,
                tool_name=tool_name,
                required_predecessors=list(prefix_ids),
                effects={marker: True},
                side_effect=workflow_side_effect,
                simulated_latency_s=latency,
            )
            checkpoints.append(node)
            nodes.append(node)
            schemas.append(
                ToolSchema(
                    name=tool_name,
                    description=(
                        f"Record a route checkpoint for the {template.title} workflow. "
                        f"{transition_semantics} {decision_semantics}"
                    ),
                    required_arguments=_required_arguments(),
                    mutating=True,
                    idempotent=True,
                    side_effect=workflow_side_effect,
                    operation_id=node_id,
                    required_completed_operations=list(prefix_ids),
                )
            )
            latency_trace[node_id] = latency
        lane_checkpoints[lane] = (checkpoints[0], checkpoints[1])

        final_predecessors = [node.node_id for node in checkpoints]
        for outcome in GOAL_VARIANTS:
            node_id = _operation_id(seed, f"lane:{lane}:final:{outcome}")
            tool_name = _tool_name(template.key, seed, f"lane:{lane}:final:{outcome}")
            transition_semantics = _semantic_text(
                _COMPLETION_AXIS_DESCRIPTION[f"a{outcome[0]}"],
                seed,
                f"lane-final-a:{lane}:{outcome}",
            )
            decision_semantics = _semantic_text(
                _COMPLETION_AXIS_DESCRIPTION[f"b{outcome[1]}"],
                seed,
                f"lane-final-b:{lane}:{outcome}",
            )
            latency = sample_mutating_service_time(rng)
            node = WorkflowNode(
                node_id=node_id,
                tool_name=tool_name,
                required_predecessors=final_predecessors,
                effects={
                    "status": "completed",
                    "route_signature": lane,
                    "goal_signature": outcome,
                },
                side_effect=workflow_side_effect,
                simulated_latency_s=latency,
            )
            lane_finals[(lane, outcome)] = node
            nodes.append(node)
            schemas.append(
                ToolSchema(
                    name=tool_name,
                    description=(
                        f"Complete the {template.title} workflow on this route. "
                        f"{transition_semantics} {decision_semantics}"
                    ),
                    required_arguments=_required_arguments(),
                    mutating=True,
                    idempotent=True,
                    side_effect=workflow_side_effect,
                    operation_id=node_id,
                    required_completed_operations=final_predecessors,
                )
            )
            latency_trace[node_id] = latency

    schemas.extend(
        (
            ToolSchema(
                name=f"{template.key}.inspect_status",
                description="Read the current workflow status without changing it.",
                required_arguments=_required_arguments(),
                mutating=False,
                idempotent=True,
                operation_id="inspect-status",
            ),
            ToolSchema(
                name=f"{template.key}.review_context",
                description="Review visible workflow context without changing it.",
                required_arguments=_required_arguments(),
                mutating=False,
                idempotent=True,
                operation_id="inspect-status",
            ),
        )
    )
    forbidden_side_effect = f"{template.key}.force_complete"
    schemas.append(
        ToolSchema(
            name=f"{template.key}.force_complete",
            description="Unsafe administrative shortcut retained as a negative candidate.",
            required_arguments=_required_arguments(),
            mutating=True,
            idempotent=False,
            side_effect=forbidden_side_effect,
            operation_id="force-complete",
            policy_allowed=False,
        )
    )

    initial_state: dict[str, Any] = {
        "entity_id": entity_id,
        "version": 0,
        "status": "pending",
        "priority": priority,
        "region": region,
        "channel": channel,
        "approval_token": None,
        "route_signature": None,
        "goal_signature": None,
        "completed_nodes": [],
        "side_effects": [],
        "processed_call_ids": [],
        "idempotency_results": {},
        "audit_log": [],
    }
    visible_observation = {
        key: value
        for key, value in initial_state.items()
        if key not in {"processed_call_ids", "idempotency_results", "audit_log"}
    }

    tasks: list[CounterfactualWorkflowTask] = []
    for variant in GOAL_VARIANTS:
        case_id = f"{group_id}-goal-{_opaque(seed, f'case:{variant}', size=5)}"
        target_ids = set(lane_prefix_ids[variant])
        target_ids.update(
            {
                lane_checkpoints[variant][0].node_id,
                lane_checkpoints[variant][1].node_id,
                lane_finals[(variant, variant)].node_id,
            }
        )
        target_nodes = [node for node in nodes if node.node_id in target_ids]
        orders = _topological_orders(target_nodes, limit=4)
        if len(orders) < 4 or any(len(order) != optimal_steps for order in orders):
            raise RuntimeError(f"v1.4 generator failed to build an optimal plan for {case_id}")
        plans = [
            _plan_calls(case_id, entity_id, plan_index, order)
            for plan_index, order in enumerate(orders)
        ]
        transition_goal = _semantic_text(
            _TRANSITION_GOAL_TEXT[variant[0]],
            seed,
            f"goal-a:{variant[0]}",
        )
        decision_goal = _semantic_text(
            _DECISION_GOAL_TEXT[variant[1]],
            seed,
            f"goal-b:{variant[1]}",
        )
        tasks.append(
            CounterfactualWorkflowTask(
                case_id=case_id,
                group_id=group_id,
                goal_variant=variant,
                split=split,
                family=template.key,
                topology=topology,
                difficulty=tier,  # type: ignore[arg-type]
                seed=seed,
                entity_id=entity_id,
                user_goal=f"For the {template.title}: {transition_goal} {decision_goal}",
                initial_state=initial_state,
                visible_observation=visible_observation,
                tool_schemas=schemas,
                workflow_nodes=nodes,
                hidden_goal_predicates=[
                    StatePredicate(path="status", operator="eq", value="completed"),
                    StatePredicate(path="route_signature", operator="eq", value=variant),
                    StatePredicate(path="goal_signature", operator="eq", value=variant),
                ],
                forbidden_side_effects=[forbidden_side_effect],
                oracle_plans=plans,
                optimal_steps=optimal_steps,
                max_steps=optimal_steps,
                latency_trace=latency_trace,
                generator_version=GENERATOR_VERSION_V14,
            )
        )
    return tasks


def generate_counterfactual_tasks(
    split: Split | str,
    groups: int,
    *,
    base_seed: int = DEVELOPMENT_BASE_SEED_V14,
) -> list[CounterfactualWorkflowTask]:
    """Generate four cases per group with balanced families."""

    if groups < 1:
        raise ValueError("groups must be positive")
    resolved_split = split if isinstance(split, Split) else Split(split)
    family_sizes = _allocate(groups, len(_FAMILY_TEMPLATES))
    tasks: list[CounterfactualWorkflowTask] = []
    for family_index, family_size in enumerate(family_sizes):
        counts = _tier_counts(family_size)
        tiers = [name for name in ("short", "medium", "long") for _ in range(counts[name])]
        random.Random(_SPLIT_OFFSETS[resolved_split] + base_seed + family_index * 10_000).shuffle(
            tiers
        )
        tier_ordinals: Counter[str] = Counter()
        for ordinal, tier in enumerate(tiers):
            tasks.extend(
                _build_group(
                    split=resolved_split,
                    family_index=family_index,
                    family_ordinal=ordinal,
                    tier=tier,
                    tier_ordinal=tier_ordinals[tier],
                    base_seed=base_seed,
                )
            )
            tier_ordinals[tier] += 1
    return sorted(tasks, key=lambda task: task.case_id)


def generate_all_splits_v14(
    *,
    train_groups: int = CANONICAL_GROUP_COUNTS[Split.TRAIN],
    dev_groups: int = CANONICAL_GROUP_COUNTS[Split.DEV],
    test_groups: int = CANONICAL_GROUP_COUNTS[Split.TEST],
    base_seed: int = DEVELOPMENT_BASE_SEED_V14,
) -> dict[Split, list[CounterfactualWorkflowTask]]:
    splits = {
        split: generate_counterfactual_tasks(split, groups, base_seed=base_seed)
        for split, groups in (
            (Split.TRAIN, train_groups),
            (Split.DEV, dev_groups),
            (Split.TEST, test_groups),
        )
    }
    assert_v14_splits_disjoint(splits)
    return splits


def _groups(
    tasks: Iterable[CounterfactualWorkflowTask],
) -> dict[str, list[CounterfactualWorkflowTask]]:
    result: dict[str, list[CounterfactualWorkflowTask]] = {}
    for task in tasks:
        result.setdefault(task.group_id, []).append(task)
    return result


def assert_v14_splits_disjoint(
    splits: Mapping[Split, Sequence[CounterfactualWorkflowTask]],
) -> None:
    seen_groups: set[str] = set()
    seen_entities: set[str] = set()
    seen_seeds: set[int] = set()
    for split, tasks in splits.items():
        grouped = _groups(tasks)
        if any(len(group) != 4 for group in grouped.values()):
            raise ValueError(f"{split.value} contains an incomplete counterfactual group")
        groups = set(grouped)
        entities = {group[0].entity_id for group in grouped.values()}
        seeds = {group[0].seed for group in grouped.values()}
        if len(entities) != len(groups) or len(seeds) != len(groups):
            raise ValueError(f"duplicate world identity inside {split.value}")
        if seen_groups & groups or seen_entities & entities or seen_seeds & seeds:
            raise ValueError(f"cross-split counterfactual leakage in {split.value}")
        seen_groups.update(groups)
        seen_entities.update(entities)
        seen_seeds.update(seeds)


def _public_candidate_signature(candidates: Sequence[ToolCall]) -> list[dict[str, Any]]:
    return [
        {"tool_name": candidate.tool_name, "arguments": candidate.arguments}
        for candidate in candidates
    ]


def _policy_payload_without_goal(policy_input: PolicyInput) -> dict[str, Any]:
    payload = policy_input.model_dump(mode="json")
    observation = payload.get("observation")
    if not isinstance(observation, dict):
        raise TypeError("PolicyInput observation must serialize to an object")
    observation["user_goal"] = "<counterfactual-goal>"
    return payload


def _assert_shared_public_process(
    group_id: str,
    group: Sequence[CounterfactualWorkflowTask],
) -> None:
    """Replay one shared path and compare the complete public decision process.

    Static task equality proves that the four cases share a world definition.
    This dynamic check additionally guards candidate ordering, masking, and
    observation serialization after every state transition on a common public
    trajectory.  The trajectory may be goal-incompatible for three cases; it
    remains executable because goal relevance is evaluator-only.
    """

    # Imported lazily to keep module initialization free of policy backends.
    from agentic_tool_rl.envs.workflow import TransactionalWorkflowEnv

    environments = [TransactionalWorkflowEnv(task) for task in group]
    masks = [ActionMask(task.tool_schemas) for task in group]
    reference_plan = group[0].oracle_plans[0]
    for step_index, reference_call in enumerate(reference_plan):
        candidate_signatures: list[list[dict[str, Any]]] = []
        decision_masks: list[list[bool]] = []
        policy_payloads: list[dict[str, Any]] = []
        candidate_sets: list[list[ToolCall]] = []
        for task, environment, action_mask in zip(group, environments, masks, strict=True):
            observation = environment.observe()
            candidates = environment.candidate_actions(include_invalid=True)
            mask = action_mask.mask(observation, candidates)
            candidate_sets.append(candidates)
            candidate_signatures.append(_public_candidate_signature(candidates))
            decision_masks.append(mask)
            policy_payloads.append(
                _policy_payload_without_goal(
                    PolicyInput.from_decision(
                        observation,
                        candidates,
                        mask,
                        task.tool_schemas,
                    )
                )
            )
        if any(value != candidate_signatures[0] for value in candidate_signatures[1:]):
            raise ValueError(f"{group_id} candidate set/order leaks the goal at step {step_index}")
        if any(value != decision_masks[0] for value in decision_masks[1:]):
            raise ValueError(f"{group_id} action mask leaks the goal at step {step_index}")
        if any(value != policy_payloads[0] for value in policy_payloads[1:]):
            raise ValueError(
                f"{group_id} PolicyInput differs outside user_goal at step {step_index}"
            )

        for environment, candidates in zip(environments, candidate_sets, strict=True):
            selected = next(
                (
                    candidate
                    for candidate in candidates
                    if candidate.tool_name == reference_call.tool_name
                    and candidate.arguments == reference_call.arguments
                ),
                None,
            )
            if selected is None:
                raise ValueError(
                    f"{group_id} shared trajectory action is absent at step {step_index}"
                )
            outcome = environment.step(selected)
            if not outcome.accepted:
                raise ValueError(f"{group_id} shared trajectory was rejected at step {step_index}")


def optimal_candidate_indices(
    task: CounterfactualWorkflowTask,
    completed_nodes: Sequence[str],
    candidates: Sequence[ToolCall],
) -> set[int]:
    """Return all unit-cost actions that reduce goal distance by one."""

    completed = set(completed_nodes)
    target_operations = {str(call.arguments["operation_id"]) for call in task.oracle_plans[0]}
    result: set[int] = set()
    for index, candidate in enumerate(candidates):
        operation = str(candidate.arguments.get("operation_id", ""))
        if (
            operation in target_operations
            and operation not in completed
            and candidate.arguments.get("entity_id") == task.entity_id
            and candidate.arguments.get("expected_version") == len(completed)
        ):
            schema = next(
                (item for item in task.tool_schemas if item.name == candidate.tool_name),
                None,
            )
            if schema is not None and set(schema.required_completed_operations).issubset(completed):
                result.add(index)
    return result


def validate_counterfactual_groups(
    tasks: Sequence[CounterfactualWorkflowTask],
) -> CounterfactualGateReport:
    """Fail closed on v1.4 public-input, oracle, and branch invariants."""

    # Imported lazily to keep the generator independent from policy backends.
    from agentic_tool_rl.envs.workflow import TransactionalWorkflowEnv

    grouped = _groups(tasks)
    oracle_solvable = 0
    expert_states = 0
    multi_positive_states = 0
    goal_incompatible_states = 0
    allowed_differences = {
        "case_id",
        "goal_variant",
        "user_goal",
        "hidden_goal_predicates",
        "oracle_plans",
    }
    for group_id, group in grouped.items():
        if len(group) != 4 or {task.goal_variant for task in group} != set(GOAL_VARIANTS):
            raise ValueError(f"{group_id} must contain exactly the four goal variants")
        reference = group[0].model_dump(mode="json")
        reference_public = {
            key: value for key, value in reference.items() if key not in allowed_differences
        }
        for task in group:
            payload = task.model_dump(mode="json")
            public = {
                key: value for key, value in payload.items() if key not in allowed_differences
            }
            if public != reference_public:
                raise ValueError(f"{group_id} differs outside the allowed counterfactual fields")
            if len(task.oracle_plans) < 4:
                raise ValueError(f"{task.case_id} must expose at least four shortest plans")
            for plan_index in range(len(task.oracle_plans)):
                oracle_report = verify_task_solvable(task, plan_index)
                if not oracle_report.solvable or oracle_report.executed_steps != task.optimal_steps:
                    raise ValueError(
                        f"oracle plan {plan_index} failed for {task.case_id}: "
                        f"{oracle_report.reason}"
                    )
            oracle_solvable += 1
        _assert_shared_public_process(group_id, group)

        # One representative goal is sufficient for topology-level counts;
        # all four cases share the same public process and path length.
        task = group[0]
        environment = TransactionalWorkflowEnv(task)
        mask = ActionMask(task)
        for oracle_call in task.oracle_plans[0]:
            candidates = environment.candidate_actions(include_invalid=True)
            decisions = mask.mask(environment.observe(), candidates)
            positives = optimal_candidate_indices(
                task,
                environment.state.get("completed_nodes", []),
                candidates,
            )
            if not positives:
                raise ValueError(f"{task.case_id} has an expert state without a positive action")
            if not all(decisions[index] for index in positives):
                raise ValueError(f"{task.case_id} masks an optimal action")
            expert_states += 1
            multi_positive_states += len(positives) >= 2
            positive_operations = {
                str(candidates[index].arguments.get("operation_id")) for index in positives
            }
            incompatible = any(
                allowed
                and environment.dry_run(candidate).valid
                and str(candidate.arguments.get("operation_id")) not in positive_operations
                and next(
                    (
                        schema.mutating
                        for schema in task.tool_schemas
                        if schema.name == candidate.tool_name
                    ),
                    False,
                )
                for candidate, allowed in zip(candidates, decisions, strict=True)
            )
            goal_incompatible_states += incompatible
            selected = next(
                candidate
                for candidate in candidates
                if candidate.tool_name == oracle_call.tool_name
                and candidate.arguments == oracle_call.arguments
            )
            outcome = environment.step(selected)
            if not outcome.accepted:
                raise ValueError(f"oracle candidate was rejected for {task.case_id}")

    report = CounterfactualGateReport(
        groups=len(grouped),
        cases=len(tasks),
        oracle_solvable=oracle_solvable,
        minimum_oracle_plans_per_case=min(
            (len(task.oracle_plans) for task in tasks),
            default=0,
        ),
        invariant_groups=len(grouped),
        multi_positive_states=multi_positive_states,
        expert_states=expert_states,
        goal_incompatible_states=goal_incompatible_states,
    )
    if report.multi_positive_fraction < 0.20:
        raise ValueError("fewer than 20% of expert states have multiple optimal actions")
    if report.goal_incompatible_fraction < 0.80:
        raise ValueError("fewer than 80% of expert states expose a safe goal-incompatible action")
    return report


__all__ = [
    "CANONICAL_GROUP_COUNTS",
    "DEVELOPMENT_BASE_SEED_V14",
    "GENERATOR_VERSION_V14",
    "GOAL_VARIANTS",
    "CounterfactualGateReport",
    "assert_v14_splits_disjoint",
    "generate_all_splits_v14",
    "generate_counterfactual_tasks",
    "optimal_candidate_indices",
    "validate_counterfactual_groups",
]
