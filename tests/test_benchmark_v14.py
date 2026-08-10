from __future__ import annotations

import re
from collections import Counter
from itertools import combinations

from agentic_tool_rl.contracts import Split
from agentic_tool_rl.envs.benchmark import generate_tasks
from agentic_tool_rl.envs.benchmark_v14 import (
    _CHECKPOINT_AXIS_DESCRIPTION,
    _COMPLETION_AXIS_DESCRIPTION,
    _DECISION_GOAL_TEXT,
    _PREFIX_AXIS_DESCRIPTION,
    _TRANSITION_GOAL_TEXT,
    DEVELOPMENT_BASE_SEED_V14,
    GENERATOR_VERSION_V14,
    GOAL_VARIANTS,
    assert_v14_splits_disjoint,
    generate_all_splits_v14,
    generate_counterfactual_tasks,
    validate_counterfactual_groups,
)
from agentic_tool_rl.envs.oracle import verify_task_solvable


def test_v13_contract_shape_remains_frozen() -> None:
    payload = generate_tasks(Split.TEST, 1)[0].model_dump(mode="json")

    assert "group_id" not in payload
    assert "goal_variant" not in payload
    assert payload["generator_version"] == "benchmark-v1.3.0"
    assert len(payload["workflow_nodes"]) == payload["optimal_steps"]


def test_counterfactual_group_differs_only_in_goal_and_hidden_targets() -> None:
    tasks = generate_counterfactual_tasks(Split.DEV, 1, base_seed=91)

    assert len(tasks) == 4
    assert {task.goal_variant for task in tasks} == set(GOAL_VARIANTS)
    assert len({task.case_id for task in tasks}) == 4
    assert len({task.group_id for task in tasks}) == 1
    assert len({task.entity_id for task in tasks}) == 1
    assert len({task.seed for task in tasks}) == 1
    assert len({task.user_goal for task in tasks}) == 4
    assert all(task.generator_version == GENERATOR_VERSION_V14 for task in tasks)
    assert all(len(task.workflow_nodes) > task.optimal_steps for task in tasks)
    assert all(len(task.oracle_plans) >= 4 for task in tasks)
    oracle_reports = [
        (task, verify_task_solvable(task, plan_index))
        for task in tasks
        for plan_index in range(len(task.oracle_plans))
    ]
    assert all(
        report.solvable and report.executed_steps == task.optimal_steps
        for task, report in oracle_reports
    )

    report = validate_counterfactual_groups(tasks)
    assert report.oracle_solvable == 4
    assert report.minimum_oracle_plans_per_case >= 4
    assert report.invariant_groups == 1
    assert report.multi_positive_fraction >= 0.20
    assert report.goal_incompatible_fraction >= 0.80


def test_v14_generator_is_deterministic_and_split_isolated() -> None:
    first = generate_all_splits_v14(
        train_groups=10,
        dev_groups=10,
        test_groups=10,
        base_seed=DEVELOPMENT_BASE_SEED_V14,
    )
    second = generate_all_splits_v14(
        train_groups=10,
        dev_groups=10,
        test_groups=10,
        base_seed=DEVELOPMENT_BASE_SEED_V14,
    )

    assert first == second
    assert_v14_splits_disjoint(first)
    assert {split: len(tasks) for split, tasks in first.items()} == {
        Split.TRAIN: 40,
        Split.DEV: 40,
        Split.TEST: 40,
    }


def test_v14_semantic_lanes_are_isomorphic_and_target_disjoint() -> None:
    tasks = generate_counterfactual_tasks(Split.DEV, 1, base_seed=91)
    reference = tasks[0]
    workflow_names = {node.tool_name for node in reference.workflow_nodes}
    descriptions = [
        schema.description.casefold()
        for schema in reference.tool_schemas
        if schema.name in workflow_names
    ]
    schema_by_operation = {
        schema.operation_id: schema
        for schema in reference.tool_schemas
        if schema.operation_id is not None
    }
    node_by_id = {node.node_id: node for node in reference.workflow_nodes}
    prefix_size = reference.optimal_steps - 3
    target_operations = {
        task.goal_variant: {str(call.arguments["operation_id"]) for call in task.oracle_plans[0]}
        for task in tasks
    }

    assert all(
        re.fullmatch(rf"{re.escape(reference.family)}\.execute_[0-9a-f]{{12}}", name)
        for name in workflow_names
    )
    assert not any(
        marker in name
        for name in workflow_names
        for marker in (
            "finalize_",
            "stage_in_bounded_batches",
            "switch_all_at_once",
            "hold_for_named_reviewer",
            "continue_after_automatic_checks",
        )
    )
    assert not any(
        marker in description
        for description in descriptions
        for marker in ("optional advisory", "decoy")
    )
    assert len(reference.workflow_nodes) == 4 * (prefix_size + 2 + 4)
    for left, right in combinations(GOAL_VARIANTS, 2):
        assert target_operations[left].isdisjoint(target_operations[right])
    assert all(
        len(operations) == reference.optimal_steps for operations in target_operations.values()
    )

    workflow_side_effects = {
        schema_by_operation[node.node_id].side_effect for node in reference.workflow_nodes
    }
    assert len(workflow_side_effects) == 1
    assert None not in workflow_side_effects

    schema_shapes = {
        (
            schema.mutating,
            schema.idempotent,
            schema.policy_allowed,
            tuple(sorted(schema.required_arguments.items())),
            schema.side_effect,
        )
        for schema in schema_by_operation.values()
        if schema.name in workflow_names
    }
    assert len(schema_shapes) == 1

    final_nodes = [node for node in reference.workflow_nodes if "status" in node.effects]
    marker_nodes = [node for node in reference.workflow_nodes if "status" not in node.effects]
    assert len(final_nodes) == 16
    assert len(marker_nodes) == 4 * (prefix_size + 2)
    assert all(
        node.effects == {next(iter(node.effects)): True}
        and re.fullmatch(r"workflow_marker_[0-9a-f]{10}", next(iter(node.effects)))
        for node in marker_nodes
    )
    assert all(
        set(node.effects) == {"status", "route_signature", "goal_signature"}
        and node.effects["status"] == "completed"
        for node in final_nodes
    )

    prefix_nodes = [node for node in marker_nodes if len(node.required_predecessors) < prefix_size]
    checkpoint_nodes = [
        node for node in marker_nodes if len(node.required_predecessors) == prefix_size
    ]
    assert len(prefix_nodes) == 4 * prefix_size
    assert len(checkpoint_nodes) == 8
    prefix_stems = Counter(
        schema_by_operation[node.node_id].description.split(". ", 1)[0] for node in prefix_nodes
    )
    assert len(prefix_stems) == prefix_size
    assert set(prefix_stems.values()) == {4}
    checkpoint_stems = {
        schema_by_operation[node.node_id].description.split(". ", 1)[0] for node in checkpoint_nodes
    }
    assert len(checkpoint_stems) == 1
    assert (
        len(
            {
                schema_by_operation[node.node_id].description.split(". ", 1)[0]
                for node in final_nodes
            }
        )
        == 1
    )
    assert all(
        len(schema_by_operation[node.node_id].description.split(". ")) == 3
        for node in reference.workflow_nodes
    )
    assert len({tuple(sorted(node.required_predecessors)) for node in final_nodes}) == 4
    assert set(
        Counter(tuple(sorted(node.required_predecessors)) for node in final_nodes).values()
    ) == {4}

    action_phrases = {
        phrase
        for bank in (
            _PREFIX_AXIS_DESCRIPTION,
            _CHECKPOINT_AXIS_DESCRIPTION,
            _COMPLETION_AXIS_DESCRIPTION,
        )
        for options in bank.values()
        for phrase in options
    }
    goal_phrases = {
        phrase
        for bank in (_TRANSITION_GOAL_TEXT, _DECISION_GOAL_TEXT)
        for options in bank.values()
        for phrase in options
    }
    assert action_phrases.isdisjoint(goal_phrases)

    lane_shapes: list[tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]] = []
    for task in tasks:
        variant = task.goal_variant
        operations = target_operations[variant]
        target_nodes = [node_by_id[operation] for operation in operations]
        target_final = [node for node in target_nodes if "status" in node.effects]
        target_checkpoints = [
            node
            for node in target_nodes
            if "status" not in node.effects and len(node.required_predecessors) == prefix_size
        ]
        target_prefixes = [
            node
            for node in target_nodes
            if "status" not in node.effects and len(node.required_predecessors) < prefix_size
        ]
        assert len(target_prefixes) == prefix_size
        assert len(target_checkpoints) == 2
        assert len(target_final) == 1
        assert target_final[0].effects == {
            "status": "completed",
            "route_signature": variant,
            "goal_signature": variant,
        }
        predicate_values = {
            predicate.path: predicate.value for predicate in task.hidden_goal_predicates
        }
        assert predicate_values["route_signature"] == variant
        assert predicate_values["goal_signature"] == variant

        final_predecessors = tuple(sorted(target_final[0].required_predecessors))
        lane_finals = [
            node
            for node in final_nodes
            if tuple(sorted(node.required_predecessors)) == final_predecessors
        ]
        assert len(lane_finals) == 4
        assert {node.effects["route_signature"] for node in lane_finals} == {variant}
        assert {node.effects["goal_signature"] for node in lane_finals} == set(GOAL_VARIANTS)

        lane_shapes.append(
            (
                tuple(sorted(len(node.required_predecessors) for node in target_prefixes)),
                tuple(sorted(len(node.required_predecessors) for node in target_checkpoints)),
                tuple(sorted(len(node.required_predecessors) for node in lane_finals)),
            )
        )
        assert any(option in task.user_goal for option in _TRANSITION_GOAL_TEXT[variant[0]])
        assert any(option in task.user_goal for option in _DECISION_GOAL_TEXT[variant[1]])
        for node in target_prefixes:
            description = schema_by_operation[node.node_id].description
            assert any(
                option in description for option in _PREFIX_AXIS_DESCRIPTION[f"a{variant[0]}"]
            )
            assert any(
                option in description for option in _PREFIX_AXIS_DESCRIPTION[f"b{variant[1]}"]
            )
        for node in target_checkpoints:
            description = schema_by_operation[node.node_id].description
            assert any(
                option in description for option in _CHECKPOINT_AXIS_DESCRIPTION[f"a{variant[0]}"]
            )
            assert any(
                option in description for option in _CHECKPOINT_AXIS_DESCRIPTION[f"b{variant[1]}"]
            )
        for node in lane_finals:
            outcome = str(node.effects["goal_signature"])
            description = schema_by_operation[node.node_id].description
            assert any(
                option in description for option in _COMPLETION_AXIS_DESCRIPTION[f"a{outcome[0]}"]
            )
            assert any(
                option in description for option in _COMPLETION_AXIS_DESCRIPTION[f"b{outcome[1]}"]
            )
    assert len(set(lane_shapes)) == 1


def test_v14_canonical_cardinality_and_test_cells() -> None:
    splits = generate_all_splits_v14()

    assert {split: len(tasks) for split, tasks in splits.items()} == {
        Split.TRAIN: 2000,
        Split.DEV: 400,
        Split.TEST: 1000,
    }
    representatives = {
        task.group_id: task for task in splits[Split.TEST] if task.goal_variant == "00"
    }
    assert len(representatives) == 250
    assert Counter(task.family for task in representatives.values()) == {
        family: 25 for family in {task.family for task in representatives.values()}
    }
    populated_cells = Counter(
        (task.family, task.difficulty, task.topology) for task in representatives.values()
    )
    assert min(populated_cells.values()) >= 5


def test_v14_topology_rotation_populates_exact_development_strata() -> None:
    splits = generate_all_splits_v14()

    assert (
        len({(task.family, task.difficulty, task.topology) for task in splits[Split.TRAIN]}) == 90
    )
    assert len({(task.family, task.difficulty, task.topology) for task in splits[Split.DEV]}) == 90
    assert len({(task.family, task.difficulty, task.topology) for task in splits[Split.TEST]}) == 30


def test_v14_canonical_test_passes_counterfactual_quality_gate() -> None:
    tasks = generate_counterfactual_tasks(Split.TEST, 250)

    report = validate_counterfactual_groups(tasks)

    assert report.groups == 250
    assert report.cases == 1000
    assert report.oracle_solvable == 1000
    assert report.minimum_oracle_plans_per_case >= 4
    assert report.multi_positive_fraction >= 0.20
    assert report.goal_incompatible_fraction >= 0.80
