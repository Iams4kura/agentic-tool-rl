from __future__ import annotations

import re
from collections import Counter, defaultdict
from pathlib import Path

import pytest

from agentic_tool_rl.contracts import Split, WorkflowTask
from agentic_tool_rl.envs.action_validity import generate_action_validity_dataset
from agentic_tool_rl.envs.benchmark import (
    GENERATOR_VERSION,
    WORKFLOW_FAMILIES,
    WORKFLOW_TOPOLOGIES,
    assert_splits_disjoint,
    generate_all_splits,
    generate_tasks,
)
from agentic_tool_rl.envs.oracle import (
    replay_oracle,
    validate_workflow_dag,
    verify_task_solvable,
)
from agentic_tool_rl.envs.persistence import (
    load_action_validity_jsonl,
    load_tasks_jsonl,
    save_tasks_jsonl,
    sha256_file,
    write_action_validity_dataset,
    write_benchmark,
)
from agentic_tool_rl.latency_profile import (
    MUTATING_SERVICE_QUANTILES_S,
    SOURCE_ARCHIVE_SHA256,
)


def test_generated_service_times_use_the_frozen_public_trace_profile() -> None:
    tasks = generate_tasks(Split.TEST, 32, base_seed=818)

    observed = {latency for task in tasks for latency in task.latency_trace.values()}

    assert observed == set(MUTATING_SERVICE_QUANTILES_S)
    assert SOURCE_ARCHIVE_SHA256 == (
        "aff8b3ca7240a41a109e4ee598e0a96e45fcb92e7b8395ac19cb3748cd260d89"
    )


def test_canonical_1000_case_distribution_and_invariants() -> None:
    tasks = generate_tasks(Split.TEST, 1000)

    assert len(tasks) == 1000
    assert len({task.case_id for task in tasks}) == 1000
    assert len({task.seed for task in tasks}) == 1000
    assert len({task.entity_id for task in tasks}) == 1000
    assert Counter(task.family for task in tasks) == {family: 100 for family in WORKFLOW_FAMILIES}

    tiers_by_family: dict[str, Counter[str]] = defaultdict(Counter)
    for task in tasks:
        tiers_by_family[task.family][task.difficulty] += 1
        assert task.generator_version == GENERATOR_VERSION
        assert task.topology in WORKFLOW_TOPOLOGIES
        assert task.optimal_steps == len(task.workflow_nodes)
        assert task.max_steps == task.optimal_steps + 4
        assert len({node.tool_name for node in task.workflow_nodes}) >= 3
        assert any(node.required_predecessors for node in task.workflow_nodes)
        assert len(task.oracle_plans) >= 2
        assert all(len(plan) == task.optimal_steps for plan in task.oracle_plans)
        assert task.forbidden_side_effects
        assert any(
            schema.side_effect in task.forbidden_side_effects for schema in task.tool_schemas
        )
        operation_ids = [node.node_id for node in task.workflow_nodes]
        assert len(operation_ids) == len(set(operation_ids))
        assert all(re.fullmatch(r"op_[0-9a-f]{20}", value) for value in operation_ids)
        assert all("step" not in value.lower() for value in operation_ids)
        schemas_by_operation = {
            schema.operation_id: schema
            for schema in task.tool_schemas
            if schema.mutating and schema.policy_allowed
        }
        assert {node.node_id: node.required_predecessors for node in task.workflow_nodes} == {
            operation_id: schema.required_completed_operations
            for operation_id, schema in schemas_by_operation.items()
        }

    assert all(
        counts == {"short": 30, "medium": 40, "long": 30} for counts in tiers_by_family.values()
    )
    ranges = {"short": range(6, 9), "medium": range(9, 12), "long": range(12, 16)}
    assert all(task.optimal_steps in ranges[task.difficulty] for task in tasks)


def test_family_topology_combinations_are_held_out_from_train_and_dev() -> None:
    splits = generate_all_splits(
        train_size=120,
        dev_size=120,
        test_size=40,
        base_seed=8181,
    )

    train_pairs = {(task.family, task.topology) for task in splits[Split.TRAIN]}
    dev_pairs = {(task.family, task.topology) for task in splits[Split.DEV]}
    test_pairs = {(task.family, task.topology) for task in splits[Split.TEST]}

    assert test_pairs.isdisjoint(train_pairs)
    assert test_pairs.isdisjoint(dev_pairs)
    assert {task.topology for tasks in splits.values() for task in tasks} == set(
        WORKFLOW_TOPOLOGIES
    )
    assert Counter(task.family for task in splits[Split.TEST]) == {
        family: 4 for family in WORKFLOW_FAMILIES
    }


def test_four_topologies_are_structurally_distinct_at_the_same_length() -> None:
    tasks = generate_tasks(Split.TEST, 1000, base_seed=5150)
    signatures: dict[str, tuple[tuple[int, ...], ...]] = {}
    for task in tasks:
        if task.optimal_steps != 10 or task.topology in signatures:
            continue
        positions = {node.node_id: index for index, node in enumerate(task.workflow_nodes)}
        signatures[task.topology] = tuple(
            tuple(sorted(positions[value] for value in node.required_predecessors))
            for node in task.workflow_nodes
        )

    assert set(signatures) == set(WORKFLOW_TOPOLOGIES)
    assert len(set(signatures.values())) == len(WORKFLOW_TOPOLOGIES)


def test_opaque_operation_ids_are_seeded_per_task() -> None:
    first = generate_tasks(Split.TEST, 1, base_seed=100)[0]
    repeated = generate_tasks(Split.TEST, 1, base_seed=100)[0]
    changed_seed = generate_tasks(Split.TEST, 1, base_seed=101)[0]

    first_ids = [node.node_id for node in first.workflow_nodes]

    assert first == repeated
    assert set(first_ids).isdisjoint(node.node_id for node in changed_seed.workflow_nodes)


def test_generator_is_deterministic_down_to_jsonl_bytes(tmp_path: Path) -> None:
    first = generate_tasks("test", 32, base_seed=701)
    second = generate_tasks("test", 32, base_seed=701)
    first_path = tmp_path / "first.jsonl"
    second_path = tmp_path / "second.jsonl"

    first_digest = save_tasks_jsonl(first, first_path)
    second_digest = save_tasks_jsonl(second, second_path)

    assert first == second
    assert first_path.read_bytes() == second_path.read_bytes()
    assert first_digest == second_digest == sha256_file(first_path)
    assert load_tasks_jsonl(first_path) == first


def test_arbitrary_smoke_size_and_split_isolation() -> None:
    splits = generate_all_splits(train_size=7, dev_size=13, test_size=32, base_seed=17)

    assert {split: len(tasks) for split, tasks in splits.items()} == {
        Split.TRAIN: 7,
        Split.DEV: 13,
        Split.TEST: 32,
    }
    assert_splits_disjoint(splits)
    for tasks in splits.values():
        counts = Counter(task.family for task in tasks)
        assert len(tasks) <= 1 or max(counts.values()) - min(counts.values()) <= 1


def test_every_canonical_task_is_oracle_solvable() -> None:
    tasks = generate_tasks(Split.TEST, 1000)
    reports = [verify_task_solvable(task) for task in tasks]

    assert all(report.solvable for report in reports)
    assert [report.executed_steps for report in reports] == [task.optimal_steps for task in tasks]


def test_oracle_accepts_both_legal_fork_orders() -> None:
    task = generate_tasks(Split.TEST, 1)[0]

    first = replay_oracle(task, plan_index=0)
    second = replay_oracle(task, plan_index=1)

    assert first[-1].success
    assert second[-1].success
    assert [call.arguments["operation_id"] for call in task.oracle_plans[0]] != [
        call.arguments["operation_id"] for call in task.oracle_plans[1]
    ]


def test_dag_validator_rejects_an_unknown_predecessor() -> None:
    task = generate_tasks(Split.TEST, 1)[0]
    bad_nodes = list(task.workflow_nodes)
    bad_nodes[0] = bad_nodes[0].model_copy(update={"required_predecessors": ["missing"]})
    malformed = task.model_copy(update={"workflow_nodes": bad_nodes})

    with pytest.raises(ValueError, match="unknown predecessors"):
        validate_workflow_dag(malformed)


def test_manifest_records_hashes_counts_strata_and_oracle_status(tmp_path: Path) -> None:
    splits = generate_all_splits(train_size=4, dev_size=4, test_size=20, base_seed=911)

    manifest = write_benchmark(tmp_path, splits, base_seed=911)
    reloaded = write_benchmark(tmp_path, splits, base_seed=911)

    assert manifest == reloaded
    assert (tmp_path / "manifest.json").is_file()
    for split, tasks in splits.items():
        entry = manifest.files[split.value]
        path = tmp_path / entry.path
        assert entry.count == len(tasks)
        assert entry.sha256 == sha256_file(path)
        assert entry.oracle_solvable == len(tasks)
        assert sum(entry.family_counts.values()) == len(tasks)
        assert sum(entry.length_counts.values()) == len(tasks)
        assert load_tasks_jsonl(path) == sorted(tasks, key=lambda task: task.case_id)


def test_contract_round_trip_is_lossless() -> None:
    task = generate_tasks(Split.DEV, 1)[0]
    payload = task.model_dump_json()

    assert WorkflowTask.model_validate_json(payload) == task


def test_canonical_action_validity_dataset_has_20k_balanced_independent_labels() -> None:
    tasks = generate_tasks(Split.TEST, 1000)

    examples = generate_action_validity_dataset(tasks)

    assert len(examples) == 20_000
    assert Counter(example.valid_label for example in examples) == {True: 10_000, False: 10_000}
    assert Counter(
        example.invalid_kind.value for example in examples if example.invalid_kind is not None
    ) == {
        "schema": 2500,
        "grounding": 2500,
        "precondition": 2500,
        "safety": 2500,
    }
    assert Counter(example.challenge_source for example in examples) == {
        "standard_candidate": 19_500,
        "hidden_ledger_collision": 500,
    }
    groups = Counter((example.case_id, example.state_index) for example in examples)
    assert len(groups) == 5000
    assert set(groups.values()) == {4}
    assert all(example.label_source == "environment_dry_run" for example in examples)


def test_action_validity_generation_and_persistence_are_deterministic(tmp_path: Path) -> None:
    tasks = generate_tasks(Split.TEST, 3, base_seed=51)

    first = generate_action_validity_dataset(tasks, seed=909)
    second = generate_action_validity_dataset(tasks, seed=909)
    manifest = write_action_validity_dataset(tmp_path, tasks, seed=909)

    assert first == second
    assert manifest.dataset_name == "action-validity-v2"
    assert manifest.count == 60
    assert manifest.valid_count == manifest.invalid_count == 30
    assert sum(manifest.challenge_counts.values()) == 60
    assert manifest.sha256 == sha256_file(tmp_path / "action_validity.jsonl")
    assert load_action_validity_jsonl(tmp_path / "action_validity.jsonl") == sorted(
        first, key=lambda example: example.sample_id
    )
    assert (tmp_path / "action_validity.manifest.json").is_file()


@pytest.mark.parametrize("size", [0, -1])
def test_generator_rejects_non_positive_sizes(size: int) -> None:
    with pytest.raises(ValueError, match="positive"):
        generate_tasks(Split.TEST, size)
