"""Deterministic artifacts and exact development protocol for benchmark-v1.4."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from agentic_tool_rl.contracts import CounterfactualWorkflowTask, Split
from agentic_tool_rl.envs.benchmark_v14 import (
    DEVELOPMENT_BASE_SEED_V14,
    GENERATOR_VERSION_V14,
    assert_v14_splits_disjoint,
    generate_all_splits_v14,
    validate_counterfactual_groups,
)
from agentic_tool_rl.envs.persistence import (
    load_tasks_jsonl,
    save_tasks_jsonl,
)
from agentic_tool_rl.evaluation.io import sha256_json, write_json_atomic
from agentic_tool_rl.package_resources import (
    IMPLEMENTATION_FINGERPRINT_SCHEMA,
    implementation_fingerprint_v2,
    load_packaged_protocol,
    packaged_protocol_sha256,
)

MANIFEST_SCHEMA_V14 = "benchmark-v1.4-development-manifest-v1"
_EXACT_DEVELOPMENT_COUNTS = {
    "train": {"groups": 500, "cases": 2000},
    "dev": {"groups": 100, "cases": 400},
    "test": {"groups": 250, "cases": 1000},
}
_DEVELOPMENT_QUALITY_GATES: dict[str, float | int] = {
    "minimum_deterministic_shortest_plans_per_case": 4,
    "minimum_goal_incompatible_state_fraction": 0.80,
    "minimum_multi_positive_expert_state_fraction": 0.20,
    "oracle_tsr": 1.0,
    "public_ready_tsr_max": 0.25,
    "strongest_public_rule_dev_tsr_strict_max": 0.80,
}


def _sequence_digest(values: Sequence[str | int]) -> str:
    return sha256_json(sorted(values, key=str))


def _development_protocol_identity(
    protocol_counts: Mapping[str, Mapping[str, int]],
    *,
    base_seed: int,
) -> dict[str, Any]:
    """Compare every implemented development-protocol field, not just sizes."""

    # The baseline registry lives in evaluation and is imported lazily to keep
    # persistence module initialization free of an env/evaluation import cycle.
    from agentic_tool_rl.evaluation.benchmark_v14 import PUBLIC_BASELINE_IDS

    protocol = load_packaged_protocol("benchmark-v1.4-development.json")
    exact_counts_match = dict(protocol_counts) == _EXACT_DEVELOPMENT_COUNTS
    checks = {
        "schema": protocol.get("schema_version")
        == "benchmark-v1.4-development-protocol-v1",
        "status": protocol.get("status") == "development-acceptance",
        "canonical_disabled": protocol.get("canonical_final_permitted") is False,
        "base_seed": protocol.get("development_base_seed") == base_seed,
        "generator": protocol.get("generator_version") == GENERATOR_VERSION_V14,
        "dataset": protocol.get("dataset")
        == {
            **_EXACT_DEVELOPMENT_COUNTS,
            "goals_per_group": 4,
        },
        "quality_gates": protocol.get("quality_gates")
        == _DEVELOPMENT_QUALITY_GATES,
        "baseline_registry": protocol.get("baselines")
        == ["O-oracle-shortest-path", *PUBLIC_BASELINE_IDS],
        "external_locks_open": (
            protocol.get("external_time_anchor_locked") is False
            and protocol.get("future_randomness_beacon_locked") is False
        ),
    }
    return {
        "schema": "benchmark-v1.4-development-protocol-v1",
        "resource": "benchmark-v1.4-development.json",
        "sha256": packaged_protocol_sha256("benchmark-v1.4-development.json"),
        "exact_counts_match": exact_counts_match,
        "checks": checks,
        "exact_development_match": exact_counts_match and all(checks.values()),
    }


def load_counterfactual_tasks_jsonl(path: str | Path) -> list[CounterfactualWorkflowTask]:
    tasks = load_tasks_jsonl(path)
    if any(not isinstance(task, CounterfactualWorkflowTask) for task in tasks):
        raise ValueError("file contains a non-v1.4 task")
    return [task for task in tasks if isinstance(task, CounterfactualWorkflowTask)]


def write_benchmark_v14(
    output_dir: str | Path,
    splits: Mapping[Split, Sequence[CounterfactualWorkflowTask]],
    *,
    base_seed: int = DEVELOPMENT_BASE_SEED_V14,
) -> Path:
    """Persist all splits and a fail-closed, explicitly non-canonical manifest."""

    assert_v14_splits_disjoint(splits)
    if set(splits) != {Split.TRAIN, Split.DEV, Split.TEST}:
        raise ValueError("v1.4 artifact requires train, dev, and test splits")
    directory = Path(output_dir)
    files: dict[str, dict[str, Any]] = {}
    protocol_counts: dict[str, dict[str, int]] = {}
    for split in (Split.TRAIN, Split.DEV, Split.TEST):
        tasks = list(splits[split])
        if any(task.split != split for task in tasks):
            raise ValueError(f"task split mismatch while writing {split.value}")
        gate = validate_counterfactual_groups(tasks)
        path = directory / f"{split.value}.jsonl"
        digest = save_tasks_jsonl(tasks, path)
        representatives = {
            task.group_id: task for task in tasks if task.goal_variant == "00"
        }
        family_counts = Counter(task.family for task in representatives.values())
        cell_counts = Counter(
            f"{task.family}|{task.difficulty}|{task.topology}"
            for task in representatives.values()
        )
        files[split.value] = {
            "path": path.name,
            "sha256": digest,
            "cases": len(tasks),
            "groups": len(representatives),
            "case_ids_sha256": _sequence_digest([task.case_id for task in tasks]),
            "group_ids_sha256": _sequence_digest(list(representatives)),
            "world_seeds_sha256": _sequence_digest(
                [task.seed for task in representatives.values()]
            ),
            "family_group_counts": dict(sorted(family_counts.items())),
            "populated_cell_group_counts": dict(sorted(cell_counts.items())),
            "counterfactual_gate": gate.to_dict(),
        }
        protocol_counts[split.value] = {
            "groups": len(representatives),
            "cases": len(tasks),
        }
    manifest: dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA_V14,
        "generator_version": GENERATOR_VERSION_V14,
        "purpose": "development-acceptance",
        "canonical": False,
        "canonical_final_permitted": False,
        "base_seed": base_seed,
        "implementation_identity": {
            "schema": IMPLEMENTATION_FINGERPRINT_SCHEMA,
            "sha256": implementation_fingerprint_v2(),
        },
        "protocol_identity": _development_protocol_identity(
            protocol_counts,
            base_seed=base_seed,
        ),
        "protocol": {
            "counterfactual_goals_per_group": 4,
            "counts": protocol_counts,
            "families": 10,
            "optimal_steps": {"minimum": 6, "maximum": 15},
            "public_ready_tsr_max": 0.25,
            "strongest_public_rule_dev_tsr_strict_max": 0.80,
            "minimum_multi_positive_fraction": 0.20,
            "minimum_goal_incompatible_state_fraction": 0.80,
            "minimum_deterministic_shortest_plans_per_case": 4,
            "external_time_anchor_locked": False,
            "future_randomness_beacon_locked": False,
        },
        "files": files,
    }
    return write_json_atomic(directory / "manifest.json", manifest)


def generate_and_write_benchmark_v14(
    output_dir: str | Path,
    *,
    train_groups: int = 500,
    dev_groups: int = 100,
    test_groups: int = 250,
    base_seed: int = DEVELOPMENT_BASE_SEED_V14,
) -> Path:
    splits = generate_all_splits_v14(
        train_groups=train_groups,
        dev_groups=dev_groups,
        test_groups=test_groups,
        base_seed=base_seed,
    )
    return write_benchmark_v14(output_dir, splits, base_seed=base_seed)


__all__ = [
    "MANIFEST_SCHEMA_V14",
    "generate_and_write_benchmark_v14",
    "load_counterfactual_tasks_jsonl",
    "write_benchmark_v14",
]
