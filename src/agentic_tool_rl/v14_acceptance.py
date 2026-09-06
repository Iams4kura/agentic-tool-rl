"""Real development-acceptance workflow for the v1.4 benchmark."""

from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path
from typing import Any

from agentic_tool_rl.contracts import Split
from agentic_tool_rl.envs.benchmark_v14 import (
    DEVELOPMENT_BASE_SEED_V14,
    generate_all_splits_v14,
)
from agentic_tool_rl.envs.persistence_v14 import write_benchmark_v14
from agentic_tool_rl.evaluation.benchmark_v14 import (
    run_baseline_suite,
    verify_baseline_suite,
    write_baseline_suite,
)
from agentic_tool_rl.evaluation.io import write_json_atomic


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_v14_development_acceptance(
    output_dir: str | Path,
    *,
    train_groups: int = 500,
    dev_groups: int = 100,
    test_groups: int = 250,
    base_seed: int = DEVELOPMENT_BASE_SEED_V14,
) -> dict[str, Any]:
    """Generate twice, validate public baselines, and persist reviewable evidence."""

    output = Path(output_dir)
    splits = generate_all_splits_v14(
        train_groups=train_groups,
        dev_groups=dev_groups,
        test_groups=test_groups,
        base_seed=base_seed,
    )
    write_json_atomic(
        output / "verification.json",
        {
            "schema_version": "benchmark-v1.4-development-verification-v1",
            "passed": False,
            "status": "publication-in-progress",
            "canonical": False,
            "canonical_final_executed": False,
            "base_seed": base_seed,
        },
    )
    manifest_path = write_benchmark_v14(output, splits, base_seed=base_seed)
    with tempfile.TemporaryDirectory(prefix="agentic-tool-rl-v14-replay-") as temporary:
        replay_root = Path(temporary)
        replay_splits = generate_all_splits_v14(
            train_groups=train_groups,
            dev_groups=dev_groups,
            test_groups=test_groups,
            base_seed=base_seed,
        )
        replay_manifest = write_benchmark_v14(
            replay_root,
            replay_splits,
            base_seed=base_seed,
        )
        deterministic_files = {
            name: _sha256(output / name) == _sha256(replay_root / name)
            for name in ("train.jsonl", "dev.jsonl", "test.jsonl", "manifest.json")
        }
        if not all(deterministic_files.values()):
            raise RuntimeError("benchmark-v1.4 generator replay is not byte deterministic")
        if manifest_path.read_bytes() != replay_manifest.read_bytes():
            raise RuntimeError("benchmark-v1.4 manifest replay mismatch")

    baseline_result = run_baseline_suite(splits[Split.DEV], random_seed=base_seed)
    baseline_report_path = write_baseline_suite(output / "dev-baselines", baseline_result)
    baseline_verification = verify_baseline_suite(
        output / "dev-baselines",
        splits[Split.DEV],
        random_seed=base_seed,
    )
    verification: dict[str, Any] = {
        "schema_version": "benchmark-v1.4-development-verification-v1",
        "passed": True,
        "canonical": False,
        "canonical_final_executed": False,
        "base_seed": base_seed,
        "counts": {
            split.value: {
                "groups": len({task.group_id for task in tasks}),
                "cases": len(tasks),
            }
            for split, tasks in splits.items()
        },
        "deterministic_replay": deterministic_files,
        "manifest": {
            "path": manifest_path.name,
            "sha256": _sha256(manifest_path),
        },
        "dev_baseline_report": {
            "path": baseline_report_path.relative_to(output).as_posix(),
            "sha256": _sha256(baseline_report_path),
            "passed": baseline_result.report["passed"],
            "independent_replay": baseline_verification,
            "oracle_tsr": baseline_result.report["oracle"],
            "strongest_public_rule_tsr": baseline_result.report[
                "strongest_public_rule_tsr"
            ],
        },
    }
    verification_path = write_json_atomic(output / "verification.json", verification)
    verification["verification_path"] = str(verification_path)
    return verification


__all__ = ["run_v14_development_acceptance"]
