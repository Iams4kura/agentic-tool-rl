"""Canonical JSONL and manifest persistence for benchmark evidence."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from agentic_tool_rl.contracts import (
    ActionValidityExample,
    ActionValidityManifest,
    BenchmarkFile,
    BenchmarkManifest,
    CounterfactualWorkflowTask,
    Split,
    WorkflowTask,
)
from agentic_tool_rl.envs.action_validity import generate_action_validity_dataset
from agentic_tool_rl.envs.benchmark import (
    DEFAULT_BASE_SEED,
    GENERATOR_VERSION,
    generate_all_splits,
)
from agentic_tool_rl.envs.oracle import verify_task_solvable


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _pretty_json(value: Any) -> str:
    return f"{json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)}\n"


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        handle.write(content)
        temporary = Path(handle.name)
    temporary.replace(path)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sequence_digest(values: Sequence[str | int]) -> str:
    payload = "\n".join(str(value) for value in sorted(values, key=str)).encode()
    return hashlib.sha256(payload).hexdigest()


def save_tasks_jsonl(tasks: Sequence[WorkflowTask], path: str | Path) -> str:
    """Write canonical, newline-delimited task JSON and return its SHA256."""

    target = Path(path)
    content = "".join(
        f"{_canonical_json(task.model_dump(mode='json'))}\n"
        for task in sorted(tasks, key=lambda item: item.case_id)
    )
    _atomic_text(target, content)
    return sha256_file(target)


def load_tasks_jsonl(path: str | Path) -> list[WorkflowTask]:
    tasks: list[WorkflowTask] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
                if not isinstance(raw, dict):
                    raise ValueError("task row must be an object")
                model = (
                    CounterfactualWorkflowTask
                    if str(raw.get("generator_version", "")).startswith("benchmark-v1.4")
                    else WorkflowTask
                )
                tasks.append(model.model_validate(raw))
            except ValueError as error:
                raise ValueError(f"invalid task JSON at line {line_number}") from error
    return tasks


def save_action_validity_jsonl(
    examples: Sequence[ActionValidityExample], path: str | Path
) -> str:
    target = Path(path)
    content = "".join(
        f"{_canonical_json(example.model_dump(mode='json'))}\n"
        for example in sorted(examples, key=lambda item: item.sample_id)
    )
    _atomic_text(target, content)
    return sha256_file(target)


def load_action_validity_jsonl(path: str | Path) -> list[ActionValidityExample]:
    examples: list[ActionValidityExample] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                examples.append(ActionValidityExample.model_validate_json(line))
            except ValueError as error:
                raise ValueError(f"invalid action-validity JSON at line {line_number}") from error
    return examples


def write_action_validity_dataset(
    output_dir: str | Path,
    tasks: Sequence[WorkflowTask],
    *,
    states_per_task: int = 5,
    candidates_per_state: int = 4,
    seed: int = 20_260_808,
) -> ActionValidityManifest:
    """Generate and persist ``action_validity.jsonl`` plus its integrity manifest."""

    examples = generate_action_validity_dataset(
        tasks,
        states_per_task=states_per_task,
        candidates_per_state=candidates_per_state,
        seed=seed,
    )
    directory = Path(output_dir)
    path = directory / "action_validity.jsonl"
    digest = save_action_validity_jsonl(examples, path)
    invalid_counts = Counter(
        example.invalid_kind.value
        for example in examples
        if example.invalid_kind is not None
    )
    challenge_counts = Counter(example.challenge_source for example in examples)
    manifest = ActionValidityManifest(
        generator_version=GENERATOR_VERSION,
        source_task_count=len(tasks),
        states_per_task=states_per_task,
        candidates_per_state=candidates_per_state,
        seed=seed,
        count=len(examples),
        valid_count=sum(example.valid_label for example in examples),
        invalid_count=sum(not example.valid_label for example in examples),
        invalid_kind_counts=dict(sorted(invalid_counts.items())),
        challenge_counts=dict(sorted(challenge_counts.items())),
        sha256=digest,
        source_cases_digest=_sequence_digest([task.case_id for task in tasks]),
    )
    _atomic_text(
        directory / "action_validity.manifest.json",
        _pretty_json(manifest.model_dump(mode="json")),
    )
    return manifest


def write_benchmark(
    output_dir: str | Path,
    splits: Mapping[Split, Sequence[WorkflowTask]] | Mapping[str, Sequence[WorkflowTask]],
    *,
    base_seed: int = DEFAULT_BASE_SEED,
    verify_oracles: bool = True,
) -> BenchmarkManifest:
    """Persist split JSONL files and a deterministic integrity manifest."""

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    files: dict[str, BenchmarkFile] = {}
    for split_like, task_sequence in splits.items():
        split = split_like if isinstance(split_like, Split) else Split(split_like)
        tasks = list(task_sequence)
        if any(task.split != split for task in tasks):
            raise ValueError(f"task split mismatch while writing {split.value}")
        path = directory / f"{split.value}.jsonl"
        digest = save_tasks_jsonl(tasks, path)
        solvable = 0
        if verify_oracles:
            reports = [verify_task_solvable(task) for task in tasks]
            failed = [report for report in reports if not report.solvable]
            if failed:
                raise ValueError(
                    f"unsolvable generated task: {failed[0].case_id}: {failed[0].reason}"
                )
            solvable = len(reports)
        family_counts = Counter(task.family for task in tasks)
        length_counts = Counter(task.difficulty for task in tasks)
        files[split.value] = BenchmarkFile(
            split=split,
            path=path.name,
            count=len(tasks),
            sha256=digest,
            family_counts=dict(sorted(family_counts.items())),
            length_counts=dict(sorted(length_counts.items())),
            seed_digest=_sequence_digest([task.seed for task in tasks]),
            entity_digest=_sequence_digest([task.entity_id for task in tasks]),
            oracle_solvable=solvable,
        )
    manifest = BenchmarkManifest(
        generator_version=GENERATOR_VERSION,
        base_seed=base_seed,
        files=dict(sorted(files.items())),
    )
    _atomic_text(
        directory / "manifest.json",
        _pretty_json(manifest.model_dump(mode="json")),
    )
    return manifest


def generate_and_write_benchmark(
    output_dir: str | Path,
    *,
    train_size: int,
    dev_size: int,
    test_size: int,
    base_seed: int = DEFAULT_BASE_SEED,
    verify_oracles: bool = True,
) -> BenchmarkManifest:
    splits = generate_all_splits(
        train_size=train_size,
        dev_size=dev_size,
        test_size=test_size,
        base_seed=base_seed,
    )
    return write_benchmark(
        output_dir,
        splits,
        base_seed=base_seed,
        verify_oracles=verify_oracles,
    )


__all__ = [
    "generate_and_write_benchmark",
    "load_action_validity_jsonl",
    "load_tasks_jsonl",
    "save_action_validity_jsonl",
    "save_tasks_jsonl",
    "sha256_file",
    "write_action_validity_dataset",
    "write_benchmark",
]
