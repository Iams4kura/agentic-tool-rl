from __future__ import annotations

import hashlib
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from agentic_tool_rl.contracts import (
    ActionValidityExample,
    InvalidActionKind,
    Split,
    WorkflowTask,
)
from agentic_tool_rl.envs.action_validity import generate_action_validity_dataset
from agentic_tool_rl.envs.benchmark import generate_tasks
from agentic_tool_rl.evaluation import (
    CANONICAL_ACTION_VALIDITY_COUNT,
    CaseIdManifestError,
    ResumeGuard,
    ResumeIntegrityError,
    RunInputHashes,
    assert_exact_case_ids,
    build_case_id_manifest,
    canonical_run_signature,
    evaluate_action_validity,
    file_sha256,
    model_parameter_digest,
    verify_case_id_manifest,
)
from agentic_tool_rl.experiment import runtime_identity, source_fingerprint


def _frozen_examples(
    count: int = 2,
) -> tuple[list[ActionValidityExample], dict[str, WorkflowTask]]:
    tasks = generate_tasks(Split.TEST, count, base_seed=131)
    examples = generate_action_validity_dataset(tasks, seed=717)
    return examples, {task.case_id: task for task in tasks}


def _hashes() -> RunInputHashes:
    return RunInputHashes(
        benchmark_sha256="1" * 64,
        config_sha256="2" * 64,
        source_sha256="3" * 64,
        checkpoint_sha256="4" * 64,
    )


def test_frozen_action_validity_is_scored_by_policy_mask() -> None:
    examples, tasks = _frozen_examples()

    result = evaluate_action_validity(
        examples,
        tasks,
        expected_count=40,
        bootstrap_samples=30,
        bootstrap_seed=19,
    )

    assert result.count == 40
    assert result.cases == 2
    assert result.valid_count == result.invalid_count == 20
    assert result.valid_recall == 1.0
    assert result.invalid_recall == 0.95
    assert result.balanced_accuracy == 0.975
    assert result.macro_f1 == pytest.approx(0.9749843652282677)
    assert result.invalid_kind_recall == {
        "schema": 1.0,
        "grounding": 1.0,
        "precondition": 1.0,
        "safety": 0.8,
    }
    assert result.confusion_matrix == {"tp": 20, "tn": 19, "fp": 1, "fn": 0}
    assert result.bootstrap["balanced_accuracy"]["low"] == 0.95
    assert result.bootstrap["balanced_accuracy"]["high"] == 1.0

    grouped = result.predictions_by_case()
    assert set(grouped) == set(tasks)
    assert all(len(rows) == 20 for rows in grouped.values())
    assert all("predicted_invalid_kind" in row for rows in grouped.values() for row in rows)


def test_evaluator_reads_frozen_label_instead_of_relabelling_with_environment() -> None:
    examples, tasks = _frozen_examples(count=1)
    valid_index = next(index for index, example in enumerate(examples) if example.valid_label)
    payload = examples[valid_index].model_dump(mode="json")
    payload.update(valid_label=False, invalid_kind=InvalidActionKind.SCHEMA.value)
    deliberately_disagreeing_label = ActionValidityExample.model_validate(payload)
    changed = list(examples)
    changed[valid_index] = deliberately_disagreeing_label

    result = evaluate_action_validity(changed, tasks)

    prediction = next(
        item
        for item in result.predictions
        if item.sample_id == deliberately_disagreeing_label.sample_id
    )
    assert prediction.valid_label is False
    assert prediction.predicted_valid is True
    assert result.confusion_matrix["fp"] == 1
    assert result.invalid_recall < 1.0


def test_evaluator_rejects_wrong_cardinality_duplicate_ids_and_tampered_state() -> None:
    examples, tasks = _frozen_examples(count=1)

    with pytest.raises(ValueError, match="expected 20000"):
        evaluate_action_validity(
            examples,
            tasks,
            expected_count=CANONICAL_ACTION_VALIDITY_COUNT,
        )
    with pytest.raises(ValueError, match="duplicate sample_id"):
        evaluate_action_validity([*examples, examples[0]], tasks)

    bad_digest = examples[0].model_copy(update={"state_sha256": "0" * 64})
    with pytest.raises(ValueError, match="state digest mismatch"):
        evaluate_action_validity([bad_digest, *examples[1:]], tasks)


def test_canonical_20k_action_validity_gate_executes() -> None:
    tasks = generate_tasks(Split.TEST, 1000, base_seed=909)
    examples = generate_action_validity_dataset(tasks, seed=909)

    result = evaluate_action_validity(
        examples,
        {task.case_id: task for task in tasks},
        expected_count=CANONICAL_ACTION_VALIDITY_COUNT,
    )

    assert result.count == 20_000
    assert result.cases == 1000
    assert result.is_canonical_20k
    assert result.valid_count == result.invalid_count == 10_000


def test_file_and_model_parameter_digests_are_deterministic(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"abc")
    assert file_sha256(artifact) == hashlib.sha256(b"abc").hexdigest()

    torch.manual_seed(3)
    model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Linear(4, 2))
    copied = deepcopy(model)
    first = model_parameter_digest(model)
    assert len(first) == 64
    assert model_parameter_digest(copied) == first
    with torch.no_grad():
        next(copied.parameters()).add_(1.0)
    assert model_parameter_digest(copied) != first


def test_source_fingerprint_covers_lock_and_build_inputs(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "configs").mkdir()
    (tmp_path / "src" / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    (tmp_path / "configs" / "run.yaml").write_text("seed: 1\n", encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
    lock = tmp_path / "uv.lock"
    lock.write_text("version = 1\n", encoding="utf-8")

    before = source_fingerprint(tmp_path)
    lock.write_text("version = 2\n", encoding="utf-8")

    assert source_fingerprint(tmp_path) != before
    assert set(runtime_identity()) == {
        "python_implementation",
        "python_version",
        "python_cache_tag",
        "torch_version",
        "torch_cuda_version",
    }


def test_canonical_run_signature_covers_all_four_hashes() -> None:
    hashes = _hashes()
    reordered = {
        "source_sha256": hashes.source_sha256,
        "checkpoint_sha256": hashes.checkpoint_sha256,
        "benchmark_sha256": hashes.benchmark_sha256,
        "config_sha256": hashes.config_sha256,
    }
    signature = canonical_run_signature(hashes)

    assert len(signature) == 64
    assert canonical_run_signature(reordered) == signature
    assert canonical_run_signature(replace(hashes, checkpoint_sha256="5" * 64)) != signature
    with pytest.raises(ValueError, match="run hash keys differ"):
        canonical_run_signature({**reordered, "extra": "6" * 64})


def test_resume_guard_allows_only_byte_identical_run_inputs(tmp_path: Path) -> None:
    trace = tmp_path / "traces.jsonl"
    trace.touch()
    guard = ResumeGuard(tmp_path / "run-integrity.json")
    hashes = _hashes()

    signature = guard.authorize(hashes, trace_paths=[trace])
    trace.write_text('{"case_id":"case-001"}\n', encoding="utf-8")

    assert guard.authorize(hashes, trace_paths=[trace]) == signature
    assert guard.verify(hashes) == signature
    with pytest.raises(ResumeIntegrityError, match="config_sha256"):
        guard.authorize(replace(hashes, config_sha256="9" * 64), trace_paths=[trace])


def test_resume_guard_refuses_unattributed_existing_trace(tmp_path: Path) -> None:
    trace = tmp_path / "traces.jsonl"
    trace.write_text('{"case_id":"case-001"}\n', encoding="utf-8")

    with pytest.raises(ResumeIntegrityError, match="without prior integrity evidence"):
        ResumeGuard(tmp_path / "missing-guard.json").authorize(
            _hashes(), trace_paths=[trace]
        )


def test_case_id_manifest_requires_exact_expected_set() -> None:
    manifest = build_case_id_manifest(["case-c", "case-a", "case-b"])

    assert manifest["case_ids"] == ["case-a", "case-b", "case-c"]
    assert verify_case_id_manifest(manifest, ["case-b", "case-c", "case-a"]) == (
        "case-a",
        "case-b",
        "case-c",
    )
    assert assert_exact_case_ids(["case-a", "case-b"], ["case-b", "case-a"]) == (
        "case-a",
        "case-b",
    )

    with pytest.raises(
        CaseIdManifestError,
        match=r"missing=\['case-x'\].*extra=\['case-c'\]",
    ):
        verify_case_id_manifest(manifest, ["case-a", "case-b", "case-x"])
    with pytest.raises(ValueError, match="duplicate"):
        build_case_id_manifest(["case-a", "case-a"])

    tampered = dict(manifest)
    tampered["sha256"] = "0" * 64
    with pytest.raises(CaseIdManifestError, match="SHA256"):
        verify_case_id_manifest(tampered, ["case-a", "case-b", "case-c"])
