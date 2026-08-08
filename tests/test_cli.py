from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import cast

import pytest
import torch
from typer.testing import CliRunner

import agentic_tool_rl.cli as cli_module
from agentic_tool_rl.cli import app
from agentic_tool_rl.config import (
    AblationConfig,
    AblationVariant,
    ExperimentConfig,
    load_config,
)
from agentic_tool_rl.contracts import WorkflowTask
from agentic_tool_rl.envs import TransactionalWorkflowEnv, load_tasks_jsonl
from agentic_tool_rl.evaluation import (
    canonical_run_signature,
    compute_metrics,
    file_sha256,
    read_jsonl,
    recompute_metrics,
    write_metrics,
)
from agentic_tool_rl.evaluation.io import canonical_json, sha256_json
from agentic_tool_rl.experiment import (
    ExperimentResult,
    prepare_benchmark,
    verify_experiment_manifest,
)
from agentic_tool_rl.features import FeatureEncoder
from agentic_tool_rl.models import ActionSample, ActorCritic, ProgressEstimator
from agentic_tool_rl.training import load_checkpoint, run_episode, save_checkpoint

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNNER = CliRunner()


def _manifest_artifact_path(manifest_path: Path, value: object) -> Path:
    assert isinstance(value, str) and value
    return manifest_path.parent / value


def _experiment_artifact_path(experiment: dict[str, object], value: object) -> Path:
    assert experiment["path_base"] == "root"
    assert isinstance(value, str) and value
    return Path(str(experiment["root"])) / value


class _ScriptedPolicy:
    """Test-only policy that follows a precomputed legal candidate sequence."""

    def __init__(self, action_indices: list[int]) -> None:
        self._action_indices = iter(action_indices)

    def act(
        self,
        state_features: torch.Tensor,
        action_features: torch.Tensor,
        action_mask: torch.Tensor,
        *,
        deterministic: bool = False,
    ) -> ActionSample:
        assert deterministic is True
        action_index = next(self._action_indices)
        assert bool(action_mask[0, action_index])
        return ActionSample(
            action_index=torch.tensor([action_index], dtype=torch.long),
            log_prob=torch.zeros(1, dtype=state_features.dtype),
            value=torch.zeros(1, dtype=state_features.dtype),
            entropy=torch.zeros(1, dtype=state_features.dtype),
            logits=torch.zeros((1, action_features.shape[1]), dtype=state_features.dtype),
        )


def _oracle_action_indices(task: WorkflowTask) -> list[int]:
    environment = TransactionalWorkflowEnv(task)
    indices: list[int] = []
    for oracle_call in task.oracle_plans[0]:
        candidates = environment.candidate_actions(include_invalid=True)
        action_index = next(
            index
            for index, candidate in enumerate(candidates)
            if candidate.tool_name == oracle_call.tool_name
            and candidate.arguments == oracle_call.arguments
        )
        indices.append(action_index)
        outcome = environment.step(candidates[action_index])
        assert outcome.accepted is True
    assert environment.evaluate().success is True
    return indices


def _oracle_trace(
    task: WorkflowTask,
    original_trace: dict[str, object],
    checkpoint_path: Path,
    config: ExperimentConfig,
    variant: AblationVariant,
) -> dict[str, object]:
    _, estimator, encoder, _ = load_checkpoint(checkpoint_path)
    scripted = _ScriptedPolicy(_oracle_action_indices(task))
    result = run_episode(
        scripted,  # type: ignore[arg-type]
        estimator,
        task,
        encoder,
        config.reward,
        trajectory_id=str(original_trace["case_id"]),
        use_action_mask=variant.action_mask,
        use_progress_reward=variant.progress_reward,
        deterministic=True,
        credit_assignment=("sequence" if variant.algorithm == "sequence_ppo" else "action"),
        gamma=config.training.ppo.gamma,
        trace_mode="compact",
    )
    return result.trace


def _refresh_trace_derived_evidence(
    manifest_path: Path,
    *,
    variant_name: str,
    traces: list[dict[str, object]],
) -> None:
    """Model an attacker refreshing every self-reported trace derivative."""

    manifest = json.loads(manifest_path.read_bytes())
    run = next(item for item in manifest["runs"] if item["variant"] == variant_name)
    trace_path = _manifest_artifact_path(manifest_path, run["traces"])
    trace_path.write_text("".join(canonical_json(row) + "\n" for row in traces), encoding="utf-8")
    evaluation = manifest["experiment_config"]["evaluation"]
    metrics = compute_metrics(
        traces,
        timeout_s=float(evaluation["timeout_s"]),
        bootstrap_samples=int(evaluation["bootstrap_samples"]),
        bootstrap_seed=int(evaluation["bootstrap_seed"]) + int(run["seed"]),
        confidence=float(evaluation["confidence"]),
    )
    metrics_path = _manifest_artifact_path(manifest_path, run["metrics"])
    write_metrics(metrics_path, metrics)
    recompute_metrics(
        trace_path,
        published_metrics_path=metrics_path,
        output_path=_manifest_artifact_path(manifest_path, run["recompute"]),
        tolerance=1e-9,
    )
    run.update(
        {
            "trace_sha256": file_sha256(trace_path),
            "cases": metrics.cases,
            "tsr": metrics.tsr,
            "successful_conditional_simulated_service_time_s": (
                metrics.successful_conditional_simulated_service_time_s
            ),
            "timeout_penalized_simulated_cost_s": (metrics.timeout_penalized_simulated_cost_s),
        }
    )
    manifest_path.write_text(canonical_json(manifest) + "\n", encoding="utf-8")


def _write_experiment_config(
    path: Path,
    *,
    train_cases: int,
    dev_cases: int,
    test_cases: int,
) -> Path:
    path.write_text(
        f"""\
kind: experiment
name: cli-e2e

benchmark:
  generator_version: benchmark-v1.3.0
  train_cases: {train_cases}
  dev_cases: {dev_cases}
  test_cases: {test_cases}
  families: 10
  min_steps: 6
  max_steps: 15
  seed: 20260808

policy:
  backend: lightweight
  observation_dim: 32
  action_dim: 32
  hidden_dim: 24

training:
  behavior_cloning:
    epochs: 1
    batch_size: 64
    learning_rate: 0.003
  ppo:
    iterations: 1
    epochs: 1
    rollout_episodes: 4
    minibatch_size: 128
    learning_rate: 0.0003
    gamma: 0.99
    gae_lambda: 0.95
    clip_ratio: 0.2
    value_coefficient: 0.5
    entropy_coefficient: 0.01
    max_grad_norm: 0.5

reward:
  task_success: 1.0
  task_failure: -1.0
  invalid_action: -0.2
  forbidden_side_effect: -1.0
  step_cost: -0.01
  progress_beta: 0.35

evaluation:
  timeout_s: 80.0
  bootstrap_samples: 0
  bootstrap_seed: 20260808
  confidence: 0.95
""",
        encoding="utf-8",
    )
    return path


def _json_output(output: str) -> dict[str, object]:
    value = json.loads(output)
    assert isinstance(value, dict)
    return value


def _standalone_artifacts(
    tmp_path: Path,
    *,
    variant_name: str,
) -> tuple[Path, Path, Path, AblationVariant]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    config_path = _write_experiment_config(
        tmp_path / "experiment.yaml",
        train_cases=2,
        dev_cases=2,
        test_cases=3,
    )
    config = load_config(config_path)
    assert isinstance(config, ExperimentConfig)
    ablation_path = PROJECT_ROOT / "configs/ablation.yaml"
    ablation = load_config(ablation_path)
    assert isinstance(ablation, AblationConfig)
    variant = next(item for item in ablation.variants if item.name == variant_name)
    benchmark_dir = tmp_path / "benchmark"
    prepare_benchmark(config, benchmark_dir)

    encoder = FeatureEncoder(
        state_dim=config.policy.observation_dim,
        action_dim=config.policy.action_dim,
    )
    model = ActorCritic(
        state_dim=encoder.state_dim,
        action_dim=encoder.action_dim,
        hidden_dim=config.policy.hidden_dim,
    )
    estimator = ProgressEstimator(encoder.state_dim, hidden_dim=64)
    checkpoint = tmp_path / f"{variant_name}.pt"
    save_checkpoint(
        checkpoint,
        model,
        estimator,
        encoder,
        seed=17,
        variant=variant.name,
        metadata={
            "schema_version": "1.0",
            "variant": variant.model_dump(mode="json"),
            "seed": 17,
        },
    )
    return config_path, benchmark_dir, checkpoint, variant


@pytest.mark.qwen
def test_qwen_dry_run_validates_wiring_without_loading_weights() -> None:
    result = RUNNER.invoke(
        app,
        [
            "qwen-dry-run",
            "--config",
            str(PROJECT_ROOT / "configs/qwen3_lora_gpu.yaml"),
        ],
    )

    assert result.exit_code == 0, result.output
    payload = _json_output(result.stdout)
    assert payload["valid"] is True
    assert payload["kind"] == "qwen_gpu"
    assert payload["model_name_or_path"] == "Qwen/Qwen3-4B"
    assert payload["would_download_weights"] is False
    assert payload["optional_dependencies"] == ["transformers", "peft", "accelerate"]


def test_generate_writes_frozen_benchmark_and_action_labels(tmp_path: Path) -> None:
    config = _write_experiment_config(
        tmp_path / "generate.yaml",
        train_cases=2,
        dev_cases=2,
        test_cases=3,
    )
    benchmark_dir = tmp_path / "benchmark"

    result = RUNNER.invoke(
        app,
        ["generate", "--config", str(config), "--output", str(benchmark_dir)],
    )

    assert result.exit_code == 0, result.output
    payload = _json_output(result.stdout)
    assert payload["benchmark_dir"] == str(benchmark_dir)
    manifest = payload["manifest"]
    action_validity = payload["action_validity"]
    assert isinstance(manifest, dict)
    assert isinstance(action_validity, dict)
    assert manifest["files"]["train"]["count"] == 2
    assert manifest["files"]["dev"]["count"] == 2
    assert manifest["files"]["test"]["count"] == 3
    assert action_validity["source_task_count"] == 3
    assert action_validity["count"] == 60
    assert action_validity["valid_count"] == action_validity["invalid_count"] == 30
    for name in (
        "train.jsonl",
        "dev.jsonl",
        "test.jsonl",
        "manifest.json",
        "action_validity.jsonl",
        "action_validity.manifest.json",
    ):
        assert (benchmark_dir / name).is_file()


def _evaluate_args(
    *,
    config: Path,
    benchmark: Path,
    checkpoint: Path,
    variant: str,
    output: Path,
) -> list[str]:
    return [
        "evaluate",
        "--checkpoint",
        str(checkpoint),
        "--variant",
        variant,
        "--ablation",
        str(PROJECT_ROOT / "configs/ablation.yaml"),
        "--config",
        str(config),
        "--benchmark-dir",
        str(benchmark),
        "--output",
        str(output),
    ]


def test_standalone_evaluate_resume_guard_binds_all_four_inputs(
    tmp_path: Path,
) -> None:
    config, benchmark, checkpoint, variant = _standalone_artifacts(
        tmp_path, variant_name="B-BC-Mask"
    )

    checkpoint_output = tmp_path / "checkpoint-output"
    first = RUNNER.invoke(
        app,
        _evaluate_args(
            config=config,
            benchmark=benchmark,
            checkpoint=checkpoint,
            variant=variant.name,
            output=checkpoint_output,
        ),
    )
    assert first.exit_code == 0, first.output
    assert (checkpoint_output / "run-integrity.json").is_file()

    test_jsonl = benchmark / "test.jsonl"
    original_test = test_jsonl.read_bytes()
    try:
        test_jsonl.write_bytes(original_test + b"\n")
        changed_test = RUNNER.invoke(
            app,
            _evaluate_args(
                config=config,
                benchmark=benchmark,
                checkpoint=checkpoint,
                variant=variant.name,
                output=checkpoint_output,
            ),
        )
        assert changed_test.exit_code != 0
        assert "benchmark test JSONL checksum mismatch" in str(
            changed_test.exception
        )
    finally:
        test_jsonl.write_bytes(original_test)

    alternate_checkpoint = tmp_path / "alternate.pt"
    model, estimator, encoder, metadata = load_checkpoint(checkpoint)
    with torch.no_grad():
        next(model.parameters()).add_(0.125)
    save_checkpoint(
        alternate_checkpoint,
        model,
        estimator,
        encoder,
        seed=17,
        variant=variant.name,
        metadata=metadata,
    )
    changed_checkpoint = RUNNER.invoke(
        app,
        _evaluate_args(
            config=config,
            benchmark=benchmark,
            checkpoint=alternate_checkpoint,
            variant=variant.name,
            output=checkpoint_output,
        ),
    )
    assert changed_checkpoint.exit_code != 0
    assert "checkpoint_sha256" in str(changed_checkpoint.exception)

    config_output = tmp_path / "config-output"
    assert (
        RUNNER.invoke(
            app,
            _evaluate_args(
                config=config,
                benchmark=benchmark,
                checkpoint=checkpoint,
                variant=variant.name,
                output=config_output,
            ),
        ).exit_code
        == 0
    )
    changed_config = tmp_path / "changed-config.yaml"
    changed_config.write_text(
        config.read_text(encoding="utf-8").replace("timeout_s: 80.0", "timeout_s: 81.0"),
        encoding="utf-8",
    )
    changed_config_result = RUNNER.invoke(
        app,
        _evaluate_args(
            config=changed_config,
            benchmark=benchmark,
            checkpoint=checkpoint,
            variant=variant.name,
            output=config_output,
        ),
    )
    assert changed_config_result.exit_code != 0
    assert "config_sha256" in str(changed_config_result.exception)

    benchmark_output = tmp_path / "benchmark-output"
    assert (
        RUNNER.invoke(
            app,
            _evaluate_args(
                config=config,
                benchmark=benchmark,
                checkpoint=checkpoint,
                variant=variant.name,
                output=benchmark_output,
            ),
        ).exit_code
        == 0
    )
    alternate_benchmark = tmp_path / "alternate-benchmark"
    shutil.copytree(benchmark, alternate_benchmark)
    alternate_manifest = alternate_benchmark / "manifest.json"
    alternate_manifest.write_bytes(alternate_manifest.read_bytes() + b"\n")
    changed_benchmark = RUNNER.invoke(
        app,
        _evaluate_args(
            config=config,
            benchmark=alternate_benchmark,
            checkpoint=checkpoint,
            variant=variant.name,
            output=benchmark_output,
        ),
    )
    assert changed_benchmark.exit_code != 0
    assert "benchmark_sha256" in str(changed_benchmark.exception)


def test_standalone_commands_reject_wrong_variant_and_use_sequence_credit(
    tmp_path: Path,
) -> None:
    config, benchmark, checkpoint, _ = _standalone_artifacts(
        tmp_path / "bc", variant_name="B-BC-Mask"
    )
    wrong_variant = RUNNER.invoke(
        app,
        [
            "run-episode",
            "--checkpoint",
            str(checkpoint),
            "--variant",
            "E-PPO-Progress-Mask",
            "--ablation",
            str(PROJECT_ROOT / "configs/ablation.yaml"),
            "--config",
            str(config),
            "--benchmark-dir",
            str(benchmark),
        ],
    )
    assert wrong_variant.exit_code == 2
    assert "checkpoint variant identity mismatch" in wrong_variant.output

    model, estimator, encoder, _ = load_checkpoint(checkpoint)
    shared_checkpoint = tmp_path / "shared-bc.pt"
    save_checkpoint(
        shared_checkpoint,
        model,
        estimator,
        encoder,
        seed=17,
        variant="shared-bc",
        metadata={"bc": {}, "progress": {}},
    )
    shared_as_variant = RUNNER.invoke(
        app,
        [
            "run-episode",
            "--checkpoint",
            str(shared_checkpoint),
            "--variant",
            "B-BC-Mask",
            "--ablation",
            str(PROJECT_ROOT / "configs/ablation.yaml"),
            "--config",
            str(config),
            "--benchmark-dir",
            str(benchmark),
        ],
    )
    assert shared_as_variant.exit_code == 2
    assert "observed 'shared-bc'" in shared_as_variant.output

    f_config, f_benchmark, f_checkpoint, f_variant = _standalone_artifacts(
        tmp_path / "sequence", variant_name="F-Sequence-PPO-Progress-Mask"
    )
    output = tmp_path / "sequence-evaluation"
    sequence = RUNNER.invoke(
        app,
        _evaluate_args(
            config=f_config,
            benchmark=f_benchmark,
            checkpoint=f_checkpoint,
            variant=f_variant.name,
            output=output,
        ),
    )
    assert sequence.exit_code == 0, sequence.output
    traces = read_jsonl(output / "traces.jsonl")
    assert traces
    assert all(trace["credit_assignment"] == "sequence" for trace in traces)
    assert all(
        any(not allowed for allowed in step["action_mask"])
        for trace in traces
        for step in trace["steps"]
    )

    episode_output = tmp_path / "sequence-episode.json"
    sequence_episode = RUNNER.invoke(
        app,
        [
            "run-episode",
            "--checkpoint",
            str(f_checkpoint),
            "--variant",
            f_variant.name,
            "--ablation",
            str(PROJECT_ROOT / "configs/ablation.yaml"),
            "--config",
            str(f_config),
            "--benchmark-dir",
            str(f_benchmark),
            "--output",
            str(episode_output),
        ],
    )
    assert sequence_episode.exit_code == 0, sequence_episode.output
    episode = json.loads(episode_output.read_text(encoding="utf-8"))
    assert episode["credit_assignment"] == "sequence"
    assert any(not allowed for step in episode["steps"] for allowed in step["action_mask"])


def test_smoke_runs_real_lightweight_loop_and_verify_run_accepts_it(
    tmp_path: Path,
) -> None:
    config = _write_experiment_config(
        tmp_path / "smoke.yaml",
        train_cases=10,
        dev_cases=4,
        test_cases=4,
    )
    benchmark_dir = tmp_path / "benchmark"
    output_dir = tmp_path / "runs"

    smoke = RUNNER.invoke(
        app,
        [
            "smoke",
            "--config",
            str(config),
            "--ablation",
            str(PROJECT_ROOT / "configs/ablation.yaml"),
            "--benchmark-dir",
            str(benchmark_dir),
            "--output",
            str(output_dir),
        ],
    )

    assert smoke.exit_code == 0, smoke.output
    payload = _json_output(smoke.stdout)
    verification = payload["verification"]
    experiment = payload["experiment"]
    assert isinstance(verification, dict)
    assert isinstance(experiment, dict)
    assert verification == {
        "passed": True,
        "runs": 2,
        "evaluation_units": 8,
        "checked": verification["checked"],
    }
    assert len(verification["checked"]) == 2
    assert experiment["variants"] == ["B-BC-Mask", "E-PPO-Progress-Mask"]
    assert experiment["path_base"] == "root"
    assert len(experiment["runs"]) == 2
    assert experiment["claim_gate"] is None
    run_root = Path(str(experiment["root"]))
    assert not (run_root / "claim-check.json").exists()
    bundled_benchmark = _experiment_artifact_path(experiment, experiment["benchmark_dir"])
    assert bundled_benchmark == run_root / "benchmark"
    run_by_variant = {run["variant"]: run for run in experiment["runs"]}
    assert run_by_variant["B-BC-Mask"]["ppo_updates"] == 0
    assert run_by_variant["B-BC-Mask"]["parameter_l2_delta"] == 0.0
    assert run_by_variant["E-PPO-Progress-Mask"]["ppo_updates"] > 0
    assert run_by_variant["E-PPO-Progress-Mask"]["parameter_l2_delta"] > 0.0
    for run in experiment["runs"]:
        assert "simulated_latency_s" not in run
        assert "balanced_accuracy" not in run
        assert "successful_conditional_simulated_service_time_s" in run
        assert "timeout_penalized_simulated_cost_s" in run
        assert not Path(str(run["directory"])).is_absolute()
        assert ".." not in Path(str(run["directory"])).parts
        metrics_path = _experiment_artifact_path(experiment, run["metrics"])
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        assert metrics["schema_version"] == "3.0"
        assert "balanced_accuracy" not in metrics
        traces = read_jsonl(_experiment_artifact_path(experiment, run["traces"]))
        assert all(trace["action_evaluations"] == [] for trace in traces)

    manifest = run_root / "run-manifest.json"
    manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert manifest_payload["schema_version"] == "3.0"
    assert manifest_payload["path_base"] == "run-manifest-parent"
    assert manifest_payload["root"] == "."
    assert manifest_payload["benchmark_dir"] == "benchmark"
    assert all(
        not Path(str(run[field])).is_absolute() and ".." not in Path(str(run[field])).parts
        for run in manifest_payload["runs"]
        for field in ("directory", "checkpoint", "traces", "metrics", "recompute")
    )
    verify = RUNNER.invoke(app, ["verify-run", "--manifest", str(manifest)])

    assert verify.exit_code == 0, verify.output
    verify_payload = _json_output(verify.stdout)
    assert verify_payload["passed"] is True
    assert verify_payload["runs"] == 2
    assert verify_payload["evaluation_units"] == 8

    # The run directory is a self-contained artifact bundle: its frozen
    # benchmark and every run artifact remain valid after relocation/rename.
    moved_root = tmp_path / "relocated" / "renamed-bundle"
    shutil.copytree(run_root, moved_root)
    moved_manifest = moved_root / "run-manifest.json"
    assert verify_experiment_manifest(moved_manifest)["passed"] is True

    original_moved_manifest = moved_manifest.read_bytes()
    path_attacks: tuple[tuple[str, str], ...] = (
        ("benchmark_dir", str(bundled_benchmark.resolve())),
        ("case_id_manifest", "../case-id-manifest.json"),
    )
    for field, forged_value in path_attacks:
        try:
            forged = json.loads(original_moved_manifest)
            forged[field] = forged_value
            moved_manifest.write_text(canonical_json(forged) + "\n", encoding="utf-8")
            with pytest.raises(ValueError, match=r"relative path|path base"):
                verify_experiment_manifest(moved_manifest)
        finally:
            moved_manifest.write_bytes(original_moved_manifest)

    for field, forged_value in (
        (
            "checkpoint",
            str(
                _experiment_artifact_path(
                    experiment, run_by_variant["B-BC-Mask"]["checkpoint"]
                ).resolve()
            ),
        ),
        ("traces", "../traces.jsonl"),
    ):
        try:
            forged = json.loads(original_moved_manifest)
            forged["runs"][0][field] = forged_value
            moved_manifest.write_text(canonical_json(forged) + "\n", encoding="utf-8")
            with pytest.raises(ValueError, match="relative path"):
                verify_experiment_manifest(moved_manifest)
        finally:
            moved_manifest.write_bytes(original_moved_manifest)

    escape_link = moved_root / "escape-benchmark"
    escape_link.symlink_to(benchmark_dir, target_is_directory=True)
    try:
        forged = json.loads(original_moved_manifest)
        forged["benchmark_dir"] = escape_link.name
        moved_manifest.write_text(canonical_json(forged) + "\n", encoding="utf-8")
        with pytest.raises(ValueError, match="escapes the artifact bundle"):
            verify_experiment_manifest(moved_manifest)
    finally:
        moved_manifest.write_bytes(original_moved_manifest)

    # Each evidence layer fails closed even when a neighbouring self-reported
    # checksum is also forged. Restore bytes between attacks so each assertion
    # isolates one verifier boundary.
    benchmark_test = bundled_benchmark / "test.jsonl"
    original = benchmark_test.read_bytes()
    try:
        benchmark_test.write_bytes(original + b"\n")
        with pytest.raises(RuntimeError, match="benchmark test JSONL checksum"):
            verify_experiment_manifest(manifest)
    finally:
        benchmark_test.write_bytes(original)

    action_jsonl = bundled_benchmark / "action_validity.jsonl"
    original = action_jsonl.read_bytes()
    try:
        action_jsonl.write_bytes(original + b"\n")
        with pytest.raises(RuntimeError, match="action-validity JSONL checksum"):
            verify_experiment_manifest(manifest)
    finally:
        action_jsonl.write_bytes(original)

    action_manifest = bundled_benchmark / "action_validity.manifest.json"
    original = action_manifest.read_bytes()
    try:
        forged = json.loads(original)
        forged["dataset_name"] = "action-validity-v1"
        action_manifest.write_text(canonical_json(forged) + "\n", encoding="utf-8")
        with pytest.raises(RuntimeError, match="unsupported action-validity"):
            verify_experiment_manifest(manifest)
    finally:
        action_manifest.write_bytes(original)

    original = action_manifest.read_bytes()
    try:
        forged = json.loads(original)
        forged["challenge_counts"]["standard_candidate"] += 1
        action_manifest.write_text(canonical_json(forged) + "\n", encoding="utf-8")
        with pytest.raises(RuntimeError, match="action-validity label summary"):
            verify_experiment_manifest(manifest)
    finally:
        action_manifest.write_bytes(original)

    case_manifest = Path(str(experiment["root"])) / "case-id-manifest.json"
    original = case_manifest.read_bytes()
    try:
        forged = json.loads(original)
        forged["case_ids"][0] = "forged-self-consistent-case"
        forged["case_ids"] = sorted(forged["case_ids"])
        identity = {
            "schema_version": forged["schema_version"],
            "count": forged["count"],
            "case_ids": forged["case_ids"],
        }
        forged["sha256"] = sha256_json(identity)
        case_manifest.write_text(canonical_json(forged) + "\n", encoding="utf-8")
        with pytest.raises(ValueError, match="case_id set mismatch"):
            verify_experiment_manifest(manifest)
    finally:
        case_manifest.write_bytes(original)

    full_directory = _experiment_artifact_path(
        experiment, run_by_variant["E-PPO-Progress-Mask"]["directory"]
    )
    training_path = full_directory / "training.json"
    expected_training = training_path.read_bytes()
    training_path.unlink()
    resumed = RUNNER.invoke(
        app,
        [
            "smoke",
            "--config",
            str(config),
            "--ablation",
            str(PROJECT_ROOT / "configs/ablation.yaml"),
            "--benchmark-dir",
            str(benchmark_dir),
            "--output",
            str(output_dir),
        ],
    )
    assert resumed.exit_code == 0, resumed.output
    assert training_path.read_bytes() == expected_training

    integrity_path = full_directory / "run-integrity.json"
    original = integrity_path.read_bytes()
    try:
        forged = json.loads(original)
        forged["run_signature"] = "0" * 64
        integrity_path.write_text(canonical_json(forged) + "\n", encoding="utf-8")
        with pytest.raises(RuntimeError, match="resume evidence mismatch"):
            verify_experiment_manifest(manifest)
    finally:
        integrity_path.write_bytes(original)

    original = training_path.read_bytes()
    try:
        forged = json.loads(original)
        forged["rollout_steps"] += 1
        training_path.write_text(canonical_json(forged) + "\n", encoding="utf-8")
        with pytest.raises(RuntimeError, match="training/checkpoint metadata mismatch"):
            verify_experiment_manifest(manifest)
    finally:
        training_path.write_bytes(original)

    checkpoint_path = _experiment_artifact_path(
        experiment, run_by_variant["E-PPO-Progress-Mask"]["checkpoint"]
    )
    original_checkpoint = checkpoint_path.read_bytes()
    original_training = training_path.read_bytes()
    original_integrity = integrity_path.read_bytes()
    original_manifest = manifest.read_bytes()
    try:
        checkpoint_payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        forged_training = json.loads(original_training)
        forged_delta = float(forged_training["parameter_l2_delta"]) + 1.0
        forged_training["parameter_l2_delta"] = forged_delta
        checkpoint_payload["metadata"] = forged_training
        torch.save(checkpoint_payload, checkpoint_path)
        training_path.write_text(canonical_json(forged_training) + "\n", encoding="utf-8")
        forged_manifest = json.loads(original_manifest)
        full_run = next(
            run for run in forged_manifest["runs"] if run["variant"] == "E-PPO-Progress-Mask"
        )
        forged_checkpoint_sha256 = file_sha256(checkpoint_path)
        full_run["checkpoint_sha256"] = forged_checkpoint_sha256
        full_run["parameter_l2_delta"] = forged_delta
        manifest.write_text(canonical_json(forged_manifest) + "\n", encoding="utf-8")
        forged_integrity = json.loads(original_integrity)
        forged_integrity["input_hashes"]["checkpoint_sha256"] = forged_checkpoint_sha256
        forged_integrity["run_signature"] = canonical_run_signature(
            forged_integrity["input_hashes"]
        )
        integrity_path.write_text(canonical_json(forged_integrity) + "\n", encoding="utf-8")
        with pytest.raises(RuntimeError, match="parameter delta does not recompute"):
            verify_experiment_manifest(manifest)
    finally:
        checkpoint_path.write_bytes(original_checkpoint)
        training_path.write_bytes(original_training)
        integrity_path.write_bytes(original_integrity)
        manifest.write_bytes(original_manifest)

    trace_path = _experiment_artifact_path(
        experiment, run_by_variant["E-PPO-Progress-Mask"]["traces"]
    )
    original_trace = trace_path.read_bytes()
    original_manifest = manifest.read_bytes()
    try:
        traces = [json.loads(line) for line in original_trace.splitlines()]
        traces[0]["steps"][0]["action"]["call_id"] += "-forged"
        trace_path.write_text(
            "".join(canonical_json(row) + "\n" for row in traces), encoding="utf-8"
        )
        forged_manifest = json.loads(original_manifest)
        full_run = next(
            run for run in forged_manifest["runs"] if run["variant"] == "E-PPO-Progress-Mask"
        )
        full_run["trace_sha256"] = file_sha256(trace_path)
        manifest.write_text(canonical_json(forged_manifest) + "\n", encoding="utf-8")
        with pytest.raises(RuntimeError, match="trace ToolCall differs from candidate"):
            verify_experiment_manifest(manifest)
    finally:
        trace_path.write_bytes(original_trace)
        manifest.write_bytes(original_manifest)

    # A fully self-consistent oracle trajectory is still invalid evidence when
    # it was not emitted by the declared checkpoint's deterministic policy.
    # Refresh trace hash, metrics, recomputation, and run-manifest scalars to
    # model an attacker who understands every existing self-reported layer.
    target_variant = "E-PPO-Progress-Mask"
    target_run = run_by_variant[target_variant]
    trace_path = _experiment_artifact_path(experiment, target_run["traces"])
    metrics_path = _experiment_artifact_path(experiment, target_run["metrics"])
    recompute_path = _experiment_artifact_path(experiment, target_run["recompute"])
    original_artifacts = {
        path: path.read_bytes() for path in (trace_path, metrics_path, recompute_path, manifest)
    }
    try:
        manifest_payload = json.loads(original_artifacts[manifest])
        experiment_config = ExperimentConfig.model_validate(manifest_payload["experiment_config"])
        variant_config = AblationVariant.model_validate(
            manifest_payload["variant_configs"][target_variant]
        )
        tasks = {task.case_id: task for task in load_tasks_jsonl(bundled_benchmark / "test.jsonl")}
        traces = [json.loads(line) for line in original_artifacts[trace_path].splitlines()]
        forged_index = -1
        for index, original_row in enumerate(traces):
            forged = _oracle_trace(
                tasks[str(original_row["case_id"])],
                original_row,
                _experiment_artifact_path(experiment, target_run["checkpoint"]),
                experiment_config,
                variant_config,
            )
            original_actions = [step["action_index"] for step in original_row["steps"]]
            forged_steps = cast(list[dict[str, object]], forged["steps"])
            forged_actions = [step["action_index"] for step in forged_steps]
            if original_actions != forged_actions:
                forged_index = index
                traces[index] = forged
                break
        assert forged_index >= 0, "smoke policy unexpectedly equals every oracle path"
        assert traces[forged_index]["success"] is True
        _refresh_trace_derived_evidence(manifest, variant_name=target_variant, traces=traces)
        with pytest.raises(RuntimeError, match="checkpoint policy trace mismatch"):
            verify_experiment_manifest(manifest)
    finally:
        for path, content in original_artifacts.items():
            path.write_bytes(content)

    # Scalar policy outputs and reward decomposition are policy evidence too;
    # none affects environment replay or aggregate metrics, so each used to be
    # forgeable after refreshing only the trace hash.
    for attacked_field in ("log_prob", "value", "reward"):
        original_artifacts = {
            path: path.read_bytes() for path in (trace_path, metrics_path, recompute_path, manifest)
        }
        try:
            traces = [json.loads(line) for line in original_artifacts[trace_path].splitlines()]
            step = traces[0]["steps"][0]
            step[attacked_field] = float(step[attacked_field]) + 0.125
            if attacked_field == "reward":
                step["reward_components"]["step"] = float(step["reward_components"]["step"]) + 0.125
            _refresh_trace_derived_evidence(manifest, variant_name=target_variant, traces=traces)
            with pytest.raises(RuntimeError, match="checkpoint policy trace mismatch"):
                verify_experiment_manifest(manifest)
        finally:
            for path, content in original_artifacts.items():
                path.write_bytes(content)

    original_manifest = manifest.read_bytes()
    try:
        forged_manifest = json.loads(original_manifest)
        forged_manifest["runs"][1]["variant"] = forged_manifest["runs"][0]["variant"]
        manifest.write_text(canonical_json(forged_manifest) + "\n", encoding="utf-8")
        with pytest.raises(RuntimeError, match="duplicate or undeclared run pair"):
            verify_experiment_manifest(manifest)
    finally:
        manifest.write_bytes(original_manifest)

    original_manifest = manifest.read_bytes()
    try:
        forged_manifest = json.loads(original_manifest)
        forged_manifest["ablation_config"]["variants"][1]["action_mask"] = False
        manifest.write_text(canonical_json(forged_manifest) + "\n", encoding="utf-8")
        with pytest.raises(ValueError, match="frozen fair-comparison matrix"):
            verify_experiment_manifest(manifest)
    finally:
        manifest.write_bytes(original_manifest)

    original_manifest = manifest.read_bytes()
    try:
        forged_manifest = json.loads(original_manifest)
        forged_manifest["runs"][0]["timeout_penalized_simulated_cost_s"] += 1.0
        manifest.write_text(canonical_json(forged_manifest) + "\n", encoding="utf-8")
        with pytest.raises(RuntimeError, match="run metric summary mismatch"):
            verify_experiment_manifest(manifest)
    finally:
        manifest.write_bytes(original_manifest)

    stale_claim = Path(str(experiment["root"])) / "claim-check.json"
    stale_claim.write_text("{}\n", encoding="utf-8")
    try:
        with pytest.raises(RuntimeError, match="stale canonical claim"):
            verify_experiment_manifest(manifest)
    finally:
        stale_claim.unlink()

    assert verify_experiment_manifest(manifest)["passed"] is True


def test_metrics_only_verify_claims_command_is_not_exposed() -> None:
    result = RUNNER.invoke(app, ["verify-claims"])

    assert result.exit_code == 2
    assert "No such command" in result.output


def test_ablate_succeeds_and_reports_when_preregistered_hypothesis_is_false(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    claim_path = tmp_path / "claim-check.json"
    claim_path.write_text(
        json.dumps({"canonical": True, "passed": False, "checks": []}),
        encoding="utf-8",
    )
    result = ExperimentResult(
        run_id="canonical-failure",
        root=str(tmp_path),
        benchmark_dir=str(tmp_path / "benchmark"),
        seeds=(17, 29, 43, 71, 101),
        variants=(
            "A-BC-Unmasked",
            "B-BC-Mask",
            "C-PPO-Sparse-Unmasked",
            "D-PPO-Sparse-Mask",
            "E-PPO-Progress-Mask",
            "F-Sequence-PPO-Progress-Mask",
        ),
        runs=(),
        claim_gate=str(claim_path),
    )
    observed: dict[str, object] = {}

    monkeypatch.setattr(cli_module, "prepare_benchmark", lambda *_args, **_kwargs: tmp_path)

    def _fake_run(*_args: object, **kwargs: object) -> ExperimentResult:
        observed.update(kwargs)
        return result

    monkeypatch.setattr(cli_module, "run_experiment_matrix", _fake_run)
    monkeypatch.setattr(
        cli_module,
        "verify_experiment_manifest",
        lambda _path: {"passed": True},
    )

    command = RUNNER.invoke(
        app,
        [
            "ablate",
            "--config",
            str(PROJECT_ROOT / "configs/cpu_full.yaml"),
            "--ablation",
            str(PROJECT_ROOT / "configs/ablation.yaml"),
            "--benchmark-dir",
            str(tmp_path / "benchmark"),
            "--output",
            str(tmp_path / "runs"),
        ],
    )

    assert command.exit_code == 0, command.output
    assert observed["require_canonical_claim"] is True
    payload = _json_output(command.stdout)
    assert payload["claim"] == {"canonical": True, "passed": False, "checks": []}


def test_ablate_rejects_smoke_sized_inputs_before_training(tmp_path: Path) -> None:
    config = _write_experiment_config(
        tmp_path / "not-canonical.yaml",
        train_cases=4,
        dev_cases=2,
        test_cases=4,
    )

    command = RUNNER.invoke(
        app,
        [
            "ablate",
            "--config",
            str(config),
            "--ablation",
            str(PROJECT_ROOT / "configs/ablation.yaml"),
            "--benchmark-dir",
            str(tmp_path / "benchmark"),
            "--output",
            str(tmp_path / "runs"),
        ],
    )

    assert command.exit_code != 0
    assert isinstance(command.exception, ValueError)
    assert "canonical claim requires" in str(command.exception)
    assert "1,000 unique test cases" in str(command.exception)
    assert not list((tmp_path / "runs").rglob("checkpoint.pt"))
