"""Public command-line entrypoints for training, evaluation, and verification."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import typer

from agentic_tool_rl.config import (
    AblationConfig,
    AblationVariant,
    ExperimentConfig,
    dry_run_config,
    load_config,
)
from agentic_tool_rl.evaluation import (
    ResumeGuard,
    RunInputHashes,
    assert_exact_case_ids,
    compute_metrics,
    file_sha256,
    recompute_metrics,
    write_json_atomic,
    write_metrics,
)
from agentic_tool_rl.experiment import (
    prepare_benchmark,
    run_experiment_matrix,
    source_fingerprint,
    validate_benchmark_evidence,
    variant_config_sha256,
    verify_experiment_manifest,
)
from agentic_tool_rl.features import FeatureEncoder
from agentic_tool_rl.models import ActorCritic, ProgressEstimator
from agentic_tool_rl.training import (
    CreditAssignment,
    evaluate_policy,
    load_checkpoint,
    run_episode,
)

app = typer.Typer(
    name="agentic-tool-rl",
    no_args_is_help=True,
    help="Action-level PPO for reproducible long-horizon tool-calling agents.",
)


def _echo(value: Any) -> None:
    typer.echo(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


def _experiment_config(path: Path) -> ExperimentConfig:
    value = load_config(path)
    if not isinstance(value, ExperimentConfig):
        raise typer.BadParameter(f"{path} is not an experiment config")
    return value


def _ablation_config(path: Path) -> AblationConfig:
    value = load_config(path)
    if not isinstance(value, AblationConfig):
        raise typer.BadParameter(f"{path} is not an ablation config")
    return value


def _selected_variant(ablation: AblationConfig, name: str) -> AblationVariant:
    matches = [variant for variant in ablation.variants if variant.name == name]
    if len(matches) != 1:
        declared = ", ".join(variant.name for variant in ablation.variants)
        raise typer.BadParameter(
            f"unknown ablation variant {name!r}; declared variants: {declared}",
            param_hint="--variant",
        )
    return matches[0]


def _load_variant_checkpoint(
    checkpoint: Path,
    variant: AblationVariant,
) -> tuple[ActorCritic, ProgressEstimator, FeatureEncoder, dict[str, Any], int]:
    """Load one declared variant checkpoint and reject shared/mislabeled artifacts."""

    try:
        model, estimator, encoder, metadata = load_checkpoint(
            checkpoint, expected_variant=variant.name
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise typer.BadParameter(str(exc), param_hint="--checkpoint") from exc
    expected_definition = variant.model_dump(mode="json")
    if metadata.get("variant") != expected_definition:
        raise typer.BadParameter(
            "checkpoint training metadata does not match the selected variant",
            param_hint="--variant",
        )
    seed = metadata.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise typer.BadParameter(
            "checkpoint training metadata is missing an integer seed",
            param_hint="--checkpoint",
        )
    return model, estimator, encoder, metadata, seed


def _variant_credit_assignment(variant: AblationVariant) -> CreditAssignment:
    return "sequence" if variant.algorithm == "sequence_ppo" else "action"


@app.command("generate")
def generate_command(
    config: Annotated[
        Path, typer.Option("--config", "-c", help="Experiment YAML config")
    ] = Path("configs/cpu_full.yaml"),
    output: Annotated[
        Path, typer.Option("--output", "-o", help="Benchmark artifact directory")
    ] = Path("artifacts/benchmark-v1"),
) -> None:
    """Generate tasks, oracle evidence, hashes, and frozen action labels."""

    experiment = _experiment_config(config)
    directory = prepare_benchmark(experiment, output)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    action_manifest = json.loads(
        (directory / "action_validity.manifest.json").read_text(encoding="utf-8")
    )
    _echo(
        {
            "benchmark_dir": str(directory),
            "manifest": manifest,
            "action_validity": action_manifest,
        }
    )


@app.command("train")
def train_command(
    config: Annotated[Path, typer.Option("--config", "-c")] = Path(
        "configs/cpu_full.yaml"
    ),
    ablation: Annotated[Path, typer.Option("--ablation", "-a")] = Path(
        "configs/ablation.yaml"
    ),
    benchmark_dir: Annotated[Path, typer.Option("--benchmark-dir")] = Path(
        "artifacts/benchmark-v1"
    ),
    output: Annotated[Path, typer.Option("--output", "-o")] = Path(
        "artifacts/runs"
    ),
    variant: Annotated[str, typer.Option("--variant")] = "E-PPO-Progress-Mask",
    seed: Annotated[int, typer.Option("--seed")] = 17,
) -> None:
    """Train and evaluate one declared ablation variant."""

    experiment = _experiment_config(config)
    matrix = _ablation_config(ablation)
    if not (benchmark_dir / "manifest.json").exists():
        prepare_benchmark(experiment, benchmark_dir)
    result = run_experiment_matrix(
        experiment,
        matrix,
        benchmark_dir=benchmark_dir,
        output_root=output,
        seeds=[seed],
        variant_names=[variant],
    )
    _echo(result.to_dict())


@app.command("smoke")
def smoke_command(
    config: Annotated[Path, typer.Option("--config", "-c")] = Path(
        "configs/smoke.yaml"
    ),
    ablation: Annotated[Path, typer.Option("--ablation", "-a")] = Path(
        "configs/ablation.yaml"
    ),
    benchmark_dir: Annotated[Path, typer.Option("--benchmark-dir")] = Path(
        "artifacts/benchmark-smoke"
    ),
    output: Annotated[Path, typer.Option("--output", "-o")] = Path(
        "artifacts/runs/smoke"
    ),
) -> None:
    """Run the CI-sized generation → BC → PPO → evaluation → recompute loop."""

    experiment = _experiment_config(config)
    matrix = _ablation_config(ablation)
    prepare_benchmark(experiment, benchmark_dir)
    result = run_experiment_matrix(
        experiment,
        matrix,
        benchmark_dir=benchmark_dir,
        output_root=output,
        seeds=[matrix.seeds[0]],
        variant_names=["B-BC-Mask", "E-PPO-Progress-Mask"],
    )
    verification = verify_experiment_manifest(Path(result.root) / "run-manifest.json")
    _echo({"experiment": result.to_dict(), "verification": verification})


@app.command("ablate")
def ablate_command(
    config: Annotated[Path, typer.Option("--config", "-c")] = Path(
        "configs/cpu_full.yaml"
    ),
    ablation: Annotated[Path, typer.Option("--ablation", "-a")] = Path(
        "configs/ablation.yaml"
    ),
    benchmark_dir: Annotated[Path, typer.Option("--benchmark-dir")] = Path(
        "artifacts/benchmark-v1"
    ),
    output: Annotated[Path, typer.Option("--output", "-o")] = Path(
        "artifacts/runs/full"
    ),
) -> None:
    """Run all six variants, five seeds, and the frozen test split."""

    experiment = _experiment_config(config)
    matrix = _ablation_config(ablation)
    prepare_benchmark(experiment, benchmark_dir)
    result = run_experiment_matrix(
        experiment,
        matrix,
        benchmark_dir=benchmark_dir,
        output_root=output,
        require_canonical_claim=True,
    )
    verification = verify_experiment_manifest(Path(result.root) / "run-manifest.json")
    if result.claim_gate is None:
        raise RuntimeError("canonical ablation did not produce claim evidence")
    claim = json.loads(
        (Path(result.root) / result.claim_gate).read_text(encoding="utf-8")
    )
    if not isinstance(claim, dict):
        raise RuntimeError("canonical claim evidence must be a JSON object")
    _echo(
        {
            "experiment": result.to_dict(),
            "verification": verification,
            "claim": claim,
        }
    )
    if claim.get("canonical") is not True:
        raise RuntimeError("canonical ablation produced structurally invalid evidence")


@app.command("run-episode")
def run_episode_command(
    checkpoint: Annotated[Path, typer.Option("--checkpoint")],
    variant_name: Annotated[str, typer.Option("--variant")],
    ablation: Annotated[Path, typer.Option("--ablation", "-a")],
    benchmark_dir: Annotated[Path, typer.Option("--benchmark-dir")] = Path(
        "artifacts/benchmark-v1"
    ),
    case_index: Annotated[int, typer.Option("--case-index", min=0)] = 0,
    output: Annotated[Path, typer.Option("--output", "-o")] = Path(
        "artifacts/demo-episode.json"
    ),
    config: Annotated[Path, typer.Option("--config", "-c")] = Path(
        "configs/cpu_full.yaml"
    ),
) -> None:
    """Execute one fully expanded, auditable policy episode."""

    experiment = _experiment_config(config)
    variant = _selected_variant(_ablation_config(ablation), variant_name)
    _, _, tasks, _ = validate_benchmark_evidence(benchmark_dir, experiment)
    if case_index >= len(tasks):
        raise typer.BadParameter(f"case-index must be smaller than {len(tasks)}")
    model, estimator, encoder, metadata, _ = _load_variant_checkpoint(
        checkpoint, variant
    )
    credit_assignment = _variant_credit_assignment(variant)
    task = tasks[case_index]
    result = run_episode(
        model,
        estimator,
        task,
        encoder,
        experiment.reward,
        trajectory_id=f"demo-{task.case_id}",
        use_action_mask=variant.action_mask,
        use_progress_reward=variant.progress_reward,
        deterministic=True,
        credit_assignment=credit_assignment,
        gamma=experiment.training.ppo.gamma,
        trace_mode="full",
    )
    write_json_atomic(output, result.trace)
    _echo(
        {
            "output": str(output),
            "variant": variant.name,
            "success": result.trace["success"],
            "metadata": metadata,
        }
    )


@app.command("evaluate")
def evaluate_command(
    checkpoint: Annotated[Path, typer.Option("--checkpoint")],
    variant_name: Annotated[str, typer.Option("--variant")],
    ablation: Annotated[Path, typer.Option("--ablation", "-a")],
    benchmark_dir: Annotated[Path, typer.Option("--benchmark-dir")] = Path(
        "artifacts/benchmark-v1"
    ),
    config: Annotated[Path, typer.Option("--config", "-c")] = Path(
        "configs/cpu_full.yaml"
    ),
    output: Annotated[Path, typer.Option("--output", "-o")] = Path(
        "artifacts/evaluation"
    ),
) -> None:
    """Evaluate a checkpoint on the frozen test set."""

    experiment = _experiment_config(config)
    variant = _selected_variant(_ablation_config(ablation), variant_name)
    model, estimator, encoder, _, seed = _load_variant_checkpoint(
        checkpoint, variant
    )
    _, _, tasks, _ = validate_benchmark_evidence(benchmark_dir, experiment)
    output.mkdir(parents=True, exist_ok=True)
    trace_path = output / "traces.jsonl"
    ResumeGuard(output / "run-integrity.json").authorize(
        RunInputHashes(
            benchmark_sha256=file_sha256(benchmark_dir / "manifest.json"),
            config_sha256=variant_config_sha256(experiment, variant, seed),
            source_sha256=source_fingerprint(),
            checkpoint_sha256=file_sha256(checkpoint),
        ),
        trace_paths=[trace_path],
    )
    credit_assignment = _variant_credit_assignment(variant)
    store = evaluate_policy(
        model,
        estimator,
        tasks,
        encoder,
        experiment.reward,
        trace_path=trace_path,
        use_action_mask=variant.action_mask,
        use_progress_reward=variant.progress_reward,
        credit_assignment=credit_assignment,
        gamma=experiment.training.ppo.gamma,
        trace_mode="compact",
    )
    assert_exact_case_ids(
        (task.case_id for task in tasks), store.completed_case_ids()
    )
    metrics = compute_metrics(
        store.records(),
        timeout_s=experiment.evaluation.timeout_s,
        bootstrap_samples=experiment.evaluation.bootstrap_samples,
        bootstrap_seed=experiment.evaluation.bootstrap_seed + seed,
        confidence=experiment.evaluation.confidence,
    )
    write_metrics(output / "metrics.json", metrics)
    _echo({"variant": variant.name, "metrics": metrics.to_dict()})


@app.command("recompute")
def recompute_command(
    traces: Annotated[Path, typer.Option("--traces")],
    metrics: Annotated[Path | None, typer.Option("--metrics")] = None,
    output: Annotated[Path, typer.Option("--output", "-o")] = Path(
        "artifacts/recompute.json"
    ),
) -> None:
    """Recompute metrics from raw traces for published-result consistency."""

    result = recompute_metrics(
        traces,
        published_metrics_path=metrics,
        output_path=output,
        tolerance=1e-9,
    )
    _echo(result.to_dict())
    if metrics is not None and not result.matches:
        raise typer.Exit(code=1)


@app.command("verify-run")
def verify_run_command(
    manifest: Annotated[Path, typer.Option("--manifest")],
) -> None:
    """Verify checksums, exact case coverage, updates, and metric recomputation."""

    _echo(verify_experiment_manifest(manifest))


@app.command("qwen-dry-run")
def qwen_dry_run_command(
    config: Annotated[Path, typer.Option("--config", "-c")] = Path(
        "configs/qwen3_lora_gpu.yaml"
    ),
) -> None:
    """Validate optional Qwen/LoRA/GPU wiring without downloading weights."""

    _echo(dry_run_config(config))


if __name__ == "__main__":
    app()
