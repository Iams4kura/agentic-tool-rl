"""Reproducible experiment orchestration and evidence manifests."""

from __future__ import annotations

import copy
import hashlib
import json
import platform
import shutil
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from math import isfinite
from pathlib import Path
from typing import Any, NoReturn

import torch

from agentic_tool_rl.config import AblationConfig, AblationVariant, ExperimentConfig
from agentic_tool_rl.contracts import (
    ActionValidityExample,
    ActionValidityManifest,
    BenchmarkManifest,
    Split,
    ToolCall,
    WorkflowTask,
)
from agentic_tool_rl.envs import (
    GENERATOR_VERSION,
    TransactionalWorkflowEnv,
    generate_action_validity_dataset,
    generate_all_splits,
    generate_and_write_benchmark,
    load_action_validity_jsonl,
    load_tasks_jsonl,
    write_action_validity_dataset,
)
from agentic_tool_rl.evaluation import (
    CANONICAL_ACTION_VALIDITY_COUNT,
    CANONICAL_SEED_COUNT,
    CANONICAL_TEST_CASE_COUNT,
    CANONICAL_VARIANT_DEFINITIONS,
    CanonicalClaimEvidence,
    ResumeGuard,
    RunInputHashes,
    assert_exact_case_ids,
    build_case_id_manifest,
    compare_metrics,
    compute_metrics,
    evaluate_action_validity,
    file_sha256,
    model_parameter_digest,
    read_jsonl,
    recompute_metrics,
    verify_canonical_claims,
    verify_case_id_manifest,
    write_claim_check,
    write_json_atomic,
    write_metrics,
)
from agentic_tool_rl.features import FeatureEncoder
from agentic_tool_rl.models import ActorCritic, ProgressEstimator
from agentic_tool_rl.training import (
    CreditAssignment,
    evaluate_policy,
    load_checkpoint,
    run_episode,
    save_checkpoint,
    train_behavior_policy,
    train_ppo_variant,
    train_progress_estimator,
)

_POLICY_TRACE_FLOAT_TOLERANCE = 1e-7
_BUNDLED_BENCHMARK_FILES = (
    "train.jsonl",
    "dev.jsonl",
    "test.jsonl",
    "manifest.json",
    "action_validity.jsonl",
    "action_validity.manifest.json",
)


@dataclass(frozen=True)
class VariantRunResult:
    variant: str
    seed: int
    directory: str
    checkpoint: str
    traces: str
    metrics: str
    recompute: str
    cases: int
    tsr: float
    successful_conditional_simulated_service_time_s: float | None
    timeout_penalized_simulated_cost_s: float
    ppo_updates: int
    checkpoint_sha256: str
    trace_sha256: str
    bc_checkpoint_sha256: str
    parameter_before_sha256: str
    parameter_after_sha256: str
    parameter_l2_delta: float
    rollout_steps: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ExperimentResult:
    run_id: str
    root: str
    benchmark_dir: str
    seeds: tuple[int, ...]
    variants: tuple[str, ...]
    runs: tuple[VariantRunResult, ...]
    claim_gate: str | None
    path_base: str = "root"

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "runs": [run.to_dict() for run in self.runs],
        }


def source_fingerprint(repo_root: str | Path | None = None) -> str:
    root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[2]
    digest = hashlib.sha256()
    files = (
        sorted((root / "src").rglob("*.py"))
        + sorted((root / "configs").glob("*.yaml"))
        + [path for name in ("pyproject.toml", "uv.lock") if (path := root / name).is_file()]
    )
    for path in files:
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def runtime_identity() -> dict[str, str | None]:
    """Return the execution identity that can affect numerical evidence."""

    return {
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "python_cache_tag": sys.implementation.cache_tag,
        "torch_version": str(torch.__version__),
        "torch_cuda_version": torch.version.cuda,
    }


def _relative_artifact_path(root: Path, path: Path) -> str:
    """Return a canonical POSIX artifact path rooted at ``root``."""

    resolved_root = root.resolve()
    resolved_path = path.resolve()
    try:
        relative = resolved_path.relative_to(resolved_root)
    except ValueError as exc:
        raise RuntimeError(f"artifact path escapes run root: {path}") from exc
    if relative == Path("."):
        raise RuntimeError("artifact path must name a file or child directory")
    return relative.as_posix()


def _resolve_artifact_path(
    root: Path,
    value: object,
    *,
    description: str,
) -> Path:
    """Resolve one canonical relative path without cwd or traversal semantics."""

    if not isinstance(value, str) or not value:
        raise ValueError(f"{description} must be a non-empty relative path")
    candidate_value = Path(value)
    if (
        candidate_value.is_absolute()
        or "\\" in value
        or any(part in {"", ".", ".."} for part in candidate_value.parts)
        or candidate_value.as_posix() != value
    ):
        raise ValueError(f"{description} must be a canonical relative path")
    resolved_root = root.resolve()
    resolved_path = (resolved_root / candidate_value).resolve()
    try:
        resolved_path.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"{description} escapes the artifact bundle") from exc
    return resolved_path


def _snapshot_benchmark_bundle(source: Path, destination: Path) -> None:
    """Copy the complete verifier input into the self-contained run bundle."""

    resolved_source = source.resolve()
    resolved_destination = destination.resolve()
    if resolved_source == resolved_destination:
        return
    if destination.is_symlink():
        raise RuntimeError("bundled benchmark directory must not be a symlink")
    destination.mkdir(parents=True, exist_ok=True)
    for name in _BUNDLED_BENCHMARK_FILES:
        source_path = resolved_source / name
        destination_path = destination / name
        if not source_path.is_file() or source_path.is_symlink():
            raise RuntimeError(f"benchmark artifact is missing or unsafe: {source_path}")
        if destination_path.is_symlink():
            raise RuntimeError(f"bundled benchmark artifact must not be a symlink: {name}")
        shutil.copy2(source_path, destination_path)


def prepare_benchmark(config: ExperimentConfig, output_dir: str | Path) -> Path:
    """Generate the frozen task and independent action-validity evidence."""

    directory = Path(output_dir)
    if config.benchmark.generator_version != GENERATOR_VERSION:
        raise ValueError(
            "config benchmark.generator_version does not match the installed generator: "
            f"{config.benchmark.generator_version!r} != {GENERATOR_VERSION!r}"
        )
    generate_and_write_benchmark(
        directory,
        train_size=config.benchmark.train_cases,
        dev_size=config.benchmark.dev_cases,
        test_size=config.benchmark.test_cases,
        base_seed=config.benchmark.seed,
        verify_oracles=True,
    )
    test_tasks = load_tasks_jsonl(directory / "test.jsonl")
    write_action_validity_dataset(directory, test_tasks, seed=config.benchmark.seed)
    return directory


def load_benchmark_splits(
    directory: str | Path,
) -> tuple[list[WorkflowTask], list[WorkflowTask], list[WorkflowTask]]:
    root = Path(directory)
    return (
        load_tasks_jsonl(root / "train.jsonl"),
        load_tasks_jsonl(root / "dev.jsonl"),
        load_tasks_jsonl(root / "test.jsonl"),
    )


def _variant_flags(variant: AblationVariant) -> tuple[bool, bool, CreditAssignment]:
    credit: CreditAssignment = "sequence" if variant.algorithm == "sequence_ppo" else "action"
    return variant.action_mask, variant.progress_reward, credit


def _aggregate_trace_metrics(
    runs: Sequence[VariantRunResult], *, root: Path, timeout_s: float
) -> dict[str, float | None]:
    """Aggregate from case traces so conditional service time is case-weighted."""

    rows = [row for run in runs for row in read_jsonl(root / run.traces)]
    metrics = compute_metrics(rows, timeout_s=timeout_s)
    return {
        "tsr": metrics.tsr,
        "successful_conditional_simulated_service_time_s": (
            metrics.successful_conditional_simulated_service_time_s
        ),
        "timeout_penalized_simulated_cost_s": (metrics.timeout_penalized_simulated_cost_s),
    }


def _comparison_trace_evidence(
    runs_by_variant: Mapping[str, Sequence[VariantRunResult]],
    *,
    root: Path,
) -> tuple[
    dict[str, dict[int, tuple[str, ...]]],
    dict[str, dict[int, dict[str, bool]]],
]:
    case_ids: dict[str, dict[int, tuple[str, ...]]] = {}
    outcomes: dict[str, dict[int, dict[str, bool]]] = {}
    for variant, runs in runs_by_variant.items():
        case_ids[variant] = {}
        outcomes[variant] = {}
        for run in runs:
            rows = read_jsonl(root / run.traces)
            case_ids[variant][run.seed] = tuple(str(row.get("case_id", "")) for row in rows)
            outcomes[variant][run.seed] = {
                str(row.get("case_id", "")): bool(row.get("success")) for row in rows
            }
    return case_ids, outcomes


def _variant_definition(variant: AblationVariant) -> tuple[str, str, bool, bool]:
    return (
        variant.name,
        variant.algorithm,
        variant.progress_reward,
        variant.action_mask,
    )


def _assert_canonical_claim_inputs(
    *,
    seeds: Sequence[int],
    variants: Sequence[AblationVariant],
    test_tasks: Sequence[WorkflowTask],
    action_validity_count: int,
    action_validity_valid_count: int,
    action_validity_invalid_count: int,
    bootstrap_samples: int,
) -> None:
    """Fail before training if a requested canonical run is structurally smaller."""

    failures: list[str] = []
    if len(seeds) != CANONICAL_SEED_COUNT or len(set(seeds)) != CANONICAL_SEED_COUNT:
        failures.append("exactly five unique seeds")
    case_ids = [task.case_id for task in test_tasks]
    if (
        len(case_ids) != CANONICAL_TEST_CASE_COUNT
        or len(set(case_ids)) != CANONICAL_TEST_CASE_COUNT
    ):
        failures.append("exactly 1,000 unique test cases")
    definitions = tuple(_variant_definition(variant) for variant in variants)
    if len(definitions) != len(CANONICAL_VARIANT_DEFINITIONS) or set(definitions) != set(
        CANONICAL_VARIANT_DEFINITIONS
    ):
        failures.append("the frozen six-variant definitions")
    if action_validity_count != CANONICAL_ACTION_VALIDITY_COUNT:
        failures.append("exactly 20,000 action-validity samples")
    if (
        action_validity_valid_count != CANONICAL_ACTION_VALIDITY_COUNT // 2
        or action_validity_invalid_count != CANONICAL_ACTION_VALIDITY_COUNT // 2
    ):
        failures.append("10,000 valid and 10,000 invalid action labels")
    if bootstrap_samples < 2:
        failures.append("at least two paired bootstrap samples")
    if failures:
        raise ValueError("canonical claim requires " + ", ".join(failures))


def _parameter_l2_delta(before: torch.nn.Module, after: torch.nn.Module) -> float:
    before_parameters = dict(before.named_parameters())
    after_parameters = dict(after.named_parameters())
    if before_parameters.keys() != after_parameters.keys():
        raise ValueError("model parameter names differ")
    squared = 0.0
    with torch.no_grad():
        for name, before_parameter in before_parameters.items():
            delta = after_parameters[name].detach().cpu() - before_parameter.detach().cpu()
            squared += float(torch.sum(delta.double().pow(2)).item())
    return float(squared**0.5)


def variant_config_sha256(config: ExperimentConfig, variant: AblationVariant, seed: int) -> str:
    """Hash the exact experiment, variant semantics, and training seed."""

    payload = {
        "experiment": config.model_dump(mode="json"),
        "variant": variant.model_dump(mode="json"),
        "seed": seed,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _canonical_run_id(
    config: ExperimentConfig,
    benchmark_dir: Path,
    seeds: Sequence[int],
    variants: Sequence[AblationVariant],
) -> str:
    manifest_digest = hashlib.sha256((benchmark_dir / "manifest.json").read_bytes()).hexdigest()
    payload = {
        "config": config.model_dump(mode="json"),
        "benchmark_manifest_sha256": manifest_digest,
        "seeds": list(seeds),
        "variants": [variant.model_dump(mode="json") for variant in variants],
        "source_sha256": source_fingerprint(),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def run_experiment_matrix(
    config: ExperimentConfig,
    ablation: AblationConfig,
    *,
    benchmark_dir: str | Path,
    output_root: str | Path,
    seeds: Sequence[int] | None = None,
    variant_names: Sequence[str] | None = None,
    require_canonical_claim: bool = False,
) -> ExperimentResult:
    """Train and evaluate a reproducible subset or the complete six-by-five matrix."""

    selected_seeds = tuple(seeds if seeds is not None else ablation.seeds)
    if len(selected_seeds) != len(set(selected_seeds)):
        raise ValueError("selected seeds must be unique")
    requested = set(variant_names) if variant_names is not None else None
    variants = tuple(
        variant for variant in ablation.variants if requested is None or variant.name in requested
    )
    if not selected_seeds or not variants:
        raise ValueError("at least one seed and variant are required")
    if requested is not None and {variant.name for variant in variants} != requested:
        missing = requested - {variant.name for variant in variants}
        raise ValueError(f"unknown ablation variants: {sorted(missing)}")
    if require_canonical_claim and selected_seeds != ablation.seeds:
        raise ValueError("canonical claim requires the pre-registered seed sequence")

    benchmark_path = Path(benchmark_dir).resolve()
    train_tasks, dev_tasks, test_tasks, action_examples = validate_benchmark_evidence(
        benchmark_path, config
    )
    action_validity = evaluate_action_validity(
        action_examples,
        {task.case_id: task for task in test_tasks},
        expected_count=len(test_tasks) * 5 * 4,
        bootstrap_samples=config.evaluation.bootstrap_samples,
        bootstrap_seed=config.evaluation.bootstrap_seed,
        confidence=config.evaluation.confidence,
    )
    if require_canonical_claim:
        _assert_canonical_claim_inputs(
            seeds=selected_seeds,
            variants=variants,
            test_tasks=test_tasks,
            action_validity_count=action_validity.count,
            action_validity_valid_count=action_validity.valid_count,
            action_validity_invalid_count=action_validity.invalid_count,
            bootstrap_samples=config.evaluation.bootstrap_samples,
        )
    run_id = _canonical_run_id(config, benchmark_path, selected_seeds, variants)
    root = (Path(output_root) / run_id).resolve()
    root.mkdir(parents=True, exist_ok=True)
    bundled_benchmark_path = root / "benchmark"
    _snapshot_benchmark_bundle(benchmark_path, bundled_benchmark_path)
    write_json_atomic(
        bundled_benchmark_path / "action_validity.metrics.json",
        action_validity.to_dict(include_predictions=False),
    )
    source_sha256 = source_fingerprint()
    benchmark_sha256 = file_sha256(benchmark_path / "manifest.json")
    write_json_atomic(
        root / "case-id-manifest.json",
        build_case_id_manifest(task.case_id for task in test_tasks),
    )
    results: list[VariantRunResult] = []

    for seed in selected_seeds:
        shared_dir = root / "shared" / f"seed-{seed}"
        shared_dir.mkdir(parents=True, exist_ok=True)
        shared_checkpoint = shared_dir / "bc-checkpoint.pt"
        if shared_checkpoint.exists():
            bc_model, estimator, encoder, shared_metadata = load_checkpoint(
                shared_checkpoint,
                expected_variant="shared-bc",
                expected_seed=seed,
            )
            bc_payload = shared_metadata.get("bc")
            progress_payload = shared_metadata.get("progress")
            if not isinstance(bc_payload, dict) or not isinstance(progress_payload, dict):
                raise RuntimeError("shared BC checkpoint is missing training metadata")
        else:
            encoder = FeatureEncoder(
                state_dim=config.policy.observation_dim,
                action_dim=config.policy.action_dim,
            )
            bc_model, bc_metrics = train_behavior_policy(train_tasks, encoder, config, seed=seed)
            estimator, progress_metrics = train_progress_estimator(
                train_tasks,
                dev_tasks,
                encoder,
                seed=seed,
                epochs=max(10, config.training.behavior_cloning.epochs * 3),
            )
            bc_payload = asdict(bc_metrics)
            progress_payload = asdict(progress_metrics)
            save_checkpoint(
                shared_checkpoint,
                bc_model,
                estimator,
                encoder,
                seed=seed,
                variant="shared-bc",
                metadata={"bc": bc_payload, "progress": progress_payload},
            )
        bc_checkpoint_sha256 = file_sha256(shared_checkpoint)
        parameter_before_sha256 = model_parameter_digest(bc_model)

        for variant in variants:
            action_mask, progress_reward, credit = _variant_flags(variant)
            variant_dir = root / variant.name / f"seed-{seed}"
            variant_dir.mkdir(parents=True, exist_ok=True)
            checkpoint = variant_dir / "checkpoint.pt"
            training_path = variant_dir / "training.json"
            if training_path.exists() and not checkpoint.exists():
                raise RuntimeError(f"partial training artifacts for {variant.name}/seed-{seed}")
            if checkpoint.exists():
                model, variant_estimator, loaded_encoder, checkpoint_metadata = load_checkpoint(
                    checkpoint,
                    expected_variant=variant.name,
                    expected_seed=seed,
                )
                if loaded_encoder.fingerprint() != encoder.fingerprint():
                    raise RuntimeError("cached variant feature encoder differs from shared BC")
                estimator = variant_estimator
                restore_training_evidence = not training_path.exists()
                if restore_training_evidence:
                    training_payload = checkpoint_metadata
                else:
                    training_payload = json.loads(training_path.read_text(encoding="utf-8"))
                    if not isinstance(training_payload, dict):
                        raise RuntimeError("training evidence must be a JSON object")
                    if checkpoint_metadata != training_payload:
                        raise RuntimeError(
                            "cached checkpoint metadata differs from training evidence"
                        )
                if training_payload.get("variant") != variant.model_dump(mode="json"):
                    raise RuntimeError("cached training evidence has a different variant config")
                if int(training_payload.get("seed", -1)) != seed:
                    raise RuntimeError("cached training evidence has a different seed")
                if training_payload.get("bc_checkpoint_sha256") != bc_checkpoint_sha256:
                    raise RuntimeError("cached variant was trained from a different BC checkpoint")
                if training_payload.get("parameter_before_sha256") != parameter_before_sha256:
                    raise RuntimeError("cached variant has a different pre-update parameter digest")
                ppo_payload = training_payload.get("ppo")
                ppo_updates = (
                    0 if not isinstance(ppo_payload, dict) else int(ppo_payload["updates"])
                )
                rollout_steps = int(training_payload["rollout_steps"])
                parameter_after_sha256 = str(training_payload["parameter_after_sha256"])
                parameter_l2_delta = float(training_payload["parameter_l2_delta"])
                if model_parameter_digest(model) != parameter_after_sha256:
                    raise RuntimeError("cached model parameters differ from training evidence")
                recomputed_delta = _parameter_l2_delta(bc_model, model)
                if abs(recomputed_delta - parameter_l2_delta) > 1e-12:
                    raise RuntimeError("cached parameter delta differs from training evidence")
                if variant.algorithm == "bc":
                    if ppo_updates != 0 or parameter_l2_delta != 0.0:
                        raise RuntimeError("cached BC variant contains PPO updates or drift")
                elif ppo_updates <= 0 or parameter_l2_delta <= 0.0:
                    raise RuntimeError("cached PPO variant has no real update evidence")
                if restore_training_evidence:
                    write_json_atomic(training_path, training_payload)
            else:
                ppo_metrics = None
                rollout_steps = 0
                if variant.algorithm == "bc":
                    model = copy.deepcopy(bc_model)
                else:
                    model, ppo_metrics, rollout_steps = train_ppo_variant(
                        bc_model,
                        estimator,
                        train_tasks,
                        encoder,
                        config,
                        seed=seed,
                        use_action_mask=action_mask,
                        use_progress_reward=progress_reward,
                        credit_assignment=credit,
                    )
                parameter_after_sha256 = model_parameter_digest(model)
                parameter_l2_delta = _parameter_l2_delta(bc_model, model)
                ppo_updates = 0 if ppo_metrics is None else ppo_metrics.updates
                if variant.algorithm == "bc":
                    if parameter_l2_delta != 0.0:
                        raise RuntimeError("BC evaluation model differs from shared BC")
                elif parameter_l2_delta <= 0.0 or ppo_updates <= 0:
                    raise RuntimeError(f"{variant.name} did not perform a real update")
                training_payload = {
                    "schema_version": "1.0",
                    "variant": variant.model_dump(mode="json"),
                    "seed": seed,
                    "bc": bc_payload,
                    "progress": progress_payload,
                    "ppo": None if ppo_metrics is None else asdict(ppo_metrics),
                    "rollout_steps": rollout_steps,
                    "bc_checkpoint_sha256": bc_checkpoint_sha256,
                    "parameter_before_sha256": parameter_before_sha256,
                    "parameter_after_sha256": parameter_after_sha256,
                    "parameter_l2_delta": parameter_l2_delta,
                }
                save_checkpoint(
                    checkpoint,
                    model,
                    estimator,
                    encoder,
                    seed=seed,
                    variant=variant.name,
                    metadata=training_payload,
                )
                write_json_atomic(training_path, training_payload)

            checkpoint_sha256 = file_sha256(checkpoint)
            trace_path = variant_dir / "traces.jsonl"
            ResumeGuard(variant_dir / "run-integrity.json").authorize(
                RunInputHashes(
                    benchmark_sha256=benchmark_sha256,
                    config_sha256=variant_config_sha256(config, variant, seed),
                    source_sha256=source_sha256,
                    checkpoint_sha256=checkpoint_sha256,
                ),
                trace_paths=[trace_path],
            )
            store = evaluate_policy(
                model,
                estimator,
                test_tasks,
                encoder,
                config.reward,
                trace_path=trace_path,
                use_action_mask=action_mask,
                use_progress_reward=progress_reward,
                credit_assignment=credit,
                gamma=config.training.ppo.gamma,
                trace_mode="compact",
            )
            assert_exact_case_ids((task.case_id for task in test_tasks), store.completed_case_ids())
            metrics = compute_metrics(
                store.records(),
                timeout_s=config.evaluation.timeout_s,
                bootstrap_samples=config.evaluation.bootstrap_samples,
                bootstrap_seed=config.evaluation.bootstrap_seed + seed,
                confidence=config.evaluation.confidence,
            )
            metrics_path = write_metrics(variant_dir / "metrics.json", metrics)
            recompute_path = variant_dir / "recompute.json"
            recomputed = recompute_metrics(
                trace_path,
                published_metrics_path=metrics_path,
                output_path=recompute_path,
            )
            if not recomputed.matches:
                raise RuntimeError(f"metric recomputation failed for {variant.name}/seed-{seed}")
            trace_sha256 = file_sha256(trace_path)
            results.append(
                VariantRunResult(
                    variant=variant.name,
                    seed=seed,
                    directory=_relative_artifact_path(root, variant_dir),
                    checkpoint=_relative_artifact_path(root, checkpoint),
                    traces=_relative_artifact_path(root, trace_path),
                    metrics=_relative_artifact_path(root, metrics_path),
                    recompute=_relative_artifact_path(root, recompute_path),
                    cases=metrics.cases,
                    tsr=metrics.tsr,
                    successful_conditional_simulated_service_time_s=(
                        metrics.successful_conditional_simulated_service_time_s
                    ),
                    timeout_penalized_simulated_cost_s=(metrics.timeout_penalized_simulated_cost_s),
                    ppo_updates=ppo_updates,
                    checkpoint_sha256=checkpoint_sha256,
                    trace_sha256=trace_sha256,
                    bc_checkpoint_sha256=bc_checkpoint_sha256,
                    parameter_before_sha256=parameter_before_sha256,
                    parameter_after_sha256=parameter_after_sha256,
                    parameter_l2_delta=parameter_l2_delta,
                    rollout_steps=rollout_steps,
                )
            )

    claim_path: Path | None = None
    if require_canonical_claim:
        comparison = ablation.canonical_comparison
        comparison_names = (
            comparison.total_system_baseline_variant,
            comparison.matched_baseline_variant,
            comparison.treatment_variant,
        )
        all_variant_runs: dict[str, list[VariantRunResult]] = {
            variant.name: [run for run in results if run.variant == variant.name]
            for variant in variants
        }
        comparison_runs: dict[str, list[VariantRunResult]] = {
            name: [run for run in results if run.variant == name] for name in comparison_names
        }
        case_ids_by_variant_seed, success_by_variant_seed = _comparison_trace_evidence(
            all_variant_runs, root=root
        )
        summaries = {
            name: _aggregate_trace_metrics(
                comparison_runs[name],
                root=root,
                timeout_s=config.evaluation.timeout_s,
            )
            for name in comparison_names
        }
        evidence = CanonicalClaimEvidence(
            seeds=selected_seeds,
            test_case_ids=tuple(task.case_id for task in test_tasks),
            variant_definitions=tuple(_variant_definition(variant) for variant in variants),
            case_ids_by_variant_seed=case_ids_by_variant_seed,
            success_by_variant_seed=success_by_variant_seed,
            action_validity_count=action_validity.count,
            action_validity_valid_count=action_validity.valid_count,
            action_validity_invalid_count=action_validity.invalid_count,
            bootstrap_samples=config.evaluation.bootstrap_samples,
            bootstrap_seed=config.evaluation.bootstrap_seed,
            confidence=config.evaluation.confidence,
        )
        claim = verify_canonical_claims(
            summaries[comparison.treatment_variant],
            summaries[comparison.matched_baseline_variant],
            summaries[comparison.total_system_baseline_variant],
            evidence,
            action_validity_balanced_accuracy=action_validity.balanced_accuracy,
        )
        claim_path = write_claim_check(root / "claim-check.json", claim)
    else:
        # The claim mode is not part of the training identity. Remove stale
        # derived claim evidence if the same inputs are intentionally rerun as
        # a diagnostic/non-canonical matrix.
        (root / "claim-check.json").unlink(missing_ok=True)

    result = ExperimentResult(
        run_id=run_id,
        root=str(root),
        benchmark_dir=_relative_artifact_path(root, bundled_benchmark_path),
        seeds=selected_seeds,
        variants=tuple(variant.name for variant in variants),
        runs=tuple(results),
        claim_gate=(None if claim_path is None else _relative_artifact_path(root, claim_path)),
    )
    manifest_payload = {
        "schema_version": "3.0",
        "source_sha256": source_sha256,
        "runtime_identity": runtime_identity(),
        "experiment_config": config.model_dump(mode="json"),
        "ablation_config": ablation.model_dump(mode="json"),
        "variant_configs": {variant.name: variant.model_dump(mode="json") for variant in variants},
        "benchmark_manifest_sha256": benchmark_sha256,
        "case_id_manifest": _relative_artifact_path(root, root / "case-id-manifest.json"),
        "action_validity_metrics": _relative_artifact_path(
            root, bundled_benchmark_path / "action_validity.metrics.json"
        ),
        "action_validity_metrics_sha256": file_sha256(
            bundled_benchmark_path / "action_validity.metrics.json"
        ),
        "expected_evaluation_units": len(selected_seeds) * len(variants) * len(test_tasks),
        "claim_mode": "canonical" if require_canonical_claim else "none",
        **result.to_dict(),
        "path_base": "run-manifest-parent",
        "root": ".",
    }
    write_json_atomic(root / "run-manifest.json", manifest_payload)
    return result


def _read_json_object(path: Path, *, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{description} is unreadable or invalid: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{description} must be a JSON object: {path}")
    return value


def _sequence_digest(values: Sequence[str | int]) -> str:
    payload = "\n".join(str(value) for value in sorted(values, key=str)).encode()
    return hashlib.sha256(payload).hexdigest()


def validate_benchmark_evidence(
    benchmark_dir: Path,
    config: ExperimentConfig,
) -> tuple[
    list[WorkflowTask],
    list[WorkflowTask],
    list[WorkflowTask],
    list[ActionValidityExample],
]:
    manifest_path = benchmark_dir / "manifest.json"
    manifest = BenchmarkManifest.model_validate(
        _read_json_object(manifest_path, description="benchmark manifest")
    )
    if manifest.generator_version != GENERATOR_VERSION:
        raise RuntimeError("benchmark generator_version differs from installed generator")
    if manifest.generator_version != config.benchmark.generator_version:
        raise RuntimeError("benchmark generator_version differs from experiment config")
    if manifest.base_seed != config.benchmark.seed:
        raise RuntimeError("benchmark base seed differs from experiment config")
    if set(manifest.files) != {split.value for split in Split}:
        raise RuntimeError("benchmark manifest must contain exactly train/dev/test")

    split_tasks: dict[Split, list[WorkflowTask]] = {}
    expected_counts = {
        Split.TRAIN: config.benchmark.train_cases,
        Split.DEV: config.benchmark.dev_cases,
        Split.TEST: config.benchmark.test_cases,
    }
    all_case_ids: set[str] = set()
    for split in Split:
        entry = manifest.files[split.value]
        expected_name = f"{split.value}.jsonl"
        if entry.split != split or entry.path != expected_name:
            raise RuntimeError(f"benchmark {split.value} entry points to an unexpected file")
        data_path = benchmark_dir / entry.path
        if file_sha256(data_path) != entry.sha256:
            raise RuntimeError(f"benchmark {split.value} JSONL checksum mismatch")
        tasks = load_tasks_jsonl(data_path)
        if len(tasks) != entry.count or entry.count != expected_counts[split]:
            raise RuntimeError(f"benchmark {split.value} count mismatch")
        if any(task.split != split for task in tasks):
            raise RuntimeError(f"benchmark {split.value} contains a wrong split label")
        if any(task.generator_version != manifest.generator_version for task in tasks):
            raise RuntimeError(f"benchmark {split.value} contains a wrong generator_version")
        case_ids = [task.case_id for task in tasks]
        if len(case_ids) != len(set(case_ids)) or all_case_ids.intersection(case_ids):
            raise RuntimeError("benchmark case ids are duplicated within or across splits")
        all_case_ids.update(case_ids)
        family_counts = dict(sorted(Counter(task.family for task in tasks).items()))
        length_counts = dict(sorted(Counter(task.difficulty for task in tasks).items()))
        if entry.family_counts != family_counts or entry.length_counts != length_counts:
            raise RuntimeError(f"benchmark {split.value} stratum summary mismatch")
        if entry.seed_digest != _sequence_digest([task.seed for task in tasks]):
            raise RuntimeError(f"benchmark {split.value} seed digest mismatch")
        if entry.entity_digest != _sequence_digest([task.entity_id for task in tasks]):
            raise RuntimeError(f"benchmark {split.value} entity digest mismatch")
        if entry.oracle_solvable != len(tasks):
            raise RuntimeError(f"benchmark {split.value} oracle evidence is incomplete")
        split_tasks[split] = tasks

    regenerated_splits = generate_all_splits(
        train_size=config.benchmark.train_cases,
        dev_size=config.benchmark.dev_cases,
        test_size=config.benchmark.test_cases,
        base_seed=config.benchmark.seed,
    )
    for split in Split:
        regenerated = sorted(regenerated_splits[split], key=lambda task: task.case_id)
        if split_tasks[split] != regenerated:
            raise RuntimeError(f"benchmark {split.value} differs from deterministic regeneration")

    test_tasks = split_tasks[Split.TEST]
    action_manifest_path = benchmark_dir / "action_validity.manifest.json"
    action_manifest = ActionValidityManifest.model_validate(
        _read_json_object(action_manifest_path, description="action-validity manifest")
    )
    action_path = benchmark_dir / "action_validity.jsonl"
    if file_sha256(action_path) != action_manifest.sha256:
        raise RuntimeError("action-validity JSONL checksum mismatch")
    action_examples = load_action_validity_jsonl(action_path)
    canonical_count = len(test_tasks) * 5 * 4
    if action_manifest.dataset_name != "action-validity-v2":
        raise RuntimeError("unsupported action-validity dataset contract")
    if action_manifest.generator_version != manifest.generator_version:
        raise RuntimeError("action-validity generator_version mismatch")
    if action_manifest.seed != config.benchmark.seed:
        raise RuntimeError("action-validity seed differs from experiment config")
    if action_manifest.source_task_count != len(test_tasks):
        raise RuntimeError("action-validity source task count mismatch")
    if action_manifest.states_per_task != 5 or action_manifest.candidates_per_state != 4:
        raise RuntimeError("action-validity canonical 5x4 sampling contract mismatch")
    if action_manifest.count != canonical_count or len(action_examples) != canonical_count:
        raise RuntimeError("action-validity count does not satisfy test_cases * 5 * 4")
    if len(test_tasks) == 1000 and canonical_count != 20_000:
        raise RuntimeError("canonical 1000-case action-validity dataset is not 20k")
    valid_count = sum(example.valid_label for example in action_examples)
    invalid_count = len(action_examples) - valid_count
    invalid_counts = dict(
        sorted(
            Counter(
                example.invalid_kind.value
                for example in action_examples
                if example.invalid_kind is not None
            ).items()
        )
    )
    challenge_counts = dict(
        sorted(Counter(example.challenge_source for example in action_examples).items())
    )
    if (
        action_manifest.valid_count != valid_count
        or action_manifest.invalid_count != invalid_count
        or action_manifest.invalid_kind_counts != invalid_counts
        or action_manifest.challenge_counts != challenge_counts
        or valid_count != invalid_count
    ):
        raise RuntimeError("action-validity label summary mismatch")
    if len(test_tasks) == 1000 and challenge_counts != {
        "hidden_ledger_collision": 500,
        "standard_candidate": 19_500,
    }:
        raise RuntimeError("canonical action-validity challenge strata mismatch")
    test_case_ids = {task.case_id for task in test_tasks}
    if {example.case_id for example in action_examples} != test_case_ids:
        raise RuntimeError("action-validity cases differ from the frozen test split")
    if any(example.split != Split.TEST for example in action_examples):
        raise RuntimeError("action-validity contains a non-test example")
    if action_manifest.source_cases_digest != _sequence_digest(list(test_case_ids)):
        raise RuntimeError("action-validity source case digest mismatch")
    sample_ids = [example.sample_id for example in action_examples]
    if len(sample_ids) != len(set(sample_ids)):
        raise RuntimeError("action-validity contains duplicate sample ids")
    group_counts = Counter((example.case_id, example.state_index) for example in action_examples)
    if len(group_counts) != len(test_tasks) * 5 or set(group_counts.values()) != {4}:
        raise RuntimeError("action-validity state/candidate cardinality mismatch")
    states_by_case: dict[str, set[int]] = {case_id: set() for case_id in test_case_ids}
    for example in action_examples:
        states_by_case[example.case_id].add(example.state_index)
    if any(states != set(range(5)) for states in states_by_case.values()):
        raise RuntimeError("action-validity state indices are not canonical 0..4")
    regenerated_actions = generate_action_validity_dataset(
        test_tasks,
        states_per_task=action_manifest.states_per_task,
        candidates_per_state=action_manifest.candidates_per_state,
        seed=action_manifest.seed,
    )
    if action_examples != sorted(regenerated_actions, key=lambda example: example.sample_id):
        raise RuntimeError("action-validity JSONL differs from deterministic regeneration")
    return (
        split_tasks[Split.TRAIN],
        split_tasks[Split.DEV],
        test_tasks,
        action_examples,
    )


def _checkpoint_payload(path: Path) -> Mapping[str, Any]:
    value = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(value, Mapping):
        raise RuntimeError(f"checkpoint payload is not a mapping: {path}")
    return value


def _replay_trace(
    trace: Mapping[str, Any],
    task: WorkflowTask,
) -> None:
    """Replay the public ToolCalls and prove summary/final-state consistency."""

    if trace.get("schema_version") != "1.0" or trace.get("case_id") != task.case_id:
        raise RuntimeError(f"trace identity mismatch for {task.case_id}")
    for field in (
        "success",
        "goal_predicates_satisfied",
    ):
        if not isinstance(trace.get(field), bool):
            raise RuntimeError(f"trace {field} must be boolean for {task.case_id}")
    if trace.get("family") != task.family or trace.get("difficulty") != task.difficulty:
        raise RuntimeError(f"trace task metadata mismatch for {task.case_id}")
    if trace.get("optimal_steps") != task.optimal_steps or trace.get("max_steps") != task.max_steps:
        raise RuntimeError(f"trace step budget mismatch for {task.case_id}")
    raw_steps = trace.get("steps")
    raw_executed = trace.get("executed_actions")
    if not isinstance(raw_steps, list) or not isinstance(raw_executed, list):
        raise RuntimeError(f"trace lacks replayable steps for {task.case_id}")
    if len(raw_steps) != len(raw_executed) or trace.get("step_count") != len(raw_steps):
        raise RuntimeError(f"trace step cardinality mismatch for {task.case_id}")

    environment = TransactionalWorkflowEnv(task)
    for index, (raw_step, raw_execution) in enumerate(zip(raw_steps, raw_executed, strict=True)):
        if not isinstance(raw_step, Mapping) or not isinstance(raw_execution, Mapping):
            raise RuntimeError(f"trace step {index} is malformed for {task.case_id}")
        if raw_step.get("step_index") != index or raw_execution.get("step_index") != index:
            raise RuntimeError(f"trace step order mismatch for {task.case_id}")
        candidates = environment.candidate_actions(include_invalid=True)
        candidate_payload = [candidate.model_dump(mode="json") for candidate in candidates]
        encoded = json.dumps(candidate_payload, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
        if raw_step.get("candidate_count") != len(candidates):
            raise RuntimeError(f"trace candidate count mismatch for {task.case_id} step {index}")
        if raw_step.get("candidates_sha256") != hashlib.sha256(encoded).hexdigest():
            raise RuntimeError(f"trace candidate digest mismatch for {task.case_id} step {index}")
        action_index = raw_step.get("action_index")
        if not isinstance(action_index, int) or isinstance(action_index, bool):
            raise RuntimeError(f"trace action index is invalid for {task.case_id} step {index}")
        if action_index < 0 or action_index >= len(candidates):
            raise RuntimeError(
                f"trace action index is out of range for {task.case_id} step {index}"
            )
        if raw_execution.get("candidate_index") != action_index:
            raise RuntimeError(f"trace selected action indices differ for {task.case_id}")
        for field in ("executed_valid", "accepted"):
            if not isinstance(raw_execution.get(field), bool):
                raise RuntimeError(f"trace execution {field} must be boolean for {task.case_id}")
        for field in ("valid_label", "predicted_valid", "done"):
            if not isinstance(raw_step.get(field), bool):
                raise RuntimeError(f"trace step {field} must be boolean for {task.case_id}")
        action = ToolCall.model_validate(raw_step.get("action"))
        if action != candidates[action_index]:
            raise RuntimeError(f"trace ToolCall differs from candidate for {task.case_id}")
        validation = environment.dry_run(action)
        if raw_step.get("valid_label") != validation.valid:
            raise RuntimeError(f"trace validity label mismatch for {task.case_id}")
        if raw_execution.get("executed_valid") != validation.valid:
            raise RuntimeError(f"trace executed validity mismatch for {task.case_id}")
        outcome = environment.step(action)
        if raw_execution.get("accepted") != outcome.accepted:
            raise RuntimeError(f"trace transaction result mismatch for {task.case_id}")
        if raw_step.get("done") != outcome.done:
            raise RuntimeError(f"trace termination flag mismatch for {task.case_id}")

    evaluation = environment.evaluate()
    if not environment.done:
        raise RuntimeError(f"trace ended before environment termination for {task.case_id}")
    if trace.get("success") != evaluation.success:
        raise RuntimeError(f"trace success mismatch for {task.case_id}")
    if trace.get("goal_predicates_satisfied") != evaluation.goal_predicates_satisfied:
        raise RuntimeError(f"trace goal status mismatch for {task.case_id}")
    if trace.get("forbidden_side_effect_count") != evaluation.forbidden_side_effect_count:
        raise RuntimeError(f"trace side-effect count mismatch for {task.case_id}")
    if trace.get("final_state") != environment.state_snapshot():
        raise RuntimeError(f"trace final state mismatch for {task.case_id}")
    latency = trace.get("simulated_latency_s")
    if not isinstance(latency, (int, float)) or isinstance(latency, bool):
        raise RuntimeError(f"trace latency is invalid for {task.case_id}")
    if abs(float(latency) - environment.simulated_latency_s) > 1e-9:
        raise RuntimeError(f"trace latency mismatch for {task.case_id}")


def _raise_policy_trace_mismatch(
    *, case_id: str, path: str, recorded: object, expected: object
) -> NoReturn:
    raise RuntimeError(
        f"checkpoint policy trace mismatch for {case_id} at {path}: "
        f"recorded={recorded!r}, expected={expected!r}"
    )


def _assert_policy_trace_equal(
    recorded: object,
    expected: object,
    *,
    case_id: str,
    path: str = "trace",
) -> None:
    """Recursively compare deterministic policy evidence with bounded float drift."""

    if isinstance(expected, Mapping):
        if not isinstance(recorded, Mapping):
            _raise_policy_trace_mismatch(
                case_id=case_id, path=path, recorded=recorded, expected=expected
            )
        recorded_keys = set(recorded)
        expected_keys = set(expected)
        if recorded_keys != expected_keys:
            _raise_policy_trace_mismatch(
                case_id=case_id,
                path=f"{path}.keys",
                recorded=sorted(str(key) for key in recorded_keys),
                expected=sorted(str(key) for key in expected_keys),
            )
        for key in sorted(expected_keys, key=str):
            _assert_policy_trace_equal(
                recorded[key],
                expected[key],
                case_id=case_id,
                path=f"{path}.{key}",
            )
        return

    if isinstance(expected, Sequence) and not isinstance(expected, (str, bytes, bytearray)):
        if not isinstance(recorded, Sequence) or isinstance(recorded, (str, bytes, bytearray)):
            _raise_policy_trace_mismatch(
                case_id=case_id, path=path, recorded=recorded, expected=expected
            )
        if len(recorded) != len(expected):
            _raise_policy_trace_mismatch(
                case_id=case_id,
                path=f"{path}.length",
                recorded=len(recorded),
                expected=len(expected),
            )
        for index, (recorded_item, expected_item) in enumerate(
            zip(recorded, expected, strict=True)
        ):
            _assert_policy_trace_equal(
                recorded_item,
                expected_item,
                case_id=case_id,
                path=f"{path}[{index}]",
            )
        return

    if isinstance(expected, bool) or isinstance(recorded, bool):
        if type(recorded) is not type(expected) or recorded != expected:
            _raise_policy_trace_mismatch(
                case_id=case_id, path=path, recorded=recorded, expected=expected
            )
        return

    if isinstance(expected, int) and isinstance(recorded, int):
        if recorded != expected:
            _raise_policy_trace_mismatch(
                case_id=case_id, path=path, recorded=recorded, expected=expected
            )
        return

    if isinstance(expected, (int, float)) and isinstance(recorded, (int, float)):
        recorded_float = float(recorded)
        expected_float = float(expected)
        if (
            not isfinite(recorded_float)
            or not isfinite(expected_float)
            or abs(recorded_float - expected_float) > _POLICY_TRACE_FLOAT_TOLERANCE
        ):
            _raise_policy_trace_mismatch(
                case_id=case_id, path=path, recorded=recorded, expected=expected
            )
        return

    if type(recorded) is not type(expected) or recorded != expected:
        _raise_policy_trace_mismatch(
            case_id=case_id, path=path, recorded=recorded, expected=expected
        )


def _verify_checkpoint_policy_trace(
    trace: Mapping[str, Any],
    task: WorkflowTask,
    *,
    model: ActorCritic,
    estimator: ProgressEstimator,
    encoder: FeatureEncoder,
    config: ExperimentConfig,
    variant: AblationVariant,
) -> None:
    """Prove that a trace was emitted by the declared deterministic policy."""

    use_action_mask, use_progress_reward, credit_assignment = _variant_flags(variant)
    expected = run_episode(
        model,
        estimator,
        task,
        encoder,
        config.reward,
        trajectory_id=task.case_id,
        use_action_mask=use_action_mask,
        use_progress_reward=use_progress_reward,
        deterministic=True,
        credit_assignment=credit_assignment,
        gamma=config.training.ppo.gamma,
        trace_mode="compact",
    ).trace
    _assert_policy_trace_equal(
        trace,
        expected,
        case_id=task.case_id,
    )


def _claim_metric_projection(trace: Mapping[str, Any]) -> dict[str, Any]:
    """Retain only fields needed for the three aggregate claim metrics."""

    return {
        "case_id": trace["case_id"],
        "family": trace["family"],
        "success": trace["success"],
        "optimal_steps": trace["optimal_steps"],
        "step_count": trace["step_count"],
        "simulated_latency_s": trace["simulated_latency_s"],
    }


def verify_experiment_manifest(path: str | Path) -> dict[str, Any]:
    """Fail closed unless every declared run is rooted in replayable evidence."""

    manifest_path = Path(path)
    payload = _read_json_object(manifest_path, description="run manifest")
    expected_manifest_keys = {
        "schema_version",
        "source_sha256",
        "runtime_identity",
        "experiment_config",
        "ablation_config",
        "variant_configs",
        "benchmark_manifest_sha256",
        "case_id_manifest",
        "action_validity_metrics",
        "action_validity_metrics_sha256",
        "expected_evaluation_units",
        "claim_mode",
        "path_base",
        "run_id",
        "root",
        "benchmark_dir",
        "seeds",
        "variants",
        "runs",
        "claim_gate",
    }
    if payload.get("schema_version") != "3.0" or set(payload) != expected_manifest_keys:
        raise ValueError("unsupported or structurally invalid run manifest")
    if payload.get("path_base") != "run-manifest-parent" or payload.get("root") != ".":
        raise ValueError("run manifest artifact path base is invalid")
    bundle_root = manifest_path.resolve().parent
    if payload.get("source_sha256") != source_fingerprint():
        raise RuntimeError("run manifest source fingerprint differs from current source")
    if payload.get("runtime_identity") != runtime_identity():
        raise RuntimeError("run manifest Python/Torch runtime identity mismatch")
    claim_mode = payload.get("claim_mode")
    if claim_mode not in {"canonical", "none"}:
        raise ValueError("run manifest claim_mode is invalid")
    if (claim_mode == "canonical") != (payload.get("claim_gate") is not None):
        raise RuntimeError("run manifest claim mode and claim evidence disagree")
    config = ExperimentConfig.model_validate(payload.get("experiment_config"))
    ablation = AblationConfig.model_validate(payload.get("ablation_config"))
    raw_seeds = payload.get("seeds")
    raw_variants = payload.get("variants")
    raw_variant_configs = payload.get("variant_configs")
    runs = payload.get("runs")
    if not isinstance(raw_seeds, list) or not raw_seeds:
        raise ValueError("run manifest has no seeds")
    if any(not isinstance(seed, int) or isinstance(seed, bool) for seed in raw_seeds):
        raise ValueError("run manifest seeds must be integers")
    seeds = tuple(raw_seeds)
    if len(seeds) != len(set(seeds)):
        raise ValueError("run manifest seeds must be unique")
    if (
        not isinstance(raw_variants, list)
        or not raw_variants
        or not all(isinstance(name, str) and name for name in raw_variants)
    ):
        raise ValueError("run manifest has invalid variants")
    variants = tuple(raw_variants)
    if len(variants) != len(set(variants)):
        raise ValueError("run manifest variants must be unique")
    if not isinstance(raw_variant_configs, dict) or set(raw_variant_configs) != set(variants):
        raise ValueError("run manifest variant configs do not cover declared variants")
    variant_configs = {
        name: AblationVariant.model_validate(raw_variant_configs[name]) for name in variants
    }
    if any(variant_configs[name].name != name for name in variants):
        raise ValueError("run manifest variant config name mismatch")
    declared_ablation_variants = {variant.name: variant for variant in ablation.variants}
    if any(
        name not in declared_ablation_variants
        or variant_configs[name] != declared_ablation_variants[name]
        for name in variants
    ):
        raise ValueError("run manifest variants differ from the frozen ablation config")
    if claim_mode == "canonical" and (
        seeds != ablation.seeds or set(variants) != set(declared_ablation_variants)
    ):
        raise RuntimeError("canonical run must use every pre-registered seed and variant")
    if not isinstance(runs, list) or not runs:
        raise ValueError("run manifest has no runs")

    benchmark_dir = _resolve_artifact_path(
        bundle_root,
        payload.get("benchmark_dir"),
        description="benchmark_dir",
    )
    if benchmark_dir != (bundle_root / "benchmark").resolve():
        raise RuntimeError("benchmark directory is not canonical for the run bundle")
    benchmark_manifest_path = benchmark_dir / "manifest.json"
    benchmark_sha256 = file_sha256(benchmark_manifest_path)
    if benchmark_sha256 != payload.get("benchmark_manifest_sha256"):
        raise RuntimeError("benchmark manifest checksum mismatch")
    _, _, test_tasks, action_examples = validate_benchmark_evidence(
        benchmark_dir, config
    )
    task_by_id = {task.case_id: task for task in test_tasks}
    expected_case_ids = tuple(sorted(task_by_id))
    case_manifest_path = _resolve_artifact_path(
        bundle_root,
        payload.get("case_id_manifest"),
        description="case_id_manifest",
    )
    if case_manifest_path != (bundle_root / "case-id-manifest.json").resolve():
        raise RuntimeError("case-id manifest path is not canonical for the run bundle")
    case_manifest = _read_json_object(case_manifest_path, description="case-id manifest")
    verify_case_id_manifest(case_manifest, expected_case_ids)

    action_evaluation = evaluate_action_validity(
        action_examples,
        task_by_id,
        expected_count=len(test_tasks) * 5 * 4,
        bootstrap_samples=config.evaluation.bootstrap_samples,
        bootstrap_seed=config.evaluation.bootstrap_seed,
        confidence=config.evaluation.confidence,
    )
    action_metrics_path = _resolve_artifact_path(
        bundle_root,
        payload.get("action_validity_metrics"),
        description="action_validity_metrics",
    )
    if (
        action_metrics_path
        != (bundle_root / "benchmark" / "action_validity.metrics.json").resolve()
    ):
        raise RuntimeError("action-validity metrics path is not canonical for the run bundle")
    if file_sha256(action_metrics_path) != payload.get("action_validity_metrics_sha256"):
        raise RuntimeError("action-validity metrics checksum mismatch")
    if _read_json_object(
        action_metrics_path, description="action-validity metrics"
    ) != action_evaluation.to_dict(include_predictions=False):
        raise RuntimeError("action-validity metrics differ from frozen data")

    ordered_variant_configs = tuple(variant_configs[name] for name in variants)
    expected_run_id = _canonical_run_id(config, benchmark_dir, seeds, ordered_variant_configs)
    if payload.get("run_id") != expected_run_id:
        raise RuntimeError("run id is not the canonical input digest")
    root = bundle_root

    expected_pairs = {(variant, seed) for variant in variants for seed in seeds}
    observed_pairs: set[tuple[str, int]] = set()
    run_keys = set(VariantRunResult.__dataclass_fields__)
    for run in runs:
        if not isinstance(run, dict) or set(run) != run_keys:
            raise ValueError("run entry has an invalid schema")
        for field in (
            "variant",
            "directory",
            "checkpoint",
            "traces",
            "metrics",
            "recompute",
            "checkpoint_sha256",
            "trace_sha256",
            "bc_checkpoint_sha256",
            "parameter_before_sha256",
            "parameter_after_sha256",
        ):
            if not isinstance(run.get(field), str) or not run[field]:
                raise ValueError(f"run field {field} must be a non-empty string")
        for field in ("seed", "cases", "ppo_updates", "rollout_steps"):
            if not isinstance(run.get(field), int) or isinstance(run[field], bool):
                raise ValueError(f"run field {field} must be an integer")
        for field in (
            "tsr",
            "timeout_penalized_simulated_cost_s",
            "parameter_l2_delta",
        ):
            value = run.get(field)
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not isfinite(float(value))
            ):
                raise ValueError(f"run field {field} must be finite numeric evidence")
        conditional_service_time = run.get("successful_conditional_simulated_service_time_s")
        if conditional_service_time is not None and (
            not isinstance(conditional_service_time, (int, float))
            or isinstance(conditional_service_time, bool)
            or not isfinite(float(conditional_service_time))
            or float(conditional_service_time) < 0.0
        ):
            raise ValueError(
                "run successful conditional simulated service time must be null or "
                "finite non-negative numeric evidence"
            )
        variant = run.get("variant")
        seed = run.get("seed")
        if not isinstance(variant, str) or not isinstance(seed, int) or isinstance(seed, bool):
            raise ValueError("run variant/seed identity is invalid")
        pair = (variant, seed)
        if pair not in expected_pairs or pair in observed_pairs:
            raise RuntimeError(f"duplicate or undeclared run pair: {pair}")
        observed_pairs.add(pair)
    if observed_pairs != expected_pairs:
        raise RuntimeError("run matrix does not uniquely cover variants x seeds")

    shared_models: dict[int, tuple[torch.nn.Module, str, str, Mapping[str, Any], str]] = {}
    observed_units = 0
    checked: list[dict[str, Any]] = []
    verified_case_ids: dict[tuple[str, int], tuple[str, ...]] = {}
    verified_successes: dict[tuple[str, int], dict[str, bool]] = {}
    comparison = ablation.canonical_comparison
    comparison_names = (
        comparison.total_system_baseline_variant,
        comparison.matched_baseline_variant,
        comparison.treatment_variant,
    )
    comparison_metric_rows: dict[str, list[dict[str, Any]]] = {
        name: [] for name in comparison_names
    }
    training_keys = {
        "schema_version",
        "variant",
        "seed",
        "bc",
        "progress",
        "ppo",
        "rollout_steps",
        "bc_checkpoint_sha256",
        "parameter_before_sha256",
        "parameter_after_sha256",
        "parameter_l2_delta",
    }
    for run in runs:
        assert isinstance(run, dict)
        variant = str(run["variant"])
        seed = int(run["seed"])
        variant_config = variant_configs[variant]
        variant_dir = _resolve_artifact_path(
            root,
            run.get("directory"),
            description=f"run directory for {variant}/seed-{seed}",
        )
        expected_variant_dir = root / variant / f"seed-{seed}"
        if variant_dir != expected_variant_dir.resolve():
            raise RuntimeError(f"run directory is not canonical for {variant}/seed-{seed}")
        checkpoint_path = _resolve_artifact_path(
            root,
            run.get("checkpoint"),
            description=f"checkpoint for {variant}/seed-{seed}",
        )
        trace_path = _resolve_artifact_path(
            root,
            run.get("traces"),
            description=f"traces for {variant}/seed-{seed}",
        )
        metrics_path = _resolve_artifact_path(
            root,
            run.get("metrics"),
            description=f"metrics for {variant}/seed-{seed}",
        )
        recompute_path = _resolve_artifact_path(
            root,
            run.get("recompute"),
            description=f"recompute evidence for {variant}/seed-{seed}",
        )
        expected_paths = {
            checkpoint_path: variant_dir / "checkpoint.pt",
            trace_path: variant_dir / "traces.jsonl",
            metrics_path: variant_dir / "metrics.json",
            recompute_path: variant_dir / "recompute.json",
        }
        if any(actual != expected.resolve() for actual, expected in expected_paths.items()):
            raise RuntimeError(f"run artifact path is not canonical for {variant}/seed-{seed}")

        if seed not in shared_models:
            shared_path = root / "shared" / f"seed-{seed}" / "bc-checkpoint.pt"
            shared_hash = file_sha256(shared_path)
            loaded_shared_model, _, shared_encoder, loaded_shared_metadata = load_checkpoint(
                shared_path,
                expected_variant="shared-bc",
                expected_seed=seed,
            )
            shared_raw = _checkpoint_payload(shared_path)
            if shared_raw.get("seed") != seed or shared_raw.get("variant") != "shared-bc":
                raise RuntimeError(f"shared BC checkpoint identity mismatch for seed {seed}")
            if set(loaded_shared_metadata) != {"bc", "progress"}:
                raise RuntimeError(f"shared BC metadata is incomplete for seed {seed}")
            shared_digest = model_parameter_digest(loaded_shared_model)
            shared_models[seed] = (
                loaded_shared_model,
                shared_hash,
                shared_digest,
                loaded_shared_metadata,
                shared_encoder.fingerprint(),
            )
        shared_model, shared_hash, shared_digest, shared_metadata, shared_feature_digest = (
            shared_models[seed]
        )
        if run.get("bc_checkpoint_sha256") != shared_hash:
            raise RuntimeError(f"shared BC file hash mismatch for {variant}/seed-{seed}")
        if run.get("parameter_before_sha256") != shared_digest:
            raise RuntimeError(f"shared BC parameter digest mismatch for {variant}/seed-{seed}")

        if file_sha256(trace_path) != run.get("trace_sha256"):
            raise RuntimeError(f"trace checksum mismatch in {trace_path}")
        checkpoint_sha256 = file_sha256(checkpoint_path)
        if checkpoint_sha256 != run.get("checkpoint_sha256"):
            raise RuntimeError(f"checkpoint checksum mismatch in {checkpoint_path}")
        (
            checkpoint_model,
            checkpoint_estimator,
            checkpoint_encoder,
            checkpoint_metadata,
        ) = load_checkpoint(
            checkpoint_path,
            expected_variant=variant,
            expected_seed=seed,
        )
        checkpoint_raw = _checkpoint_payload(checkpoint_path)
        if checkpoint_raw.get("seed") != seed or checkpoint_raw.get("variant") != variant:
            raise RuntimeError(f"checkpoint identity mismatch in {checkpoint_path}")
        if checkpoint_encoder.fingerprint() != shared_feature_digest:
            raise RuntimeError(
                f"checkpoint feature encoder differs from shared BC in {checkpoint_path}"
            )
        after_digest = model_parameter_digest(checkpoint_model)
        if after_digest != run.get("parameter_after_sha256"):
            raise RuntimeError(f"checkpoint parameter digest mismatch in {checkpoint_path}")

        training_path = variant_dir / "training.json"
        training_payload = _read_json_object(training_path, description="training evidence")
        if set(training_payload) != training_keys or checkpoint_metadata != training_payload:
            raise RuntimeError(f"training/checkpoint metadata mismatch in {checkpoint_path}")
        if training_payload.get("schema_version") != "1.0":
            raise RuntimeError(f"unsupported training evidence in {training_path}")
        if training_payload.get("variant") != variant_config.model_dump(mode="json"):
            raise RuntimeError(f"training variant config mismatch in {training_path}")
        if training_payload.get("seed") != seed:
            raise RuntimeError(f"training seed mismatch in {training_path}")
        if training_payload.get("bc") != shared_metadata.get("bc") or training_payload.get(
            "progress"
        ) != shared_metadata.get("progress"):
            raise RuntimeError(f"training bootstrap metadata mismatch in {training_path}")
        for field in (
            "bc_checkpoint_sha256",
            "parameter_before_sha256",
            "parameter_after_sha256",
            "parameter_l2_delta",
            "rollout_steps",
        ):
            if training_payload.get(field) != run.get(field):
                raise RuntimeError(f"{field} mismatch in {checkpoint_path}")
        recomputed_delta = _parameter_l2_delta(shared_model, checkpoint_model)
        recorded_delta = run.get("parameter_l2_delta")
        if not isinstance(recorded_delta, (int, float)) or isinstance(recorded_delta, bool):
            raise RuntimeError(f"parameter delta is invalid in {checkpoint_path}")
        if abs(recomputed_delta - float(recorded_delta)) > 1e-12:
            raise RuntimeError(f"parameter delta does not recompute in {checkpoint_path}")
        ppo_payload = training_payload.get("ppo")
        if variant_config.algorithm == "bc":
            if (
                ppo_payload is not None
                or run.get("ppo_updates") != 0
                or run.get("rollout_steps") != 0
                or float(recorded_delta) != 0.0
                or after_digest != shared_digest
            ):
                raise RuntimeError("BC run contains PPO updates, rollout steps, or drift")
        else:
            if not isinstance(ppo_payload, dict):
                raise RuntimeError(f"PPO metadata is missing in {checkpoint_path}")
            ppo_keys = {
                "policy_loss",
                "value_loss",
                "entropy",
                "approximate_kl",
                "clip_fraction",
                "grad_norm",
                "total_loss",
                "updates",
                "stopped_early",
            }
            if set(ppo_payload) != ppo_keys:
                raise RuntimeError(f"PPO metadata schema mismatch in {checkpoint_path}")
            if (
                ppo_payload.get("updates") != run.get("ppo_updates")
                or not isinstance(run.get("ppo_updates"), int)
                or isinstance(run.get("ppo_updates"), bool)
                or int(run["ppo_updates"]) <= 0
                or not isinstance(run.get("rollout_steps"), int)
                or isinstance(run.get("rollout_steps"), bool)
                or int(run["rollout_steps"]) <= 0
                or float(recorded_delta) <= 0.0
            ):
                raise RuntimeError(f"PPO update/rollout evidence mismatch in {checkpoint_path}")

        config_sha256 = variant_config_sha256(config, variant_config, seed)
        ResumeGuard(variant_dir / "run-integrity.json").verify(
            RunInputHashes(
                benchmark_sha256=benchmark_sha256,
                config_sha256=config_sha256,
                source_sha256=str(payload["source_sha256"]),
                checkpoint_sha256=checkpoint_sha256,
            )
        )
        trace_rows = read_jsonl(trace_path)
        assert_exact_case_ids(
            expected_case_ids, (str(row.get("case_id", "")) for row in trace_rows)
        )
        for trace in trace_rows:
            case_id = str(trace["case_id"])
            _replay_trace(trace, task_by_id[case_id])
            _verify_checkpoint_policy_trace(
                trace,
                task_by_id[case_id],
                model=checkpoint_model,
                estimator=checkpoint_estimator,
                encoder=checkpoint_encoder,
                config=config,
                variant=variant_config,
            )

        if claim_mode == "canonical":
            verified_case_ids[(variant, seed)] = tuple(
                str(row["case_id"]) for row in trace_rows
            )
            verified_successes[(variant, seed)] = {
                str(row["case_id"]): bool(row["success"]) for row in trace_rows
            }
            if variant in comparison_metric_rows:
                comparison_metric_rows[variant].extend(
                    _claim_metric_projection(row) for row in trace_rows
                )

        published_metrics = _read_json_object(
            metrics_path, description="published run metrics"
        )
        recomputed_metrics = compute_metrics(
            trace_rows,
            timeout_s=config.evaluation.timeout_s,
            bootstrap_samples=config.evaluation.bootstrap_samples,
            bootstrap_seed=config.evaluation.bootstrap_seed + seed,
            confidence=config.evaluation.confidence,
        )
        recomputed = compare_metrics(recomputed_metrics, published_metrics, tolerance=1e-9)
        if not recomputed.matches:
            raise RuntimeError(f"metric mismatch in {trace_path}")
        if (
            _read_json_object(recompute_path, description="metric recomputation evidence")
            != recomputed.to_dict()
        ):
            raise RuntimeError(f"recompute evidence mismatch in {recompute_path}")
        metric_fields = {
            "cases": recomputed.metrics.cases,
            "tsr": recomputed.metrics.tsr,
            "successful_conditional_simulated_service_time_s": (
                recomputed.metrics.successful_conditional_simulated_service_time_s
            ),
            "timeout_penalized_simulated_cost_s": (
                recomputed.metrics.timeout_penalized_simulated_cost_s
            ),
        }
        if any(run.get(name) != value for name, value in metric_fields.items()):
            raise RuntimeError(f"run metric summary mismatch in {trace_path}")
        observed_units += recomputed.metrics.cases
        checked.append(
            {
                "variant": variant,
                "seed": seed,
                "cases": recomputed.metrics.cases,
                "matches": True,
            }
        )

    expected_units = len(expected_pairs) * len(test_tasks)
    if payload.get("expected_evaluation_units") != expected_units:
        raise RuntimeError("declared evaluation-unit count is not derived from matrix x test split")
    if observed_units != expected_units:
        raise RuntimeError(f"expected {expected_units} evaluation units, observed {observed_units}")
    claim_path = root / "claim-check.json"
    if claim_mode == "canonical":
        declared_claim_path = _resolve_artifact_path(
            root,
            payload.get("claim_gate"),
            description="canonical claim evidence",
        )
        if declared_claim_path != claim_path.resolve():
            raise RuntimeError("canonical claim path is not rooted in the run directory")
        all_variant_runs = {
            name: [run for run in runs if run["variant"] == name] for name in variants
        }
        if any(len(values) != len(seeds) for values in all_variant_runs.values()):
            raise RuntimeError("canonical claim lacks paired A-F seed runs")
        case_ids_by_variant_seed: dict[str, dict[int, tuple[str, ...]]] = {}
        success_by_variant_seed: dict[str, dict[int, dict[str, bool]]] = {}
        for name in variants:
            case_ids_by_variant_seed[name] = {}
            success_by_variant_seed[name] = {}
            for run in all_variant_runs[name]:
                run_seed = int(run["seed"])
                case_ids_by_variant_seed[name][run_seed] = verified_case_ids[(name, run_seed)]
                success_by_variant_seed[name][run_seed] = verified_successes[(name, run_seed)]

        summaries: dict[str, dict[str, float | None]] = {}
        for name in comparison_names:
            aggregate = compute_metrics(
                comparison_metric_rows[name], timeout_s=config.evaluation.timeout_s
            )
            summaries[name] = {
                "tsr": aggregate.tsr,
                "successful_conditional_simulated_service_time_s": (
                    aggregate.successful_conditional_simulated_service_time_s
                ),
                "timeout_penalized_simulated_cost_s": (
                    aggregate.timeout_penalized_simulated_cost_s
                ),
            }
        evidence = CanonicalClaimEvidence(
            seeds=seeds,
            test_case_ids=expected_case_ids,
            variant_definitions=tuple(
                _variant_definition(variant_configs[name]) for name in variants
            ),
            case_ids_by_variant_seed=case_ids_by_variant_seed,
            success_by_variant_seed=success_by_variant_seed,
            action_validity_count=action_evaluation.count,
            action_validity_valid_count=action_evaluation.valid_count,
            action_validity_invalid_count=action_evaluation.invalid_count,
            bootstrap_samples=config.evaluation.bootstrap_samples,
            bootstrap_seed=config.evaluation.bootstrap_seed,
            confidence=config.evaluation.confidence,
        )
        recomputed_claim = verify_canonical_claims(
            summaries[comparison.treatment_variant],
            summaries[comparison.matched_baseline_variant],
            summaries[comparison.total_system_baseline_variant],
            evidence,
            action_validity_balanced_accuracy=action_evaluation.balanced_accuracy,
        )
        if not recomputed_claim.canonical:
            raise RuntimeError("canonical claim lacks the frozen 5x6x1000/20k evidence")
        if (
            _read_json_object(claim_path, description="canonical claim evidence")
            != recomputed_claim.to_dict()
        ):
            raise RuntimeError("canonical claim evidence does not recompute")
    elif claim_path.exists():
        raise RuntimeError("non-canonical run contains stale canonical claim evidence")
    return {
        "passed": True,
        "runs": len(runs),
        "evaluation_units": observed_units,
        "checked": checked,
    }


__all__ = [
    "ExperimentResult",
    "VariantRunResult",
    "load_benchmark_splits",
    "prepare_benchmark",
    "run_experiment_matrix",
    "runtime_identity",
    "source_fingerprint",
    "validate_benchmark_evidence",
    "variant_config_sha256",
    "verify_experiment_manifest",
]
