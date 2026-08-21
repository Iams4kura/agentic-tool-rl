"""Strict configuration schemas shared by CPU and optional Qwen paths."""

from __future__ import annotations

from importlib import resources
from pathlib import Path
from typing import Any, Literal, TypeAlias

import yaml  # type: ignore[import-untyped]
from pydantic import BaseModel, ConfigDict, Field, PositiveFloat, PositiveInt, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class BenchmarkConfig(StrictModel):
    generator_version: str = "benchmark-v1"
    train_cases: PositiveInt
    dev_cases: PositiveInt
    test_cases: PositiveInt
    families: PositiveInt = 10
    min_steps: PositiveInt = 6
    max_steps: PositiveInt = 15
    seed: int

    @model_validator(mode="after")
    def validate_step_range(self) -> BenchmarkConfig:
        if (self.families, self.min_steps, self.max_steps) != (10, 6, 15):
            raise ValueError(
                "the installed benchmark generator is frozen to families=10, "
                "min_steps=6, and max_steps=15"
            )
        return self


class LightweightPolicyConfig(StrictModel):
    backend: Literal["lightweight"] = "lightweight"
    observation_dim: PositiveInt = 64
    action_dim: PositiveInt = 64
    hidden_dim: PositiveInt = 128


class BehaviorCloningConfig(StrictModel):
    epochs: PositiveInt
    batch_size: PositiveInt
    learning_rate: PositiveFloat


class PPOConfig(StrictModel):
    iterations: PositiveInt = 1
    epochs: PositiveInt
    rollout_episodes: PositiveInt
    minibatch_size: PositiveInt
    learning_rate: PositiveFloat
    gamma: float = Field(ge=0.0, le=1.0)
    gae_lambda: float = Field(ge=0.0, le=1.0)
    clip_ratio: PositiveFloat
    value_coefficient: float = Field(ge=0.0)
    entropy_coefficient: float = Field(ge=0.0)
    max_grad_norm: PositiveFloat


class TrainingConfig(StrictModel):
    behavior_cloning: BehaviorCloningConfig
    ppo: PPOConfig


class RewardConfig(StrictModel):
    task_success: float
    task_failure: float
    invalid_action: float
    forbidden_side_effect: float
    step_cost: float
    progress_beta: float = Field(ge=0.0)


class EvaluationConfig(StrictModel):
    timeout_s: PositiveFloat = 30.0
    bootstrap_samples: int = Field(default=1000, ge=0)
    bootstrap_seed: int = 20260808
    confidence: float = Field(default=0.95, gt=0.0, lt=1.0)

    @model_validator(mode="after")
    def validate_bootstrap_samples(self) -> EvaluationConfig:
        if self.bootstrap_samples == 1:
            raise ValueError("evaluation.bootstrap_samples must be zero or at least two")
        return self


class ExperimentConfig(StrictModel):
    kind: Literal["experiment"]
    name: str
    benchmark: BenchmarkConfig
    policy: LightweightPolicyConfig
    training: TrainingConfig
    reward: RewardConfig
    evaluation: EvaluationConfig


class AblationVariant(StrictModel):
    name: str
    algorithm: Literal["bc", "action_ppo", "sequence_ppo"]
    progress_reward: bool
    action_mask: bool


class CanonicalComparisonConfig(StrictModel):
    """Pre-registered comparison used by the canonical ablation report."""

    treatment_variant: Literal["E-PPO-Progress-Mask"]
    matched_baseline_variant: Literal["B-BC-Mask"]
    total_system_baseline_variant: Literal["A-BC-Unmasked"]
    primary_metric: Literal["tsr"]
    hypothesis: Literal["paired_tsr_difference_ci_lower_gt_zero"]


class AblationConfig(StrictModel):
    kind: Literal["ablation"]
    name: str
    seeds: tuple[int, ...]
    canonical_comparison: CanonicalComparisonConfig
    variants: tuple[AblationVariant, ...]

    @model_validator(mode="after")
    def validate_matrix(self) -> AblationConfig:
        if len(self.seeds) != 5 or len(set(self.seeds)) != 5:
            raise ValueError("ablation requires exactly five unique seeds")
        names = [variant.name for variant in self.variants]
        if len(names) != 6 or len(set(names)) != 6:
            raise ValueError("ablation requires exactly six uniquely named variants")
        expected = {
            ("A-BC-Unmasked", "bc", False, False),
            ("B-BC-Mask", "bc", False, True),
            ("C-PPO-Sparse-Unmasked", "action_ppo", False, False),
            ("D-PPO-Sparse-Mask", "action_ppo", False, True),
            ("E-PPO-Progress-Mask", "action_ppo", True, True),
            ("F-Sequence-PPO-Progress-Mask", "sequence_ppo", True, True),
        }
        observed = {
            (
                variant.name,
                variant.algorithm,
                variant.progress_reward,
                variant.action_mask,
            )
            for variant in self.variants
        }
        if observed != expected:
            raise ValueError("ablation variants must match the frozen fair-comparison matrix")
        return self


class QwenModelConfig(StrictModel):
    model_name_or_path: str
    revision: str = "main"
    dtype: Literal["float16", "bfloat16", "float32"] = "bfloat16"
    trust_remote_code: bool = False
    attention_implementation: Literal["eager", "sdpa", "flash_attention_2"] = "sdpa"
    max_input_tokens: PositiveInt = 4096
    max_new_tokens: PositiveInt = 128
    value_head: Literal["separate_linear"] = "separate_linear"


class LoRAConfig(StrictModel):
    rank: PositiveInt = 16
    alpha: PositiveInt = 32
    dropout: float = Field(default=0.05, ge=0.0, lt=1.0)
    target_modules: tuple[str, ...]

    @model_validator(mode="after")
    def validate_targets(self) -> LoRAConfig:
        if not self.target_modules or len(set(self.target_modules)) != len(self.target_modules):
            raise ValueError("lora.target_modules must be non-empty and unique")
        return self

    def to_peft_kwargs(self) -> dict[str, Any]:
        """Translate readable YAML names to PEFT's constructor names."""

        return {
            "r": self.rank,
            "lora_alpha": self.alpha,
            "lora_dropout": self.dropout,
            "target_modules": list(self.target_modules),
            "task_type": "CAUSAL_LM",
        }


class DistributedConfig(StrictModel):
    strategy: Literal["single_gpu", "ddp", "fsdp"]
    num_gpus: PositiveInt
    gradient_checkpointing: bool = True
    mixed_precision: Literal["no", "fp16", "bf16"] = "bf16"

    @model_validator(mode="after")
    def validate_strategy(self) -> DistributedConfig:
        if self.strategy == "single_gpu" and self.num_gpus != 1:
            raise ValueError("single_gpu strategy requires num_gpus=1")
        if self.strategy in {"ddp", "fsdp"} and self.num_gpus < 2:
            raise ValueError(f"{self.strategy} strategy requires at least two GPUs")
        return self


class QwenTrainingConfig(StrictModel):
    per_device_batch_size: PositiveInt
    gradient_accumulation_steps: PositiveInt
    learning_rate: PositiveFloat
    max_steps: PositiveInt
    save_steps: PositiveInt


class QwenGPUConfig(StrictModel):
    kind: Literal["qwen_gpu"]
    name: str
    seed: int
    model: QwenModelConfig
    lora: LoRAConfig
    distributed: DistributedConfig
    training: QwenTrainingConfig


ConfigDocument: TypeAlias = ExperimentConfig | AblationConfig | QwenGPUConfig
PackagedConfigName: TypeAlias = Literal[
    "ablation.yaml",
    "cpu_full.yaml",
    "qwen3_lora_gpu.yaml",
    "smoke.yaml",
]
PACKAGED_CONFIG_NAMES: frozenset[str] = frozenset(
    {
        "ablation.yaml",
        "cpu_full.yaml",
        "qwen3_lora_gpu.yaml",
        "smoke.yaml",
    }
)


def _parse_raw(text: str, *, source: str) -> dict[str, Any]:
    value = yaml.safe_load(text)
    if not isinstance(value, dict):
        raise ValueError(f"config {source} must contain a YAML object")
    return value


def _validate_document(raw: dict[str, Any]) -> ConfigDocument:
    kind = raw.get("kind")
    if kind == "experiment":
        return ExperimentConfig.model_validate(raw)
    if kind == "ablation":
        return AblationConfig.model_validate(raw)
    if kind == "qwen_gpu":
        return QwenGPUConfig.model_validate(raw)
    raise ValueError(f"unsupported config kind {kind!r}")


def load_config(path: str | Path) -> ConfigDocument:
    """Load an explicit user path without any cwd or package fallback."""

    source = Path(path)
    return _validate_document(
        _parse_raw(source.read_text(encoding="utf-8"), source=str(source))
    )


def load_packaged_config(name: PackagedConfigName | str) -> ConfigDocument:
    """Load one immutable default from the installed package resources."""

    if name not in PACKAGED_CONFIG_NAMES:
        raise ValueError(f"unknown packaged config {name!r}")
    resource = resources.files("agentic_tool_rl").joinpath("resources", "configs", name)
    source = f"package resource agentic_tool_rl/resources/configs/{name}"
    try:
        text = resource.read_text(encoding="utf-8")
    except (FileNotFoundError, ModuleNotFoundError) as exc:
        raise RuntimeError(f"required {source} is missing") from exc
    return _validate_document(_parse_raw(text, source=source))


def dry_run_config(path: str | Path | ConfigDocument) -> dict[str, Any]:
    """Validate wiring without importing model libraries or downloading weights."""

    config = (
        path
        if isinstance(path, (ExperimentConfig, AblationConfig, QwenGPUConfig))
        else load_config(path)
    )
    result: dict[str, Any] = {
        "valid": True,
        "kind": config.kind,
        "name": config.name,
        "would_download_weights": False,
    }
    if isinstance(config, QwenGPUConfig):
        result.update(
            {
                "model_name_or_path": config.model.model_name_or_path,
                "revision": config.model.revision,
                "num_gpus": config.distributed.num_gpus,
                "strategy": config.distributed.strategy,
                "lora_rank": config.lora.rank,
                "optional_dependencies": ["transformers", "peft", "accelerate"],
            }
        )
    elif isinstance(config, AblationConfig):
        result.update({"variants": len(config.variants), "seeds": len(config.seeds)})
    else:
        result.update(
            {
                "backend": config.policy.backend,
                "test_cases": config.benchmark.test_cases,
                "bootstrap_samples": config.evaluation.bootstrap_samples,
            }
        )
    return result


__all__ = [
    "PACKAGED_CONFIG_NAMES",
    "AblationConfig",
    "AblationVariant",
    "BenchmarkConfig",
    "CanonicalComparisonConfig",
    "ConfigDocument",
    "EvaluationConfig",
    "ExperimentConfig",
    "LightweightPolicyConfig",
    "LoRAConfig",
    "PackagedConfigName",
    "QwenGPUConfig",
    "dry_run_config",
    "load_config",
    "load_packaged_config",
]
