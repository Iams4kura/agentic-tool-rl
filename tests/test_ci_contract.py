from __future__ import annotations

import tomllib
from pathlib import Path

from agentic_tool_rl.config import AblationConfig, ExperimentConfig, load_config

ROOT = Path(__file__).resolve().parents[1]


def test_default_ci_is_lightweight_only() -> None:
    workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    ci_recipe = makefile.split("\nci:", maxsplit=1)[1].split(
        "benchmark:", maxsplit=1
    )[0]

    assert "make verify-smoke" in workflow
    assert "qwen" not in workflow.lower()
    assert "qwen" not in ci_recipe.lower()


def test_qwen_contract_is_manual_and_excluded_from_default_pytest() -> None:
    workflow = (ROOT / ".github/workflows/qwen-adapter.yml").read_text(
        encoding="utf-8"
    )
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    pytest_options = pyproject["tool"]["pytest"]["ini_options"]

    assert "workflow_dispatch:" in workflow
    assert "push:" not in workflow
    assert "pull_request:" not in workflow
    assert "make qwen-check" in workflow
    assert '-m "not qwen"' in pytest_options["addopts"]


def test_full_benchmark_invokes_canonical_ablate_command() -> None:
    workflow = (ROOT / ".github/workflows/full-benchmark.yml").read_text(
        encoding="utf-8"
    )
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")

    assert "run: make benchmark" in workflow
    assert "agentic-tool-rl ablate" in makefile


def test_public_configs_do_not_contain_inert_identity_fields() -> None:
    experiment = load_config(ROOT / "configs/cpu_full.yaml")
    ablation = load_config(ROOT / "configs/ablation.yaml")

    assert isinstance(experiment, ExperimentConfig)
    assert isinstance(ablation, AblationConfig)
    assert {"seed", "artifacts_dir"}.isdisjoint(experiment.model_dump())
    assert "recurrent" not in experiment.policy.model_dump()
    assert "base_config" not in ablation.model_dump()
