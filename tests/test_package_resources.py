from __future__ import annotations

import hashlib
import importlib
import json
from importlib import resources
from pathlib import Path

import pytest
from typer.testing import CliRunner

import agentic_tool_rl.cli as cli_module
import agentic_tool_rl.config as config_module
from agentic_tool_rl.cli import app

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNNER = CliRunner()
PACKAGED_CONFIGS = (
    "ablation.yaml",
    "cpu_full.yaml",
    "qwen3_lora_gpu.yaml",
    "smoke.yaml",
)
PROTOCOL_NAME = "benchmark-v1.4-development.json"


def test_packaged_configs_are_canonical_cwd_independent_resources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resource_root = resources.files("agentic_tool_rl").joinpath("resources", "configs")
    for name in PACKAGED_CONFIGS:
        packaged = resource_root.joinpath(name).read_bytes()
        assert packaged == (PROJECT_ROOT / "configs" / name).read_bytes()
        loaded = config_module.load_packaged_config(name)
        assert loaded.name

    empty_cwd = tmp_path / "empty-cwd"
    empty_cwd.mkdir()
    monkeypatch.chdir(empty_cwd)
    result = RUNNER.invoke(app, ["qwen-dry-run"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["kind"] == "qwen_gpu"
    assert payload["would_download_weights"] is False


def test_all_cli_default_resolvers_are_cwd_independent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    empty_cwd = tmp_path / "empty-cwd"
    empty_cwd.mkdir()
    monkeypatch.chdir(empty_cwd)

    assert cli_module._experiment_config(None).name == "cpu-full"
    assert cli_module._experiment_config(None, default_resource="smoke.yaml").name == "cpu-smoke"
    assert cli_module._ablation_config(None).name == "benchmark-v1-six-way"


def test_explicit_config_path_wins_and_missing_path_never_falls_back(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    packaged = (
        resources.files("agentic_tool_rl")
        .joinpath("resources", "configs", "qwen3_lora_gpu.yaml")
        .read_text(encoding="utf-8")
    )
    explicit = tmp_path / "custom-qwen.yaml"
    explicit.write_text(
        packaged.replace("name: qwen3-4b-step-ppo-lora", "name: explicit-user-config"),
        encoding="utf-8",
    )

    selected = RUNNER.invoke(app, ["qwen-dry-run", "--config", str(explicit)])
    assert selected.exit_code == 0, selected.output
    assert json.loads(selected.stdout)["name"] == "explicit-user-config"

    empty_cwd = tmp_path / "empty-cwd"
    empty_cwd.mkdir()
    monkeypatch.chdir(empty_cwd)
    missing = RUNNER.invoke(
        app,
        ["qwen-dry-run", "--config", "configs/qwen3_lora_gpu.yaml"],
    )
    assert missing.exit_code != 0
    assert isinstance(missing.exception, FileNotFoundError)


def test_packaged_v14_protocol_resolver_preserves_exact_bytes_and_hash() -> None:
    package_resources = importlib.import_module("agentic_tool_rl.package_resources")
    expected = (PROJECT_ROOT / "docs" / "protocol" / PROTOCOL_NAME).read_bytes()

    assert package_resources.packaged_protocol_bytes(PROTOCOL_NAME) == expected
    assert (
        package_resources.packaged_protocol_sha256(PROTOCOL_NAME)
        == hashlib.sha256(expected).hexdigest()
    )
    document = package_resources.load_packaged_protocol(PROTOCOL_NAME)
    assert document["generator_version"] == "benchmark-v1.4.0"
    assert document["canonical_final_permitted"] is False
    assert document["quality_gates"]["minimum_deterministic_shortest_plans_per_case"] == 4
    assert "R4-reviewer-composite-shortcut-v1" in document["baselines"]
    assert "R5-source-aware-family-stage-v1" in document["baselines"]
    assert "R6-public-downstream-centrality-v1" in document["baselines"]


def test_implementation_fingerprint_v2_is_layout_independent_and_complete() -> None:
    package_resources = importlib.import_module("agentic_tool_rl.package_resources")

    members = package_resources.implementation_fingerprint_members()
    digest = package_resources.implementation_fingerprint_v2()

    assert members == tuple(sorted(members))
    assert "agentic_tool_rl/py.typed" in members
    assert "agentic_tool_rl/resources/configs/smoke.yaml" in members
    assert "agentic_tool_rl/resources/protocols/benchmark-v1.4-development.json" in members
    assert all("__pycache__" not in member and not member.endswith(".pyc") for member in members)
    assert len(digest) == 64
    int(digest, 16)
