from __future__ import annotations

import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

from agentic_tool_rl.config import (
    AblationConfig,
    ExperimentConfig,
    QwenGPUConfig,
    dry_run_config,
    load_config,
)
from agentic_tool_rl.models.adapters import (
    BackendContractError,
    EncodedPolicyInput,
    FakeQwenRuntime,
    GenerationResult,
    PolicyBackend,
    PolicyOutput,
    QwenAdapter,
    parse_tool_call,
)

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.qwen


def test_qwen_module_does_not_eagerly_import_optional_dependencies() -> None:
    # The imports above exercised the public adapter module. Core imports still
    # must not initialize heavyweight optional libraries.
    assert "transformers" not in sys.modules
    assert "peft" not in sys.modules


def test_fake_qwen_runtime_exercises_full_policy_contract() -> None:
    runtime = FakeQwenRuntime(
        [
            GenerationResult(
                text='```json\n{"tool_name":"approve","arguments":{"request_id":"r-7"}}\n```',
                log_prob=-0.25,
                value=0.7,
            )
        ]
    )
    adapter = QwenAdapter(runtime)
    candidates = [
        {"tool_name": "reject", "arguments": {"request_id": "r-7"}},
        {"tool_name": "approve", "arguments": {"request_id": "r-7"}},
    ]

    output = adapter.act(
        {"task_id": "t-1", "visible_state": {"approved": False}},
        candidates,
        deterministic=True,
    )

    assert output == PolicyOutput(
        action_index=1,
        tool_call={"tool_name": "approve", "arguments": {"request_id": "r-7"}},
        log_prob=-0.25,
        value=0.7,
        raw_output=runtime._outputs[0].text,
    )
    assert "visible_state" in runtime.prompts[0]
    assert "approve" in runtime.prompts[0]
    assert output.to_dict()["value"] == pytest.approx(0.7)


def test_qwen_prompt_contains_only_public_observation_and_candidate_dtos() -> None:
    adapter = QwenAdapter(
        FakeQwenRuntime([GenerationResult('{"tool_name":"read","arguments":{}}', 0.0, 0.0)])
    )
    encoded = adapter.encode(
        {
            "task_id": "hidden-case-id",
            "case_id": "hidden-case-id",
            "user_goal": "inspect the request",
            "visible_state": {
                "status": "pending",
                "processed_call_ids": ["secret-ledger-entry"],
                "valid_label": True,
            },
            "workflow_nodes": [{"safety_token": "hidden-token"}],
        },
        [
            {
                "tool_name": "read",
                "arguments": {},
                "call_id": "candidate-valid-label-must-not-leak",
                "valid_label": True,
                "schema_label": "valid",
                "schema": {"candidate_valid": True},
            }
        ],
    )

    assert encoded.observation == {
        "user_goal": "inspect the request",
        "visible_state": {"status": "pending"},
    }
    assert encoded.candidates == ({"tool_name": "read", "arguments": {}},)
    for secret in (
        "call_id",
        "candidate-valid-label-must-not-leak",
        "valid_label",
        "schema_label",
        "workflow_nodes",
        "hidden-case-id",
        "hidden-token",
        "secret-ledger-entry",
    ):
        assert secret not in encoded.prompt


def test_parser_accepts_openai_arguments_string_and_rejects_prose_only() -> None:
    assert parse_tool_call(
        '<tool_call>{"name":"lookup","arguments":"{\\"id\\": 3}"}</tool_call>'
    ) == {"tool_name": "lookup", "arguments": {"id": 3}}
    with pytest.raises(BackendContractError, match="structured tool call"):
        parse_tool_call("I think the lookup tool would be useful.")


def test_adapter_rejects_ungrounded_or_incomplete_runtime_output() -> None:
    adapter = QwenAdapter(
        FakeQwenRuntime([GenerationResult('{"tool_name":"delete","arguments":{}}', -1.0, 0.0)])
    )
    with pytest.raises(BackendContractError, match="grounded candidates"):
        adapter.act({"state": "safe"}, [{"tool_name": "read"}])

    class BrokenRuntime:
        def generate(self, prompt: str, *, deterministic: bool) -> str:
            return '{"tool_name":"read","arguments":{}}'

    with pytest.raises(BackendContractError, match="GenerationResult"):
        QwenAdapter(BrokenRuntime()).act({"state": "safe"}, [{"tool_name": "read"}])


def test_policy_backend_contract_is_backend_neutral() -> None:
    class FakePolicy(PolicyBackend):
        def encode(
            self,
            observation: Mapping[str, Any],
            candidates: Sequence[Mapping[str, Any]],
        ) -> EncodedPolicyInput:
            return EncodedPolicyInput("fake", observation, tuple(candidates))

        def act(
            self,
            observation: Mapping[str, Any],
            candidates: Sequence[Mapping[str, Any]],
            *,
            deterministic: bool = False,
        ) -> PolicyOutput:
            call = dict(candidates[0])
            return PolicyOutput(0, call, -0.1, 0.2)

    backend: PolicyBackend = FakePolicy()
    output = backend.act({"state": 1}, [{"tool_name": "read", "arguments": {}}], deterministic=True)
    assert output.action_index == 0
    assert output.log_prob == pytest.approx(-0.1)


def test_all_shipped_configs_validate_and_qwen_dry_run_is_offline() -> None:
    cpu = load_config(ROOT / "configs/cpu_full.yaml")
    ablation = load_config(ROOT / "configs/ablation.yaml")
    qwen = load_config(ROOT / "configs/qwen3_lora_gpu.yaml")

    assert isinstance(cpu, ExperimentConfig)
    assert cpu.benchmark.test_cases == 1000
    assert isinstance(ablation, AblationConfig)
    assert len(ablation.variants) == 6
    assert len(ablation.seeds) == 5
    assert ablation.canonical_comparison.treatment_variant == "E-PPO-Progress-Mask"
    assert ablation.canonical_comparison.matched_baseline_variant == "B-BC-Mask"
    assert (
        ablation.canonical_comparison.total_system_baseline_variant
        == "A-BC-Unmasked"
    )
    assert isinstance(qwen, QwenGPUConfig)
    assert qwen.lora.to_peft_kwargs()["r"] == 16
    assert qwen.lora.to_peft_kwargs()["task_type"] == "CAUSAL_LM"
    report = dry_run_config(ROOT / "configs/qwen3_lora_gpu.yaml")
    assert report == {
        "valid": True,
        "kind": "qwen_gpu",
        "name": "qwen3-4b-step-ppo-lora",
        "would_download_weights": False,
        "model_name_or_path": "Qwen/Qwen3-4B",
        "revision": "1cfa9a7208912126459214e8b04321603b3df60c",
        "num_gpus": 2,
        "strategy": "fsdp",
        "lora_rank": 16,
        "optional_dependencies": ["transformers", "peft", "accelerate"],
    }
