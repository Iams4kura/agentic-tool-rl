from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from agentic_tool_rl.config import (
    AblationConfig,
    ExperimentConfig,
    QwenGPUConfig,
    dry_run_config,
    load_config,
)
from agentic_tool_rl.contracts import Split
from agentic_tool_rl.envs.benchmark import generate_tasks
from agentic_tool_rl.envs.workflow import TransactionalWorkflowEnv
from agentic_tool_rl.features import FeatureEncoder
from agentic_tool_rl.grounding.action_mask import ActionMask
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
from agentic_tool_rl.policy_input import PolicyInput

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.qwen


def _policy_input(*, base_seed: int = 1601) -> PolicyInput:
    task = generate_tasks(Split.TEST, 1, base_seed=base_seed)[0]
    environment = TransactionalWorkflowEnv(task)
    observation = environment.observe()
    candidates = environment.candidate_actions()
    public_mask = ActionMask(task.tool_schemas).mask(observation, candidates)
    return PolicyInput.from_decision(
        observation,
        candidates,
        public_mask,
        task.tool_schemas,
    )


def test_qwen_module_does_not_eagerly_import_optional_dependencies() -> None:
    # The imports above exercised the public adapter module. Core imports still
    # must not initialize heavyweight optional libraries.
    assert "transformers" not in sys.modules
    assert "peft" not in sys.modules


def test_fake_qwen_runtime_exercises_full_policy_contract() -> None:
    policy_input = _policy_input()
    selected_index = next(
        index
        for index, candidate in enumerate(policy_input.candidates)
        if candidate.action_mask
    )
    selected = policy_input.candidates[selected_index].tool_call.model_dump(mode="json")
    runtime = FakeQwenRuntime(
        [
            GenerationResult(
                text=f"```json\n{json.dumps(selected, sort_keys=True)}\n```",
                log_prob=-0.25,
                value=0.7,
            )
        ]
    )
    adapter = QwenAdapter(runtime)

    output = adapter.act(policy_input, deterministic=True)

    assert output == PolicyOutput(
        action_index=selected_index,
        tool_call=selected,
        log_prob=-0.25,
        value=0.7,
        raw_output=runtime._outputs[0].text,
    )
    assert "POLICY_INPUT=" in runtime.prompts[0]
    assert policy_input.sha256() in runtime.prompts[0]
    assert output.to_dict()["value"] == pytest.approx(0.7)


def test_qwen_prompt_contains_only_public_observation_and_candidate_dtos() -> None:
    policy_input = _policy_input(base_seed=1602)
    adapter = QwenAdapter(
        FakeQwenRuntime(
            [
                GenerationResult(
                    json.dumps(
                        policy_input.candidates[0].tool_call.model_dump(mode="json"),
                        sort_keys=True,
                    ),
                    0.0,
                    0.0,
                )
            ]
        )
    )
    encoded = adapter.encode(policy_input)
    feature_input = FeatureEncoder(32, 32).encode_policy_input(policy_input)

    assert encoded.policy_input is policy_input
    assert encoded.policy_input_sha256 == feature_input.policy_input_sha256
    assert encoded.prompt.endswith(
        f"POLICY_INPUT={policy_input.canonical_bytes().decode('utf-8')}"
    )
    assert encoded.observation == policy_input.observation.model_dump(mode="json")
    assert encoded.candidates == tuple(
        candidate.tool_call.model_dump(mode="json")
        for candidate in policy_input.candidates
    )
    assert '"description"' in encoded.prompt
    assert '"mutating"' in encoded.prompt
    assert '"required_completed_operations"' in encoded.prompt
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


def test_qwen_prompt_is_invariant_to_hidden_observation_and_call_id_replacement() -> None:
    task = generate_tasks(Split.TEST, 1, base_seed=1607)[0]
    environment = TransactionalWorkflowEnv(task)
    observation = environment.observe()
    candidates = environment.candidate_actions()
    public_mask = ActionMask(task.tool_schemas).mask(observation, candidates)
    baseline = PolicyInput.from_decision(
        observation,
        candidates,
        public_mask,
        task.tool_schemas,
    )
    changed_observation = observation.model_dump(mode="json")
    changed_observation["task_id"] = "replacement-hidden-task"
    changed_observation["visible_state"] = {
        **changed_observation["visible_state"],
        "processed_call_ids": ["replacement-secret-ledger"],
        "oracle_distance": 0,
        "positive_label": True,
    }
    changed_candidates = [
        candidate.model_copy(update={"call_id": f"replacement-call-{index}"})
        for index, candidate in enumerate(candidates)
    ]
    changed = PolicyInput.from_decision(
        changed_observation,
        changed_candidates,
        public_mask,
        task.tool_schemas,
    )
    adapter = QwenAdapter(
        FakeQwenRuntime(
            [GenerationResult('{"tool_name":"unused","arguments":{}}', 0.0, 0.0)]
        )
    )

    assert adapter.encode(changed).prompt == adapter.encode(baseline).prompt


def test_parser_accepts_openai_arguments_string_and_rejects_prose_only() -> None:
    assert parse_tool_call(
        '<tool_call>{"name":"lookup","arguments":"{\\"id\\": 3}"}</tool_call>'
    ) == {"tool_name": "lookup", "arguments": {"id": 3}}
    with pytest.raises(BackendContractError, match="structured tool call"):
        parse_tool_call("I think the lookup tool would be useful.")


def test_adapter_rejects_ungrounded_or_incomplete_runtime_output() -> None:
    policy_input = _policy_input(base_seed=1603)
    adapter = QwenAdapter(
        FakeQwenRuntime([GenerationResult('{"tool_name":"delete","arguments":{}}', -1.0, 0.0)])
    )
    with pytest.raises(BackendContractError, match="grounded candidates"):
        adapter.act(policy_input)

    class BrokenRuntime:
        def generate(self, prompt: str, *, deterministic: bool) -> str:
            return '{"tool_name":"read","arguments":{}}'

    with pytest.raises(BackendContractError, match="GenerationResult"):
        QwenAdapter(BrokenRuntime()).act(policy_input)

    with pytest.raises(TypeError, match="only PolicyInput"):
        adapter.encode(object())  # type: ignore[arg-type]


def test_adapter_rejects_a_grounded_but_masked_candidate() -> None:
    policy_input = _policy_input(base_seed=1604)
    masked = next(candidate for candidate in policy_input.candidates if not candidate.action_mask)
    output = json.dumps(masked.tool_call.model_dump(mode="json"), sort_keys=True)
    adapter = QwenAdapter(FakeQwenRuntime([GenerationResult(output, -1.0, 0.0)]))

    with pytest.raises(BackendContractError, match="masked candidate"):
        adapter.act(policy_input)


def test_explicit_legacy_entry_builds_the_same_canonical_policy_input() -> None:
    task = generate_tasks(Split.TEST, 1, base_seed=1606)[0]
    environment = TransactionalWorkflowEnv(task)
    observation = environment.observe()
    candidates = environment.candidate_actions()
    public_mask = ActionMask(task.tool_schemas).mask(observation, candidates)
    expected = PolicyInput.from_decision(
        observation,
        candidates,
        public_mask,
        task.tool_schemas,
    )
    adapter = QwenAdapter(
        FakeQwenRuntime(
            [GenerationResult('{"tool_name":"unused","arguments":{}}', 0.0, 0.0)]
        )
    )

    encoded = adapter.encode_legacy(
        observation,
        candidates,
        action_mask=public_mask,
        tool_schemas=task.tool_schemas,
    )

    assert encoded.policy_input.canonical_bytes() == expected.canonical_bytes()


def test_policy_backend_contract_is_backend_neutral() -> None:
    class FakePolicy(PolicyBackend):
        def encode(self, policy_input: PolicyInput) -> EncodedPolicyInput:
            return EncodedPolicyInput("fake", policy_input)

        def act(
            self,
            policy_input: PolicyInput,
            *,
            deterministic: bool = False,
        ) -> PolicyOutput:
            action_index = next(
                index
                for index, candidate in enumerate(policy_input.candidates)
                if candidate.action_mask
            )
            call = policy_input.candidates[action_index].tool_call.model_dump(mode="json")
            return PolicyOutput(action_index, call, -0.1, 0.2)

    backend: PolicyBackend = FakePolicy()
    policy_input = _policy_input(base_seed=1605)
    output = backend.act(policy_input, deterministic=True)
    assert policy_input.candidates[output.action_index].action_mask
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
