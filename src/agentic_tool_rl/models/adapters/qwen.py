"""Lazy Qwen adapter for structured tool-call policies.

Importing this module never imports ``transformers`` or ``peft``. Optional
dependencies and model weights are touched only by ``from_pretrained``.
"""

from __future__ import annotations

import importlib.util
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from agentic_tool_rl.models.adapters.base import (
    BackendContractError,
    EncodedPolicyInput,
    PolicyBackend,
    PolicyOutput,
)
from agentic_tool_rl.policy_input import PolicyInput


@dataclass(frozen=True)
class GenerationResult:
    """Runtime output required for an action-level PPO transition."""

    text: str
    log_prob: float
    value: float

    def __post_init__(self) -> None:
        if not isfinite(self.log_prob) or not isfinite(self.value):
            raise BackendContractError("runtime log_prob and value must be finite")


@runtime_checkable
class QwenRuntime(Protocol):
    def generate(self, prompt: str, *, deterministic: bool) -> GenerationResult:
        """Generate exactly one structured action with its policy statistics."""


class QwenAdapter(PolicyBackend):
    """Translate one canonical ``PolicyInput`` to and from a Qwen runtime."""

    def __init__(self, runtime: QwenRuntime) -> None:
        if not isinstance(runtime, QwenRuntime):
            raise TypeError("runtime must implement generate(prompt, deterministic=...)")
        self.runtime = runtime

    @staticmethod
    def optional_dependencies_available() -> bool:
        return (
            importlib.util.find_spec("transformers") is not None
            and importlib.util.find_spec("peft") is not None
        )

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        *,
        revision: str = "main",
        dtype: str = "bfloat16",
        device_map: str | Mapping[str, Any] = "auto",
        trust_remote_code: bool = False,
        max_new_tokens: int = 128,
        lora: Mapping[str, Any] | None = None,
        value_head_path: str | Path | None = None,
    ) -> QwenAdapter:
        """Create the real runtime, importing optional dependencies lazily."""

        runtime = TransformersQwenRuntime.from_pretrained(
            model_name_or_path,
            revision=revision,
            dtype=dtype,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
            max_new_tokens=max_new_tokens,
            lora=lora,
            value_head_path=value_head_path,
        )
        return cls(runtime)

    def encode(self, policy_input: PolicyInput) -> EncodedPolicyInput:
        if not isinstance(policy_input, PolicyInput):
            raise TypeError("QwenAdapter.encode accepts only PolicyInput")
        canonical = policy_input.canonical_bytes().decode("utf-8")
        prompt = (
            "You are a structured tool-calling policy. Use only POLICY_INPUT and choose "
            "exactly one candidate whose action_mask permits it. Return one JSON object and "
            "no prose: "
            '{"tool_name":"...","arguments":{...}}.\n'
            f"POLICY_INPUT_SHA256={policy_input.sha256()}\n"
            f"POLICY_INPUT={canonical}"
        )
        return EncodedPolicyInput(prompt, policy_input)

    def act(
        self,
        policy_input: PolicyInput,
        *,
        deterministic: bool = False,
    ) -> PolicyOutput:
        encoded = self.encode(policy_input)
        generated = self.runtime.generate(encoded.prompt, deterministic=deterministic)
        if not isinstance(generated, GenerationResult):
            raise BackendContractError(
                "QwenRuntime.generate must return GenerationResult with log_prob and value"
            )
        tool_call = parse_tool_call(generated.text)
        action_index = match_candidate(tool_call, encoded.candidates)
        if not policy_input.candidates[action_index].action_mask:
            raise BackendContractError("generated tool call selects a masked candidate")
        return PolicyOutput(
            action_index=action_index,
            tool_call=tool_call,
            log_prob=generated.log_prob,
            value=generated.value,
            raw_output=generated.text,
        )

    def encode_legacy(
        self,
        observation: Any,
        candidates: Sequence[Any],
        *,
        action_mask: Sequence[bool],
        tool_schemas: Sequence[Any],
    ) -> EncodedPolicyInput:
        """Explicit migration adapter for callers that still hold four components."""

        return self.encode(
            PolicyInput.from_decision(
                observation,
                candidates,
                action_mask,
                tool_schemas,
            )
        )

    def act_legacy(
        self,
        observation: Any,
        candidates: Sequence[Any],
        *,
        action_mask: Sequence[bool],
        tool_schemas: Sequence[Any],
        deterministic: bool = False,
    ) -> PolicyOutput:
        """Explicit compatibility entry; normal execution must call ``act(PolicyInput)``."""

        policy_input = PolicyInput.from_decision(
            observation,
            candidates,
            action_mask,
            tool_schemas,
        )
        return self.act(policy_input, deterministic=deterministic)


def _json_objects(text: str) -> Sequence[Mapping[str, Any]]:
    decoder = json.JSONDecoder()
    objects: list[Mapping[str, Any]] = []
    for match in re.finditer(r"\{", text):
        try:
            value, _ = decoder.raw_decode(text[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(value, Mapping):
            objects.append(value)
    return objects


def parse_tool_call(text: str) -> dict[str, Any]:
    """Parse a single JSON/fenced/<tool_call> result into the canonical schema."""

    if not isinstance(text, str) or not text.strip():
        raise BackendContractError("model output is empty")
    candidates = _json_objects(text)
    for value in candidates:
        name = value.get("tool_name", value.get("name"))
        arguments = value.get("arguments", value.get("parameters"))
        if not isinstance(name, str) or not name:
            continue
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError as exc:
                raise BackendContractError("tool-call arguments string is not valid JSON") from exc
        if not isinstance(arguments, Mapping):
            raise BackendContractError("tool-call arguments must be a JSON object")
        return {"tool_name": name, "arguments": dict(arguments)}
    raise BackendContractError("model output does not contain a structured tool call")


def _candidate_name(candidate: Mapping[str, Any]) -> str | None:
    nested = candidate.get("tool_call")
    if isinstance(nested, Mapping):
        nested_name = nested.get("tool_name", nested.get("name"))
        return nested_name if isinstance(nested_name, str) else None
    value = candidate.get("tool_name", candidate.get("name"))
    return value if isinstance(value, str) else None


def _candidate_arguments(candidate: Mapping[str, Any]) -> Mapping[str, Any] | None:
    nested = candidate.get("tool_call")
    source = nested if isinstance(nested, Mapping) else candidate
    value = source.get("arguments")
    return value if isinstance(value, Mapping) else None


def match_candidate(tool_call: Mapping[str, Any], candidates: Sequence[Mapping[str, Any]]) -> int:
    """Ground a generated call in the supplied candidate set.

    Fixed calls (candidates with ``arguments``) require an exact argument match.
    Tool-schema candidates without fixed arguments match by tool name.
    """

    name = tool_call.get("tool_name")
    arguments = tool_call.get("arguments")
    for index, candidate in enumerate(candidates):
        if _candidate_name(candidate) != name:
            continue
        expected_arguments = _candidate_arguments(candidate)
        if expected_arguments is None or dict(expected_arguments) == arguments:
            return index
    raise BackendContractError("generated tool call is not one of the grounded candidates")


class FakeQwenRuntime:
    """Small deterministic runtime for contract tests and offline dry-runs."""

    def __init__(self, outputs: Sequence[GenerationResult]) -> None:
        if not outputs:
            raise ValueError("outputs must not be empty")
        self._outputs = list(outputs)
        self.prompts: list[str] = []

    def generate(self, prompt: str, *, deterministic: bool) -> GenerationResult:
        self.prompts.append(prompt)
        index = min(len(self.prompts) - 1, len(self._outputs) - 1)
        return self._outputs[index]


class TransformersQwenRuntime:
    """Optional Hugging Face runtime with an explicit scalar value head."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        value_head: Any,
        *,
        max_new_tokens: int,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.value_head = value_head
        self.max_new_tokens = max_new_tokens

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        *,
        revision: str,
        dtype: str,
        device_map: str | Mapping[str, Any],
        trust_remote_code: bool,
        max_new_tokens: int,
        lora: Mapping[str, Any] | None,
        value_head_path: str | Path | None,
    ) -> TransformersQwenRuntime:
        if not QwenAdapter.optional_dependencies_available():
            raise ImportError(
                "Qwen support is optional; install with `pip install 'agentic-tool-rl[qwen]'`"
            )
        # These imports are intentionally inside the constructor path.
        import torch
        from peft import LoraConfig, get_peft_model  # type: ignore[import-not-found]
        from transformers import (  # type: ignore[import-not-found]
            AutoModelForCausalLM,
            AutoTokenizer,
        )

        if not hasattr(torch, dtype):
            raise ValueError(f"unsupported torch dtype {dtype!r}")
        torch_dtype = getattr(torch, dtype)
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            revision=revision,
            trust_remote_code=trust_remote_code,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            revision=revision,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
        )
        if lora:
            model = get_peft_model(model, LoraConfig(**dict(lora)))
        hidden_size = int(model.config.hidden_size)
        value_head = torch.nn.Linear(hidden_size, 1, bias=True).to(
            device=next(model.parameters()).device,
            dtype=next(model.parameters()).dtype,
        )
        torch.nn.init.zeros_(value_head.weight)
        torch.nn.init.zeros_(value_head.bias)
        if value_head_path is not None:
            state = torch.load(value_head_path, map_location="cpu", weights_only=True)
            value_head.load_state_dict(state)
        return cls(model, tokenizer, value_head, max_new_tokens=max_new_tokens)

    def generate(self, prompt: str, *, deterministic: bool) -> GenerationResult:
        import torch

        encoded = self.tokenizer(prompt, return_tensors="pt")
        device = next(self.model.parameters()).device
        encoded = {name: tensor.to(device) for name, tensor in encoded.items()}
        prompt_length = int(encoded["input_ids"].shape[1])
        with torch.no_grad():
            generated = self.model.generate(
                **encoded,
                max_new_tokens=self.max_new_tokens,
                do_sample=not deterministic,
                return_dict_in_generate=True,
                output_scores=True,
            )
            sequence = generated.sequences
            transition_scores = self.model.compute_transition_scores(
                sequence, generated.scores, normalize_logits=True
            )
            forward = self.model(sequence, output_hidden_states=True, use_cache=False)
            final_hidden = forward.hidden_states[-1][:, -1, :]
            value = float(self.value_head(final_hidden).squeeze().item())
        new_tokens = sequence[0, prompt_length:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        log_prob = float(transition_scores[0].sum().item())
        return GenerationResult(text=text, log_prob=log_prob, value=value)
