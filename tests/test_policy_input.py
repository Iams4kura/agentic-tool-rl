from __future__ import annotations

import json
from copy import deepcopy

import pytest
import torch

from agentic_tool_rl.contracts import Split
from agentic_tool_rl.envs.benchmark import generate_tasks
from agentic_tool_rl.envs.workflow import TransactionalWorkflowEnv
from agentic_tool_rl.features import FeatureEncoder
from agentic_tool_rl.grounding.action_mask import ActionMask
from agentic_tool_rl.policy_input import PolicyInput


def _initial_decision():  # type: ignore[no-untyped-def]
    task = generate_tasks(Split.TEST, 1, base_seed=1401)[0]
    environment = TransactionalWorkflowEnv(task)
    observation = environment.observe()
    candidates = environment.candidate_actions()
    action_mask = ActionMask(task.tool_schemas).mask(observation, candidates)
    return task, observation, candidates, action_mask


def test_policy_input_has_stable_canonical_bytes_and_no_evidence_ids() -> None:
    task, observation, candidates, action_mask = _initial_decision()

    policy_input = PolicyInput.from_decision(
        observation,
        candidates,
        action_mask,
        task.tool_schemas,
    )
    rebuilt = PolicyInput.model_validate_json(policy_input.canonical_bytes())
    encoded = policy_input.canonical_bytes().decode("utf-8")

    assert rebuilt.canonical_bytes() == policy_input.canonical_bytes()
    assert rebuilt.sha256() == policy_input.sha256()
    assert observation.task_id not in encoded
    assert task.entity_id not in encoded
    assert "call_id" not in encoded
    assert all(candidate.call_id not in encoded for candidate in candidates)
    assert [candidate.action_mask for candidate in policy_input.candidates] == action_mask
    assert all(
        candidate.tool_call.tool_name == candidate.tool_schema.name
        for candidate in policy_input.candidates
    )


def test_hidden_task_replacement_and_private_nested_fields_do_not_change_input() -> None:
    task, observation, candidates, action_mask = _initial_decision()
    baseline = PolicyInput.from_decision(
        observation,
        candidates,
        action_mask,
        task.tool_schemas,
    )
    changed_hidden_task = task.model_copy(
        update={
            "workflow_nodes": [],
            "hidden_goal_predicates": [],
            "forbidden_side_effects": ["replacement.hidden.effect"],
            "oracle_plans": [],
            "latency_trace": {"hidden": 999.0},
        }
    )
    observation_with_private_fields = observation.model_dump(mode="json")
    observation_with_private_fields["task_id"] = "replacement-private-task-id"
    observation_with_private_fields["visible_state"] = {
        **observation_with_private_fields["visible_state"],
        "processed_call_ids": ["secret-call"],
        "idempotency_results": {"secret": "ledger"},
        "audit_log": ["private"],
        "oracle_distance": 0,
        "positive_label": True,
    }
    renamed_calls = [
        candidate.model_copy(update={"call_id": f"replacement-{index}"})
        for index, candidate in enumerate(candidates)
    ]

    changed = PolicyInput.from_decision(
        observation_with_private_fields,
        renamed_calls,
        action_mask,
        changed_hidden_task.tool_schemas,
    )

    assert changed.canonical_bytes() == baseline.canonical_bytes()
    encoder = FeatureEncoder(32, 32)
    baseline_features = encoder.encode_policy_input(baseline)
    changed_features = encoder.encode_policy_input(changed)
    assert torch.equal(changed_features.state, baseline_features.state)
    assert torch.equal(changed_features.actions, baseline_features.actions)
    assert torch.equal(changed_features.mask, baseline_features.mask)


def test_consistent_entity_renaming_preserves_policy_input_bytes() -> None:
    task, observation, candidates, action_mask = _initial_decision()
    baseline = PolicyInput.from_decision(
        observation,
        candidates,
        action_mask,
        task.tool_schemas,
    )
    replacement = "renamed-entity-with-no-shared-prefix"
    raw_observation = observation.model_dump(mode="json")
    raw_observation["task_id"] = "renamed-case"
    raw_observation["user_goal"] = raw_observation["user_goal"].replace(
        task.entity_id,
        replacement,
    )
    raw_observation["available_entities"] = [replacement]
    raw_observation["visible_state"]["entity_id"] = replacement
    raw_candidates = []
    for index, candidate in enumerate(candidates):
        raw = deepcopy(candidate.model_dump(mode="json"))
        raw["call_id"] = f"renamed-call-{index}"
        for key, value in raw["arguments"].items():
            if isinstance(value, str):
                raw["arguments"][key] = value.replace(task.entity_id, replacement)
        raw_candidates.append(raw)

    renamed = PolicyInput.from_decision(
        raw_observation,
        raw_candidates,
        action_mask,
        task.tool_schemas,
    )

    assert renamed.canonical_bytes() == baseline.canonical_bytes()


def test_policy_input_rejects_candidate_without_corresponding_schema() -> None:
    task, observation, candidates, action_mask = _initial_decision()

    with pytest.raises(ValueError, match="unknown public schema"):
        PolicyInput.from_decision(
            observation,
            candidates,
            action_mask,
            task.tool_schemas[1:],
        )


def test_policy_input_canonical_json_rejects_nan() -> None:
    task, observation, candidates, action_mask = _initial_decision()
    raw = observation.model_dump(mode="json")
    raw["visible_state"]["bad"] = float("nan")

    with pytest.raises(ValueError, match="finite JSON"):
        PolicyInput.from_decision(raw, candidates, action_mask, task.tool_schemas)

    # Keep this assertion explicit: canonical bytes are JSON, not repr/pickle.
    valid = PolicyInput.from_decision(observation, candidates, action_mask, task.tool_schemas)
    assert json.loads(valid.canonical_bytes())["schema_version"] == "policy-input-v1"
