from __future__ import annotations

import torch

from agentic_tool_rl.contracts import Split
from agentic_tool_rl.envs.benchmark import generate_tasks
from agentic_tool_rl.envs.workflow import TransactionalWorkflowEnv
from agentic_tool_rl.features import FeatureEncoder
from agentic_tool_rl.grounding.action_mask import ActionMask
from agentic_tool_rl.policy_input import PolicyInput


def test_feature_encoder_is_deterministic_and_ignores_evidence_call_id() -> None:
    encoder = FeatureEncoder(state_dim=32, action_dim=32)
    observation = {
        "task_id": "test-order-1",
        "step_index": 2,
        "user_goal": "complete order",
        "visible_state": {
            "entity_id": "order-1",
            "version": 2,
            "completed_nodes": ["op_a1b2c3", "op_d4e5f6"],
        },
    }
    first = {
        "tool_name": "order.reserve_inventory",
        "arguments": {
            "entity_id": "order-1",
            "operation_id": "op_1234567890abcdef1234",
            "expected_version": 2,
        },
        "call_id": "candidate-valid-label-must-not-leak",
    }
    second = {**first, "call_id": "candidate-schema-another-label"}

    state_a = encoder.encode_state(observation)
    state_b = encoder.encode_state(dict(observation))
    state_with_private_ledger = encoder.encode_state(
        {
            **observation,
            "visible_state": {
                **observation["visible_state"],
                "processed_call_ids": ["candidate-valid-private"],
                "idempotency_results": {"private-key": "private-node"},
                "valid_label": True,
            },
        }
    )
    action_a = encoder.encode_action(first)
    action_b = encoder.encode_action(second)

    assert torch.equal(state_a, state_b)
    assert torch.equal(state_a, state_with_private_ledger)
    assert torch.equal(action_a, action_b)
    assert encoder.fingerprint() == encoder.fingerprint()


def test_decision_features_validate_mask_and_shapes() -> None:
    encoder = FeatureEncoder(state_dim=24, action_dim=20)
    observation = {"visible_state": {"version": 0, "completed_nodes": []}}
    candidates = [
        {
            "tool_name": "workflow.open",
            "arguments": {
                "entity_id": "x",
                "operation_id": "op_11111111111111111111",
                "expected_version": 0,
            },
        },
        {
            "tool_name": "workflow.close",
            "arguments": {
                "entity_id": "x",
                "operation_id": "op_22222222222222222222",
                "expected_version": 0,
            },
        },
    ]

    features = encoder.encode_decision(observation, candidates, [True, False])
    states, actions, mask = features.as_batch()

    assert states.shape == (1, 24)
    assert actions.shape == (1, 2, 20)
    assert mask.tolist() == [[True, False]]


def test_decision_features_encode_public_entity_grounding_relation() -> None:
    encoder = FeatureEncoder(state_dim=32, action_dim=32)
    observation = {
        "available_entities": ["visible-entity"],
        "visible_state": {"entity_id": "visible-entity", "version": 0},
    }
    candidates = [
        {
            "tool_name": "workflow.open",
            "arguments": {
                "entity_id": "visible-entity",
                "operation_id": "op_11111111111111111111",
                "expected_version": 0,
            },
        },
        {
            "tool_name": "workflow.open",
            "arguments": {
                "entity_id": "missing-entity",
                "operation_id": "op_11111111111111111111",
                "expected_version": 0,
            },
        },
    ]

    features = encoder.encode_decision(observation, candidates)

    assert features.actions[:, 1].tolist() == [1.0, 0.0]
    assert torch.equal(
        encoder.encode_action(candidates[0]),
        encoder.encode_action(candidates[1]),
    )


def test_policy_input_action_features_consume_public_schema_content() -> None:
    task = generate_tasks(Split.TEST, 1, base_seed=1510)[0]
    environment = TransactionalWorkflowEnv(task)
    observation = environment.observe()
    candidates = environment.candidate_actions()
    public_mask = ActionMask(task.tool_schemas).mask(observation, candidates)
    policy_input = PolicyInput.from_decision(
        observation,
        candidates,
        public_mask,
        task.tool_schemas,
    )
    encoder = FeatureEncoder(state_dim=64, action_dim=128)
    original = encoder.encode_policy_input(policy_input)
    candidate = policy_input.candidates[0]

    def changed_action(**schema_updates: object) -> torch.Tensor:
        changed_schema = candidate.tool_schema.model_copy(update=schema_updates)
        changed_candidate = candidate.model_copy(update={"tool_schema": changed_schema})
        changed_input = policy_input.model_copy(
            update={
                "candidates": (changed_candidate, *policy_input.candidates[1:]),
            }
        )
        return encoder.encode_policy_input(changed_input).actions[0]

    assert original.policy_input_sha256 == policy_input.sha256()
    assert not torch.equal(
        original.actions[0],
        changed_action(description="semantically distinct cancellation operation"),
    )
    assert original.actions[0, 2] != changed_action(
        mutating=not candidate.tool_schema.mutating
    )[2]
    assert original.actions[0, 3] != changed_action(
        idempotent=not candidate.tool_schema.idempotent
    )[3]
    assert original.actions[0, 6] != changed_action(
        required_completed_operations=("op_aaaaaaaaaaaaaaaaaaaa",)
    )[6]


def test_encode_policy_input_uses_public_mask_with_explicit_ablation_override() -> None:
    task = generate_tasks(Split.TEST, 1, base_seed=1511)[0]
    environment = TransactionalWorkflowEnv(task)
    observation = environment.observe()
    candidates = environment.candidate_actions()
    public_mask = ActionMask(task.tool_schemas).mask(observation, candidates)
    policy_input = PolicyInput.from_decision(
        observation,
        candidates,
        public_mask,
        task.tool_schemas,
    )
    encoder = FeatureEncoder(32, 32)

    masked = encoder.encode_policy_input(policy_input)
    unmasked = encoder.encode_policy_input(
        policy_input,
        selection_mask=[True] * len(candidates),
    )

    assert masked.mask.tolist() == public_mask
    assert unmasked.mask.all()
    assert torch.equal(masked.state, unmasked.state)
    assert torch.equal(masked.actions, unmasked.actions)


def test_state_features_use_goal_semantics_but_redact_case_identifiers() -> None:
    encoder = FeatureEncoder(state_dim=128, action_dim=32)

    def observation(entity_id: str, intent: str) -> dict[str, object]:
        return {
            "task_id": f"case-{entity_id}",
            "step_index": 0,
            "max_steps": 10,
            "remaining_steps": 10,
            "user_goal": f"{intent} for {entity_id}",
            "available_entities": [entity_id],
            "visible_state": {
                "entity_id": entity_id,
                "version": 0,
                "completed_nodes": [],
            },
        }

    first = encoder.encode_state(observation("invoice-0001", "approve invoice"))
    renamed = encoder.encode_state(observation("invoice-9876", "approve invoice"))
    different_goal = encoder.encode_state(observation("invoice-0001", "cancel invoice"))

    assert torch.equal(first, renamed)
    assert not torch.equal(first, different_goal)


def test_opaque_id_renaming_preserves_match_statistics_without_ordinal_signal() -> None:
    encoder = FeatureEncoder(state_dim=64, action_dim=64)
    first_observation = {
        "step_index": 2,
        "max_steps": 10,
        "remaining_steps": 8,
        "user_goal": "complete the workflow",
        "message": "unfinished prerequisite op_aaaaaaaaaaaaaaaaaaaa",
        "visible_state": {
            "version": 1,
            "completed_nodes": ["op_aaaaaaaaaaaaaaaaaaaa"],
            "status": "pending",
        },
    }
    renamed_observation = {
        **first_observation,
        "message": "unfinished prerequisite op_11111111111111111111",
        "visible_state": {
            **first_observation["visible_state"],
            "completed_nodes": ["op_11111111111111111111"],
        },
    }
    first_action = {
        "tool_name": "workflow.reserve_resource",
        "arguments": {
            "entity_id": "entity-1",
            "operation_id": "op_aaaaaaaaaaaaaaaaaaaa",
            "expected_version": 1,
        },
    }
    renamed_action = {
        **first_action,
        "arguments": {
            **first_action["arguments"],
            "operation_id": "op_11111111111111111111",
        },
    }
    first_state = encoder.encode_state(first_observation)
    renamed_state = encoder.encode_state(renamed_observation)
    first_action_features = encoder.encode_action(first_action)
    renamed_action_features = encoder.encode_action(renamed_action)
    operation_start = encoder.state_dim - encoder.operation_buckets

    assert torch.equal(
        first_state[:operation_start],
        renamed_state[:operation_start],
    )
    assert torch.equal(
        first_action_features[:operation_start],
        renamed_action_features[:operation_start],
    )
    assert (
        torch.dot(first_state[operation_start:], first_action_features[operation_start:]).item()
        == torch.dot(
            renamed_state[operation_start:], renamed_action_features[operation_start:]
        ).item()
    )
    assert torch.linalg.vector_norm(first_state[operation_start:]).item() == 1.0
    non_replay = {
        **first_action,
        "arguments": {
            **first_action["arguments"],
            "operation_id": "op_bbbbbbbbbbbbbbbbbbbb",
        },
    }
    non_replay_features = encoder.encode_action(non_replay)
    assert torch.dot(
        first_state[operation_start:], first_action_features[operation_start:]
    ) > torch.dot(first_state[operation_start:], non_replay_features[operation_start:])
    ordinal_a = {
        **first_action,
        "arguments": {**first_action["arguments"], "operation_id": "step-01"},
    }
    ordinal_b = {
        **first_action,
        "arguments": {**first_action["arguments"], "operation_id": "step-99"},
    }
    assert torch.equal(
        encoder.encode_action(ordinal_a),
        encoder.encode_action(ordinal_b),
    )
    assert first_action_features[1].item() == 0.0
