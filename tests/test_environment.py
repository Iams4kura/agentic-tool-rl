from __future__ import annotations

from collections import Counter
from copy import deepcopy

import pytest

from agentic_tool_rl.contracts import InvalidActionKind, Split, ToolCall
from agentic_tool_rl.envs.benchmark import WORKFLOW_TOPOLOGIES, generate_tasks
from agentic_tool_rl.envs.workflow import TransactionalWorkflowEnv
from agentic_tool_rl.grounding.action_mask import ActionMask
from agentic_tool_rl.grounding.candidates import build_candidate_actions
from agentic_tool_rl.latency_profile import READ_ONLY_QUERY_S


@pytest.fixture
def task():  # type: ignore[no-untyped-def]
    return generate_tasks(Split.TEST, 1, base_seed=404)[0]


def _candidate_by_kind(environment: TransactionalWorkflowEnv) -> dict[InvalidActionKind, ToolCall]:
    result: dict[InvalidActionKind, ToolCall] = {}
    for candidate in environment.candidate_actions():
        label = environment.dry_run(candidate)
        if not label.valid and label.invalid_kind is not None:
            result[label.invalid_kind] = candidate
    return result


def test_real_environment_replays_oracle_to_final_state(task) -> None:  # type: ignore[no-untyped-def]
    environment = TransactionalWorkflowEnv(task)

    outcomes = [environment.step(call) for call in task.oracle_plans[0]]

    assert all(outcome.accepted for outcome in outcomes)
    assert outcomes[-1].done and outcomes[-1].success
    assert environment.evaluate().success
    assert environment.state["status"] == "completed"
    assert environment.state["version"] == task.optimal_steps
    assert len(environment.state["audit_log"]) == task.optimal_steps
    assert environment.simulated_latency_s == pytest.approx(sum(task.latency_trace.values()))


def test_action_mask_and_independent_dry_run_cover_all_four_error_classes(task) -> None:  # type: ignore[no-untyped-def]
    environment = TransactionalWorkflowEnv(task)
    observation = environment.observe()
    candidates = environment.candidate_actions()
    action_mask = ActionMask(task)

    oracle_labels = [environment.dry_run(candidate) for candidate in candidates]
    mask_labels = action_mask.evaluate(observation, candidates)
    invalid_counts = Counter(
        label.invalid_kind for label in oracle_labels if label.invalid_kind is not None
    )

    assert set(invalid_counts) == set(InvalidActionKind)
    assert [label.valid for label in oracle_labels] == [label.valid for label in mask_labels]
    assert [label.invalid_kind for label in oracle_labels] == [
        label.invalid_kind for label in mask_labels
    ]
    assert any(label.valid for label in oracle_labels)


def test_action_mask_is_invariant_to_hidden_task_fields(task) -> None:  # type: ignore[no-untyped-def]
    environment = TransactionalWorkflowEnv(task)
    observation = environment.observe()
    candidates = environment.candidate_actions()
    baseline = ActionMask(task).evaluate(observation, candidates)

    hidden_nodes = [
        node.model_copy(
            update={
                "required_predecessors": ["hidden-predecessor"],
                "safety_token": "hidden-replacement-token",
                "tool_name": f"hidden.{node.node_id}",
            }
        )
        for node in task.workflow_nodes
    ]
    changed_hidden_task = task.model_copy(
        update={
            "entity_id": "hidden-replacement-entity",
            "workflow_nodes": hidden_nodes,
            "forbidden_side_effects": ["hidden.replacement.side_effect"],
            "oracle_plans": [],
            "hidden_goal_predicates": [],
        }
    )

    changed = ActionMask(changed_hidden_task).evaluate(observation, candidates)

    assert baseline == changed
    assert build_candidate_actions(task, observation.visible_state) == build_candidate_actions(
        changed_hidden_task, observation.visible_state
    )
    assert not hasattr(ActionMask(task), "task")
    assert not hasattr(ActionMask(task), "_nodes")


def test_action_mask_enforces_public_schema_prerequisites(task) -> None:  # type: ignore[no-untyped-def]
    environment = TransactionalWorkflowEnv(task)
    future = task.oracle_plans[0][3].model_copy(
        update={
            "arguments": {
                **task.oracle_plans[0][3].arguments,
                "expected_version": 0,
            }
        }
    )

    decision = ActionMask(task).validate(environment.observe(), future)

    assert not decision.valid
    assert decision.invalid_kind == InvalidActionKind.PRECONDITION


def test_precondition_distractor_uses_public_unsatisfied_prerequisites(task) -> None:  # type: ignore[no-untyped-def]
    environment = TransactionalWorkflowEnv(task)
    observation = environment.observe()
    candidate = _candidate_by_kind(environment)[InvalidActionKind.PRECONDITION]
    schema = next(schema for schema in task.tool_schemas if schema.name == candidate.tool_name)

    assert candidate.arguments["expected_version"] == observation.visible_state["version"]
    assert not set(schema.required_completed_operations).issubset(
        observation.visible_state["completed_nodes"]
    )
    assert not environment.dry_run(candidate).valid
    assert not ActionMask(task).validate(observation, candidate).valid


def test_public_precondition_distractor_generalizes_across_topologies() -> None:
    pool = [
        *generate_tasks(Split.TRAIN, 40, base_seed=990),
        *generate_tasks(Split.TEST, 40, base_seed=990),
    ]
    observed_topologies = {task.topology for task in pool}

    for topology in WORKFLOW_TOPOLOGIES:
        selected = next(task for task in pool if task.topology == topology)
        environment = TransactionalWorkflowEnv(selected)
        observation = environment.observe()
        candidate = _candidate_by_kind(environment)[InvalidActionKind.PRECONDITION]
        schema = next(
            schema for schema in selected.tool_schemas if schema.name == candidate.tool_name
        )

        assert candidate.arguments["expected_version"] == observation.visible_state["version"]
        assert not set(schema.required_completed_operations).issubset(
            observation.visible_state["completed_nodes"]
        )
        assert environment.dry_run(candidate).invalid_kind == InvalidActionKind.PRECONDITION
        assert (
            ActionMask(selected).validate(observation, candidate).invalid_kind
            == InvalidActionKind.PRECONDITION
        )

    assert observed_topologies == set(WORKFLOW_TOPOLOGIES)


def test_public_prerequisite_distractors_outnumber_stale_version_fallbacks() -> None:
    blocked_count = 0
    stale_count = 0
    for selected in generate_tasks(Split.TEST, 40, base_seed=991):
        environment = TransactionalWorkflowEnv(selected)
        for oracle_call in selected.oracle_plans[0]:
            observation = environment.observe()
            candidate = _candidate_by_kind(environment)[InvalidActionKind.PRECONDITION]
            if candidate.arguments["expected_version"] == observation.visible_state["version"]:
                schema = next(
                    schema for schema in selected.tool_schemas if schema.name == candidate.tool_name
                )
                assert not set(schema.required_completed_operations).issubset(
                    observation.visible_state["completed_nodes"]
                )
                blocked_count += 1
            else:
                stale_count += 1
            assert environment.step(oracle_call).accepted

    assert blocked_count > stale_count


def test_candidate_order_is_publicly_seeded_deterministic_and_not_fixed() -> None:
    tasks = generate_tasks(Split.TEST, 40, base_seed=606)
    signatures: set[tuple[str, ...]] = set()

    for selected in tasks:
        environment = TransactionalWorkflowEnv(selected)
        first = environment.candidate_actions()
        repeated = environment.candidate_actions()
        assert first == repeated
        signatures.add(
            tuple(call.call_id.removeprefix("candidate-").split("-", 1)[0] for call in first)
        )

    assert len(signatures) > len(WORKFLOW_TOPOLOGIES)
    assert any(signature[0] != "valid" for signature in signatures)


def test_context_matched_candidate_is_legal_but_does_not_advance_goal(task) -> None:  # type: ignore[no-untyped-def]
    environment = TransactionalWorkflowEnv(task)
    context = next(
        candidate
        for candidate in environment.candidate_actions()
        if "candidate-context" in candidate.call_id
    )
    before = environment.state_snapshot()

    assert environment.dry_run(context).valid
    assert ActionMask(task).validate(environment.observe(), context).valid
    outcome = environment.step(context)

    assert outcome.accepted
    assert not outcome.success
    assert environment.state_snapshot() == before


@pytest.mark.parametrize("kind", list(InvalidActionKind))
def test_forced_invalid_calls_are_rejected_without_business_state_mutation(task, kind) -> None:  # type: ignore[no-untyped-def]
    environment = TransactionalWorkflowEnv(task)
    candidate = _candidate_by_kind(environment)[kind]
    before = environment.state_snapshot()

    outcome = environment.step(candidate)

    assert not outcome.accepted
    assert outcome.invalid_kind == kind
    assert environment.state_snapshot() == before
    assert environment.steps_taken == 1


def test_environment_labels_do_not_call_the_policy_mask(task, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    environment = TransactionalWorkflowEnv(task)
    invalid = _candidate_by_kind(environment)[InvalidActionKind.SCHEMA]

    def explode(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("policy mask must not be the environment judge")

    monkeypatch.setattr(ActionMask, "validate", explode)

    assert not environment.dry_run(invalid).valid
    assert not environment.step(invalid).accepted


def test_exact_call_replay_is_idempotent(task) -> None:  # type: ignore[no-untyped-def]
    environment = TransactionalWorkflowEnv(task)
    call = task.oracle_plans[0][0]
    first = environment.step(call)
    after_first = environment.state_snapshot()

    reordered = call.model_copy(
        update={"arguments": dict(reversed(tuple(call.arguments.items())))}
    )
    replay = environment.step(reordered)

    assert first.accepted and replay.accepted
    assert replay.info["idempotent_replay"] is True
    assert environment.state_snapshot() == after_first
    audit_entry = environment.state["audit_log"][0]
    request_sha256 = audit_entry["request_sha256"]
    assert isinstance(request_sha256, str) and len(request_sha256) == 64
    assert set(request_sha256) <= set("0123456789abcdef")
    assert "request" not in audit_entry


@pytest.mark.parametrize(
    "collision_kind",
    ["different-operation", "different-arguments", "read-only-request"],
)
def test_call_id_reuse_requires_the_exact_original_request(task, collision_kind) -> None:  # type: ignore[no-untyped-def]
    environment = TransactionalWorkflowEnv(task)
    original = task.oracle_plans[0][0]
    assert environment.step(original).accepted
    after_original = environment.state_snapshot()

    if collision_kind == "different-operation":
        conflicting = task.oracle_plans[0][1].model_copy(
            update={"call_id": original.call_id}
        )
    elif collision_kind == "different-arguments":
        conflicting = original.model_copy(
            update={
                "arguments": {
                    **original.arguments,
                    "expected_version": int(original.arguments["expected_version"]) + 1,
                }
            }
        )
    else:
        conflicting = next(
            candidate
            for candidate in environment.candidate_actions()
            if candidate.tool_name.endswith(".inspect_status")
        ).model_copy(update={"call_id": original.call_id})

    label = environment.dry_run(conflicting)
    outcome = environment.step(conflicting)

    assert not label.valid
    assert label.invalid_kind == InvalidActionKind.SAFETY
    assert "call_id" in label.reason
    assert not outcome.accepted
    assert outcome.invalid_kind == InvalidActionKind.SAFETY
    assert environment.state_snapshot() == after_original


def test_call_id_request_identity_is_type_sensitive(task) -> None:  # type: ignore[no-untyped-def]
    original = task.oracle_plans[0][0]
    schemas = [
        schema.model_copy(update={"additional_properties": True})
        if schema.name == original.tool_name
        else schema
        for schema in task.tool_schemas
    ]
    environment = TransactionalWorkflowEnv(task.model_copy(update={"tool_schemas": schemas}))
    first = original.model_copy(
        update={"arguments": {**original.arguments, "metadata": {"flag": True}}}
    )
    assert environment.step(first).accepted
    after_first = environment.state_snapshot()
    collision = first.model_copy(
        update={"arguments": {**first.arguments, "metadata": {"flag": 1}}}
    )

    outcome = environment.step(collision)

    assert not outcome.accepted
    assert outcome.invalid_kind == InvalidActionKind.SAFETY
    assert environment.state_snapshot() == after_first


def test_non_json_call_id_request_is_rejected_without_mutation(task) -> None:  # type: ignore[no-untyped-def]
    original = task.oracle_plans[0][0]
    schemas = [
        schema.model_copy(update={"additional_properties": True})
        if schema.name == original.tool_name
        else schema
        for schema in task.tool_schemas
    ]
    task_with_extras = task.model_copy(update={"tool_schemas": schemas})
    cyclic: list[object] = []
    cyclic.append(cyclic)
    deeply_nested: object = "leaf"
    for _ in range(2_000):
        deeply_nested = [deeply_nested]

    for malformed_value in ({1}, cyclic, float("nan"), {1: "bad-key"}, deeply_nested):
        environment = TransactionalWorkflowEnv(task_with_extras)
        malformed = original.model_copy(
            update={
                "arguments": {
                    **original.arguments,
                    "metadata": {"bad": malformed_value},
                }
            }
        )
        before = environment.state_snapshot()

        outcome = environment.step(malformed)

        assert not outcome.accepted
        assert outcome.invalid_kind == InvalidActionKind.SCHEMA
        assert "canonical JSON" in outcome.reason
        assert environment.state_snapshot() == before


def test_mask_allows_legal_but_non_progressing_policy_distractors(task) -> None:  # type: ignore[no-untyped-def]
    environment = TransactionalWorkflowEnv(task)
    observation = environment.observe()
    query = next(
        candidate
        for candidate in environment.candidate_actions()
        if candidate.tool_name.endswith(".inspect_status")
    )
    before = environment.state_snapshot()

    assert environment.dry_run(query).valid
    assert ActionMask(task).validate(observation, query).valid
    outcome = environment.step(query)

    assert outcome.accepted
    assert not outcome.done
    assert outcome.simulated_latency_s == READ_ONLY_QUERY_S
    assert environment.state_snapshot() == before

    advancing = task.oracle_plans[0][0]
    environment = TransactionalWorkflowEnv(task)
    environment.step(advancing)
    replay = next(
        candidate
        for candidate in environment.candidate_actions()
        if "candidate-replay" in candidate.call_id
    )
    after_progress = environment.state_snapshot()

    assert environment.dry_run(replay).valid
    assert ActionMask(task).validate(environment.observe(), replay).valid
    assert environment.step(replay).accepted
    assert environment.state_snapshot() == after_progress


def test_candidates_reveal_approval_only_after_it_is_visible(task) -> None:  # type: ignore[no-untyped-def]
    environment = TransactionalWorkflowEnv(task)
    plan = task.oracle_plans[0]
    for call in plan[: task.optimal_steps - 2]:
        assert environment.step(call).accepted

    assert environment.observe().visible_state["approval_token"] is None
    assert all(
        "approval_token" not in candidate.arguments for candidate in environment.candidate_actions()
    )

    assert environment.step(plan[task.optimal_steps - 2]).accepted
    visible_approval = environment.observe().visible_state["approval_token"]
    final = next(
        candidate
        for candidate in environment.candidate_actions()
        if "approval_token" in candidate.arguments
    )

    assert isinstance(visible_approval, str)
    assert final.arguments["approval_token"] == visible_approval


def test_alternative_legal_order_is_judged_by_final_state(task) -> None:  # type: ignore[no-untyped-def]
    environment = TransactionalWorkflowEnv(task)

    for call in task.oracle_plans[1]:
        outcome = environment.step(call)

    assert outcome.success
    assert environment.evaluate().goal_predicates_satisfied


def test_harmful_shortcut_is_rejected_before_any_side_effect(task) -> None:  # type: ignore[no-untyped-def]
    environment = TransactionalWorkflowEnv(task)
    shortcut = _candidate_by_kind(environment)[InvalidActionKind.SAFETY]
    before = deepcopy(environment.state_snapshot())

    outcome = environment.step(shortcut)

    assert not outcome.accepted
    assert outcome.invalid_kind == InvalidActionKind.SAFETY
    assert environment.state_snapshot() == before
    assert environment.evaluate().forbidden_side_effect_count == 0


def test_precondition_rejects_valid_tool_called_out_of_order(task) -> None:  # type: ignore[no-untyped-def]
    environment = TransactionalWorkflowEnv(task)
    future = task.oracle_plans[0][3].model_copy(
        update={
            "arguments": {
                **task.oracle_plans[0][3].arguments,
                "expected_version": 0,
            }
        }
    )

    label = environment.dry_run(future)

    assert not label.valid
    assert label.invalid_kind == InvalidActionKind.PRECONDITION


def test_observation_hides_internal_ledgers(task) -> None:  # type: ignore[no-untyped-def]
    environment = TransactionalWorkflowEnv(task)
    observation = environment.observe()

    assert observation.task_id == task.case_id
    assert observation.available_entities == [task.entity_id]
    assert "processed_call_ids" not in observation.visible_state
    assert "idempotency_results" not in observation.visible_state
    assert "audit_log" not in observation.visible_state


def test_step_after_terminal_requires_explicit_reset(task) -> None:  # type: ignore[no-untyped-def]
    environment = TransactionalWorkflowEnv(task)
    for call in task.oracle_plans[0]:
        environment.step(call)

    with pytest.raises(RuntimeError, match="terminated"):
        environment.step(task.oracle_plans[0][-1])

    observation = environment.reset()
    assert observation.step_index == 0
    assert not observation.done
    assert environment.state["status"] == "pending"
