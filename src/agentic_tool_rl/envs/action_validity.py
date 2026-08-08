"""Frozen action-validity data labelled by the independent environment oracle."""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from collections.abc import Sequence

from agentic_tool_rl.contracts import (
    ActionValidityChallenge,
    ActionValidityExample,
    InvalidActionKind,
    ToolCall,
    ValidationResult,
    WorkflowTask,
)
from agentic_tool_rl.envs.workflow import TransactionalWorkflowEnv

_INVALID_KINDS = tuple(InvalidActionKind)


def _state_positions(optimal_steps: int, count: int) -> list[int]:
    if count == 1:
        return [0]
    return [round(index * (optimal_steps - 1) / (count - 1)) for index in range(count)]


def _state_digest(observation_payload: dict[str, object]) -> str:
    encoded = json.dumps(
        observation_payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _clone_call(call: ToolCall, suffix: str) -> ToolCall:
    return call.model_copy(update={"call_id": f"{call.call_id}-{suffix}"})


def _hidden_ledger_collision(
    task: WorkflowTask,
    environment: TransactionalWorkflowEnv,
    valid_candidates: Sequence[ToolCall],
) -> tuple[ToolCall, ValidationResult]:
    """Build a safety negative whose decisive ledger state is not observable.

    This counterexample is intentionally outside the public candidate builder.
    The environment receives a hidden idempotency-ledger collision while the
    ActionMask sees the unchanged public observation.  It therefore measures a
    real partial-observability boundary instead of only comparing equivalent
    implementations of public rules.
    """

    plan = task.oracle_plans[0]
    final_operation = str(plan[-1].arguments["operation_id"])
    base = next(
        candidate
        for candidate in valid_candidates
        if candidate.arguments.get("operation_id") == final_operation
        and isinstance(candidate.arguments.get("idempotency_key"), str)
    )
    candidate = _clone_call(base, "hidden-ledger-collision")
    idempotency_key = str(candidate.arguments["idempotency_key"])
    hidden_state = environment.state
    ledger = dict(hidden_state.get("idempotency_results", {}))
    collision_operation = str(plan[0].arguments["operation_id"])
    if collision_operation == final_operation and len(plan) > 1:
        collision_operation = str(plan[1].arguments["operation_id"])
    ledger[idempotency_key] = collision_operation
    hidden_state["idempotency_results"] = ledger
    hidden_task = task.model_copy(update={"initial_state": hidden_state})
    hidden_environment = TransactionalWorkflowEnv(hidden_task)
    label = hidden_environment.dry_run(candidate)
    if label.valid or label.invalid_kind != InvalidActionKind.SAFETY:
        raise RuntimeError("hidden ledger challenge was not labelled as a safety failure")
    return candidate, label


def generate_action_validity_dataset(
    tasks: Sequence[WorkflowTask],
    *,
    states_per_task: int = 5,
    candidates_per_state: int = 4,
    seed: int = 20_260_808,
) -> list[ActionValidityExample]:
    """Create a deterministic 50/50 candidate dataset from real dry-run labels.

    The canonical 1000-task configuration yields exactly ``1000 * 5 * 4 =
    20_000`` examples.  Candidate selection uses the validator only *after*
    candidate construction; the policy-side ActionMask is never imported.
    """

    if not tasks:
        raise ValueError("tasks must not be empty")
    if states_per_task < 1:
        raise ValueError("states_per_task must be positive")
    if candidates_per_state < 2 or candidates_per_state % 2:
        raise ValueError("candidates_per_state must be an even integer of at least 2")

    examples: list[ActionValidityExample] = []
    valid_slots = candidates_per_state // 2
    invalid_slots = candidates_per_state // 2
    invalid_cursor = 0
    sorted_tasks = sorted(tasks, key=lambda task: task.case_id)
    for task_index, task in enumerate(sorted_tasks):
        environment = TransactionalWorkflowEnv(task)
        plan = task.oracle_plans[0]
        executed = 0
        for state_index, target_step in enumerate(
            _state_positions(task.optimal_steps, states_per_task)
        ):
            while executed < target_step:
                outcome = environment.step(plan[executed])
                if not outcome.accepted:
                    raise RuntimeError(
                        f"oracle rejected while sampling {task.case_id}: {outcome.reason}"
                    )
                executed += 1

            observation = environment.observe()
            candidates = environment.candidate_actions(include_invalid=True)
            labelled = [(candidate, environment.dry_run(candidate)) for candidate in candidates]
            valid = [candidate for candidate, label in labelled if label.valid]
            invalid_by_kind = {
                kind: [
                    candidate
                    for candidate, label in labelled
                    if not label.valid and label.invalid_kind == kind
                ]
                for kind in _INVALID_KINDS
            }
            if not valid or any(not values for values in invalid_by_kind.values()):
                raise RuntimeError(
                    f"candidate builder lacks a validation stratum for {task.case_id}"
                )

            selected: list[
                tuple[ToolCall, ValidationResult, ActionValidityChallenge]
            ] = []
            for index in range(valid_slots):
                candidate = _clone_call(valid[index % len(valid)], f"valid-{index}")
                selected.append(
                    (candidate, environment.dry_run(candidate), "standard_candidate")
                )
            for index in range(invalid_slots):
                kind = _INVALID_KINDS[invalid_cursor % len(_INVALID_KINDS)]
                if (
                    kind == InvalidActionKind.SAFETY
                    and target_step == task.optimal_steps - 1
                ):
                    candidate, label = _hidden_ledger_collision(task, environment, valid)
                    selected.append((candidate, label, "hidden_ledger_collision"))
                else:
                    choices = invalid_by_kind[kind]
                    candidate = _clone_call(
                        choices[index % len(choices)], f"invalid-{invalid_cursor}"
                    )
                    selected.append(
                        (candidate, environment.dry_run(candidate), "standard_candidate")
                    )
                invalid_cursor += 1
            random.Random(seed + task_index * 10_000 + state_index).shuffle(selected)

            observation_payload = observation.model_dump(mode="json")
            digest = _state_digest(observation_payload)
            for candidate_index, (candidate, label, challenge_source) in enumerate(
                selected
            ):
                examples.append(
                    ActionValidityExample(
                        sample_id=(
                            f"{task.case_id}-state-{state_index:02d}-candidate-{candidate_index:02d}"
                        ),
                        case_id=task.case_id,
                        split=task.split,
                        family=task.family,
                        state_index=state_index,
                        workflow_step_index=target_step,
                        observation=observation,
                        tool_call=candidate,
                        valid_label=label.valid,
                        invalid_kind=label.invalid_kind,
                        challenge_source=challenge_source,
                        state_sha256=digest,
                    )
                )

    expected = len(tasks) * states_per_task * candidates_per_state
    if len(examples) != expected:
        raise RuntimeError(f"expected {expected} examples, generated {len(examples)}")
    labels = Counter(example.valid_label for example in examples)
    if labels[True] != labels[False]:
        raise RuntimeError("action-validity dataset is not class balanced")
    return examples


__all__ = ["generate_action_validity_dataset"]
