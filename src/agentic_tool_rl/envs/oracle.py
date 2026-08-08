"""DAG validation and real environment replay for benchmark solvability."""

from __future__ import annotations

from dataclasses import dataclass

from agentic_tool_rl.contracts import StepOutcome, WorkflowTask
from agentic_tool_rl.envs.workflow import TransactionalWorkflowEnv


@dataclass(frozen=True, slots=True)
class OracleReport:
    case_id: str
    solvable: bool
    plan_index: int
    executed_steps: int
    reason: str = ""


def validate_workflow_dag(task: WorkflowTask) -> None:
    node_ids = {node.node_id for node in task.workflow_nodes}
    if len(node_ids) != len(task.workflow_nodes):
        raise ValueError(f"duplicate workflow node in {task.case_id}")
    tool_names = {schema.name for schema in task.tool_schemas}
    for node in task.workflow_nodes:
        unknown = set(node.required_predecessors) - node_ids
        if unknown:
            raise ValueError(f"{node.node_id} references unknown predecessors: {sorted(unknown)}")
        if node.node_id in node.required_predecessors:
            raise ValueError(f"{node.node_id} depends on itself")
        if node.tool_name not in tool_names:
            raise ValueError(f"{node.node_id} references unknown tool {node.tool_name}")

    completed: set[str] = set()
    remaining = set(node_ids)
    while remaining:
        ready = {
            node.node_id
            for node in task.workflow_nodes
            if node.node_id in remaining
            and set(node.required_predecessors).issubset(completed)
        }
        if not ready:
            raise ValueError(f"workflow DAG contains a cycle in {task.case_id}")
        completed.update(ready)
        remaining -= ready


def replay_oracle(task: WorkflowTask, plan_index: int = 0) -> list[StepOutcome]:
    validate_workflow_dag(task)
    try:
        plan = task.oracle_plans[plan_index]
    except IndexError as error:
        raise ValueError(f"oracle plan {plan_index} does not exist for {task.case_id}") from error
    environment = TransactionalWorkflowEnv(task)
    outcomes: list[StepOutcome] = []
    for call in plan:
        outcome = environment.step(call)
        outcomes.append(outcome)
        if not outcome.accepted:
            raise ValueError(
                f"oracle plan {plan_index} rejected at step {len(outcomes) - 1}: {outcome.reason}"
            )
        if outcome.done:
            break
    if not environment.evaluate().success:
        raise ValueError(f"oracle plan {plan_index} does not satisfy {task.case_id}")
    return outcomes


def verify_task_solvable(task: WorkflowTask, plan_index: int = 0) -> OracleReport:
    try:
        outcomes = replay_oracle(task, plan_index)
    except (RuntimeError, ValueError) as error:
        return OracleReport(
            case_id=task.case_id,
            solvable=False,
            plan_index=plan_index,
            executed_steps=0,
            reason=str(error),
        )
    return OracleReport(
        case_id=task.case_id,
        solvable=True,
        plan_index=plan_index,
        executed_steps=len(outcomes),
    )


__all__ = ["OracleReport", "replay_oracle", "validate_workflow_dag", "verify_task_solvable"]
