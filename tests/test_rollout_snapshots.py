from __future__ import annotations

import pytest
import torch

from agentic_tool_rl.contracts import StepRecord as ContractRecord
from agentic_tool_rl.contracts import ToolCall
from agentic_tool_rl.rollout import RolloutBuffer, StepRecord


@pytest.mark.parametrize("portable", [False, True])
def test_rollout_evidence_isolated_from_reused_inputs(portable: bool) -> None:
    observation = {"state": {"ids": ["before"]}}
    next_observation = {"state": {"ids": ["after"]}}
    candidate = ToolCall(tool_name="lookup", arguments={"ids": ["original"]})
    info = {"result": {"accepted": [True]}}
    state_features = torch.tensor([1.0, 2.0])
    action_features = torch.tensor([[3.0, 4.0]])
    buffer = RolloutBuffer()
    if portable:
        contract = ContractRecord(
            trajectory_id="snapshot", step_index=0, observation=observation,
            next_observation=next_observation, candidates=[candidate],
            action_index=0, action_mask=[True], log_prob=-0.5, value=0.1,
            reward=1.0, done=True, info=info,
        )
        record = buffer.add_contract(
            contract, state_features=state_features, action_features=action_features,
        )
        # Mutate exactly the objects retained by the portable contract, even if
        # Pydantic copied some caller containers during validation.
        observation = contract.observation
        next_observation = contract.next_observation
        candidate = contract.candidates[0]
        info = contract.info
    else:
        record = StepRecord(
            trajectory_id="snapshot", step_index=0, observation=observation,
            next_observation=next_observation, candidates=(candidate,),
            action_index=0, action_mask=(True,), log_prob=-0.5, value=0.1,
            reward=1.0, done=True, info=info,
            state_features=state_features, action_features=action_features,
        )
        buffer.add(record)
    expected = record.to_trace_dict()
    observation["state"]["ids"].append("changed")
    next_observation["state"]["ids"].clear()
    candidate.arguments["ids"].append("changed")
    info["result"]["accepted"][0] = False
    state_features.fill_(9)
    action_features.fill_(9)

    assert buffer.records[0].to_trace_dict() == expected
    assert isinstance(record.action, ToolCall)
    batch = buffer.as_ppo_batch()
    assert batch.states.tolist() == [[1.0, 2.0]]
    assert batch.action_features.tolist() == [[[3.0, 4.0]]]

    exported = record.to_trace_dict()
    exported["observation"]["state"]["ids"].clear()
    exported["next_observation"]["state"]["ids"].append("changed")
    exported["candidates"][0]["arguments"]["ids"].clear()
    exported["info"]["result"]["accepted"].clear()
    assert record.to_trace_dict() == expected
