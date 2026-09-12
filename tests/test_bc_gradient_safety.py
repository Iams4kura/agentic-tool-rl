from __future__ import annotations

import copy

import pytest
import torch

from agentic_tool_rl.algorithms import (
    BCConfig,
    BehaviorCloningTrainer,
    ExpertBatch,
    PolicyBatch,
)
from agentic_tool_rl.models import ActorCritic


@pytest.mark.parametrize("expert", [False, True], ids=["single-label", "multi-positive"])
@pytest.mark.parametrize("bad_gradient", [float("nan"), float("inf"), 1e30])
def test_bc_rejects_nonfinite_gradient_norm_before_optimizer_step(
    expert: bool, bad_gradient: float
) -> None:
    torch.manual_seed(73)
    model = ActorCritic(state_dim=3, action_dim=2, hidden_dim=12)
    states = torch.randn(4, 3)
    features = torch.randn(4, 3, 2)
    mask = torch.ones(4, 3, dtype=torch.bool)
    batch: PolicyBatch | ExpertBatch
    if expert:
        positives = torch.tensor([[True, True, False]]).expand(4, -1)
        batch = ExpertBatch(states, features, mask, positives)
    else:
        batch = PolicyBatch(states, features, mask, torch.zeros(4, dtype=torch.long))
    trainer = BehaviorCloningTrainer(model, BCConfig(epochs=1, batch_size=4))
    # Populate Adam moments first: a failed later update must preserve them too.
    trainer.update(batch)
    parameters = [parameter.detach().clone() for parameter in model.parameters()]
    optimizer_before = copy.deepcopy(trainer.optimizer.state_dict())
    handle = next(model.parameters()).register_hook(
        lambda gradient: torch.full_like(gradient, bad_gradient)
    )
    try:
        with pytest.raises(FloatingPointError, match="BC gradient norm is non-finite"):
            trainer.update(batch)
    finally:
        handle.remove()

    for expected, actual in zip(parameters, model.parameters(), strict=True):
        assert torch.equal(expected, actual)
    optimizer_after = trainer.optimizer.state_dict()
    assert optimizer_before["param_groups"] == optimizer_after["param_groups"]
    for key, state in optimizer_before["state"].items():
        for name, expected in state.items():
            actual = optimizer_after["state"][key][name]
            if isinstance(expected, torch.Tensor):
                assert torch.equal(expected, actual)
            else:
                assert expected == actual
    # The next call clears rejected gradients and remains usable.
    assert trainer.update(batch).updates == 1
    assert all(bool(torch.isfinite(parameter).all()) for parameter in model.parameters())
