from __future__ import annotations

import torch
import torch.nn.functional as F

from agentic_tool_rl.algorithms import (
    BCConfig,
    BehaviorCloningTrainer,
    ExpertBatch,
    PolicyBatch,
    multi_positive_bc_loss,
)
from agentic_tool_rl.models import ActorCritic


def _expert_batch(*, multiple_positives: bool) -> ExpertBatch:
    generator = torch.Generator().manual_seed(20260810)
    states = torch.randn(8, 3, generator=generator)
    action_features = torch.randn(8, 4, 2, generator=generator)
    candidate_mask = torch.ones(8, 4, dtype=torch.bool)
    candidate_mask[::2, 3] = False
    positive_mask = torch.zeros_like(candidate_mask)
    positive_mask[:, 0] = True
    if multiple_positives:
        positive_mask[:, 1] = True
    return ExpertBatch(
        states=states,
        action_features=action_features,
        candidate_mask=candidate_mask,
        positive_mask=positive_mask,
    )


def test_single_positive_loss_and_update_reduce_to_v13_single_label_bc() -> None:
    expert = _expert_batch(multiple_positives=False)
    actions = expert.positive_mask.to(torch.long).argmax(dim=1)

    logits = torch.tensor(
        [[2.0, -1.0, 0.5, -torch.inf], [-4.0, 3.0, 1.0, 0.0]],
        dtype=torch.float64,
    )
    positives = torch.tensor(
        [[True, False, False, False], [False, True, False, False]]
    )
    expected_loss = F.cross_entropy(logits, positives.to(torch.long).argmax(dim=1))
    assert torch.allclose(multi_positive_bc_loss(logits, positives), expected_loss)

    policy = PolicyBatch(
        states=expert.states,
        action_features=expert.action_features,
        action_masks=expert.candidate_mask,
        actions=actions,
    )
    torch.manual_seed(9)
    single_model = ActorCritic(state_dim=3, action_dim=2, hidden_dim=12)
    expert_model = ActorCritic(state_dim=3, action_dim=2, hidden_dim=12)
    expert_model.load_state_dict(single_model.state_dict())
    config = BCConfig(learning_rate=1e-3, epochs=2, batch_size=4, seed=17)

    single_metrics = BehaviorCloningTrainer(single_model, config).update(policy)
    expert_metrics = BehaviorCloningTrainer(expert_model, config).update(expert)

    assert single_metrics.updates == expert_metrics.updates
    assert single_metrics.loss == expert_metrics.loss
    assert single_metrics.accuracy == expert_metrics.accuracy
    assert single_metrics.entropy == expert_metrics.entropy
    assert single_metrics.grad_norm == expert_metrics.grad_norm
    for single_parameter, expert_parameter in zip(
        single_model.parameters(), expert_model.parameters(), strict=True
    ):
        assert torch.equal(single_parameter, expert_parameter)


def test_multiple_positives_are_optimized_as_one_set_without_mutual_penalty() -> None:
    logits = torch.tensor([[4.0, -3.0, 1.0]], dtype=torch.float64, requires_grad=True)
    first_only = torch.tensor([[True, False, False]])
    two_positives = torch.tensor([[True, True, False]])
    all_positives = torch.tensor([[True, True, True]])

    single_loss = multi_positive_bc_loss(logits, first_only)
    expanded_loss = multi_positive_bc_loss(logits, two_positives)
    all_positive_loss = multi_positive_bc_loss(logits, all_positives)

    assert expanded_loss < single_loss
    assert torch.allclose(all_positive_loss, torch.zeros_like(all_positive_loss), atol=1e-12)
    all_positive_loss.backward()
    assert logits.grad is not None
    assert torch.allclose(logits.grad, torch.zeros_like(logits.grad), atol=1e-12)


def test_set_loss_is_invariant_to_joint_candidate_position_swaps() -> None:
    logits = torch.tensor(
        [[2.5, -0.5, 1.25, -3.0], [-1.0, 4.0, 0.25, 2.0]],
        dtype=torch.float64,
    )
    positives = torch.tensor(
        [[True, False, True, False], [False, True, False, True]]
    )
    permutation = torch.tensor([3, 1, 0, 2])

    original = multi_positive_bc_loss(logits, positives)
    permuted = multi_positive_bc_loss(
        logits[:, permutation], positives[:, permutation]
    )

    assert torch.allclose(original, permuted, rtol=0.0, atol=1e-15)


def test_set_loss_moves_monotonically_with_positive_and_negative_logits() -> None:
    logits = torch.tensor([[0.2, -0.4, 1.1]], dtype=torch.float64)
    positives = torch.tensor([[True, True, False]])
    baseline = multi_positive_bc_loss(logits, positives)

    raised_positive = logits.clone()
    raised_positive[0, 1] += 0.75
    raised_negative = logits.clone()
    raised_negative[0, 2] += 0.75

    assert multi_positive_bc_loss(raised_positive, positives) < baseline
    assert multi_positive_bc_loss(raised_negative, positives) > baseline


def test_expert_metrics_record_candidate_set_quality_without_changing_legacy_fields() -> None:
    expert = _expert_batch(multiple_positives=True)
    config = BCConfig(learning_rate=1e-3, epochs=1, batch_size=4, seed=23)
    expert_metrics = BehaviorCloningTrainer(
        ActorCritic(state_dim=3, action_dim=2, hidden_dim=12), config
    ).update(expert)

    assert expert_metrics.set_nll is not None and expert_metrics.set_nll >= 0.0
    assert expert_metrics.positive_probability_mass is not None
    assert 0.0 <= expert_metrics.positive_probability_mass <= 1.0
    assert (
        expert_metrics.top1_in_positive_set_accuracy
        == expert_metrics.accuracy
    )
    assert expert_metrics.mean_positive_count == 2.0

    legacy = PolicyBatch(
        states=expert.states,
        action_features=expert.action_features,
        action_masks=expert.candidate_mask,
        actions=expert.positive_mask.to(torch.long).argmax(dim=1),
    )
    legacy_metrics = BehaviorCloningTrainer(
        ActorCritic(state_dim=3, action_dim=2, hidden_dim=12), config
    ).update(legacy)
    assert legacy_metrics.set_nll is None
    assert legacy_metrics.positive_probability_mass is None
    assert legacy_metrics.top1_in_positive_set_accuracy is None
    assert legacy_metrics.mean_positive_count is None


def test_expert_batch_rejects_a_row_without_positive_candidate() -> None:
    batch = _expert_batch(multiple_positives=True)
    positive_mask = batch.positive_mask.clone()
    positive_mask[3] = False
    invalid = ExpertBatch(
        states=batch.states,
        action_features=batch.action_features,
        candidate_mask=batch.candidate_mask,
        positive_mask=positive_mask,
    )

    try:
        invalid.validate()
    except ValueError as error:
        assert "at least one positive" in str(error)
    else:
        raise AssertionError("ExpertBatch accepted a row without a positive candidate")


def test_expert_batch_rejects_positive_outside_candidate_mask() -> None:
    batch = _expert_batch(multiple_positives=False)
    positive_mask = batch.positive_mask.clone()
    positive_mask[0, 3] = True
    invalid = ExpertBatch(
        states=batch.states,
        action_features=batch.action_features,
        candidate_mask=batch.candidate_mask,
        positive_mask=positive_mask,
    )

    try:
        invalid.validate()
    except ValueError as error:
        assert "subset" in str(error)
    else:
        raise AssertionError("ExpertBatch accepted a masked-out positive candidate")
