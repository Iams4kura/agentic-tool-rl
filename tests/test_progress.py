from __future__ import annotations

import math

import torch

from agentic_tool_rl.models import (
    ProgressEstimator,
    compute_progress_metrics,
)


def test_progress_metrics_are_calculated_from_predictions() -> None:
    probabilities = torch.tensor([0.05, 0.2, 0.8, 0.95])
    labels = torch.tensor([0.0, 0.0, 1.0, 1.0])
    metrics = compute_progress_metrics(probabilities, labels, num_bins=5)
    assert math.isclose(metrics.brier, 0.02125, rel_tol=1e-5)
    assert metrics.ece < 0.2
    assert metrics.auroc == 1.0
    assert metrics.loss > 0


def test_progress_estimator_supervised_training_and_freeze() -> None:
    torch.manual_seed(23)
    negative = torch.randn(64, 3) * 0.25 - 1.0
    positive = torch.randn(64, 3) * 0.25 + 1.0
    features = torch.cat((negative, positive))
    labels = torch.cat((torch.zeros(64), torch.ones(64)))
    model = ProgressEstimator(input_dim=3, hidden_dim=16)
    with torch.no_grad():
        before = compute_progress_metrics(model.probabilities(features), labels)
    parameters_before = [parameter.detach().clone() for parameter in model.parameters()]
    metrics = model.fit(
        features,
        labels,
        epochs=30,
        learning_rate=5e-3,
        batch_size=32,
        seed=29,
        freeze_after=True,
    )
    assert metrics.brier < before.brier
    assert metrics.brier < 0.02
    assert metrics.auroc > 0.99
    assert model.frozen
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )
    assert any(
        not torch.equal(old, new.detach())
        for old, new in zip(parameters_before, model.parameters(), strict=True)
    )


def test_progress_auroc_reports_undefined_single_class() -> None:
    metrics = compute_progress_metrics(torch.tensor([0.1, 0.3]), torch.zeros(2))
    assert math.isnan(metrics.auroc)
