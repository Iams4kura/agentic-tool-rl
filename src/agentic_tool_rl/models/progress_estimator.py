"""Supervised success-probability estimator used for potential shaping."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ProgressMetrics:
    loss: float
    brier: float
    ece: float
    auroc: float


def _binary_auroc(probabilities: Tensor, labels: Tensor) -> float:
    """Compute tie-aware binary AUROC without an optional sklearn dependency."""

    probabilities = probabilities.detach().flatten().to(dtype=torch.float64, device="cpu")
    labels = labels.detach().flatten().to(dtype=torch.int64, device="cpu")
    positives = int((labels == 1).sum())
    negatives = int((labels == 0).sum())
    if positives == 0 or negatives == 0:
        return float("nan")

    order = torch.argsort(probabilities, stable=True)
    sorted_probabilities = probabilities[order]
    ranks = torch.empty_like(probabilities)
    start = 0
    while start < len(sorted_probabilities):
        end = start + 1
        while (
            end < len(sorted_probabilities)
            and sorted_probabilities[end] == sorted_probabilities[start]
        ):
            end += 1
        average_rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = average_rank
        start = end
    positive_rank_sum = ranks[labels == 1].sum().item()
    return float(
        (positive_rank_sum - positives * (positives + 1) / 2.0)
        / (positives * negatives)
    )


def compute_progress_metrics(
    probabilities: Tensor,
    labels: Tensor,
    *,
    num_bins: int = 10,
) -> ProgressMetrics:
    """Compute BCE, Brier score, ECE and AUROC from probabilities."""

    if num_bins <= 0:
        raise ValueError("num_bins must be positive")
    probabilities = probabilities.detach().flatten().to(dtype=torch.float64, device="cpu")
    labels = labels.detach().flatten().to(dtype=torch.float64, device="cpu")
    if probabilities.numel() == 0 or probabilities.shape != labels.shape:
        raise ValueError("probabilities and labels must be non-empty vectors of equal size")
    if bool(((probabilities < 0) | (probabilities > 1)).any()):
        raise ValueError("probabilities must lie in [0, 1]")
    if bool(((labels != 0) & (labels != 1)).any()):
        raise ValueError("labels must be binary")

    eps = torch.finfo(probabilities.dtype).eps
    loss = F.binary_cross_entropy(probabilities.clamp(eps, 1 - eps), labels).item()
    brier = torch.mean((probabilities - labels) ** 2).item()
    ece = 0.0
    boundaries = torch.linspace(0.0, 1.0, num_bins + 1, dtype=torch.float64)
    for index in range(num_bins):
        lower, upper = boundaries[index], boundaries[index + 1]
        in_bin = (probabilities >= lower) & (
            probabilities <= upper if index == num_bins - 1 else probabilities < upper
        )
        count = int(in_bin.sum())
        if count:
            confidence = probabilities[in_bin].mean()
            accuracy = labels[in_bin].mean()
            ece += count / probabilities.numel() * abs(confidence.item() - accuracy.item())
    return ProgressMetrics(
        loss=float(loss),
        brier=float(brier),
        ece=float(ece),
        auroc=_binary_auroc(probabilities, labels),
    )


class ProgressEstimator(nn.Module):
    """Predict ``P(task succeeds | visible state, goal)`` from public features."""

    def __init__(self, input_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        if min(input_dim, hidden_dim) <= 0:
            raise ValueError("input_dim and hidden_dim must be positive")
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: Tensor) -> Tensor:
        if features.ndim != 2 or features.shape[-1] != self.input_dim:
            raise ValueError(f"features must have shape [B, {self.input_dim}]")
        return cast(Tensor, self.network(features)).squeeze(-1)

    def probabilities(self, features: Tensor) -> Tensor:
        return torch.sigmoid(self(features))

    @property
    def frozen(self) -> bool:
        return all(not parameter.requires_grad for parameter in self.parameters())

    def freeze(self) -> ProgressEstimator:
        self.requires_grad_(False)
        self.eval()
        return self

    def unfreeze(self) -> ProgressEstimator:
        self.requires_grad_(True)
        self.train()
        return self

    def fit(
        self,
        features: Tensor,
        labels: Tensor,
        *,
        epochs: int = 100,
        learning_rate: float = 3e-3,
        batch_size: int = 64,
        seed: int = 0,
        freeze_after: bool = True,
    ) -> ProgressMetrics:
        """Perform genuine mini-batch supervised updates and return calibration metrics."""

        if epochs <= 0 or learning_rate <= 0 or batch_size <= 0:
            raise ValueError("epochs, learning_rate and batch_size must be positive")
        if features.ndim != 2 or features.shape[-1] != self.input_dim:
            raise ValueError(f"features must have shape [N, {self.input_dim}]")
        labels = labels.flatten().to(device=features.device, dtype=features.dtype)
        if labels.shape[0] != features.shape[0] or features.shape[0] == 0:
            raise ValueError("features and labels must be non-empty and aligned")
        if bool(((labels != 0) & (labels != 1)).any()):
            raise ValueError("labels must be binary")

        self.unfreeze()
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        for _ in range(epochs):
            permutation = torch.randperm(features.shape[0], generator=generator)
            for start in range(0, features.shape[0], batch_size):
                indices = permutation[start : start + batch_size].to(features.device)
                logits = self(features[indices])
                loss = F.binary_cross_entropy_with_logits(logits, labels[indices])
                optimizer.zero_grad(set_to_none=True)
                loss.backward()  # type: ignore[no-untyped-call]
                optimizer.step()

        self.eval()
        with torch.no_grad():
            metrics = compute_progress_metrics(self.probabilities(features), labels)
        if freeze_after:
            self.freeze()
        return metrics
