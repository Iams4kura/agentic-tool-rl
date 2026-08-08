"""Frozen service-time profile derived from a public production trace.

The source is the Microsoft Azure Functions 2019 trace.  Its daily function
duration rows contain an invocation count and an average execution time in
milliseconds.  We aggregate all fourteen days, weight each row by invocation
count, and freeze empirical quantiles instead of tuning latency against model
results.  Azure documents that these durations exclude cold-start and network
time, so this benchmark describes *simulated service execution time*, not wall
clock latency of an end-user Agent.
"""

from __future__ import annotations

import random

PROFILE_ID = "azure-functions-2019-invocation-weighted-v1"
SOURCE_URL = (
    "https://github.com/Azure/AzurePublicDataset/releases/download/"
    "dataset-functions-2019/"
    "azurefunctions_dataset2019_azurefunctions-dataset2019.tar.xz"
)
SOURCE_ARCHIVE_SHA256 = "aff8b3ca7240a41a109e4ee598e0a96e45fcb92e7b8395ac19cb3748cd260d89"
SOURCE_LICENSE = "CC-BY-4.0"
SOURCE_ROWS = 662_922
SOURCE_INVOCATIONS = 12_481_740_344

# Seconds.  Mutating business operations use the frozen p85..p95 profile;
# rejected validation, exact replay, and read-only inspection use p75, p50,
# and p80 respectively. These choices were frozen before the benchmark-v1.2.0
# pilot and are inherited unchanged by benchmark-v1.3.0.
MUTATING_SERVICE_QUANTILES_S = (1.592, 1.633, 1.658, 1.686, 1.734)
INVALID_VALIDATION_S = 0.334
IDEMPOTENT_REPLAY_S = 0.140
READ_ONLY_QUERY_S = 0.902


def sample_mutating_service_time(rng: random.Random) -> float:
    """Sample one frozen empirical quantile using the task-local RNG."""

    return rng.choice(MUTATING_SERVICE_QUANTILES_S)


__all__ = [
    "IDEMPOTENT_REPLAY_S",
    "INVALID_VALIDATION_S",
    "MUTATING_SERVICE_QUANTILES_S",
    "PROFILE_ID",
    "READ_ONLY_QUERY_S",
    "SOURCE_ARCHIVE_SHA256",
    "SOURCE_INVOCATIONS",
    "SOURCE_LICENSE",
    "SOURCE_ROWS",
    "SOURCE_URL",
    "sample_mutating_service_time",
]
