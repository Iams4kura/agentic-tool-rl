from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest


def _load_script() -> ModuleType:
    path = Path(__file__).parents[1] / "scripts" / "derive_azure_latency_profile.py"
    spec = importlib.util.spec_from_file_location("derive_azure_latency_profile", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_invocation_weighted_quantiles_use_nearest_rank() -> None:
    module = _load_script()

    quantiles, rows, invocations = module.weighted_quantiles(
        [(0.1, 1), (0.2, 3), (0.9, 1)],
        (0.5, 0.8, 1.0),
    )

    assert rows == 3
    assert invocations == 5
    assert quantiles == {"p50": 0.2, "p80": 0.2, "p100": 0.9}


def test_empty_or_invalid_quantile_input_fails_closed() -> None:
    module = _load_script()

    with pytest.raises(ValueError, match="no valid duration rows"):
        module.weighted_quantiles([], (0.5,))
    with pytest.raises(ValueError, match="quantiles"):
        module.weighted_quantiles([(0.1, 1)], (0.0,))
