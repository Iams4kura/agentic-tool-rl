#!/usr/bin/env python3
"""Generate and quality-gate non-canonical benchmark-v1.4 development artifacts."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from agentic_tool_rl.envs.benchmark_v14 import DEVELOPMENT_BASE_SEED_V14
from agentic_tool_rl.v14_acceptance import run_v14_development_acceptance


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/benchmark-v1.4-development"),
    )
    parser.add_argument("--train-groups", type=int, default=500)
    parser.add_argument("--dev-groups", type=int, default=100)
    parser.add_argument("--test-groups", type=int, default=250)
    parser.add_argument("--base-seed", type=int, default=DEVELOPMENT_BASE_SEED_V14)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if min(args.train_groups, args.dev_groups, args.test_groups) < 1:
        raise SystemExit("group counts must be positive")
    verification = run_v14_development_acceptance(
        args.output,
        train_groups=args.train_groups,
        dev_groups=args.dev_groups,
        test_groups=args.test_groups,
        base_seed=args.base_seed,
    )
    print(json.dumps(verification, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
