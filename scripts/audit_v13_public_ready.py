"""Reproduce and independently verify the frozen v1.3 public-ready shortcut."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agentic_tool_rl.envs.benchmark import generate_tasks
from agentic_tool_rl.envs.persistence import load_tasks_jsonl
from agentic_tool_rl.evaluation.public_baselines import (
    MANIFEST_FILENAME,
    TRACE_FILENAME,
    V13_DEFAULT_BASE_SEED,
    VERIFICATION_FILENAME,
    verify_public_ready_audit,
    write_public_ready_audit,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run R1 public-ready greedy on frozen benchmark-v1.3 test cases and write "
            "per-case replay evidence."
        )
    )
    parser.add_argument(
        "--benchmark-test",
        type=Path,
        help="Optional frozen test.jsonl; otherwise regenerate v1.3 deterministically.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/audits/v1.3-public-ready"),
        help="Output directory for trace, manifest, and independent verification.",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=1000,
        help="Generated test size when --benchmark-test is omitted (default: 1000).",
    )
    parser.add_argument(
        "--base-seed",
        type=int,
        default=V13_DEFAULT_BASE_SEED,
        help=f"Frozen benchmark base seed (default: {V13_DEFAULT_BASE_SEED}).",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Do not rewrite evidence; independently replay and verify existing output.",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.size < 1:
        raise SystemExit("--size must be positive")
    tasks = (
        load_tasks_jsonl(args.benchmark_test)
        if args.benchmark_test is not None
        else generate_tasks("test", args.size, base_seed=args.base_seed)
    )
    if not args.verify_only:
        write_public_ready_audit(args.output, tasks, base_seed=args.base_seed)
    verification = verify_public_ready_audit(args.output, tasks, base_seed=args.base_seed)
    result = {
        "trace": str(args.output / TRACE_FILENAME),
        "manifest": str(args.output / MANIFEST_FILENAME),
        "verification": str(args.output / VERIFICATION_FILENAME),
        **verification,
    }
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
