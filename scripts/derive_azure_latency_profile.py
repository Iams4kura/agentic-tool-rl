#!/usr/bin/env python3
"""Reproduce frozen duration quantiles from Azure Functions Trace 2019.

The input can be either the official ``.tar.xz`` archive or a directory that
contains the 14 ``function_durations_percentiles.*.csv`` files. Only Python's
standard library is required; the script never downloads data by itself.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import tarfile
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import TextIO

OFFICIAL_ARCHIVE_SHA256 = (
    "aff8b3ca7240a41a109e4ee598e0a96e45fcb92e7b8395ac19cb3748cd260d89"
)
QUANTILES = (0.5, 0.75, 0.8, 0.85, 0.875, 0.9, 0.925, 0.95, 0.975)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _csv_rows(handle: TextIO) -> Iterator[tuple[float, int]]:
    reader = csv.DictReader(handle)
    if reader.fieldnames is None or not {"Average", "Count"}.issubset(reader.fieldnames):
        raise ValueError("duration CSV must contain Average and Count columns")
    for row in reader:
        try:
            average_ms = float(row["Average"])
            count = int(float(row["Count"]))
        except (TypeError, ValueError):
            continue
        if math.isfinite(average_ms) and average_ms >= 0 and count > 0:
            yield average_ms / 1000.0, count


def _directory_rows(directory: Path) -> Iterator[tuple[float, int]]:
    paths = sorted(directory.rglob("function_durations_percentiles.*.csv"))
    if len(paths) != 14:
        raise ValueError(f"expected 14 duration CSV files, found {len(paths)}")
    for path in paths:
        with path.open(encoding="utf-8", newline="") as handle:
            yield from _csv_rows(handle)


def _archive_rows(archive: Path) -> Iterator[tuple[float, int]]:
    with tarfile.open(archive, mode="r:xz") as bundle:
        members = sorted(
            (
                member
                for member in bundle.getmembers()
                if member.isfile()
                and Path(member.name).name.startswith("function_durations_percentiles.")
                and member.name.endswith(".csv")
            ),
            key=lambda member: member.name,
        )
        if len(members) != 14:
            raise ValueError(f"expected 14 duration CSV members, found {len(members)}")
        for member in members:
            extracted = bundle.extractfile(member)
            if extracted is None:
                raise ValueError(f"could not read archive member {member.name!r}")
            with extracted, io.TextIOWrapper(extracted, encoding="utf-8", newline="") as handle:
                yield from _csv_rows(handle)


def weighted_quantiles(
    rows: Iterable[tuple[float, int]], quantiles: Sequence[float] = QUANTILES
) -> tuple[dict[str, float], int, int]:
    """Return nearest-rank invocation-weighted quantiles in seconds."""

    requested = tuple(float(value) for value in quantiles)
    if not requested or any(not 0 < value <= 1 for value in requested):
        raise ValueError("quantiles must be non-empty and in (0, 1]")
    values = sorted(rows)
    if not values:
        raise ValueError("no valid duration rows")
    invocation_count = sum(weight for _, weight in values)
    targets = [value * invocation_count for value in requested]
    result: dict[str, float] = {}
    cumulative = 0
    target_index = 0
    for duration_s, weight in values:
        cumulative += weight
        while target_index < len(targets) and cumulative >= targets[target_index]:
            key = f"p{requested[target_index] * 100:g}".replace(".", "_")
            result[key] = duration_s
            target_index += 1
    return result, len(values), invocation_count


def derive(source: Path, *, verify_official_hash: bool) -> dict[str, object]:
    if source.is_file():
        archive_sha256 = file_sha256(source)
        if verify_official_hash and archive_sha256 != OFFICIAL_ARCHIVE_SHA256:
            raise ValueError(
                "archive SHA256 mismatch: "
                f"expected {OFFICIAL_ARCHIVE_SHA256}, got {archive_sha256}"
            )
        rows = _archive_rows(source)
    elif source.is_dir():
        archive_sha256 = None
        rows = _directory_rows(source)
    else:
        raise ValueError(f"source does not exist: {source}")
    quantiles, row_count, invocation_count = weighted_quantiles(rows)
    return {
        "archive_sha256": archive_sha256,
        "filter": "finite Average >= 0 and Count > 0",
        "rows": row_count,
        "invocations": invocation_count,
        "weight": "Count",
        "unit": "seconds",
        "quantiles": quantiles,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="official .tar.xz or extracted directory")
    parser.add_argument(
        "--skip-official-hash-check",
        action="store_true",
        help="allow a non-official archive (directory inputs never have an archive hash)",
    )
    args = parser.parse_args()
    payload = derive(
        args.source,
        verify_official_hash=not args.skip_official_hash_check,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
