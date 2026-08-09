#!/usr/bin/env python3
"""Build and validate deterministic Python distribution artifacts."""

from __future__ import annotations

import argparse
import hashlib
import re
import stat
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

PYTHON_VERSION = "3.11.15"
_MAX_MEMBER_BYTES = 64 * 1024 * 1024
_MAX_ARCHIVE_BYTES = 256 * 1024 * 1024
_WINDOWS_DRIVE = re.compile(r"^[A-Za-z]:")
_PRIVATE_KEY_MARKERS = tuple(
    b"-----BEGIN " + key_type + b"PRIVATE KEY-----"
    for key_type in (b"", b"ENCRYPTED ", b"RSA ", b"DSA ", b"EC ", b"OPENSSH ")
)
_PRIVATE_KEY_NAMES = {
    "id_dsa",
    "id_ecdsa",
    "id_ed25519",
    "id_rsa",
    "private-key",
    "private_key",
}


class DistributionCheckError(RuntimeError):
    """Raised when a built distribution violates the release contract."""


@dataclass(frozen=True)
class BuildArtifacts:
    wheel: Path
    sdist: Path


def sha256_file(path: Path) -> str:
    """Return the SHA256 digest of ``path`` without loading it into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _archive_parts(name: str) -> tuple[str, ...]:
    if not name or "\x00" in name:
        raise DistributionCheckError("archive member has an empty or NUL-containing name")
    if "\\" in name:
        raise DistributionCheckError(f"archive member uses a backslash separator: {name!r}")
    if name.startswith("/") or _WINDOWS_DRIVE.match(name):
        raise DistributionCheckError(f"archive member has an absolute path: {name!r}")

    normalized = name[:-1] if name.endswith("/") else name
    parts = tuple(normalized.split("/"))
    if not normalized or any(part in {"", ".", ".."} for part in parts):
        raise DistributionCheckError(f"archive member has a non-canonical path: {name!r}")
    return parts


def _check_sensitive_path(parts: tuple[str, ...]) -> None:
    for part in parts:
        lowered = part.casefold()
        if lowered == ".coverage" or lowered.startswith(".coverage."):
            raise DistributionCheckError(f"coverage database is forbidden: {'/'.join(parts)!r}")
        if lowered != ".env.example" and (
            lowered == ".env" or lowered.startswith(".env.")
        ):
            raise DistributionCheckError(f"environment file is forbidden: {'/'.join(parts)!r}")
        if (
            lowered in _PRIVATE_KEY_NAMES
            or lowered.endswith((".key", ".p12", ".pfx"))
            or "private_key" in lowered
            or "private-key" in lowered
        ):
            raise DistributionCheckError(f"private-key path is forbidden: {'/'.join(parts)!r}")


def _check_private_key_payload(name: str, payload: bytes) -> None:
    if any(marker in payload for marker in _PRIVATE_KEY_MARKERS):
        raise DistributionCheckError(f"private-key material is forbidden: {name!r}")


def _register_member(
    name: str,
    *,
    seen: set[str],
) -> tuple[str, ...]:
    parts = _archive_parts(name)
    _check_sensitive_path(parts)
    canonical = "/".join(parts).casefold()
    if canonical in seen:
        raise DistributionCheckError(f"archive contains a duplicate member path: {name!r}")
    seen.add(canonical)
    return parts


def validate_wheel(path: Path) -> None:
    """Validate wheel integrity, paths, member types, and sensitive contents."""

    seen: set[str] = set()
    total_size = 0
    try:
        with zipfile.ZipFile(path) as archive:
            members = archive.infolist()
            if not members:
                raise DistributionCheckError(f"wheel is empty: {path}")

            for member in members:
                _register_member(member.filename, seen=seen)
                mode = member.external_attr >> 16
                if stat.S_ISLNK(mode):
                    raise DistributionCheckError(
                        f"wheel contains a symbolic link: {member.filename!r}"
                    )
                if member.flag_bits & 0x1:
                    raise DistributionCheckError(
                        f"wheel contains an encrypted member: {member.filename!r}"
                    )
                if member.file_size > _MAX_MEMBER_BYTES:
                    raise DistributionCheckError(
                        f"wheel member exceeds the size limit: {member.filename!r}"
                    )
                total_size += member.file_size
                if total_size > _MAX_ARCHIVE_BYTES:
                    raise DistributionCheckError("wheel exceeds the uncompressed size limit")
                if not member.is_dir():
                    _check_private_key_payload(member.filename, archive.read(member))

            bad_member = archive.testzip()
            if bad_member is not None:
                raise DistributionCheckError(f"wheel CRC check failed: {bad_member!r}")
    except (OSError, RuntimeError, zipfile.BadZipFile) as exc:
        raise DistributionCheckError(f"cannot inspect wheel {path}: {exc}") from exc


def validate_sdist(path: Path) -> None:
    """Validate sdist integrity, paths, member types, and sensitive contents."""

    seen: set[str] = set()
    total_size = 0
    try:
        with tarfile.open(path, mode="r:gz") as archive:
            members = archive.getmembers()
            if not members:
                raise DistributionCheckError(f"sdist is empty: {path}")

            for member in members:
                _register_member(member.name, seen=seen)
                if not (member.isfile() or member.isdir()):
                    raise DistributionCheckError(
                        f"sdist contains a link or special file: {member.name!r}"
                    )
                if member.size > _MAX_MEMBER_BYTES:
                    raise DistributionCheckError(
                        f"sdist member exceeds the size limit: {member.name!r}"
                    )
                total_size += member.size
                if total_size > _MAX_ARCHIVE_BYTES:
                    raise DistributionCheckError("sdist exceeds the uncompressed size limit")
                if member.isfile():
                    stream = archive.extractfile(member)
                    if stream is None:
                        raise DistributionCheckError(f"cannot read sdist member: {member.name!r}")
                    _check_private_key_payload(member.name, stream.read())
    except (OSError, tarfile.TarError) as exc:
        raise DistributionCheckError(f"cannot inspect sdist {path}: {exc}") from exc


def find_build_artifacts(directory: Path) -> BuildArtifacts:
    """Resolve the single wheel and sdist produced in a build directory."""

    wheels = sorted(directory.glob("*.whl"))
    sdists = sorted(directory.glob("*.tar.gz"))
    unexpected = sorted(
        path.name
        for path in directory.iterdir()
        if path.is_file() and path not in {*wheels, *sdists}
    )
    if len(wheels) != 1 or len(sdists) != 1 or unexpected:
        raise DistributionCheckError(
            "build output must contain exactly one wheel and one sdist; "
            f"found wheels={len(wheels)}, sdists={len(sdists)}, unexpected={unexpected}"
        )
    return BuildArtifacts(wheel=wheels[0], sdist=sdists[0])


def compare_build_outputs(first: Path, second: Path) -> dict[str, str]:
    """Inspect two builds and return hashes only when they are byte-identical."""

    first_artifacts = find_build_artifacts(first)
    second_artifacts = find_build_artifacts(second)
    pairs = {
        "wheel": (first_artifacts.wheel, second_artifacts.wheel),
        "sdist": (first_artifacts.sdist, second_artifacts.sdist),
    }
    hashes: dict[str, str] = {}
    for kind, (first_path, second_path) in pairs.items():
        if first_path.name != second_path.name:
            raise DistributionCheckError(
                f"{kind} filenames differ: {first_path.name!r} != {second_path.name!r}"
            )
        validator = validate_wheel if kind == "wheel" else validate_sdist
        validator(first_path)
        validator(second_path)
        first_hash = sha256_file(first_path)
        second_hash = sha256_file(second_path)
        if first_hash != second_hash:
            raise DistributionCheckError(
                f"{kind} is not reproducible: {first_hash} != {second_hash}"
            )
        hashes[kind] = first_hash
    return hashes


def _run_build(*, uv: str, project_root: Path, constraints: Path, output: Path) -> None:
    command = [
        uv,
        "build",
        "--python",
        PYTHON_VERSION,
        "--out-dir",
        str(output),
        "--no-create-gitignore",
        "--build-constraints",
        str(constraints),
        "--require-hashes",
        "--no-sources",
    ]
    try:
        subprocess.run(command, cwd=project_root, check=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise DistributionCheckError(f"distribution build failed: {exc}") from exc


def run_distribution_check(
    *,
    project_root: Path,
    uv: str,
    build_constraints: Path,
) -> dict[str, str]:
    """Build the project twice in isolation and validate matching artifacts."""

    root = project_root.resolve()
    constraints = build_constraints
    if not constraints.is_absolute():
        constraints = root / constraints
    constraints = constraints.resolve()
    if not (root / "pyproject.toml").is_file():
        raise DistributionCheckError(f"project root has no pyproject.toml: {root}")
    if not constraints.is_file():
        raise DistributionCheckError(f"build constraints file does not exist: {constraints}")

    with tempfile.TemporaryDirectory(prefix="agentic-tool-rl-dist-") as temporary:
        temporary_root = Path(temporary)
        first = temporary_root / "first"
        second = temporary_root / "second"
        first.mkdir()
        second.mkdir()
        _run_build(uv=uv, project_root=root, constraints=constraints, output=first)
        _run_build(uv=uv, project_root=root, constraints=constraints, output=second)
        return compare_build_outputs(first, second)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument("--uv", default="uv")
    parser.add_argument(
        "--build-constraints",
        type=Path,
        default=Path("build-constraints.txt"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        hashes = run_distribution_check(
            project_root=args.project_root,
            uv=args.uv,
            build_constraints=args.build_constraints,
        )
    except DistributionCheckError as exc:
        print(f"distribution check failed: {exc}", file=sys.stderr)
        return 1
    print("distribution check passed")
    for kind in sorted(hashes):
        print(f"{kind}_sha256={hashes[kind]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
