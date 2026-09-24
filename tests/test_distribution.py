from __future__ import annotations

import importlib
import io
import shutil
import stat
import sys
import tarfile
import tomllib
import zipfile
from collections.abc import Callable
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
distribution = importlib.import_module("scripts.check_distribution")
DistributionCheckError = distribution.DistributionCheckError
compare_build_outputs = distribution.compare_build_outputs
validate_sdist = distribution.validate_sdist
validate_wheel = distribution.validate_wheel


def test_reproducible_toolchain_is_pinned() -> None:
    uv_config = tomllib.loads((ROOT / "uv.toml").read_text(encoding="utf-8"))
    assert (ROOT / ".python-version").read_text(encoding="utf-8").strip() == "3.11.15"
    assert uv_config["required-version"] == "==0.11.29"
    assert distribution.PYTHON_VERSION == "3.11.15"

    constraint_lines = {
        line.split("==", maxsplit=1)[0]
        for line in (ROOT / "build-constraints.txt").read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#")
    }
    assert constraint_lines == {
        "hatchling",
        "packaging",
        "pathspec",
        "pluggy",
        "trove-classifiers",
    }
    for line in (ROOT / "build-constraints.txt").read_text(encoding="utf-8").splitlines():
        if line and not line.startswith("#"):
            assert " --hash=sha256:" in line
            assert len(line.rsplit("sha256:", maxsplit=1)[1]) == 64


def test_workflows_enforce_locked_read_only_toolchain() -> None:
    workflow_names = ("ci.yml", "full-benchmark.yml", "qwen-adapter.yml")
    workflows = {
        name: (ROOT / ".github/workflows" / name).read_text(encoding="utf-8")
        for name in workflow_names
    }
    for workflow in workflows.values():
        assert 'version: "0.11.29"' in workflow
        assert 'python-version: "3.11.15"' in workflow
        assert "persist-credentials: false" in workflow
        assert "run: uv lock --check" in workflow
        assert "uv sync --extra dev --locked" in workflow
        assert "runs-on: ubuntu-24.04" in workflow
        assert "ubuntu-latest" not in workflow

    assert "  push:\n    branches:\n      - main\n" in workflows["ci.yml"]
    assert "run: make package-check" in workflows["ci.yml"]
    assert "portable-verification-macos:" in workflows["ci.yml"]
    assert "runs-on: macos-14" in workflows["ci.yml"]
    assert "test_smoke_runs_real_lightweight_loop_and_verify_run_accepts_it" in workflows[
        "ci.yml"
    ]
    assert "cancel-in-progress: false" in workflows["full-benchmark.yml"]
    assert "cd artifacts" in workflows["full-benchmark.yml"]
    assert "find benchmark-v1 runs/full" in workflows["full-benchmark.yml"]
    assert "> full-benchmark-SHA256SUMS" in workflows["full-benchmark.yml"]
    assert "cancel-in-progress: true" in workflows["qwen-adapter.yml"]


def test_make_ci_contains_distribution_gate_but_not_optional_layers() -> None:
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    ci_recipe = makefile.split("\nci:", maxsplit=1)[1].split("benchmark:", maxsplit=1)[0]
    assert "$(MAKE) lock-check" in ci_recipe
    assert "$(MAKE) package-check" in ci_recipe
    assert "qwen" not in ci_recipe.casefold()


def _write_wheel(path: Path, members: dict[str, bytes]) -> None:
    with zipfile.ZipFile(path, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, payload in members.items():
            archive.writestr(name, payload)


def _write_sdist(path: Path, members: dict[str, bytes]) -> None:
    with tarfile.open(path, mode="w:gz") as archive:
        for name, payload in members.items():
            member = tarfile.TarInfo(name)
            member.size = len(payload)
            archive.addfile(member, io.BytesIO(payload))


def _write_safe_pair(directory: Path, *, payload: bytes = b"safe\n") -> None:
    directory.mkdir()
    _write_wheel(directory / "demo-0.1.0-py3-none-any.whl", {"demo/data.txt": payload})
    _write_sdist(directory / "demo-0.1.0.tar.gz", {"demo-0.1.0/data.txt": payload})


def test_safe_wheel_and_sdist_are_accepted(tmp_path: Path) -> None:
    wheel = tmp_path / "demo.whl"
    sdist = tmp_path / "demo.tar.gz"
    _write_wheel(wheel, {"demo/__init__.py": b""})
    _write_sdist(sdist, {"demo-0.1.0/demo/__init__.py": b""})

    validate_wheel(wheel)
    validate_sdist(sdist)


def test_documented_environment_example_is_not_treated_as_a_secret(tmp_path: Path) -> None:
    sdist = tmp_path / "example.tar.gz"
    _write_sdist(sdist, {"demo-0.1.0/.env.example": b"TOKEN=replace-me\n"})

    validate_sdist(sdist)


@pytest.mark.parametrize(
    "member_name",
    [
        "/absolute.txt",
        "../escape.txt",
        "demo/../../escape.txt",
        "C:/escape.txt",
        "demo\\escape.txt",
        "demo//ambiguous.txt",
    ],
)
def test_wheel_rejects_unsafe_member_paths(tmp_path: Path, member_name: str) -> None:
    wheel = tmp_path / "unsafe.whl"
    _write_wheel(wheel, {member_name: b"unsafe"})

    with pytest.raises(DistributionCheckError, match="archive member"):
        validate_wheel(wheel)


@pytest.mark.parametrize(
    "member_name",
    [
        "/absolute.txt",
        "../escape.txt",
        "demo/../../escape.txt",
        "C:/escape.txt",
        "demo\\escape.txt",
        "demo//ambiguous.txt",
    ],
)
def test_sdist_rejects_unsafe_member_paths(tmp_path: Path, member_name: str) -> None:
    sdist = tmp_path / "unsafe.tar.gz"
    _write_sdist(sdist, {member_name: b"unsafe"})

    with pytest.raises(DistributionCheckError, match="archive member"):
        validate_sdist(sdist)


@pytest.mark.parametrize(
    "member_name",
    [
        "demo/.coverage",
        "demo/.coverage.worker",
        "demo/.env",
        "demo/.env.production",
        "demo/id_ed25519",
        "demo/signing.key",
    ],
)
def test_sdist_rejects_sensitive_file_names(tmp_path: Path, member_name: str) -> None:
    sdist = tmp_path / "sensitive.tar.gz"
    _write_sdist(sdist, {member_name: b"not actually secret"})

    with pytest.raises(DistributionCheckError, match="forbidden"):
        validate_sdist(sdist)


@pytest.mark.parametrize(
    "validator, suffix",
    [(validate_wheel, ".whl"), (validate_sdist, ".tar.gz")],
)
@pytest.mark.parametrize(
    "private_key",
    [
        # Build the marker at runtime so the distribution scanner does not
        # mistake this regression fixture itself for leaked key material.
        b"-----BEGIN " + b"OPENSSH PRIVATE KEY-----\nsecret",
        b"-----BEGIN " + b"ENCRYPTED PRIVATE KEY-----\nsecret",
    ],
)
def test_archives_reject_private_key_material(
    tmp_path: Path,
    validator: Callable[[Path], None],
    suffix: str,
    private_key: bytes,
) -> None:
    archive_path = tmp_path / f"secret{suffix}"
    members = {"demo/notes.txt": private_key}
    if suffix == ".whl":
        _write_wheel(archive_path, members)
    else:
        _write_sdist(archive_path, members)

    with pytest.raises(DistributionCheckError, match="private-key material"):
        validator(archive_path)


def test_sdist_rejects_links(tmp_path: Path) -> None:
    sdist = tmp_path / "link.tar.gz"
    with tarfile.open(sdist, mode="w:gz") as archive:
        member = tarfile.TarInfo("demo-0.1.0/link")
        member.type = tarfile.SYMTYPE
        member.linkname = "target"
        archive.addfile(member)

    with pytest.raises(DistributionCheckError, match="link or special file"):
        validate_sdist(sdist)


def test_wheel_rejects_symbolic_links(tmp_path: Path) -> None:
    wheel = tmp_path / "link.whl"
    with zipfile.ZipFile(wheel, mode="w") as archive:
        member = zipfile.ZipInfo("demo/link")
        member.create_system = 3
        member.external_attr = (stat.S_IFLNK | 0o777) << 16
        archive.writestr(member, "target")

    with pytest.raises(DistributionCheckError, match="symbolic link"):
        validate_wheel(wheel)


def test_two_identical_builds_have_matching_hashes(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    _write_safe_pair(first)
    second.mkdir()
    for artifact in first.iterdir():
        shutil.copy2(artifact, second / artifact.name)

    hashes = compare_build_outputs(first, second)

    assert set(hashes) == {"sdist", "wheel"}
    assert all(len(value) == 64 for value in hashes.values())


def test_different_builds_are_rejected(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    _write_safe_pair(first, payload=b"first\n")
    _write_safe_pair(second, payload=b"second\n")

    with pytest.raises(DistributionCheckError, match="not reproducible"):
        compare_build_outputs(first, second)
