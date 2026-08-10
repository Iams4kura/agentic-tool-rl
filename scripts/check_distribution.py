#!/usr/bin/env python3
"""Build and validate deterministic Python distribution artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from email.parser import Parser
from pathlib import Path
from typing import Any

PYTHON_VERSION = "3.11.15"
REQUIRED_PACKAGED_CONFIGS = (
    "ablation.yaml",
    "cpu_full.yaml",
    "qwen3_lora_gpu.yaml",
    "smoke.yaml",
)
REQUIRED_PACKAGED_PROTOCOLS = ("benchmark-v1.4-development.json",)
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


def validate_packaged_configs(path: Path, *, source_directory: Path | None = None) -> None:
    """Require all CLI defaults in the wheel and optionally verify their exact bytes."""

    prefix = "agentic_tool_rl/resources/configs"
    try:
        with zipfile.ZipFile(path) as archive:
            members = set(archive.namelist())
            for name in REQUIRED_PACKAGED_CONFIGS:
                member = f"{prefix}/{name}"
                if member not in members:
                    raise DistributionCheckError(f"wheel is missing packaged config {member!r}")
                if source_directory is not None:
                    source = source_directory / name
                    if not source.is_file():
                        raise DistributionCheckError(
                            f"source packaged config does not exist: {source}"
                        )
                    if archive.read(member) != source.read_bytes():
                        raise DistributionCheckError(
                            f"wheel packaged config differs from explicit source config: {name}"
                        )
    except (OSError, RuntimeError, zipfile.BadZipFile) as exc:
        if isinstance(exc, DistributionCheckError):
            raise
        raise DistributionCheckError(f"cannot inspect packaged configs in {path}: {exc}") from exc


def validate_packaged_protocols(
    path: Path,
    *,
    source_directory: Path | None = None,
) -> None:
    """Require exact benchmark protocols needed to bind generated manifests."""

    prefix = "agentic_tool_rl/resources/protocols"
    try:
        with zipfile.ZipFile(path) as archive:
            members = set(archive.namelist())
            for name in REQUIRED_PACKAGED_PROTOCOLS:
                member = f"{prefix}/{name}"
                if member not in members:
                    raise DistributionCheckError(
                        f"wheel is missing packaged protocol {member!r}"
                    )
                if source_directory is not None:
                    source = source_directory / name
                    if not source.is_file():
                        raise DistributionCheckError(
                            f"source packaged protocol does not exist: {source}"
                        )
                    if archive.read(member) != source.read_bytes():
                        raise DistributionCheckError(
                            f"wheel packaged protocol differs from source protocol: {name}"
                        )
    except (OSError, RuntimeError, zipfile.BadZipFile) as exc:
        if isinstance(exc, DistributionCheckError):
            raise
        raise DistributionCheckError(
            f"cannot inspect packaged protocols in {path}: {exc}"
        ) from exc


def wheel_runtime_requirements(path: Path) -> tuple[str, ...]:
    """Read the exact ``Requires-Dist`` entries that drive wheel installation."""

    try:
        with zipfile.ZipFile(path) as archive:
            metadata_members = [
                name
                for name in archive.namelist()
                if name.endswith(".dist-info/METADATA")
            ]
            if len(metadata_members) != 1:
                raise DistributionCheckError(
                    "wheel must contain exactly one dist-info METADATA file"
                )
            document = Parser().parsestr(
                archive.read(metadata_members[0]).decode("utf-8")
            )
    except (OSError, UnicodeDecodeError, zipfile.BadZipFile) as exc:
        raise DistributionCheckError(f"cannot read wheel dependency metadata: {exc}") from exc
    requirements = tuple(document.get_all("Requires-Dist", []))
    if not requirements:
        raise DistributionCheckError("wheel metadata contains no runtime dependencies")
    return requirements


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


def _run_checked(
    command: Sequence[str],
    *,
    cwd: Path,
    env: dict[str, str],
    description: str,
) -> str:
    try:
        completed = subprocess.run(
            list(command),
            cwd=cwd,
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        stdout = exc.stdout if isinstance(exc, subprocess.CalledProcessError) else ""
        stderr = exc.stderr if isinstance(exc, subprocess.CalledProcessError) else ""
        raise DistributionCheckError(
            f"installed-wheel {description} failed: {exc}; stdout={stdout!r}; stderr={stderr!r}"
        ) from exc
    return completed.stdout


def _json_object(output: str, *, description: str) -> dict[str, Any]:
    try:
        value = json.loads(output)
    except json.JSONDecodeError as exc:
        raise DistributionCheckError(
            f"installed-wheel {description} emitted invalid JSON"
        ) from exc
    if not isinstance(value, dict):
        raise DistributionCheckError(
            f"installed-wheel {description} must emit a JSON object"
        )
    return value


def _export_locked_runtime_constraints(
    *,
    uv: str,
    root: Path,
    output: Path,
    env: dict[str, str],
) -> str:
    """Export exact runtime pins from ``uv.lock`` for wheel dependency resolution."""

    _run_checked(
        [
            uv,
            "export",
            "--project",
            str(root),
            "--locked",
            "--no-dev",
            "--no-emit-project",
            "--no-hashes",
            "--no-annotate",
            "--no-header",
            "--no-sources",
            "--output-file",
            str(output),
        ],
        cwd=root,
        env=env,
        description="locked runtime constraint export",
    )
    if not output.is_file() or not output.read_text(encoding="utf-8").strip():
        raise DistributionCheckError("uv.lock runtime constraint export is empty")
    return sha256_file(output)


def _resolve_probe_path(value: object) -> Path:
    if not isinstance(value, str) or not value:
        raise DistributionCheckError("installed-wheel isolation probe contains an invalid path")
    return Path(value).resolve()


def _require_inside_venv(path: Path, venv: Path, *, description: str) -> None:
    try:
        path.relative_to(venv)
    except ValueError as exc:
        raise DistributionCheckError(
            f"installed-wheel {description} escapes the temporary venv: {path}"
        ) from exc


def validate_isolated_site_probe(probe: Mapping[str, Any], *, venv: Path) -> None:
    """Reject host/user site-packages and external ``.pth`` path targets."""

    if probe.get("user_site_enabled") is not False:
        raise DistributionCheckError("installed-wheel user site must be disabled")
    site_packages = probe.get("site_packages")
    sys_path = probe.get("sys_path")
    pth_entries = probe.get("pth_entries")
    if not isinstance(site_packages, list) or not isinstance(sys_path, list):
        raise DistributionCheckError("installed-wheel isolation probe has invalid path lists")
    if not isinstance(pth_entries, list):
        raise DistributionCheckError("installed-wheel isolation probe has invalid .pth evidence")
    resolved_venv = venv.resolve()
    for value in site_packages:
        _require_inside_venv(
            _resolve_probe_path(value),
            resolved_venv,
            description="site-packages path",
        )
    for value in sys_path:
        if not isinstance(value, str):
            raise DistributionCheckError("installed-wheel sys.path contains a non-string")
        if "site-packages" in value.casefold():
            _require_inside_venv(
                _resolve_probe_path(value),
                resolved_venv,
                description="sys.path site-packages entry",
            )
    for item in pth_entries:
        if not isinstance(item, Mapping):
            raise DistributionCheckError("installed-wheel .pth evidence must contain objects")
        pth_path = _resolve_probe_path(item.get("path"))
        _require_inside_venv(pth_path, resolved_venv, description=".pth file")
        entries = item.get("entries")
        if not isinstance(entries, list):
            raise DistributionCheckError("installed-wheel .pth entries must be a list")
        for entry in entries:
            if not isinstance(entry, str) or not entry:
                raise DistributionCheckError("installed-wheel .pth entry is invalid")
            # Some locked dependencies (notably setuptools) own executable
            # bootstrap entries such as distutils-precedence.pth.  Their
            # effects are checked through the post-start sys.path below; they
            # are not a host-site injection performed by this contract.
            if entry.lstrip().startswith("import "):
                continue
            candidate = Path(entry)
            if not candidate.is_absolute():
                candidate = pth_path.parent / candidate
            _require_inside_venv(
                candidate.resolve(),
                resolved_venv,
                description=".pth target",
            )


def run_installed_wheel_contract(
    *,
    wheel: Path,
    uv: str,
    root: Path,
    expected_implementation_fingerprint: str,
) -> dict[str, Any]:
    """Install a wheel into a temp venv and exercise it outside the source tree."""

    runtime_requirements = wheel_runtime_requirements(wheel)
    with tempfile.TemporaryDirectory(prefix="agentic-tool-rl-wheel-contract-") as temporary:
        contract_root = Path(temporary)
        venv = contract_root / "venv"
        empty_cwd = contract_root / "empty-cwd"
        empty_cwd.mkdir()
        bootstrap_env = os.environ.copy()
        uv_cache_dir = _run_checked(
            [uv, "cache", "dir"],
            cwd=root,
            env=bootstrap_env,
            description="uv cache discovery",
        ).strip()
        try:
            subprocess.run(
                [sys.executable, "-m", "venv", "--without-pip", str(venv)],
                cwd=root,
                env=bootstrap_env,
                check=True,
                capture_output=True,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            raise DistributionCheckError(f"installed-wheel venv creation failed: {exc}") from exc

        scripts = venv / ("Scripts" if os.name == "nt" else "bin")
        python = scripts / ("python.exe" if os.name == "nt" else "python")
        probe_env = os.environ.copy()
        for name in ("PYTHONHOME", "PYTHONPATH", "VIRTUAL_ENV"):
            probe_env.pop(name, None)
        isolated_home = contract_root / "isolated-home"
        isolated_home.mkdir()
        probe_env["HOME"] = str(isolated_home)
        probe_env["USERPROFILE"] = str(isolated_home)
        probe_env["XDG_CACHE_HOME"] = str(isolated_home / ".cache")
        probe_env["UV_CACHE_DIR"] = uv_cache_dir
        probe_env["UV_NO_CONFIG"] = "1"
        probe_env["PYTHONNOUSERSITE"] = "1"
        probe_env["PATH"] = os.pathsep.join((str(scripts), probe_env.get("PATH", "")))

        runtime_constraints = contract_root / "runtime-constraints.txt"
        runtime_constraints_sha256 = _export_locked_runtime_constraints(
            uv=uv,
            root=root,
            output=runtime_constraints,
            env=probe_env,
        )
        _run_checked(
            [
                uv,
                "pip",
                "install",
                "--python",
                str(python),
                "--constraints",
                str(runtime_constraints),
                "--torch-backend",
                "cpu",
                "--link-mode",
                "copy",
                str(wheel),
            ],
            cwd=empty_cwd,
            env=probe_env,
            description="metadata-driven locked installation",
        )
        _run_checked(
            [uv, "pip", "check", "--python", str(python)],
            cwd=empty_cwd,
            env=probe_env,
            description="dependency consistency check",
        )

        console = scripts / ("agentic-tool-rl.exe" if os.name == "nt" else "agentic-tool-rl")
        help_output = _run_checked(
            [str(console), "--help"],
            cwd=empty_cwd,
            env=probe_env,
            description="help command",
        )
        if "qwen-dry-run" not in help_output or "semantic-canary" not in help_output:
            raise DistributionCheckError("installed-wheel help is missing required commands")

        qwen_output = _run_checked(
            [str(console), "qwen-dry-run"],
            cwd=empty_cwd,
            env=probe_env,
            description="qwen-dry-run command",
        )
        qwen = _json_object(qwen_output, description="qwen-dry-run command")
        if qwen.get("kind") != "qwen_gpu" or qwen.get("would_download_weights") is not False:
            raise DistributionCheckError("installed-wheel qwen resource contract is invalid")

        semantic_output = _run_checked(
            [str(console), "semantic-canary"],
            cwd=empty_cwd,
            env=probe_env,
            description="semantic-canary command",
        )
        semantic = _json_object(semantic_output, description="semantic-canary command")
        gate = semantic.get("gate")
        if not isinstance(gate, dict) or gate.get("passed") is not True:
            raise DistributionCheckError("installed-wheel semantic canary did not pass")

        probe_code = """
import json
import importlib.metadata
import site
import sys
from pathlib import Path
import agentic_tool_rl
from agentic_tool_rl.config import load_packaged_config
from agentic_tool_rl.package_resources import (
    implementation_fingerprint_v2,
    packaged_protocol_sha256,
)
smoke = load_packaged_config("smoke.yaml")
ablation = load_packaged_config("ablation.yaml")
site_packages = [str(Path(value).resolve()) for value in site.getsitepackages()]
pth_entries = []
for directory in site_packages:
    for pth in sorted(Path(directory).glob("*.pth")):
        entries = [
            line.strip()
            for line in pth.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        pth_entries.append({"path": str(pth.resolve()), "entries": entries})
print(json.dumps({
    "module": str(Path(agentic_tool_rl.__file__).resolve()),
    "prefix": str(Path(sys.prefix).resolve()),
    "smoke": smoke.name,
    "ablation": ablation.name,
    "protocol_sha256": packaged_protocol_sha256("benchmark-v1.4-development.json"),
    "implementation_fingerprint": implementation_fingerprint_v2(),
    "requires_dist": importlib.metadata.requires("agentic-tool-rl") or [],
    "site_packages": site_packages,
    "sys_path": list(sys.path),
    "user_site_enabled": site.ENABLE_USER_SITE,
    "pth_entries": pth_entries,
}))
"""
        resource_output = _run_checked(
            [str(python), "-c", probe_code],
            cwd=empty_cwd,
            env=probe_env,
            description="resource/import probe",
        )
        try:
            probe = _json_object(resource_output, description="resource/import probe")
            module_path = Path(str(probe["module"])).resolve()
            module_path.relative_to(venv.resolve())
            if Path(str(probe["prefix"])).resolve() != venv.resolve():
                raise ValueError("sys.prefix is not the temporary venv")
        except (KeyError, ValueError) as exc:
            raise DistributionCheckError(
                "installed-wheel import did not resolve from the temporary venv"
            ) from exc
        validate_isolated_site_probe(probe, venv=venv)
        installed_requirements = probe.get("requires_dist")
        if (
            not isinstance(installed_requirements, list)
            or tuple(installed_requirements) != runtime_requirements
        ):
            raise DistributionCheckError(
                "installed-wheel Requires-Dist differs from the built wheel metadata"
            )
        if probe.get("smoke") != "cpu-smoke" or probe.get("ablation") != "benchmark-v1-six-way":
            raise DistributionCheckError("installed-wheel packaged resource names are invalid")
        protocol_sha256 = probe.get("protocol_sha256")
        if not isinstance(protocol_sha256, str) or len(protocol_sha256) != 64:
            raise DistributionCheckError("installed-wheel protocol hash is invalid")
        if probe.get("implementation_fingerprint") != expected_implementation_fingerprint:
            raise DistributionCheckError(
                "installed-wheel implementation fingerprint differs from checkout"
            )

        config_path = contract_root / "smoke.yaml"
        try:
            with zipfile.ZipFile(wheel) as archive:
                config_path.write_bytes(
                    archive.read("agentic_tool_rl/resources/configs/smoke.yaml")
                )
        except (OSError, KeyError, zipfile.BadZipFile) as exc:
            raise DistributionCheckError(
                "installed-wheel contract could not materialize its packaged smoke config"
            ) from exc

        benchmark_dir = contract_root / "benchmark"
        train_output = contract_root / "training"
        evaluation_dir = contract_root / "evaluation"
        recompute_path = contract_root / "recompute.json"
        generate = _json_object(
            _run_checked(
                [
                    str(console),
                    "generate",
                    "--config",
                    str(config_path),
                    "--output",
                    str(benchmark_dir),
                ],
                cwd=empty_cwd,
                env=probe_env,
                description="generate command",
            ),
            description="generate command",
        )
        if Path(str(generate.get("benchmark_dir", ""))).resolve() != benchmark_dir.resolve():
            raise DistributionCheckError(
                "installed-wheel generate command returned an unexpected benchmark path"
            )

        train = _json_object(
            _run_checked(
                [
                    str(console),
                    "train",
                    "--config",
                    str(config_path),
                    "--benchmark-dir",
                    str(benchmark_dir),
                    "--output",
                    str(train_output),
                    "--variant",
                    "B-BC-Mask",
                    "--seed",
                    "17",
                ],
                cwd=empty_cwd,
                env=probe_env,
                description="train command",
            ),
            description="train command",
        )
        runs = train.get("runs")
        if not isinstance(runs, list) or len(runs) != 1 or not isinstance(runs[0], dict):
            raise DistributionCheckError(
                "installed-wheel train command did not return exactly one run"
            )
        run = runs[0]
        run_root = Path(str(train.get("root", ""))).resolve()
        try:
            run_root.relative_to(train_output.resolve())
        except ValueError as exc:
            raise DistributionCheckError(
                "installed-wheel train command escaped its output directory"
            ) from exc
        checkpoint = run_root / str(run.get("checkpoint", ""))
        if (
            run.get("variant") != "B-BC-Mask"
            or run.get("cases") != 32
            or not checkpoint.is_file()
        ):
            raise DistributionCheckError(
                "installed-wheel train command returned invalid run evidence"
            )

        evaluate = _json_object(
            _run_checked(
                [
                    str(console),
                    "evaluate",
                    "--checkpoint",
                    str(checkpoint),
                    "--variant",
                    "B-BC-Mask",
                    "--config",
                    str(config_path),
                    "--benchmark-dir",
                    str(benchmark_dir),
                    "--output",
                    str(evaluation_dir),
                ],
                cwd=empty_cwd,
                env=probe_env,
                description="evaluate command",
            ),
            description="evaluate command",
        )
        metrics = evaluate.get("metrics")
        if (
            evaluate.get("variant") != "B-BC-Mask"
            or not isinstance(metrics, dict)
            or metrics.get("cases") != 32
        ):
            raise DistributionCheckError(
                "installed-wheel evaluate command returned invalid metric evidence"
            )

        recompute = _json_object(
            _run_checked(
                [
                    str(console),
                    "recompute",
                    "--traces",
                    str(evaluation_dir / "traces.jsonl"),
                    "--metrics",
                    str(evaluation_dir / "metrics.json"),
                    "--output",
                    str(recompute_path),
                ],
                cwd=empty_cwd,
                env=probe_env,
                description="recompute command",
            ),
            description="recompute command",
        )
        if recompute.get("matches") is not True or not recompute_path.is_file():
            raise DistributionCheckError(
                "installed-wheel recompute command did not match evaluation metrics"
            )
        if any(empty_cwd.iterdir()):
            raise DistributionCheckError(
                "installed-wheel commands unexpectedly wrote into the empty working directory"
            )
        return {
            "module": str(module_path),
            "prefix": str(probe["prefix"]),
            "cwd": str(empty_cwd),
            "implementation_fingerprint": expected_implementation_fingerprint,
            "runtime_constraints_sha256": runtime_constraints_sha256,
            "runtime_requirements": len(runtime_requirements),
            "host_site_injection": False,
            "semantic_gate": str(gate.get("criterion")),
            "cli_loop": {
                "generate": True,
                "train": True,
                "evaluate": True,
                "recompute": True,
                "cases": run["cases"],
                "variant": run["variant"],
            },
        }


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
        hashes = compare_build_outputs(first, second)
        artifacts = find_build_artifacts(first)
        validate_packaged_configs(artifacts.wheel, source_directory=root / "configs")
        validate_packaged_protocols(
            artifacts.wheel,
            source_directory=root / "docs" / "protocol",
        )
        from agentic_tool_rl.package_resources import implementation_fingerprint_v2

        contract = run_installed_wheel_contract(
            wheel=artifacts.wheel,
            uv=uv,
            root=root,
            expected_implementation_fingerprint=implementation_fingerprint_v2(),
        )
        print(f"installed_wheel_contract={json.dumps(contract, sort_keys=True)}")
        return hashes


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
