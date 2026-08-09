#!/usr/bin/env python3
"""Verify one published evidence release with its exact historical verifier.

This module is intentionally outside ``src/``.  It is a bootstrapper, not an
alternative evidence verifier: it authenticates a release asset and an
annotated Git tag, recreates the declared runtime, then delegates to the
tagged ``verify-run`` implementation without weakening any of its gates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform as host_platform
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import NoReturn

_ROOT = Path(__file__).resolve().parents[1]
_TRUSTED_REPOSITORY = "Iams4kura/agentic-tool-rl"
_TRUSTED_DESCRIPTOR_FILES = {
    "v0.1.0": (
        "v0.1.0.evidence.json",
        "2f39d267fd5ecb271c5912fcc200c2b8a94b06de6f0b55b9d515475ecb0f3655",
    ),
}
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_GIT_OID_RE = re.compile(r"[0-9a-f]{40}")
_RELEASE_RE = re.compile(r"v[0-9]+\.[0-9]+\.[0-9]+")
_REPOSITORY_RE = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+")
_RUN_ID_RE = re.compile(r"[0-9a-f]{16}")
_UV_VERSION_RE = re.compile(r"uv ([0-9]+\.[0-9]+\.[0-9]+)(?:\s.*)?")
_CHECKSUM_LINE_RE = re.compile(r"([0-9a-f]{64})  \./(.+)")
_MAX_DOWNLOAD_BYTES = 2 * 1024**3
_MAX_TAR_BYTES = 8 * 1024**3
_MAX_TAR_MEMBERS = 10_000


class ReleaseVerificationError(RuntimeError):
    """Raised when any release identity or verification gate fails."""


@dataclass(frozen=True)
class TagIdentity:
    name: str
    object_oid: str
    commit_oid: str


@dataclass(frozen=True)
class SourceIdentity:
    fingerprint_schema: str
    sha256: str


@dataclass(frozen=True)
class RuntimeIdentity:
    python_implementation: str
    python_version: str
    python_cache_tag: str
    torch_version: str
    torch_cuda_version: str | None

    def to_dict(self) -> dict[str, str | None]:
        return {
            "python_implementation": self.python_implementation,
            "python_version": self.python_version,
            "python_cache_tag": self.python_cache_tag,
            "torch_version": self.torch_version,
            "torch_cuda_version": self.torch_cuda_version,
        }


@dataclass(frozen=True)
class ToolchainIdentity:
    uv_version: str


@dataclass(frozen=True)
class PlatformIdentity:
    system: str
    machine: str


@dataclass(frozen=True)
class ArchiveIdentity:
    name: str
    size_bytes: int
    sha256: str
    root: str


@dataclass(frozen=True)
class FilesManifestIdentity:
    name: str
    size_bytes: int
    sha256: str
    entry_count: int


@dataclass(frozen=True)
class RunManifestIdentity:
    path: str
    sha256: str
    run_id: str
    runs: int
    evaluation_units: int


@dataclass(frozen=True)
class ReleaseDescriptor:
    schema_version: str
    release: str
    repository: str
    tag: TagIdentity
    source: SourceIdentity
    runtime: RuntimeIdentity
    toolchain: ToolchainIdentity
    replay_platform: PlatformIdentity
    archive: ArchiveIdentity
    files_manifest: FilesManifestIdentity
    run_manifest: RunManifestIdentity


AssetFetcher = Callable[[str, int, Path], None]
Decompressor = Callable[[Path, Path], None]


def _fail(message: str) -> NoReturn:
    raise ReleaseVerificationError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            _fail(f"JSON contains duplicate field {key!r}")
        result[key] = value
    return result


def _read_json_bytes(payload: bytes, *, description: str) -> object:
    try:
        return json.loads(payload, object_pairs_hook=_strict_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReleaseVerificationError(f"{description} is not strict UTF-8 JSON") from exc


def _read_json(path: Path, *, description: str) -> object:
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise ReleaseVerificationError(f"could not read {description}: {path}") from exc
    return _read_json_bytes(payload, description=description)


def _mapping(value: object, *, description: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        _fail(f"{description} must be a JSON object")
    return value


def _exact_keys(
    value: Mapping[str, object], expected: set[str], *, description: str
) -> None:
    observed = set(value)
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        _fail(f"{description} fields differ: missing={missing}, extra={extra}")


def _string(value: object, *, description: str) -> str:
    if not isinstance(value, str) or not value:
        _fail(f"{description} must be a non-empty string")
    return value


def _integer(value: object, *, description: str, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        _fail(f"{description} must be an integer >= {minimum}")
    return value


def _digest(value: object, *, description: str) -> str:
    result = _string(value, description=description)
    if _SHA256_RE.fullmatch(result) is None:
        _fail(f"{description} must be a lowercase SHA256")
    return result


def _git_oid(value: object, *, description: str) -> str:
    result = _string(value, description=description)
    if _GIT_OID_RE.fullmatch(result) is None:
        _fail(f"{description} must be a lowercase 40-character Git object ID")
    return result


def _safe_component(value: object, *, description: str) -> str:
    result = _string(value, description=description)
    if (
        result in {".", ".."}
        or "/" in result
        or "\\" in result
        or any(ord(character) < 32 for character in result)
    ):
        _fail(f"{description} must be one canonical path component")
    return result


def _relative_posix_path(value: object, *, description: str) -> str:
    result = _string(value, description=description)
    path = PurePosixPath(result)
    if (
        path.is_absolute()
        or "\\" in result
        or path.as_posix() != result
        or any(part in {"", ".", ".."} for part in path.parts)
        or any(ord(character) < 32 for character in result)
    ):
        _fail(f"{description} must be a canonical relative POSIX path")
    return result


def parse_descriptor(value: object) -> ReleaseDescriptor:
    """Parse a descriptor with exact schemas at every nesting level."""

    root = _mapping(value, description="release descriptor")
    _exact_keys(
        root,
        {
            "schema_version",
            "release",
            "repository",
            "tag",
            "source",
            "runtime",
            "toolchain",
            "replay_platform",
            "archive",
            "files_manifest",
            "run_manifest",
        },
        description="release descriptor",
    )
    schema_version = _string(root["schema_version"], description="schema_version")
    if schema_version != "release-evidence-v1":
        _fail("unsupported release descriptor schema")
    release = _string(root["release"], description="release")
    if _RELEASE_RE.fullmatch(release) is None:
        _fail("release must be a canonical semantic version tag")
    repository = _string(root["repository"], description="repository")
    if _REPOSITORY_RE.fullmatch(repository) is None or any(
        part in {".", ".."} for part in repository.split("/")
    ):
        _fail("repository must be a canonical owner/name slug")

    raw_tag = _mapping(root["tag"], description="tag")
    _exact_keys(raw_tag, {"name", "object_oid", "commit_oid"}, description="tag")
    tag_name = _string(raw_tag["name"], description="tag.name")
    if tag_name != release:
        _fail("tag.name must equal release")
    tag = TagIdentity(
        name=tag_name,
        object_oid=_git_oid(raw_tag["object_oid"], description="tag.object_oid"),
        commit_oid=_git_oid(raw_tag["commit_oid"], description="tag.commit_oid"),
    )

    raw_source = _mapping(root["source"], description="source")
    _exact_keys(raw_source, {"fingerprint_schema", "sha256"}, description="source")
    fingerprint_schema = _string(
        raw_source["fingerprint_schema"], description="source.fingerprint_schema"
    )
    if fingerprint_schema != "source-fingerprint-v1":
        _fail("unsupported source fingerprint schema")
    source = SourceIdentity(
        fingerprint_schema=fingerprint_schema,
        sha256=_digest(raw_source["sha256"], description="source.sha256"),
    )

    raw_runtime = _mapping(root["runtime"], description="runtime")
    runtime_keys = {
        "python_implementation",
        "python_version",
        "python_cache_tag",
        "torch_version",
        "torch_cuda_version",
    }
    _exact_keys(raw_runtime, runtime_keys, description="runtime")
    raw_cuda = raw_runtime["torch_cuda_version"]
    if raw_cuda is not None and (not isinstance(raw_cuda, str) or not raw_cuda):
        _fail("runtime.torch_cuda_version must be null or a non-empty string")
    runtime = RuntimeIdentity(
        python_implementation=_string(
            raw_runtime["python_implementation"],
            description="runtime.python_implementation",
        ),
        python_version=_string(
            raw_runtime["python_version"], description="runtime.python_version"
        ),
        python_cache_tag=_string(
            raw_runtime["python_cache_tag"], description="runtime.python_cache_tag"
        ),
        torch_version=_string(
            raw_runtime["torch_version"], description="runtime.torch_version"
        ),
        torch_cuda_version=raw_cuda,
    )

    raw_toolchain = _mapping(root["toolchain"], description="toolchain")
    _exact_keys(raw_toolchain, {"uv_version"}, description="toolchain")
    uv_version = _string(
        raw_toolchain["uv_version"], description="toolchain.uv_version"
    )
    if re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", uv_version) is None:
        _fail("toolchain.uv_version must be a canonical release version")
    toolchain = ToolchainIdentity(uv_version=uv_version)

    raw_platform = _mapping(root["replay_platform"], description="replay_platform")
    _exact_keys(
        raw_platform,
        {"system", "machine"},
        description="replay_platform",
    )
    replay_platform = PlatformIdentity(
        system=_string(
            raw_platform["system"], description="replay_platform.system"
        ),
        machine=_string(
            raw_platform["machine"], description="replay_platform.machine"
        ),
    )

    raw_archive = _mapping(root["archive"], description="archive")
    _exact_keys(
        raw_archive,
        {"name", "size_bytes", "sha256", "root"},
        description="archive",
    )
    archive = ArchiveIdentity(
        name=_safe_component(raw_archive["name"], description="archive.name"),
        size_bytes=_integer(raw_archive["size_bytes"], description="archive.size_bytes"),
        sha256=_digest(raw_archive["sha256"], description="archive.sha256"),
        root=_safe_component(raw_archive["root"], description="archive.root"),
    )
    if not archive.name.endswith(".tar.zst"):
        _fail("archive.name must end in .tar.zst")

    raw_files = _mapping(root["files_manifest"], description="files_manifest")
    _exact_keys(
        raw_files,
        {"name", "size_bytes", "sha256", "entry_count"},
        description="files_manifest",
    )
    files_manifest = FilesManifestIdentity(
        name=_safe_component(raw_files["name"], description="files_manifest.name"),
        size_bytes=_integer(
            raw_files["size_bytes"], description="files_manifest.size_bytes"
        ),
        sha256=_digest(raw_files["sha256"], description="files_manifest.sha256"),
        entry_count=_integer(
            raw_files["entry_count"], description="files_manifest.entry_count"
        ),
    )
    if not files_manifest.name.endswith(".files.sha256"):
        _fail("files_manifest.name must end in .files.sha256")

    raw_run = _mapping(root["run_manifest"], description="run_manifest")
    _exact_keys(
        raw_run,
        {"path", "sha256", "run_id", "runs", "evaluation_units"},
        description="run_manifest",
    )
    run_id = _string(raw_run["run_id"], description="run_manifest.run_id")
    if _RUN_ID_RE.fullmatch(run_id) is None:
        _fail("run_manifest.run_id must be a lowercase 16-character digest")
    run_manifest = RunManifestIdentity(
        path=_relative_posix_path(raw_run["path"], description="run_manifest.path"),
        sha256=_digest(raw_run["sha256"], description="run_manifest.sha256"),
        run_id=run_id,
        runs=_integer(raw_run["runs"], description="run_manifest.runs"),
        evaluation_units=_integer(
            raw_run["evaluation_units"], description="run_manifest.evaluation_units"
        ),
    )
    return ReleaseDescriptor(
        schema_version=schema_version,
        release=release,
        repository=repository,
        tag=tag,
        source=source,
        runtime=runtime,
        toolchain=toolchain,
        replay_platform=replay_platform,
        archive=archive,
        files_manifest=files_manifest,
        run_manifest=run_manifest,
    )


def load_trusted_descriptor(release: str) -> ReleaseDescriptor:
    """Load one allowlisted descriptor; callers cannot supply a path or ref."""

    identity = _TRUSTED_DESCRIPTOR_FILES.get(release)
    if identity is None:
        _fail(f"release is not allowlisted: {release!r}")
    filename, expected_sha256 = identity
    path = _ROOT / "releases" / filename
    if not path.is_file() or path.is_symlink():
        _fail(f"trusted descriptor is missing or unsafe: {path}")
    if _sha256(path) != expected_sha256:
        _fail("trusted descriptor checksum mismatch")
    descriptor = parse_descriptor(_read_json(path, description="release descriptor"))
    if descriptor.release != release:
        _fail("allowlisted release and descriptor identity differ")
    if descriptor.repository != _TRUSTED_REPOSITORY:
        _fail("descriptor repository is not the trusted repository")
    return descriptor


def _verify_asset(path: Path, *, size_bytes: int, sha256: str, description: str) -> None:
    try:
        observed_size = path.stat().st_size
    except OSError as exc:
        raise ReleaseVerificationError(f"could not stat {description}: {path}") from exc
    if not path.is_file() or path.is_symlink():
        _fail(f"{description} is missing or unsafe")
    if observed_size != size_bytes:
        _fail(
            f"{description} size mismatch: expected {size_bytes}, got {observed_size}"
        )
    observed_digest = _sha256(path)
    if observed_digest != sha256:
        _fail(
            f"{description} SHA256 mismatch: expected {sha256}, got {observed_digest}"
        )


def _download_asset(
    descriptor: ReleaseDescriptor, name: str, size_bytes: int, destination: Path
) -> None:
    quoted_tag = urllib.parse.quote(descriptor.tag.name, safe="")
    quoted_name = urllib.parse.quote(name, safe="")
    url = (
        f"https://github.com/{descriptor.repository}/releases/download/"
        f"{quoted_tag}/{quoted_name}"
    )
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "agentic-tool-rl-release-verifier/1"},
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    received = 0
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            final_url = urllib.parse.urlparse(response.geturl())
            if final_url.scheme != "https":
                _fail("release asset redirected to a non-HTTPS URL")
            with destination.open("xb") as output:
                while chunk := response.read(1024 * 1024):
                    received += len(chunk)
                    if received > size_bytes or received > _MAX_DOWNLOAD_BYTES:
                        _fail("release asset exceeds its declared size")
                    output.write(chunk)
    except ReleaseVerificationError:
        raise
    except (OSError, urllib.error.URLError) as exc:
        raise ReleaseVerificationError(f"could not download fixed release asset {name}") from exc


def _parse_files_manifest(
    path: Path, identity: FilesManifestIdentity
) -> dict[str, str]:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise ReleaseVerificationError("files manifest is not valid UTF-8") from exc
    entries: dict[str, str] = {}
    for line_number, line in enumerate(text.splitlines(), start=1):
        match = _CHECKSUM_LINE_RE.fullmatch(line)
        if match is None:
            _fail(f"invalid files manifest line {line_number}")
        digest, raw_path = match.groups()
        canonical = _relative_posix_path(
            raw_path, description=f"files manifest path on line {line_number}"
        )
        if canonical in entries:
            _fail(f"duplicate files manifest path: {canonical}")
        entries[canonical] = digest
    if len(entries) != identity.entry_count:
        _fail(
            "files manifest entry count mismatch: "
            f"expected {identity.entry_count}, got {len(entries)}"
        )
    return entries


def _zstd_decompress(source: Path, destination: Path) -> None:
    executable = shutil.which("zstd")
    if executable is None:
        _fail("zstd is required to decompress the fixed release asset")
    destination.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    process: subprocess.Popen[bytes] | None = None
    try:
        with destination.open("xb") as output:
            process = subprocess.Popen(
                [executable, "--quiet", "--decompress", "--stdout", str(source.resolve())],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            if process.stdout is None or process.stderr is None:
                _fail("could not capture zstd output")
            while chunk := process.stdout.read(1024 * 1024):
                total += len(chunk)
                if total > _MAX_TAR_BYTES:
                    process.kill()
                    _fail("decompressed archive exceeds the safety limit")
                output.write(chunk)
            stderr = process.stderr.read().decode("utf-8", errors="replace")
            return_code = process.wait()
            if return_code != 0:
                _fail(f"zstd failed: {stderr[-500:].strip()}")
    except ReleaseVerificationError:
        if process is not None and process.poll() is None:
            process.kill()
            process.wait()
        destination.unlink(missing_ok=True)
        raise
    except OSError as exc:
        destination.unlink(missing_ok=True)
        raise ReleaseVerificationError("could not decompress release archive") from exc


def _allowed_archive_directories(root: str, files: Mapping[str, str]) -> set[str]:
    allowed = {root}
    for relative in files:
        path = PurePosixPath(root) / relative
        for parent in path.parents:
            if parent == PurePosixPath("."):
                break
            allowed.add(parent.as_posix())
    return allowed


def _safe_extract_tar(
    tar_path: Path,
    destination_parent: Path,
    *,
    root_name: str,
    expected_files: Mapping[str, str],
) -> Path:
    """Extract only the exact regular-file set declared by the checksum list."""

    expected_paths = set(expected_files)
    allowed_directories = _allowed_archive_directories(root_name, expected_files)
    seen_members: set[str] = set()
    file_members: dict[str, tarfile.TarInfo] = {}
    total_size = 0
    try:
        # Opening and entering are split so format errors become domain errors.
        bundle = tarfile.open(tar_path, mode="r:")  # noqa: SIM115
    except (OSError, tarfile.TarError) as exc:
        raise ReleaseVerificationError("decompressed payload is not a valid tar archive") from exc
    with bundle:
        try:
            members = bundle.getmembers()
        except (OSError, tarfile.TarError) as exc:
            raise ReleaseVerificationError("could not read tar member table") from exc
        if len(members) > _MAX_TAR_MEMBERS:
            _fail("tar archive contains too many members")
        for member in members:
            name = member.name
            path = PurePosixPath(name)
            if (
                path.is_absolute()
                or "\\" in name
                or path.as_posix() != name
                or any(part in {"", ".", ".."} for part in path.parts)
                or any(ord(character) < 32 for character in name)
            ):
                _fail(f"tar member is not a canonical relative path: {name!r}")
            if name in seen_members:
                _fail(f"tar archive contains duplicate member: {name}")
            seen_members.add(name)
            if not path.parts or path.parts[0] != root_name:
                _fail(f"tar member escapes the declared archive root: {name}")
            if member.issym() or member.islnk() or member.isdev() or member.isfifo():
                _fail(f"tar archive contains a link or device member: {name}")
            if member.isdir():
                if name not in allowed_directories:
                    _fail(f"tar archive contains an extra directory: {name}")
                continue
            if not member.isfile() or member.type not in {tarfile.REGTYPE, tarfile.AREGTYPE}:
                _fail(f"tar archive contains an unsupported member type: {name}")
            if len(path.parts) == 1:
                _fail("tar archive root cannot be a regular file")
            relative = PurePosixPath(*path.parts[1:]).as_posix()
            if relative not in expected_paths:
                _fail(f"tar archive contains an extra file: {relative}")
            if member.size < 0:
                _fail(f"tar member has a negative size: {relative}")
            total_size += member.size
            if total_size > _MAX_TAR_BYTES:
                _fail("tar archive expands beyond the safety limit")
            file_members[relative] = member
        observed_paths = set(file_members)
        if observed_paths != expected_paths:
            missing = sorted(expected_paths - observed_paths)
            extra = sorted(observed_paths - expected_paths)
            _fail(f"tar file set differs: missing={missing}, extra={extra}")

        destination_parent.mkdir(parents=True, exist_ok=True)
        if destination_parent.is_symlink():
            _fail("extraction destination must not be a symlink")
        root = destination_parent / root_name
        if root.exists() or root.is_symlink():
            _fail("archive extraction root already exists")
        root.mkdir(mode=0o700)
        for relative in sorted(file_members):
            member = file_members[relative]
            extracted = bundle.extractfile(member)
            if extracted is None:
                _fail(f"could not read tar member: {relative}")
            destination = root.joinpath(*PurePosixPath(relative).parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            digest = hashlib.sha256()
            observed_size = 0
            try:
                with extracted, destination.open("xb") as output:
                    while chunk := extracted.read(1024 * 1024):
                        observed_size += len(chunk)
                        digest.update(chunk)
                        output.write(chunk)
            except OSError as exc:
                raise ReleaseVerificationError(
                    f"could not safely extract tar member: {relative}"
                ) from exc
            if observed_size != member.size:
                _fail(f"tar member size changed while extracting: {relative}")
            if digest.hexdigest() != expected_files[relative]:
                _fail(f"tar member SHA256 mismatch: {relative}")
    return root


def _run_text(
    arguments: Sequence[str],
    *,
    cwd: Path | None = None,
    env: Mapping[str, str] | None = None,
    timeout: int = 600,
    description: str,
) -> str:
    try:
        result = subprocess.run(
            list(arguments),
            cwd=cwd,
            env=None if env is None else dict(env),
            text=True,
            capture_output=True,
            check=False,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ReleaseVerificationError(f"could not execute {description}") from exc
    if result.returncode != 0:
        detail = (result.stderr or result.stdout)[-1000:].strip()
        _fail(f"{description} failed with exit {result.returncode}: {detail}")
    return result.stdout.strip()


def _checkout_exact_tag(
    descriptor: ReleaseDescriptor, destination: Path, *, repository: str
) -> Path:
    if destination.exists() or destination.is_symlink():
        _fail("tag checkout destination already exists")
    _run_text(
        ["git", "init", "--quiet", str(destination)],
        description="git init",
    )
    _run_text(
        ["git", "-C", str(destination), "remote", "add", "origin", repository],
        description="git remote add",
    )
    tag_ref = f"refs/tags/{descriptor.tag.name}"
    _run_text(
        [
            "git",
            "-C",
            str(destination),
            "fetch",
            "--quiet",
            "--no-tags",
            "--depth=1",
            "origin",
            f"{tag_ref}:{tag_ref}",
        ],
        timeout=600,
        description="fetch fixed annotated tag",
    )
    object_type = _run_text(
        ["git", "-C", str(destination), "cat-file", "-t", tag_ref],
        description="read tag object type",
    )
    if object_type != "tag":
        _fail("release tag is not an annotated tag object")
    object_oid = _run_text(
        ["git", "-C", str(destination), "rev-parse", "--verify", tag_ref],
        description="resolve annotated tag object",
    )
    if object_oid != descriptor.tag.object_oid:
        _fail(
            "annotated tag object mismatch: "
            f"expected {descriptor.tag.object_oid}, got {object_oid}"
        )
    commit_oid = _run_text(
        ["git", "-C", str(destination), "rev-parse", "--verify", f"{tag_ref}^{{commit}}"],
        description="peel annotated tag",
    )
    if commit_oid != descriptor.tag.commit_oid:
        _fail(
            f"tag commit mismatch: expected {descriptor.tag.commit_oid}, got {commit_oid}"
        )
    _run_text(
        [
            "git",
            "-C",
            str(destination),
            "checkout",
            "--quiet",
            "--detach",
            descriptor.tag.commit_oid,
        ],
        description="checkout exact verifier commit",
    )
    head = _run_text(
        ["git", "-C", str(destination), "rev-parse", "--verify", "HEAD"],
        description="read detached verifier HEAD",
    )
    if head != descriptor.tag.commit_oid:
        _fail("detached verifier checkout is not the declared commit")
    status = _run_text(
        ["git", "-C", str(destination), "status", "--porcelain", "--untracked-files=all"],
        description="inspect verifier checkout",
    )
    if status:
        _fail("detached verifier checkout is not clean")
    observed_source = _source_fingerprint_v1(destination)
    if observed_source != descriptor.source.sha256:
        _fail(
            "tagged source fingerprint mismatch: "
            f"expected {descriptor.source.sha256}, got {observed_source}"
        )
    return destination


def _source_fingerprint_v1(root: Path) -> str:
    digest = hashlib.sha256()
    files = (
        sorted((root / "src").rglob("*.py"))
        + sorted((root / "configs").glob("*.yaml"))
        + [
            path
            for name in ("pyproject.toml", "uv.lock")
            if (path := root / name).is_file()
        ]
    )
    for path in files:
        if path.is_symlink():
            _fail(f"source fingerprint input must not be a symlink: {path}")
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _preflight_run_manifest(path: Path, descriptor: ReleaseDescriptor) -> None:
    _verify_asset(
        path,
        size_bytes=path.stat().st_size,
        sha256=descriptor.run_manifest.sha256,
        description="run manifest",
    )
    payload = _mapping(_read_json(path, description="run manifest"), description="run manifest")
    if payload.get("schema_version") != "3.0":
        _fail("run manifest schema differs from the registered release")
    if payload.get("source_sha256") != descriptor.source.sha256:
        _fail("run manifest source identity differs from the descriptor")
    if payload.get("runtime_identity") != descriptor.runtime.to_dict():
        _fail("run manifest runtime identity differs from the descriptor")
    if payload.get("run_id") != descriptor.run_manifest.run_id:
        _fail("run manifest run_id differs from the descriptor")
    if payload.get("expected_evaluation_units") != descriptor.run_manifest.evaluation_units:
        _fail("run manifest evaluation-unit count differs from the descriptor")
    runs = payload.get("runs")
    if not isinstance(runs, list) or len(runs) != descriptor.run_manifest.runs:
        _fail("run manifest run count differs from the descriptor")


_RUNTIME_PROBE = """
import json
import platform
import sys
import torch
print(json.dumps({
    "python_implementation": platform.python_implementation(),
    "python_version": platform.python_version(),
    "python_cache_tag": sys.implementation.cache_tag,
    "torch_version": str(torch.__version__),
    "torch_cuda_version": torch.version.cuda,
}, sort_keys=True))
""".strip()


def _run_tagged_verifier(
    descriptor: ReleaseDescriptor,
    source: Path,
    manifest: Path,
    environment: Path,
    *,
    uv_executable: str,
    base_environment: Mapping[str, str] | None = None,
) -> dict[str, object]:
    env = dict(os.environ if base_environment is None else base_environment)
    env["UV_MANAGED_PYTHON"] = "1"
    env["UV_PROJECT_ENVIRONMENT"] = str(environment.resolve())
    uv_version_output = _run_text(
        [uv_executable, "--version"],
        cwd=source,
        env=env,
        description="inspect uv version",
    )
    version_match = _UV_VERSION_RE.fullmatch(uv_version_output)
    observed_uv_version = None if version_match is None else version_match.group(1)
    if observed_uv_version != descriptor.toolchain.uv_version:
        _fail(
            "uv version mismatch: "
            f"expected {descriptor.toolchain.uv_version}, got {uv_version_output!r}"
        )
    _run_text(
        [uv_executable, "python", "install", descriptor.runtime.python_version],
        cwd=source,
        env=env,
        timeout=1200,
        description="install exact managed Python",
    )
    uv_prefix = [
        uv_executable,
        "run",
        "--project",
        str(source),
        "--locked",
        "--python",
        descriptor.runtime.python_version,
        "--managed-python",
    ]
    runtime_stdout = _run_text(
        [*uv_prefix, "python", "-c", _RUNTIME_PROBE],
        cwd=source,
        env=env,
        timeout=1800,
        description="probe tagged verifier runtime",
    )
    observed_runtime = _read_json_bytes(
        runtime_stdout.encode("utf-8"), description="runtime probe output"
    )
    if observed_runtime != descriptor.runtime.to_dict():
        _fail(
            "tagged verifier runtime mismatch: "
            f"expected {descriptor.runtime.to_dict()}, got {observed_runtime}"
        )
    verifier_stdout = _run_text(
        [
            *uv_prefix,
            "agentic-tool-rl",
            "verify-run",
            "--manifest",
            str(manifest.resolve()),
        ],
        cwd=source,
        env=env,
        timeout=7200,
        description="tagged checkpoint-bound verifier",
    )
    result = _mapping(
        _read_json_bytes(
            verifier_stdout.encode("utf-8"), description="tagged verifier output"
        ),
        description="tagged verifier output",
    )
    _exact_keys(
        result,
        {"passed", "runs", "evaluation_units", "checked"},
        description="tagged verifier output",
    )
    if result["passed"] is not True:
        _fail("tagged verifier did not report passed=true")
    if result["runs"] != descriptor.run_manifest.runs:
        _fail("tagged verifier returned the wrong run count")
    if result["evaluation_units"] != descriptor.run_manifest.evaluation_units:
        _fail("tagged verifier returned the wrong evaluation-unit count")
    checked = result["checked"]
    if not isinstance(checked, list) or len(checked) != descriptor.run_manifest.runs:
        _fail("tagged verifier returned incomplete checked-run evidence")
    return dict(result)


def _verify_descriptor(
    descriptor: ReleaseDescriptor,
    workspace: Path,
    *,
    repository: str,
    fetch_asset: AssetFetcher,
    decompress: Decompressor,
    uv_executable: str,
    base_environment: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Internal orchestration seam used by local, network-free tests."""

    observed_platform = PlatformIdentity(
        system=host_platform.system(),
        machine=host_platform.machine(),
    )
    if observed_platform != descriptor.replay_platform:
        _fail(
            "replay platform mismatch: "
            f"expected {descriptor.replay_platform}, got {observed_platform}"
        )
    if workspace.exists() and any(workspace.iterdir()):
        _fail("release verification workspace must be empty")
    workspace.mkdir(parents=True, exist_ok=True)
    downloads = workspace / "downloads"
    downloads.mkdir()
    archive_path = downloads / descriptor.archive.name
    files_manifest_path = downloads / descriptor.files_manifest.name
    fetch_asset(descriptor.archive.name, descriptor.archive.size_bytes, archive_path)
    fetch_asset(
        descriptor.files_manifest.name,
        descriptor.files_manifest.size_bytes,
        files_manifest_path,
    )
    _verify_asset(
        archive_path,
        size_bytes=descriptor.archive.size_bytes,
        sha256=descriptor.archive.sha256,
        description="release archive",
    )
    _verify_asset(
        files_manifest_path,
        size_bytes=descriptor.files_manifest.size_bytes,
        sha256=descriptor.files_manifest.sha256,
        description="files manifest",
    )
    expected_files = _parse_files_manifest(files_manifest_path, descriptor.files_manifest)
    registered_run_digest = expected_files.get(descriptor.run_manifest.path)
    if registered_run_digest != descriptor.run_manifest.sha256:
        _fail("files manifest does not bind the registered run manifest")

    tar_path = workspace / "evidence.tar"
    decompress(archive_path, tar_path)
    if not tar_path.is_file() or tar_path.is_symlink():
        _fail("decompressor did not produce a safe regular tar file")
    evidence_root = _safe_extract_tar(
        tar_path,
        workspace / "evidence",
        root_name=descriptor.archive.root,
        expected_files=expected_files,
    )
    run_manifest_path = evidence_root.joinpath(
        *PurePosixPath(descriptor.run_manifest.path).parts
    )
    _preflight_run_manifest(run_manifest_path, descriptor)
    source = _checkout_exact_tag(
        descriptor,
        workspace / "verifier-source",
        repository=repository,
    )
    result = _run_tagged_verifier(
        descriptor,
        source,
        run_manifest_path,
        workspace / "verifier-environment",
        uv_executable=uv_executable,
        base_environment=base_environment,
    )
    return {
        "schema_version": "release-verification-result-v1",
        "release": descriptor.release,
        "tag_object_oid": descriptor.tag.object_oid,
        "commit_oid": descriptor.tag.commit_oid,
        "source_sha256": descriptor.source.sha256,
        "archive_sha256": descriptor.archive.sha256,
        "run_manifest_sha256": descriptor.run_manifest.sha256,
        "run_id": descriptor.run_manifest.run_id,
        "passed": result["passed"],
        "runs": result["runs"],
        "evaluation_units": result["evaluation_units"],
    }


def verify_trusted_release(release: str) -> dict[str, object]:
    descriptor = load_trusted_descriptor(release)
    uv_executable = shutil.which("uv")
    if uv_executable is None:
        _fail("uv is required to recreate the tagged verifier runtime")
    repository = f"https://github.com/{descriptor.repository}.git"
    with tempfile.TemporaryDirectory(prefix="agentic-tool-rl-release-") as temporary:
        return _verify_descriptor(
            descriptor,
            Path(temporary),
            repository=repository,
            fetch_asset=lambda name, size, destination: _download_asset(
                descriptor, name, size, destination
            ),
            decompress=_zstd_decompress,
            uv_executable=uv_executable,
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify allowlisted release evidence with its exact tagged verifier."
    )
    parser.add_argument(
        "--release",
        required=True,
        choices=tuple(sorted(_TRUSTED_DESCRIPTOR_FILES)),
        help="Allowlisted evidence release.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        result = verify_trusted_release(arguments.release)
    except ReleaseVerificationError as exc:
        print(f"release verification failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
