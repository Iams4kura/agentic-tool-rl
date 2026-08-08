from __future__ import annotations

import copy
import hashlib
import io
import json
import os
import platform
import shutil
import subprocess
import sys
import tarfile
from dataclasses import dataclass, replace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import verify_release as verifier


def _run_git(repository: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        text=True,
        capture_output=True,
    )
    return result.stdout.strip()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _write_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value)


def _regular_member(name: str, value: bytes) -> tuple[tarfile.TarInfo, bytes]:
    member = tarfile.TarInfo(name)
    member.size = len(value)
    member.mode = 0o644
    return member, value


def _special_member(
    name: str, member_type: bytes, *, linkname: str = ""
) -> tuple[tarfile.TarInfo, bytes]:
    member = tarfile.TarInfo(name)
    member.type = member_type
    member.linkname = linkname
    member.size = 0
    if member_type == tarfile.CHRTYPE:
        member.devmajor = 1
        member.devminor = 3
    return member, b""


def _write_tar(path: Path, members: list[tuple[tarfile.TarInfo, bytes]]) -> None:
    with tarfile.open(path, mode="w:") as bundle:
        for member, value in members:
            bundle.addfile(member, io.BytesIO(value) if member.isfile() else None)


def _create_source_repository(root: Path) -> tuple[Path, str, str, str]:
    repository = root / "source-repository"
    repository.mkdir()
    _run_git(repository, "init", "--quiet")
    _run_git(repository, "config", "user.name", "Release Test")
    _run_git(repository, "config", "user.email", "release-test@example.invalid")
    _write_bytes(repository / "src/fake_policy.py", b'IDENTITY = "tagged"\n')
    _write_bytes(repository / "configs/cpu.yaml", b"mode: tagged\n")
    _write_bytes(repository / "pyproject.toml", b"[project]\nname='fake-verifier'\n")
    _write_bytes(repository / "uv.lock", b"version = 1\n")
    _run_git(repository, "add", ".")
    _run_git(repository, "commit", "--quiet", "-m", "tagged verifier")
    _run_git(repository, "tag", "-a", "v0.1.0", "-m", "fixed evidence verifier")
    tag_object = _run_git(repository, "rev-parse", "refs/tags/v0.1.0")
    tag_commit = _run_git(repository, "rev-parse", "refs/tags/v0.1.0^{commit}")
    source_sha256 = verifier._source_fingerprint_v1(repository)

    _write_bytes(repository / "src/fake_policy.py", b'IDENTITY = "evolved-main"\n')
    _run_git(repository, "add", "src/fake_policy.py")
    _run_git(repository, "commit", "--quiet", "-m", "evolve main")
    assert _run_git(repository, "rev-parse", "HEAD") != tag_commit
    return repository, tag_object, tag_commit, source_sha256


def _create_fake_uv(root: Path) -> Path:
    executable = root / "fake-bin/uv"
    source = """#!/usr/bin/env python3
import json
import os
import sys

args = sys.argv[1:]
with open(os.environ["FAKE_UV_LOG"], "a", encoding="utf-8") as handle:
    handle.write(json.dumps({
        "args": args,
        "cwd": os.getcwd(),
        "managed": os.environ.get("UV_MANAGED_PYTHON"),
        "environment": os.environ.get("UV_PROJECT_ENVIRONMENT"),
    }, sort_keys=True) + "\\n")
if args == ["--version"]:
    print(os.environ["FAKE_UV_VERSION"])
    raise SystemExit(0)
if args[:2] == ["python", "install"]:
    raise SystemExit(0)
if "python" in args and "-c" in args:
    print(os.environ["FAKE_RUNTIME_JSON"])
    raise SystemExit(0)
if "agentic-tool-rl" in args and "verify-run" in args:
    print(os.environ["FAKE_RESULT_JSON"])
    raise SystemExit(0)
print("unexpected fake uv invocation", file=sys.stderr)
raise SystemExit(9)
"""
    _write_bytes(executable, source.encode("utf-8"))
    executable.chmod(0o755)
    return executable


@dataclass(frozen=True)
class FakeRelease:
    descriptor: verifier.ReleaseDescriptor
    repository: Path
    assets: Path
    uv_executable: Path
    environment: dict[str, str]
    log_path: Path

    def fetch_asset(self, name: str, size_bytes: int, destination: Path) -> None:
        del size_bytes
        shutil.copyfile(self.assets / name, destination)


@pytest.fixture
def fake_release(tmp_path: Path) -> FakeRelease:
    repository, tag_object, tag_commit, source_sha256 = _create_source_repository(
        tmp_path
    )
    runtime = {
        "python_implementation": "CPython",
        "python_version": "3.11.15",
        "python_cache_tag": "cpython-311",
        "torch_version": "2.13.0",
        "torch_cuda_version": None,
    }
    run_manifest = {
        "schema_version": "3.0",
        "source_sha256": source_sha256,
        "runtime_identity": runtime,
        "run_id": "0123456789abcdef",
        "expected_evaluation_units": 4,
        "runs": [{"run": 1}, {"run": 2}],
    }
    evidence_files = {
        "payload.txt": b"frozen evidence\n",
        "run-manifest.json": (
            json.dumps(run_manifest, sort_keys=True, separators=(",", ":")) + "\n"
        ).encode("utf-8"),
    }
    checksums = {
        name: _sha256_bytes(value) for name, value in evidence_files.items()
    }
    files_manifest = "".join(
        f"{checksums[name]}  ./{name}\n" for name in sorted(checksums)
    ).encode("utf-8")
    assets = tmp_path / "assets"
    assets.mkdir()
    archive_name = "fake-evidence.tar.zst"
    archive = assets / archive_name
    _write_tar(
        archive,
        [
            _regular_member(f"fake-evidence/{name}", evidence_files[name])
            for name in sorted(evidence_files)
        ],
    )
    files_name = "fake-evidence.files.sha256"
    files_path = assets / files_name
    files_path.write_bytes(files_manifest)
    descriptor_value = {
        "schema_version": "release-evidence-v1",
        "release": "v0.1.0",
        "repository": "local/fake-release",
        "tag": {
            "name": "v0.1.0",
            "object_oid": tag_object,
            "commit_oid": tag_commit,
        },
        "source": {
            "fingerprint_schema": "source-fingerprint-v1",
            "sha256": source_sha256,
        },
        "runtime": runtime,
        "toolchain": {"uv_version": "0.11.29"},
        "replay_platform": {
            "system": platform.system(),
            "machine": platform.machine(),
        },
        "archive": {
            "name": archive_name,
            "size_bytes": archive.stat().st_size,
            "sha256": verifier._sha256(archive),
            "root": "fake-evidence",
        },
        "files_manifest": {
            "name": files_name,
            "size_bytes": files_path.stat().st_size,
            "sha256": verifier._sha256(files_path),
            "entry_count": len(checksums),
        },
        "run_manifest": {
            "path": "run-manifest.json",
            "sha256": checksums["run-manifest.json"],
            "run_id": run_manifest["run_id"],
            "runs": 2,
            "evaluation_units": 4,
        },
    }
    descriptor = verifier.parse_descriptor(descriptor_value)
    uv_executable = _create_fake_uv(tmp_path)
    log_path = tmp_path / "fake-uv.jsonl"
    environment = dict(os.environ)
    environment["FAKE_UV_LOG"] = str(log_path)
    environment["FAKE_UV_VERSION"] = "uv 0.11.29 (fake build)"
    environment["FAKE_RUNTIME_JSON"] = json.dumps(runtime, sort_keys=True)
    environment["FAKE_RESULT_JSON"] = json.dumps(
        {
            "passed": True,
            "runs": 2,
            "evaluation_units": 4,
            "checked": [{"run": 1}, {"run": 2}],
        },
        sort_keys=True,
    )
    return FakeRelease(
        descriptor=descriptor,
        repository=repository,
        assets=assets,
        uv_executable=uv_executable,
        environment=environment,
        log_path=log_path,
    )


def _copy_tar(source: Path, destination: Path) -> None:
    shutil.copyfile(source, destination)


def _verify_fake(
    fake: FakeRelease,
    workspace: Path,
    *,
    descriptor: verifier.ReleaseDescriptor | None = None,
    environment: dict[str, str] | None = None,
) -> dict[str, object]:
    return verifier._verify_descriptor(
        fake.descriptor if descriptor is None else descriptor,
        workspace,
        repository=str(fake.repository),
        fetch_asset=fake.fetch_asset,
        decompress=_copy_tar,
        uv_executable=str(fake.uv_executable),
        base_environment=fake.environment if environment is None else environment,
    )


def test_production_descriptor_is_strict_and_binds_every_release_identity() -> None:
    descriptor = verifier.load_trusted_descriptor("v0.1.0")

    assert descriptor.repository == "Iams4kura/agentic-tool-rl"
    assert descriptor.tag.object_oid == "98945d0a0dca8f3d14f6854ccb19b87e4bf11f9e"
    assert descriptor.tag.commit_oid == "7c3fe39c8bcad902d05bddcdd9273e9848f0507f"
    assert descriptor.source.sha256 == (
        "75eb32b7a94178616eaa977f0c42f147143af904fd09d69928cde000bbfe40d3"
    )
    assert descriptor.runtime.python_version == "3.11.15"
    assert descriptor.runtime.torch_version == "2.13.0"
    assert descriptor.toolchain.uv_version == "0.11.29"
    assert descriptor.replay_platform == verifier.PlatformIdentity(
        system="Darwin", machine="arm64"
    )
    assert descriptor.archive.sha256 == (
        "0de48d58521aebadef8abdf990616c5dc5787de56c0f27a4d9c9bfa9033ff56f"
    )
    assert descriptor.run_manifest.sha256 == (
        "5153952e0e7cef78da1338a37b0b57b53b75e7b38174d4d5584cafabeca38b19"
    )
    assert descriptor.run_manifest.runs == 30
    assert descriptor.run_manifest.evaluation_units == 30_000


@pytest.mark.parametrize(
    ("section", "field"),
    [
        (None, "unexpected"),
        ("tag", "ref"),
        ("source", "commit"),
        ("runtime", "platform"),
        ("toolchain", "installer"),
        ("replay_platform", "kernel"),
        ("archive", "url"),
        ("files_manifest", "skip_hash"),
        ("run_manifest", "ignore_source"),
    ],
)
def test_descriptor_rejects_extra_fields(section: str | None, field: str) -> None:
    value = json.loads(
        (verifier._ROOT / "releases/v0.1.0.evidence.json").read_text(encoding="utf-8")
    )
    target = value if section is None else value[section]
    target[field] = "forbidden"

    with pytest.raises(verifier.ReleaseVerificationError, match="extra"):
        verifier.parse_descriptor(value)


def test_descriptor_rejects_duplicate_json_fields() -> None:
    with pytest.raises(verifier.ReleaseVerificationError, match="duplicate field"):
        verifier._read_json_bytes(
            b'{"schema_version":"one","schema_version":"two"}',
            description="duplicate fixture",
        )


@pytest.mark.parametrize(
    "forbidden_argument",
    ["--ref", "--url", "--repository", "--ignore-source", "--ignore-runtime", "--skip-hash"],
)
def test_cli_has_no_free_ref_url_or_bypass_inputs(forbidden_argument: str) -> None:
    with pytest.raises(SystemExit):
        verifier._parser().parse_args(
            ["--release", "v0.1.0", forbidden_argument, "forbidden"]
        )


def test_safe_extract_accepts_exact_regular_file_set(tmp_path: Path) -> None:
    value = b"bound evidence"
    expected = {"payload.txt": _sha256_bytes(value)}
    archive = tmp_path / "good.tar"
    _write_tar(archive, [_regular_member("root/payload.txt", value)])

    root = verifier._safe_extract_tar(
        archive,
        tmp_path / "output",
        root_name="root",
        expected_files=expected,
    )

    assert (root / "payload.txt").read_bytes() == value


@pytest.mark.parametrize(
    ("case", "members", "expected", "message"),
    [
        (
            "absolute",
            [_regular_member("/absolute.txt", b"x")],
            {"payload.txt": _sha256_bytes(b"x")},
            "canonical relative path",
        ),
        (
            "parent",
            [_regular_member("root/../escape.txt", b"x")],
            {"payload.txt": _sha256_bytes(b"x")},
            "canonical relative path",
        ),
        (
            "symlink",
            [_special_member("root/payload.txt", tarfile.SYMTYPE, linkname="target")],
            {"payload.txt": _sha256_bytes(b"x")},
            "link or device",
        ),
        (
            "hardlink",
            [_special_member("root/payload.txt", tarfile.LNKTYPE, linkname="root/target")],
            {"payload.txt": _sha256_bytes(b"x")},
            "link or device",
        ),
        (
            "device",
            [_special_member("root/payload.txt", tarfile.CHRTYPE)],
            {"payload.txt": _sha256_bytes(b"x")},
            "link or device",
        ),
        (
            "duplicate",
            [
                _regular_member("root/payload.txt", b"x"),
                _regular_member("root/payload.txt", b"x"),
            ],
            {"payload.txt": _sha256_bytes(b"x")},
            "duplicate member",
        ),
        (
            "extra-file",
            [
                _regular_member("root/payload.txt", b"x"),
                _regular_member("root/extra.txt", b"y"),
            ],
            {"payload.txt": _sha256_bytes(b"x")},
            "extra file",
        ),
        (
            "missing-file",
            [_regular_member("root/payload.txt", b"x")],
            {
                "payload.txt": _sha256_bytes(b"x"),
                "missing.txt": _sha256_bytes(b"y"),
            },
            "missing=",
        ),
        (
            "extra-directory",
            [
                _regular_member("root/payload.txt", b"x"),
                _special_member("root/unused", tarfile.DIRTYPE),
            ],
            {"payload.txt": _sha256_bytes(b"x")},
            "extra directory",
        ),
    ],
)
def test_safe_extract_rejects_unsafe_or_non_exact_archives(
    tmp_path: Path,
    case: str,
    members: list[tuple[tarfile.TarInfo, bytes]],
    expected: dict[str, str],
    message: str,
) -> None:
    archive = tmp_path / f"{case}.tar"
    _write_tar(archive, members)

    with pytest.raises(verifier.ReleaseVerificationError, match=message):
        verifier._safe_extract_tar(
            archive,
            tmp_path / f"output-{case}",
            root_name="root",
            expected_files=expected,
        )
    assert not (tmp_path / "escape.txt").exists()
    assert not (tmp_path / "absolute.txt").exists()


def test_files_manifest_rejects_duplicate_paths(tmp_path: Path) -> None:
    digest = _sha256_bytes(b"value")
    path = tmp_path / "duplicate.files.sha256"
    path.write_text(f"{digest}  ./same.txt\n{digest}  ./same.txt\n", encoding="utf-8")
    identity = verifier.FilesManifestIdentity(
        name="duplicate.files.sha256",
        size_bytes=path.stat().st_size,
        sha256=verifier._sha256(path),
        entry_count=2,
    )

    with pytest.raises(verifier.ReleaseVerificationError, match="duplicate"):
        verifier._parse_files_manifest(path, identity)


def test_full_local_flow_uses_tagged_source_after_main_evolves(
    fake_release: FakeRelease, tmp_path: Path
) -> None:
    result = _verify_fake(fake_release, tmp_path / "verification")

    assert result["passed"] is True
    assert result["runs"] == 2
    assert result["evaluation_units"] == 4
    checkout = tmp_path / "verification/verifier-source"
    assert _run_git(checkout, "rev-parse", "HEAD") == fake_release.descriptor.tag.commit_oid
    assert (checkout / "src/fake_policy.py").read_text(encoding="utf-8") == (
        'IDENTITY = "tagged"\n'
    )
    assert (fake_release.repository / "src/fake_policy.py").read_text(
        encoding="utf-8"
    ) == 'IDENTITY = "evolved-main"\n'
    invocations = [
        json.loads(line)
        for line in fake_release.log_path.read_text(encoding="utf-8").splitlines()
    ]
    assert invocations[0]["args"] == ["--version"]
    assert invocations[1]["args"] == ["python", "install", "3.11.15"]
    assert all(invocation["managed"] == "1" for invocation in invocations)
    run_invocations = [item for item in invocations if item["args"][0] == "run"]
    assert len(run_invocations) == 2
    for invocation in run_invocations:
        arguments = invocation["args"]
        assert "--locked" in arguments
        assert arguments[arguments.index("--python") + 1] == "3.11.15"
        assert "--managed-python" in arguments
        assert Path(invocation["cwd"]) == checkout
    verifier_arguments = run_invocations[-1]["args"]
    assert "agentic-tool-rl" in verifier_arguments
    assert "verify-run" in verifier_arguments
    assert all("ignore" not in argument for argument in verifier_arguments)
    assert all("skip" not in argument for argument in verifier_arguments)


def test_archive_tamper_fails_before_decompression_or_execution(
    fake_release: FakeRelease, tmp_path: Path
) -> None:
    archive = fake_release.assets / fake_release.descriptor.archive.name
    archive.write_bytes(archive.read_bytes() + b"tamper")
    decompressed = False

    def forbidden_decompress(source: Path, destination: Path) -> None:
        del source, destination
        nonlocal decompressed
        decompressed = True

    with pytest.raises(verifier.ReleaseVerificationError, match="size mismatch"):
        verifier._verify_descriptor(
            fake_release.descriptor,
            tmp_path / "tampered",
            repository=str(fake_release.repository),
            fetch_asset=fake_release.fetch_asset,
            decompress=forbidden_decompress,
            uv_executable=str(fake_release.uv_executable),
            base_environment=fake_release.environment,
        )
    assert decompressed is False
    assert not fake_release.log_path.exists()


def test_replay_platform_mismatch_fails_before_workspace_or_download(
    fake_release: FakeRelease, tmp_path: Path
) -> None:
    wrong_platform = verifier.PlatformIdentity(
        system=f"not-{platform.system()}",
        machine=platform.machine(),
    )
    descriptor = replace(fake_release.descriptor, replay_platform=wrong_platform)
    workspace = tmp_path / "wrong-platform"

    with pytest.raises(verifier.ReleaseVerificationError, match="platform mismatch"):
        _verify_fake(fake_release, workspace, descriptor=descriptor)

    assert not workspace.exists()
    assert not fake_release.log_path.exists()


def test_moved_annotated_tag_object_is_rejected(
    fake_release: FakeRelease, tmp_path: Path
) -> None:
    wrong_tag = replace(fake_release.descriptor.tag, object_oid="0" * 40)
    descriptor = replace(fake_release.descriptor, tag=wrong_tag)

    with pytest.raises(verifier.ReleaseVerificationError, match="tag object mismatch"):
        _verify_fake(fake_release, tmp_path / "wrong-tag", descriptor=descriptor)
    assert not fake_release.log_path.exists()


def test_wrong_peeled_commit_is_rejected(
    fake_release: FakeRelease, tmp_path: Path
) -> None:
    wrong_tag = replace(fake_release.descriptor.tag, commit_oid="0" * 40)
    descriptor = replace(fake_release.descriptor, tag=wrong_tag)

    with pytest.raises(verifier.ReleaseVerificationError, match="tag commit mismatch"):
        _verify_fake(fake_release, tmp_path / "wrong-commit", descriptor=descriptor)
    assert not fake_release.log_path.exists()


def test_runtime_mismatch_fails_before_inner_verifier(
    fake_release: FakeRelease, tmp_path: Path
) -> None:
    environment = dict(fake_release.environment)
    wrong_runtime = fake_release.descriptor.runtime.to_dict()
    wrong_runtime["python_version"] = "3.11.16"
    environment["FAKE_RUNTIME_JSON"] = json.dumps(wrong_runtime, sort_keys=True)

    with pytest.raises(verifier.ReleaseVerificationError, match="runtime mismatch"):
        _verify_fake(
            fake_release,
            tmp_path / "wrong-runtime",
            environment=environment,
        )
    invocations = [
        json.loads(line)
        for line in fake_release.log_path.read_text(encoding="utf-8").splitlines()
    ]
    assert len(invocations) == 3
    assert all("agentic-tool-rl" not in item["args"] for item in invocations)


def test_uv_version_mismatch_fails_before_environment_creation(
    fake_release: FakeRelease, tmp_path: Path
) -> None:
    environment = dict(fake_release.environment)
    environment["FAKE_UV_VERSION"] = "uv 0.12.0 (unexpected)"

    with pytest.raises(verifier.ReleaseVerificationError, match="uv version mismatch"):
        _verify_fake(
            fake_release,
            tmp_path / "wrong-uv",
            environment=environment,
        )
    invocations = [
        json.loads(line)
        for line in fake_release.log_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [item["args"] for item in invocations] == [["--version"]]


def test_inner_verifier_wrong_counts_are_rejected(
    fake_release: FakeRelease, tmp_path: Path
) -> None:
    environment = dict(fake_release.environment)
    environment["FAKE_RESULT_JSON"] = json.dumps(
        {
            "passed": True,
            "runs": 1,
            "evaluation_units": 4,
            "checked": [{"run": 1}],
        },
        sort_keys=True,
    )

    with pytest.raises(verifier.ReleaseVerificationError, match="wrong run count"):
        _verify_fake(
            fake_release,
            tmp_path / "wrong-counts",
            environment=environment,
        )


def test_trusted_loader_rejects_unknown_release() -> None:
    with pytest.raises(verifier.ReleaseVerificationError, match="not allowlisted"):
        verifier.load_trusted_descriptor("v9.9.9")


def test_descriptor_copy_cannot_add_a_download_url() -> None:
    value = json.loads(
        (verifier._ROOT / "releases/v0.1.0.evidence.json").read_text(encoding="utf-8")
    )
    forged = copy.deepcopy(value)
    forged["archive"]["url"] = "https://example.invalid/malicious"

    with pytest.raises(verifier.ReleaseVerificationError, match="extra"):
        verifier.parse_descriptor(forged)
