from __future__ import annotations

import hashlib
import importlib
import sys
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
distribution = importlib.import_module("scripts.check_distribution")


def _write_wheel(path: Path, members: dict[str, bytes]) -> None:
    with zipfile.ZipFile(path, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, payload in members.items():
            archive.writestr(name, payload)


def test_wheel_resource_contract_requires_every_default_config(tmp_path: Path) -> None:
    wheel = tmp_path / "demo.whl"
    _write_wheel(wheel, {"agentic_tool_rl/__init__.py": b""})

    with pytest.raises(distribution.DistributionCheckError, match="packaged config"):
        distribution.validate_packaged_configs(wheel)

    members = {
        "agentic_tool_rl/__init__.py": b"",
        **{
            f"agentic_tool_rl/resources/configs/{name}": f"name: {name}\n".encode()
            for name in distribution.REQUIRED_PACKAGED_CONFIGS
        },
    }
    _write_wheel(wheel, members)
    distribution.validate_packaged_configs(wheel)


def test_wheel_resource_contract_requires_v14_protocol(tmp_path: Path) -> None:
    wheel = tmp_path / "demo.whl"
    _write_wheel(wheel, {"agentic_tool_rl/__init__.py": b""})

    with pytest.raises(distribution.DistributionCheckError, match="packaged protocol"):
        distribution.validate_packaged_protocols(wheel)

    _write_wheel(
        wheel,
        {
            "agentic_tool_rl/__init__.py": b"",
            "agentic_tool_rl/resources/protocols/benchmark-v1.4-development.json": b"{}\n",
        },
    )
    distribution.validate_packaged_protocols(wheel)


def test_wheel_runtime_requirements_come_from_dist_info_metadata(tmp_path: Path) -> None:
    wheel = tmp_path / "demo.whl"
    _write_wheel(
        wheel,
        {
            "demo/__init__.py": b"",
            "demo-1.0.dist-info/METADATA": (
                b"Metadata-Version: 2.4\n"
                b"Name: demo\n"
                b"Version: 1.0\n"
                b"Requires-Dist: numpy>=1.26,<3\n"
                b"Requires-Dist: torch>=2.3,<3\n"
                b"\n"
            ),
        },
    )

    assert distribution.wheel_runtime_requirements(wheel) == (
        "numpy>=1.26,<3",
        "torch>=2.3,<3",
    )


def test_locked_runtime_constraints_are_exported_from_uv_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "constraints.txt"
    commands: list[list[str]] = []

    def fake_run_checked(
        command: list[str],
        *,
        cwd: Path,
        env: dict[str, str],
        description: str,
    ) -> str:
        del cwd, env, description
        commands.append(command)
        output.write_text("numpy==2.4.6\ntorch==2.13.0\n", encoding="utf-8")
        return ""

    monkeypatch.setattr(distribution, "_run_checked", fake_run_checked)
    digest = distribution._export_locked_runtime_constraints(
        uv="uv",
        root=tmp_path,
        output=output,
        env={},
    )

    assert commands == [
        [
            "uv",
            "export",
            "--project",
            str(tmp_path),
            "--locked",
            "--no-dev",
            "--no-emit-project",
            "--no-hashes",
            "--no-annotate",
            "--no-header",
            "--no-sources",
            "--output-file",
            str(output),
        ]
    ]
    assert digest == hashlib.sha256(output.read_bytes()).hexdigest()


def test_isolation_probe_rejects_host_site_and_pth_injection(tmp_path: Path) -> None:
    venv = tmp_path / "venv"
    venv_site = venv / "lib" / "python3.11" / "site-packages"
    valid = {
        "user_site_enabled": False,
        "site_packages": [str(venv_site)],
        "sys_path": [str(venv_site)],
        "pth_entries": [],
    }
    distribution.validate_isolated_site_probe(valid, venv=venv)

    host_site = tmp_path / "host" / "site-packages"
    with pytest.raises(distribution.DistributionCheckError, match="escapes"):
        distribution.validate_isolated_site_probe(
            {**valid, "sys_path": [str(venv_site), str(host_site)]},
            venv=venv,
        )

    with pytest.raises(distribution.DistributionCheckError, match=r"\.pth target"):
        distribution.validate_isolated_site_probe(
            {
                **valid,
                "pth_entries": [
                    {
                        "path": str(venv_site / "host-injection.pth"),
                        "entries": [str(host_site)],
                    }
                ],
            },
            venv=venv,
        )
