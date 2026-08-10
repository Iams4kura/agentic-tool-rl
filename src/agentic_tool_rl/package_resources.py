"""Exact package-resource resolvers used by installed CLI and manifests."""

from __future__ import annotations

import hashlib
import json
from importlib import resources
from importlib.resources.abc import Traversable
from typing import Any, Literal, TypeAlias

PackagedProtocolName: TypeAlias = Literal["benchmark-v1.4-development.json"]
PACKAGED_PROTOCOL_NAMES: frozenset[str] = frozenset(
    {"benchmark-v1.4-development.json"}
)
IMPLEMENTATION_FINGERPRINT_SCHEMA = "implementation-fingerprint-v2"
_IMPLEMENTATION_SUFFIXES = (".py", ".typed", ".yaml", ".json")


def _implementation_members(
    directory: Traversable,
    *,
    prefix: str,
) -> list[tuple[str, bytes]]:
    members: list[tuple[str, bytes]] = []
    for child in sorted(directory.iterdir(), key=lambda item: item.name):
        if child.name == "__pycache__":
            continue
        relative = f"{prefix}/{child.name}" if prefix else child.name
        if child.is_dir():
            members.extend(_implementation_members(child, prefix=relative))
        elif child.name.endswith(_IMPLEMENTATION_SUFFIXES):
            members.append((relative, child.read_bytes()))
    return members


def implementation_fingerprint_members() -> tuple[str, ...]:
    """Return the canonical behavior-affecting member list for audit evidence."""

    package = resources.files("agentic_tool_rl")
    return tuple(
        name
        for name, _ in _implementation_members(
            package,
            prefix="agentic_tool_rl",
        )
    )


def implementation_fingerprint_v2() -> str:
    """Hash behavior code and built-in resources independent of install layout."""

    package = resources.files("agentic_tool_rl")
    digest = hashlib.sha256()
    for name, payload in _implementation_members(package, prefix="agentic_tool_rl"):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(payload)
        digest.update(b"\0")
    return digest.hexdigest()


def packaged_protocol_bytes(name: PackagedProtocolName | str) -> bytes:
    """Return exact protocol bytes from the installed distribution."""

    if name not in PACKAGED_PROTOCOL_NAMES:
        raise ValueError(f"unknown packaged protocol {name!r}")
    resource = resources.files("agentic_tool_rl").joinpath(
        "resources", "protocols", name
    )
    try:
        return resource.read_bytes()
    except (FileNotFoundError, ModuleNotFoundError) as exc:
        raise RuntimeError(f"required packaged protocol {name!r} is missing") from exc


def packaged_protocol_sha256(name: PackagedProtocolName | str) -> str:
    """Hash the exact protocol bytes that a generated manifest must bind."""

    return hashlib.sha256(packaged_protocol_bytes(name)).hexdigest()


def load_packaged_protocol(name: PackagedProtocolName | str) -> dict[str, Any]:
    """Decode one packaged protocol while retaining byte-level hash access."""

    try:
        value = json.loads(packaged_protocol_bytes(name))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"packaged protocol {name!r} is invalid JSON") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"packaged protocol {name!r} must contain a JSON object")
    return value


__all__ = [
    "IMPLEMENTATION_FINGERPRINT_SCHEMA",
    "PACKAGED_PROTOCOL_NAMES",
    "PackagedProtocolName",
    "implementation_fingerprint_members",
    "implementation_fingerprint_v2",
    "load_packaged_protocol",
    "packaged_protocol_bytes",
    "packaged_protocol_sha256",
]
