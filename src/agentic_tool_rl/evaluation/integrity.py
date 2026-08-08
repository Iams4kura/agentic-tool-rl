"""Cryptographic evidence identities and fail-closed resume protection."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable, Mapping
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import torch

from agentic_tool_rl.evaluation.io import canonical_json, sha256_json

_DIGEST_KEYS = (
    "benchmark_sha256",
    "config_sha256",
    "source_sha256",
    "checkpoint_sha256",
)
_HEX_DIGITS = frozenset("0123456789abcdef")
_RESUME_SCHEMA = "resume-guard-v1"
_CASE_MANIFEST_SCHEMA = "case-id-manifest-v1"


class ResumeIntegrityError(RuntimeError):
    """Raised when prior traces cannot be proven to belong to this run."""


class CaseIdManifestError(ValueError):
    """Raised when a case-id manifest is incomplete, extra, or corrupted."""


def _validate_sha256(value: object, *, field_name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _HEX_DIGITS for character in value)
    ):
        raise ValueError(f"{field_name} must be a lowercase 64-character SHA256 digest")
    return value


def file_sha256(path: str | Path) -> str:
    """Return the SHA256 of a regular file using bounded memory."""

    source = Path(path)
    if not source.is_file():
        raise ValueError(f"not a regular file: {source}")
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_parameter_digest(model: torch.nn.Module) -> str:
    """Hash parameter names, metadata, and raw values in canonical name order.

    Buffers and optimizer state are intentionally excluded: this digest answers
    the narrow question "are these model parameters byte-for-byte identical?".
    A checkpoint file identity remains separately covered by ``file_sha256``.
    """

    digest = hashlib.sha256()
    named_parameters = sorted(model.named_parameters(remove_duplicate=False))
    for name, parameter in named_parameters:
        tensor = parameter.detach()
        if tensor.device.type == "meta":
            raise ValueError(f"cannot digest meta parameter {name!r}")
        if tensor.layout != torch.strided:
            raise ValueError(f"cannot digest non-strided parameter {name!r}")
        metadata = canonical_json(
            {
                "name": name,
                "dtype": str(tensor.dtype),
                "shape": list(tensor.shape),
                "requires_grad": parameter.requires_grad,
            }
        ).encode("utf-8")
        raw = (
            tensor.resolve_conj()
            .resolve_neg()
            .cpu()
            .contiguous()
            .reshape(-1)
            .view(torch.uint8)
            .numpy()
            .tobytes(order="C")
        )
        digest.update(len(metadata).to_bytes(8, "big"))
        digest.update(metadata)
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class RunInputHashes:
    """The four immutable identities required to resume an evaluation run."""

    benchmark_sha256: str
    config_sha256: str
    source_sha256: str
    checkpoint_sha256: str

    def __post_init__(self) -> None:
        for name in _DIGEST_KEYS:
            _validate_sha256(getattr(self, name), field_name=name)

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> RunInputHashes:
        keys = set(value)
        expected = set(_DIGEST_KEYS)
        if keys != expected:
            missing = sorted(expected - keys)
            extra = sorted(keys - expected)
            raise ValueError(f"run hash keys differ: missing={missing}, extra={extra}")
        return cls(
            benchmark_sha256=_validate_sha256(
                value["benchmark_sha256"], field_name="benchmark_sha256"
            ),
            config_sha256=_validate_sha256(
                value["config_sha256"], field_name="config_sha256"
            ),
            source_sha256=_validate_sha256(
                value["source_sha256"], field_name="source_sha256"
            ),
            checkpoint_sha256=_validate_sha256(
                value["checkpoint_sha256"], field_name="checkpoint_sha256"
            ),
        )

    def to_dict(self) -> dict[str, str]:
        return {name: getattr(self, name) for name in _DIGEST_KEYS}


def _coerce_run_hashes(value: RunInputHashes | Mapping[str, object]) -> RunInputHashes:
    return value if isinstance(value, RunInputHashes) else RunInputHashes.from_mapping(value)


def canonical_run_signature(value: RunInputHashes | Mapping[str, object]) -> str:
    """Return a stable full-length signature over all resume-critical hashes."""

    hashes = _coerce_run_hashes(value)
    return sha256_json(
        {
            "schema_version": _RESUME_SCHEMA,
            "input_hashes": hashes.to_dict(),
        }
    )


def _resume_payload(hashes: RunInputHashes) -> dict[str, Any]:
    return {
        "schema_version": _RESUME_SCHEMA,
        "input_hashes": hashes.to_dict(),
        "run_signature": canonical_run_signature(hashes),
    }


class ResumeGuard:
    """Authorize a fresh trace or prove that a resumed trace has identical inputs.

    Call :meth:`authorize` before constructing or opening the trace store.  A
    pre-existing non-empty trace without a guard is rejected because its origin
    cannot be established retroactively.
    """

    def __init__(self, manifest_path: str | Path) -> None:
        self.manifest_path = Path(manifest_path)

    def verify(self, current: RunInputHashes | Mapping[str, object]) -> str:
        hashes = _coerce_run_hashes(current)
        expected = _resume_payload(hashes)
        try:
            observed = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise ResumeIntegrityError(f"resume guard is missing: {self.manifest_path}") from exc
        except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ResumeIntegrityError(
                f"resume guard is unreadable or corrupt: {self.manifest_path}"
            ) from exc
        if not isinstance(observed, dict):
            raise ResumeIntegrityError("resume guard must contain a JSON object")
        if observed != expected:
            observed_hashes = observed.get("input_hashes")
            mismatches: list[str] = []
            if isinstance(observed_hashes, dict):
                for name in _DIGEST_KEYS:
                    if observed_hashes.get(name) != getattr(hashes, name):
                        mismatches.append(name)
            if observed.get("schema_version") != _RESUME_SCHEMA:
                mismatches.append("schema_version")
            if observed.get("run_signature") != expected["run_signature"]:
                mismatches.append("run_signature")
            unexpected = set(observed) - set(expected)
            if unexpected:
                mismatches.append("unexpected_fields")
            detail = ", ".join(dict.fromkeys(mismatches)) or "manifest_structure"
            raise ResumeIntegrityError(f"resume evidence mismatch: {detail}")
        return str(expected["run_signature"])

    def authorize(
        self,
        current: RunInputHashes | Mapping[str, object],
        *,
        trace_paths: Iterable[str | Path] = (),
    ) -> str:
        """Create a guard for a new run or verify it before trace resume."""

        hashes = _coerce_run_hashes(current)
        if self.manifest_path.exists():
            return self.verify(hashes)

        non_empty_traces: list[str] = []
        for trace_like in trace_paths:
            trace_path = Path(trace_like)
            if trace_path.exists() and not trace_path.is_file():
                raise ResumeIntegrityError(f"trace path is not a file: {trace_path}")
            if trace_path.is_file() and trace_path.stat().st_size > 0:
                non_empty_traces.append(str(trace_path))
        if non_empty_traces:
            raise ResumeIntegrityError(
                "cannot resume non-empty traces without prior integrity evidence: "
                f"{sorted(non_empty_traces)}"
            )

        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        payload = (canonical_json(_resume_payload(hashes)) + "\n").encode("utf-8")
        temporary_path: Path | None = None
        try:
            with NamedTemporaryFile(
                dir=self.manifest_path.parent,
                prefix=f".{self.manifest_path.name}.",
                delete=False,
            ) as handle:
                temporary_path = Path(handle.name)
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            with suppress(FileExistsError):
                # Linking a complete temporary file gives us atomic create-if-
                # absent semantics: a concurrent, different run cannot replace
                # evidence after another run has already been authorized.
                os.link(temporary_path, self.manifest_path)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
        return self.verify(hashes)


def _normalize_case_ids(case_ids: Iterable[str], *, name: str) -> tuple[str, ...]:
    if isinstance(case_ids, str):
        raise ValueError(f"{name} must be an iterable of case ids, not a string")
    values = tuple(case_ids)
    if any(not isinstance(case_id, str) or not case_id for case_id in values):
        raise ValueError(f"{name} must contain non-empty strings")
    if len(values) != len(set(values)):
        raise ValueError(f"{name} contains duplicate case_id values")
    return tuple(sorted(values))


def assert_exact_case_ids(
    expected_case_ids: Iterable[str],
    observed_case_ids: Iterable[str],
) -> tuple[str, ...]:
    """Fail unless observed IDs are exactly the expected set."""

    expected = _normalize_case_ids(expected_case_ids, name="expected_case_ids")
    observed = _normalize_case_ids(observed_case_ids, name="observed_case_ids")
    missing = sorted(set(expected) - set(observed))
    extra = sorted(set(observed) - set(expected))
    if missing or extra:
        raise CaseIdManifestError(f"case_id set mismatch: missing={missing}, extra={extra}")
    return expected


def build_case_id_manifest(case_ids: Iterable[str]) -> dict[str, Any]:
    """Build a canonical manifest that includes every expected case id."""

    normalized = _normalize_case_ids(case_ids, name="case_ids")
    identity = {
        "schema_version": _CASE_MANIFEST_SCHEMA,
        "count": len(normalized),
        "case_ids": list(normalized),
    }
    return {**identity, "sha256": sha256_json(identity)}


def verify_case_id_manifest(
    manifest: Mapping[str, object],
    expected_case_ids: Iterable[str],
) -> tuple[str, ...]:
    """Verify manifest integrity and exact expected/observed set equality."""

    expected_keys = {"schema_version", "count", "case_ids", "sha256"}
    if set(manifest) != expected_keys:
        missing_keys = sorted(expected_keys - set(manifest))
        extra_keys = sorted(set(manifest) - expected_keys)
        raise CaseIdManifestError(
            f"case manifest keys differ: missing={missing_keys}, extra={extra_keys}"
        )
    if manifest.get("schema_version") != _CASE_MANIFEST_SCHEMA:
        raise CaseIdManifestError("unsupported case-id manifest schema")
    raw_case_ids = manifest.get("case_ids")
    if not isinstance(raw_case_ids, list):
        raise CaseIdManifestError("case manifest case_ids must be a list")
    try:
        observed = _normalize_case_ids(raw_case_ids, name="manifest.case_ids")
    except ValueError as exc:
        raise CaseIdManifestError(str(exc)) from exc
    if list(observed) != raw_case_ids:
        raise CaseIdManifestError("case manifest case_ids are not in canonical order")
    if manifest.get("count") != len(observed):
        raise CaseIdManifestError("case manifest count does not match case_ids")
    identity = {
        "schema_version": _CASE_MANIFEST_SCHEMA,
        "count": len(observed),
        "case_ids": list(observed),
    }
    expected_digest = sha256_json(identity)
    if manifest.get("sha256") != expected_digest:
        raise CaseIdManifestError("case manifest SHA256 mismatch")
    return assert_exact_case_ids(expected_case_ids, observed)


__all__ = [
    "CaseIdManifestError",
    "ResumeGuard",
    "ResumeIntegrityError",
    "RunInputHashes",
    "assert_exact_case_ids",
    "build_case_id_manifest",
    "canonical_run_signature",
    "file_sha256",
    "model_parameter_digest",
    "verify_case_id_manifest",
]
