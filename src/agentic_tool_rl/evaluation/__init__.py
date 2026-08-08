"""Evaluation, evidence-based consistency recomputation, and claim verification."""

from agentic_tool_rl.evaluation.action_validity import (
    CANONICAL_ACTION_VALIDITY_COUNT,
    ActionValidityEvaluation,
    ActionValidityPrediction,
    evaluate_action_validity,
)
from agentic_tool_rl.evaluation.bootstrap import (
    cluster_bootstrap,
    paired_seed_case_bootstrap_tsr_difference,
)
from agentic_tool_rl.evaluation.claim_gate import (
    CANONICAL_SEED_COUNT,
    CANONICAL_TEST_CASE_COUNT,
    CANONICAL_VARIANT_DEFINITIONS,
    CanonicalClaimEvidence,
    ClaimCheckItem,
    ClaimGateResult,
    verify_canonical_claims,
    write_claim_check,
)
from agentic_tool_rl.evaluation.integrity import (
    CaseIdManifestError,
    ResumeGuard,
    ResumeIntegrityError,
    RunInputHashes,
    assert_exact_case_ids,
    build_case_id_manifest,
    canonical_run_signature,
    file_sha256,
    model_parameter_digest,
    verify_case_id_manifest,
)
from agentic_tool_rl.evaluation.io import read_jsonl, write_json_atomic
from agentic_tool_rl.evaluation.metrics import (
    EvaluationMetrics,
    TraceSchemaError,
    compute_metrics,
    write_metrics,
)
from agentic_tool_rl.evaluation.recompute import (
    MetricDifference,
    RecomputeResult,
    diff_metrics,
    recompute_metrics,
)
from agentic_tool_rl.evaluation.trace_store import (
    CorruptTraceStoreError,
    TraceConflictError,
    TraceStore,
)

__all__ = [
    "CANONICAL_ACTION_VALIDITY_COUNT",
    "CANONICAL_SEED_COUNT",
    "CANONICAL_TEST_CASE_COUNT",
    "CANONICAL_VARIANT_DEFINITIONS",
    "ActionValidityEvaluation",
    "ActionValidityPrediction",
    "CanonicalClaimEvidence",
    "CaseIdManifestError",
    "ClaimCheckItem",
    "ClaimGateResult",
    "CorruptTraceStoreError",
    "EvaluationMetrics",
    "MetricDifference",
    "RecomputeResult",
    "ResumeGuard",
    "ResumeIntegrityError",
    "RunInputHashes",
    "TraceConflictError",
    "TraceSchemaError",
    "TraceStore",
    "assert_exact_case_ids",
    "build_case_id_manifest",
    "canonical_run_signature",
    "cluster_bootstrap",
    "compute_metrics",
    "diff_metrics",
    "evaluate_action_validity",
    "file_sha256",
    "model_parameter_digest",
    "paired_seed_case_bootstrap_tsr_difference",
    "read_jsonl",
    "recompute_metrics",
    "verify_canonical_claims",
    "verify_case_id_manifest",
    "write_claim_check",
    "write_json_atomic",
    "write_metrics",
]
