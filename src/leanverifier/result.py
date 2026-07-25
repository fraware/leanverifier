"""Verification result records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

from leanverifier.manifest import validate_result_obj

THEOREM = "LeanVerifier.AffineBinaryClassifier.affine_margin_robustness"
CLAIM = "local_linf_binary_classification_robustness"

AxiomAuditStatus = Literal["passed", "failed", "not-run"]

STATUS_TO_EXIT = {
    "verified": 0,
    "not-certified": 2,
    "invalid-input": 3,
    "build-failure": 4,
    "axiom-failure": 5,
    "digest-mismatch": 6,
    "internal-error": 7,
    "timeout": 8,
}


@dataclass
class VerificationResult:
    status: str
    exit_code: int
    model_digest: str
    request_digest: str
    lean_toolchain: str
    mathlib_revision: str
    generator_version: str
    evidence_bundle_digest: str
    build_exit_code: int | None = None
    axiom_audit_exit_code: int | None = None
    axiom_audit: AxiomAuditStatus = "not-run"
    claim: str = CLAIM
    theorem: str = THEOREM
    message: str = ""
    generated_module: str | None = None
    generated_module_digest: str | None = None
    duration_ms: int | None = None
    schema_version: str = "leanverifier.result.v1"

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        for key in (
            "generated_module",
            "generated_module_digest",
            "duration_ms",
        ):
            if data[key] is None:
                del data[key]
        if not data["message"]:
            del data["message"]
        validate_result_obj(data)
        expected = STATUS_TO_EXIT.get(data["status"])
        if expected is None or data["exit_code"] != expected:
            raise ValueError(
                f"status/exit_code inconsistency: {data['status']!r} vs {data['exit_code']}"
            )
        return data
