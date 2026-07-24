"""Verification result records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from leanverifier.manifest import validate_result_obj

THEOREM = "LeanVerifier.AffineBinaryClassifier.affine_margin_robustness"

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
    toolchain: str
    theorem: str = THEOREM
    message: str = ""
    generated_module: str | None = None
    schema_version: str = "leanverifier.result.v1"

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        if data["generated_module"] is None:
            del data["generated_module"]
        if not data["message"]:
            del data["message"]
        validate_result_obj(data)
        return data
