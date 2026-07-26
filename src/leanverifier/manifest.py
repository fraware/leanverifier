"""Load and validate model/request manifests against schemas + semantic rules."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from leanverifier.canonicalize import (
    digest_obj,
    loads_strict_json,
    parse_rational,
)
from leanverifier.errors import InvalidInputError
from leanverifier.limits import (
    CODE_RESOURCE_COMMENT_CHARS,
    CODE_RESOURCE_DIMENSION,
    CODE_RESOURCE_MANIFEST_BYTES,
    MAX_COMMENT_CHARS,
    MAX_DIMENSION,
    MAX_MANIFEST_BYTES,
)
from leanverifier.resources import load_schema_json

_MODEL_VALIDATOR = Draft202012Validator(load_schema_json("model.schema.json"))
_REQUEST_VALIDATOR = Draft202012Validator(load_schema_json("request.schema.json"))
_RESULT_VALIDATOR = Draft202012Validator(load_schema_json("result.schema.json"))


@dataclass(frozen=True)
class ModelManifest:
    dimension: int
    weights: list[Fraction]
    bias: Fraction
    raw: dict[str, Any]
    digest: str


@dataclass(frozen=True)
class RequestManifest:
    dimension: int
    reference_input: list[Fraction]
    epsilon: Fraction
    model_digest: str
    raw: dict[str, Any]
    digest: str


def _schema_errors(validator: Draft202012Validator, obj: Any) -> None:
    errors = sorted(validator.iter_errors(obj), key=lambda e: list(e.path))
    if errors:
        raise InvalidInputError(errors[0].message)


def _check_dimension(dim: Any) -> int:
    if not isinstance(dim, int) or isinstance(dim, bool):
        raise InvalidInputError("dimension must be an integer")
    if dim < 1:
        raise InvalidInputError(
            "dimension must be >= 1",
            code=CODE_RESOURCE_DIMENSION,
        )
    if dim > MAX_DIMENSION:
        raise InvalidInputError(
            f"dimension exceeds maximum {MAX_DIMENSION}",
            code=CODE_RESOURCE_DIMENSION,
        )
    return dim


def _check_comment(obj: dict[str, Any]) -> None:
    comment = obj.get("comment")
    if comment is None:
        return
    if not isinstance(comment, str):
        raise InvalidInputError("comment must be a string")
    if len(comment) > MAX_COMMENT_CHARS:
        raise InvalidInputError(
            f"comment exceeds {MAX_COMMENT_CHARS} characters",
            code=CODE_RESOURCE_COMMENT_CHARS,
        )


def _read_manifest_bytes(path: Path) -> bytes:
    """Read manifest bytes only after enforcing the size bound (F-011)."""
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise InvalidInputError(f"cannot read manifest: {path}") from exc
    if size > MAX_MANIFEST_BYTES:
        raise InvalidInputError(
            f"manifest exceeds {MAX_MANIFEST_BYTES} bytes",
            code=CODE_RESOURCE_MANIFEST_BYTES,
        )
    raw = path.read_bytes()
    if len(raw) > MAX_MANIFEST_BYTES:
        raise InvalidInputError(
            f"manifest exceeds {MAX_MANIFEST_BYTES} bytes",
            code=CODE_RESOURCE_MANIFEST_BYTES,
        )
    return raw


def parse_model(obj: Any) -> ModelManifest:
    if not isinstance(obj, dict):
        raise InvalidInputError("model must be an object")
    _schema_errors(_MODEL_VALIDATOR, obj)
    _check_comment(obj)
    dim = _check_dimension(obj["dimension"])
    weights_s = obj["weights"]
    if len(weights_s) != dim:
        raise InvalidInputError("weights length must equal dimension")
    if len(weights_s) == 0:
        raise InvalidInputError("weights must be non-empty", code=CODE_RESOURCE_DIMENSION)
    weights = [parse_rational(w) for w in weights_s]
    bias = parse_rational(obj["bias"])
    digest = digest_obj(obj)
    return ModelManifest(dim, weights, bias, obj, digest)


def parse_request(obj: Any) -> RequestManifest:
    if not isinstance(obj, dict):
        raise InvalidInputError("request must be an object")
    _schema_errors(_REQUEST_VALIDATOR, obj)
    _check_comment(obj)
    dim = _check_dimension(obj["dimension"])
    xs = obj["reference_input"]
    if len(xs) != dim:
        raise InvalidInputError("reference_input length must equal dimension")
    if len(xs) == 0:
        raise InvalidInputError(
            "reference_input must be non-empty",
            code=CODE_RESOURCE_DIMENSION,
        )
    reference = [parse_rational(x) for x in xs]
    eps = parse_rational(obj["epsilon"], nonnegative=True)
    digest = digest_obj(obj)
    return RequestManifest(dim, reference, eps, obj["model_digest"], obj, digest)


def load_model_path(path: Path) -> ModelManifest:
    raw = _read_manifest_bytes(path)
    return parse_model(loads_strict_json(raw))


def load_request_path(path: Path) -> RequestManifest:
    raw = _read_manifest_bytes(path)
    return parse_request(loads_strict_json(raw))


def validate_result_obj(obj: Any) -> None:
    _schema_errors(_RESULT_VALIDATOR, obj)
