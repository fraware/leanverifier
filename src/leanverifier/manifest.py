"""Load and validate model/request manifests against schemas + semantic rules."""

from __future__ import annotations

import json
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

_SCHEMA_DIR = Path(__file__).resolve().parents[2] / "schemas"


def _load_schema(name: str) -> dict[str, Any]:
    path = _SCHEMA_DIR / name
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise InvalidInputError(f"schema root must be object: {name}")
    return data


_MODEL_VALIDATOR = Draft202012Validator(_load_schema("model.schema.json"))
_REQUEST_VALIDATOR = Draft202012Validator(_load_schema("request.schema.json"))
_RESULT_VALIDATOR = Draft202012Validator(_load_schema("result.schema.json"))


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


def parse_model(obj: Any) -> ModelManifest:
    if not isinstance(obj, dict):
        raise InvalidInputError("model must be an object")
    _schema_errors(_MODEL_VALIDATOR, obj)
    dim = obj["dimension"]
    weights_s = obj["weights"]
    if len(weights_s) != dim:
        raise InvalidInputError("weights length must equal dimension")
    weights = [parse_rational(w) for w in weights_s]
    bias = parse_rational(obj["bias"])
    digest = digest_obj(obj)
    return ModelManifest(dim, weights, bias, obj, digest)


def parse_request(obj: Any) -> RequestManifest:
    if not isinstance(obj, dict):
        raise InvalidInputError("request must be an object")
    _schema_errors(_REQUEST_VALIDATOR, obj)
    dim = obj["dimension"]
    xs = obj["reference_input"]
    if len(xs) != dim:
        raise InvalidInputError("reference_input length must equal dimension")
    reference = [parse_rational(x) for x in xs]
    eps = parse_rational(obj["epsilon"], nonnegative=True)
    digest = digest_obj(obj)
    return RequestManifest(dim, reference, eps, obj["model_digest"], obj, digest)


def load_model_path(path: Path) -> ModelManifest:
    raw = path.read_bytes()
    return parse_model(loads_strict_json(raw))


def load_request_path(path: Path) -> RequestManifest:
    raw = path.read_bytes()
    return parse_request(loads_strict_json(raw))


def validate_result_obj(obj: Any) -> None:
    _schema_errors(_RESULT_VALIDATOR, obj)
