"""Canonical rational strings and JSON byte digests."""

from __future__ import annotations

import hashlib
import json
import re
from fractions import Fraction
from typing import Any

from leanverifier.errors import InvalidInputError
from leanverifier.limits import (
    CODE_RESOURCE_RATIONAL_CHARS,
    CODE_UTF8_INVALID,
    MAX_RATIONAL_CHARS,
)

_RAT_RE = re.compile(r"^(0|-?[1-9][0-9]*|-?[1-9][0-9]*/[1-9][0-9]*)$")
_NONNEG_RAT_RE = re.compile(r"^(0|[1-9][0-9]*|[1-9][0-9]*/[1-9][0-9]*)$")


def parse_rational(text: str, *, nonnegative: bool = False) -> Fraction:
    if not isinstance(text, str):
        raise InvalidInputError("rational must be a string")
    if len(text) > MAX_RATIONAL_CHARS:
        raise InvalidInputError(
            f"rational exceeds {MAX_RATIONAL_CHARS} characters",
            code=CODE_RESOURCE_RATIONAL_CHARS,
        )
    pattern = _NONNEG_RAT_RE if nonnegative else _RAT_RE
    if not pattern.fullmatch(text):
        raise InvalidInputError(f"non-canonical or illegal rational: {text!r}")
    value = Fraction(text)
    if nonnegative and value < 0:
        raise InvalidInputError(f"expected nonnegative rational: {text!r}")
    # Reject non-canonical representations that still match a looser parse
    if canonicalize_rational(value) != text:
        raise InvalidInputError(f"rational not in canonical form: {text!r}")
    return value


def canonicalize_rational(value: Fraction) -> str:
    # Exact reduction only — never approximate via limit_denominator (F-009).
    value = Fraction(value)
    if value == 0:
        return "0"
    num, den = value.numerator, value.denominator
    if den == 1:
        return str(num)
    return f"{num}/{den}"


def _sort_keys(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _sort_keys(obj[k]) for k in sorted(obj)}
    if isinstance(obj, list):
        return [_sort_keys(x) for x in obj]
    return obj


def _reject_non_canonical_json_values(obj: Any, path: str = "$") -> None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            _reject_non_canonical_json_values(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            _reject_non_canonical_json_values(v, f"{path}[{i}]")
    elif isinstance(obj, float):
        raise InvalidInputError(f"JSON floats are forbidden at {path}")
    elif isinstance(obj, bool):
        return
    elif isinstance(obj, int):
        # dimension fields may be ints; rationals must be strings
        return


def loads_strict_json(raw: str | bytes) -> Any:
    if isinstance(raw, bytes):
        try:
            raw = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise InvalidInputError("input must be UTF-8", code=CODE_UTF8_INVALID) from exc
    try:
        obj = json.loads(raw, object_pairs_hook=_reject_duplicates)
    except json.JSONDecodeError as exc:
        raise InvalidInputError(f"invalid JSON: {exc}") from exc
    _reject_non_canonical_json_values(obj)
    return obj


def _reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in pairs:
        if k in out:
            raise InvalidInputError(f"duplicate JSON key: {k!r}")
        out[k] = v
    return out


def canonical_json_bytes(obj: Any) -> bytes:
    sorted_obj = _sort_keys(obj)
    text = json.dumps(
        sorted_obj,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return text.encode("utf-8")


def digest_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def digest_obj(obj: Any) -> str:
    return digest_bytes(canonical_json_bytes(obj))


def l1_norm(weights: list[Fraction]) -> Fraction:
    return sum((abs(w) for w in weights), Fraction(0))


def score(weights: list[Fraction], bias: Fraction, x: list[Fraction]) -> Fraction:
    if len(weights) != len(x):
        raise InvalidInputError("dimension mismatch in score")
    return sum((w * xi for w, xi in zip(weights, x, strict=True)), bias)


def classify(weights: list[Fraction], bias: Fraction, x: list[Fraction]) -> bool:
    """Strict positive score classification (matches Lean ``classify``)."""
    return score(weights, bias, x) > 0


def linf_ball(x: list[Fraction], eps: Fraction, y: list[Fraction]) -> bool:
    """True iff ``y`` lies in the closed L∞ ball of radius ``eps`` around ``x``."""
    if len(x) != len(y):
        raise InvalidInputError("dimension mismatch in linf_ball")
    if eps < 0:
        raise InvalidInputError("epsilon must be nonnegative")
    return all(abs(yi - xi) <= eps for xi, yi in zip(x, y, strict=True))


def margin_ok(weights: list[Fraction], bias: Fraction, x: list[Fraction], eps: Fraction) -> bool:
    if eps < 0:
        raise InvalidInputError("epsilon must be nonnegative")
    return abs(score(weights, bias, x)) > eps * l1_norm(weights)


def module_token(digest: str) -> str:
    """Derive 16 hex chars from a sha256 digest string."""
    if not digest.startswith("sha256:") or len(digest) != 7 + 64:
        raise InvalidInputError("bad digest for module token")
    return digest[7:23]
