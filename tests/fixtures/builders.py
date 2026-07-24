"""Shared fixture builders for tests."""

from __future__ import annotations

from fractions import Fraction
from typing import Any

from leanverifier.canonicalize import canonicalize_rational, digest_obj


def make_model(
    weights: list[Fraction | str | int],
    bias: Fraction | str | int = 0,
    *,
    comment: str | None = None,
) -> dict[str, Any]:
    w = [canonicalize_rational(Fraction(x)) for x in weights]
    obj: dict[str, Any] = {
        "schema_version": "leanverifier.model.v1",
        "model_kind": "affine_binary",
        "dimension": len(w),
        "weights": w,
        "bias": canonicalize_rational(Fraction(bias)),
    }
    if comment is not None:
        obj["comment"] = comment
    return obj


def make_request(
    model_obj: dict[str, Any],
    reference: list[Fraction | str | int],
    epsilon: Fraction | str | int,
) -> dict[str, Any]:
    xs = [canonicalize_rational(Fraction(x)) for x in reference]
    if len(xs) != model_obj["dimension"]:
        raise ValueError("dim mismatch")
    return {
        "schema_version": "leanverifier.request.v1",
        "model_digest": digest_obj(model_obj),
        "dimension": model_obj["dimension"],
        "reference_input": xs,
        "epsilon": canonicalize_rational(Fraction(epsilon)),
    }
