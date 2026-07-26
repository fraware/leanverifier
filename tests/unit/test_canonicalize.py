from __future__ import annotations

from fractions import Fraction

import pytest

from leanverifier.canonicalize import (
    canonicalize_rational,
    digest_obj,
    parse_rational,
)
from leanverifier.errors import InvalidInputError

pytestmark = pytest.mark.unit


def test_canonicalize_basics() -> None:
    assert canonicalize_rational(Fraction(2, 4)) == "1/2"
    assert canonicalize_rational(Fraction(0, 5)) == "0"
    assert canonicalize_rational(Fraction(3, 1)) == "3"


def test_canonicalize_preserves_large_denominators() -> None:
    """F-009: never approximate via limit_denominator; denom > 1e6 stays exact."""
    huge = Fraction(1, 10**7 + 3)  # > 1_000_000; not reduced away
    assert huge.denominator > 1_000_000
    text = canonicalize_rational(huge)
    assert text == f"1/{huge.denominator}"
    assert parse_rational(text) == huge
    # Round-trip through Fraction without approximation.
    assert canonicalize_rational(Fraction(text)) == text


def test_parse_rejects_noncanonical() -> None:
    with pytest.raises(InvalidInputError):
        parse_rational("2/4")
    with pytest.raises(InvalidInputError):
        parse_rational("3.0")
    with pytest.raises(InvalidInputError):
        parse_rational("1/0")
    with pytest.raises(InvalidInputError):
        parse_rational("+1")


def test_digest_key_reorder_stable() -> None:
    a = {"b": 1, "a": 2, "schema_version": "x"}
    b = {"a": 2, "b": 1, "schema_version": "x"}
    assert digest_obj(a) == digest_obj(b)


def test_digest_weight_reorder_changes() -> None:
    a = {"weights": ["1", "2"], "bias": "0"}
    b = {"weights": ["2", "1"], "bias": "0"}
    assert digest_obj(a) != digest_obj(b)
