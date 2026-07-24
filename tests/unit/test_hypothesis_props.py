from __future__ import annotations

from fractions import Fraction

from hypothesis import given, settings
from hypothesis import strategies as st

from leanverifier.canonicalize import canonicalize_rational, l1_norm, margin_ok, score


@given(
    weights=st.lists(st.fractions(max_denominator=20), min_size=0, max_size=4),
    bias=st.fractions(max_denominator=20),
    x=st.lists(st.fractions(max_denominator=20), min_size=0, max_size=4),
    eps=st.fractions(min_value=0, max_denominator=20),
)
@settings(max_examples=50)
def test_margin_implies_lipschitz_room(weights, bias, x, eps) -> None:  # noqa: ANN001
    if len(weights) != len(x):
        return
    if margin_ok(weights, bias, x, eps):
        assert abs(score(weights, bias, x)) > eps * l1_norm(weights)


def test_small_grid_enumeration() -> None:
    # Executable validation for tiny discrete grid
    weights = [Fraction(1), Fraction(-1)]
    bias = Fraction(0)
    eps = Fraction(1, 10)
    for a in range(-2, 3):
        for b in range(-2, 3):
            x = [Fraction(a), Fraction(b)]
            if margin_ok(weights, bias, x, eps):
                s = score(weights, bias, x)
                assert abs(s) > eps * l1_norm(weights)


def test_canonicalize_roundtrip_integers() -> None:
    for n in range(-5, 6):
        assert canonicalize_rational(Fraction(n)) == ("0" if n == 0 else str(n))
