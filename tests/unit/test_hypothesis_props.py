"""Hypothesis and grid properties for L∞ margin classification invariance (F-013/R-05)."""

from __future__ import annotations

from fractions import Fraction

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from leanverifier.canonicalize import (
    canonicalize_rational,
    classify,
    l1_norm,
    linf_ball,
    margin_ok,
    score,
)

pytestmark = [pytest.mark.unit, pytest.mark.hypothesis]


@st.composite
def aligned_weights_input(
    draw: st.DrawFn, *, dim: int | None = None
) -> tuple[list[Fraction], Fraction, list[Fraction], Fraction]:
    n = dim if dim is not None else draw(st.integers(min_value=1, max_value=3))
    weights = draw(
        st.lists(st.fractions(min_value=-5, max_value=5, max_denominator=8), min_size=n, max_size=n)
    )
    bias = draw(st.fractions(min_value=-5, max_value=5, max_denominator=8))
    x = draw(
        st.lists(st.fractions(min_value=-5, max_value=5, max_denominator=8), min_size=n, max_size=n)
    )
    eps = draw(st.fractions(min_value=0, max_value=2, max_denominator=8))
    return weights, bias, x, eps


@given(data=aligned_weights_input())
@settings(max_examples=80, deadline=None)
def test_margin_implies_lipschitz_room(
    data: tuple[list[Fraction], Fraction, list[Fraction], Fraction],
) -> None:
    weights, bias, x, eps = data
    if margin_ok(weights, bias, x, eps):
        assert abs(score(weights, bias, x)) > eps * l1_norm(weights)


@given(data=aligned_weights_input())
@settings(max_examples=60, deadline=None)
def test_classification_invariant_on_linf_perturbations(
    data: tuple[list[Fraction], Fraction, list[Fraction], Fraction],
) -> None:
    """When the margin premise holds, every L∞ perturbation preserves classification.

    This asserts the executable robustness property (classify equality on the ball),
    not a tautological restatement of ``margin_ok``.
    """
    weights, bias, x, eps = data
    assume(margin_ok(weights, bias, x, eps))
    ref = classify(weights, bias, x)
    n = len(x)
    # Enumerate axis-aligned L∞ corners within ε.
    coords = [Fraction(0), eps, -eps] if eps > 0 else [Fraction(0)]
    from itertools import product

    for delta_tuple in product(coords, repeat=n):
        y = [xi + di for xi, di in zip(x, delta_tuple, strict=True)]
        assert linf_ball(x, eps, y)
        assert classify(weights, bias, y) == ref


def test_small_grid_enumeration_classification() -> None:
    """Executable validation: full small discrete grid when margin holds."""
    weights = [Fraction(1), Fraction(-1)]
    bias = Fraction(0)
    eps = Fraction(1, 10)
    for a in range(-2, 3):
        for b in range(-2, 3):
            x = [Fraction(a), Fraction(b)]
            if not margin_ok(weights, bias, x, eps):
                continue
            ref = classify(weights, bias, x)
            # Perturb each coordinate by {-eps, 0, +eps}
            for da in (-eps, Fraction(0), eps):
                for db in (-eps, Fraction(0), eps):
                    y = [x[0] + da, x[1] + db]
                    assert linf_ball(x, eps, y)
                    assert classify(weights, bias, y) == ref


def test_canonicalize_roundtrip_integers() -> None:
    for n in range(-5, 6):
        assert canonicalize_rational(Fraction(n)) == ("0" if n == 0 else str(n))
