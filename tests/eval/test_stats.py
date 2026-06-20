"""TDD anchors for the Clopper-Pearson exact-binomial bounds (the deterministic core).

The bounds gate every M0 decision, so they are pinned to KNOWN, hand-derivable values
rather than to whatever scipy happens to return. The two clean closed forms used as anchors:

  - lower(n, n, alpha)  == alpha ** (1 / n)        (all-success one-sided lower bound)
  - upper(0, n, alpha)  == 1 - alpha ** (1 / n)    (all-failure one-sided upper bound)

Reference points (pre-checked against scipy):
  - clopper_pearson_lower(50, 50, 0.10) ~= 0.9550   ( == 0.10 ** (1 / 50) )
  - clopper_pearson_upper(0, 200, 0.05) ~= 0.01487  ( == 1 - 0.05 ** (1 / 200) )
"""

from __future__ import annotations

import math

import pytest

from citeproof.eval.stats import clopper_pearson_lower, clopper_pearson_upper


def test_lower_all_success_known_value() -> None:
    # 50/50 successes, 90% one-sided -> 0.10 ** (1/50) ~= 0.9550.
    got = clopper_pearson_lower(50, 50, 0.10)
    assert math.isclose(got, 0.9550, abs_tol=1e-3)
    assert math.isclose(got, 0.10 ** (1 / 50), abs_tol=1e-12)


def test_upper_all_failure_known_value() -> None:
    # 0/200 failures, 95% one-sided -> 1 - 0.05 ** (1/200) ~= 0.01487.
    got = clopper_pearson_upper(0, 200, 0.05)
    assert math.isclose(got, 0.01487, abs_tol=1e-3)
    assert math.isclose(got, 1 - 0.05 ** (1 / 200), abs_tol=1e-12)


def test_lower_zero_successes_is_zero() -> None:
    # No successes -> the lower bound is exactly 0.0 by definition.
    for n in (1, 10, 200):
        for alpha in (0.05, 0.10):
            assert clopper_pearson_lower(0, n, alpha) == 0.0


def test_upper_all_successes_is_one() -> None:
    # All successes -> the upper bound is exactly 1.0 by definition.
    for n in (1, 10, 200):
        for alpha in (0.05, 0.10):
            assert clopper_pearson_upper(n, n, alpha) == 1.0


def test_lower_monotonic_in_x() -> None:
    # More successes can only raise the lower bound.
    n, alpha = 50, 0.10
    bounds = [clopper_pearson_lower(x, n, alpha) for x in range(n + 1)]
    assert all(a <= b for a, b in zip(bounds, bounds[1:], strict=False))
    # And strictly increasing across a representative interior stretch.
    assert clopper_pearson_lower(10, n, alpha) < clopper_pearson_lower(20, n, alpha)
    assert clopper_pearson_lower(20, n, alpha) < clopper_pearson_lower(30, n, alpha)


def test_upper_monotonic_in_x() -> None:
    # More successes can only raise the upper bound.
    n, alpha = 50, 0.10
    bounds = [clopper_pearson_upper(x, n, alpha) for x in range(n + 1)]
    assert all(a <= b for a, b in zip(bounds, bounds[1:], strict=False))
    assert clopper_pearson_upper(10, n, alpha) < clopper_pearson_upper(20, n, alpha)
    assert clopper_pearson_upper(20, n, alpha) < clopper_pearson_upper(30, n, alpha)


def test_lower_49_of_50_clears_the_090_bar() -> None:
    # One error out of 50 at the 0.90 gate: the lower bound stays above 0.90,
    # so the >= 0.90 bar tolerates ~1 mistake at n=50 (the pre-registered design point).
    got = clopper_pearson_lower(49, 50, 0.10)
    assert 0.90 < got < 0.93


def test_bounds_bracket_the_point_estimate() -> None:
    # For an interior count, lower <= x/n <= upper (two one-sided bounds at the same alpha).
    n, alpha = 100, 0.10
    for x in (1, 25, 50, 75, 99):
        lo = clopper_pearson_lower(x, n, alpha)
        hi = clopper_pearson_upper(x, n, alpha)
        assert lo <= x / n <= hi


@pytest.mark.parametrize(
    "x,n,alpha",
    [
        (-1, 10, 0.10),  # x < 0
        (11, 10, 0.10),  # x > n
        (1, 0, 0.10),  # n <= 0
        (1, 10, 0.0),  # alpha <= 0
        (1, 10, 1.0),  # alpha >= 1
    ],
)
def test_guards_reject_bad_input(x: int, n: int, alpha: float) -> None:
    with pytest.raises(ValueError):
        clopper_pearson_lower(x, n, alpha)
    with pytest.raises(ValueError):
        clopper_pearson_upper(x, n, alpha)
