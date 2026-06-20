"""Clopper-Pearson exact-binomial confidence bounds: the deterministic M0 core.

Every M0 GO/NO-GO bar is a one-sided Clopper-Pearson bound (90% for precision/recall, 95%
for false-OK), fixed in advance. This module is pure and deterministic: given (x successes,
n trials, alpha) it returns the exact beta-quantile bound.

The Clopper-Pearson interval inverts the binomial tail via the beta distribution:
  - one-sided lower bound: Beta(x, n - x + 1).ppf(alpha),         with lower(0, n) == 0.0
  - one-sided upper bound: Beta(x + 1, n - x).ppf(1 - alpha),     with upper(n, n) == 1.0

The two degenerate cases (x == 0 lower, x == n upper) are returned exactly rather than via
the quantile so the bounds are crisp 0.0 / 1.0.
"""

from __future__ import annotations

from scipy.stats import beta


def _validate(x: int, n: int, alpha: float) -> None:
    if n <= 0:
        raise ValueError(f"n must be a positive integer, got {n}")
    if x < 0 or x > n:
        raise ValueError(f"x must satisfy 0 <= x <= n, got x={x}, n={n}")
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in the open interval (0, 1), got {alpha}")


def clopper_pearson_lower(x: int, n: int, alpha: float) -> float:
    """One-sided Clopper-Pearson lower bound on a binomial proportion.

    Args:
        x: number of successes (0 <= x <= n).
        n: number of trials (n > 0).
        alpha: one-sided tail mass (0 < alpha < 1); 0.10 gives a 90% lower bound.

    Returns:
        The lower confidence bound. Exactly 0.0 when x == 0.
    """
    _validate(x, n, alpha)
    if x == 0:
        return 0.0
    return float(beta.ppf(alpha, x, n - x + 1))


def clopper_pearson_upper(x: int, n: int, alpha: float) -> float:
    """One-sided Clopper-Pearson upper bound on a binomial proportion.

    Args:
        x: number of successes (0 <= x <= n).
        n: number of trials (n > 0).
        alpha: one-sided tail mass (0 < alpha < 1); 0.05 gives a 95% upper bound.

    Returns:
        The upper confidence bound. Exactly 1.0 when x == n.
    """
    _validate(x, n, alpha)
    if x == n:
        return 1.0
    return float(beta.ppf(1.0 - alpha, x + 1, n - x))
