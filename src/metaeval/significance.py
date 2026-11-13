"""Williams test for the difference between two dependent correlations.

Two metrics scored on the same outputs give two correlations with the human score that
share one variable and are therefore not independent. The Williams test takes the
correlation between the two metrics into account; Graham and Baldwin (2014) use it to
compare metrics at system level. It is a test for Pearson correlations and assumes
normally distributed scores.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
from numpy.typing import ArrayLike

from metaeval.coefficients import pearson
from metaeval.grouping import _matrices, _merge_near_ties

__all__ = ["WilliamsTest", "student_t_sf", "williams", "williams_test"]


def _beta_continued_fraction(a: float, b: float, x: float) -> float:
    """Continued fraction of the incomplete beta function (modified Lentz method)."""
    tiny = 1e-300
    c = 1.0
    d = 1.0 - (a + b) * x / (a + 1.0)
    d = 1.0 / (d if abs(d) > tiny else tiny)
    result = d
    for m in range(1, 1000):
        even = m * (b - m) * x / ((a + 2 * m - 1.0) * (a + 2 * m))
        d = 1.0 + even * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + even / c
        c = c if abs(c) > tiny else tiny
        result *= d * c
        odd = -(a + m) * (a + b + m) * x / ((a + 2 * m) * (a + 2 * m + 1.0))
        d = 1.0 + odd * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + odd / c
        c = c if abs(c) > tiny else tiny
        step = d * c
        result *= step
        if abs(step - 1.0) < 1e-15:
            break
    return result


def _regularized_beta(a: float, b: float, x: float) -> float:
    """Regularised incomplete beta function ``I_x(a, b)``."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    front = math.exp(
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _beta_continued_fraction(a, b, x) / a
    return 1.0 - front * _beta_continued_fraction(b, a, 1.0 - x) / b


def student_t_sf(t: float, df: float) -> float:
    """Upper tail ``P(T > t)`` of Student's t distribution with ``df`` degrees of freedom."""
    if math.isnan(t) or not df > 0:
        return math.nan
    if math.isinf(t):
        return 0.0 if t > 0 else 1.0
    tail = 0.5 * _regularized_beta(df / 2.0, 0.5, df / (df + t * t))
    return tail if t >= 0 else 1.0 - tail


@dataclass(frozen=True)
class WilliamsTest:
    """Result of the Williams test for ``r_a`` against ``r_b``.

    ``r_a`` and ``r_b`` are the Pearson correlations of the two metrics with the human
    score and ``r_ab`` the correlation between the metrics, over ``n`` points.
    """

    r_a: float
    r_b: float
    r_ab: float
    n: int
    t: float
    df: int
    p_value: float
    alternative: str
    grouping: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            key: None if isinstance(value, float) and math.isnan(value) else value
            for key, value in asdict(self).items()
        }


def williams_test(
    r_a: float, r_b: float, r_ab: float, n: int, *, alternative: str = "two-sided"
) -> WilliamsTest:
    """Williams test from three correlations and the sample size.

    ``t = (r_a - r_b) * sqrt((n - 1) * (1 + r_ab)) /
    sqrt(2 * K * (n - 1) / (n - 3) + (r_a + r_b)^2 / 4 * (1 - r_ab)^3)`` with
    ``K = 1 - r_a^2 - r_b^2 - r_ab^2 + 2 * r_a * r_b * r_ab``, referred to Student's t
    with ``n - 3`` degrees of freedom. ``alternative`` is ``"two-sided"`` or ``"greater"``
    (``r_a`` is larger than ``r_b``).
    """
    if alternative not in ("two-sided", "greater"):
        raise ValueError("alternative must be 'two-sided' or 'greater'")
    t = p_value = math.nan
    df = n - 3
    if df > 0 and not any(math.isnan(r) for r in (r_a, r_b, r_ab)):
        k = 1.0 - r_a**2 - r_b**2 - r_ab**2 + 2.0 * r_a * r_b * r_ab
        variance = 2.0 * k * (n - 1) / df + (r_a + r_b) ** 2 / 4.0 * (1.0 - r_ab) ** 3
        if variance > 0:
            t = (r_a - r_b) * math.sqrt((n - 1) * (1.0 + r_ab)) / math.sqrt(variance)
            upper = student_t_sf(t, df)
            p_value = upper if alternative == "greater" else 2.0 * min(upper, 1.0 - upper)
    return WilliamsTest(r_a, r_b, r_ab, n, t, df, p_value, alternative)


def williams(
    metric_a: ArrayLike,
    metric_b: ArrayLike,
    human: ArrayLike,
    *,
    grouping: str,
    alternative: str = "two-sided",
) -> WilliamsTest:
    """Williams test for two metrics at the ``system`` or the ``pooled`` level.

    The three Pearson correlations are taken over the outputs that have all three
    scores (their system means at system level). The test needs a single correlation
    over ``n`` points, so the grouped levels are not available.
    """
    if grouping not in ("system", "pooled"):
        raise ValueError("the Williams test needs grouping 'system' or 'pooled'")
    a, h, _ = _matrices(metric_a, human)
    b, _, _ = _matrices(metric_b, human)
    complete = ~np.isnan(a) & ~np.isnan(b) & ~np.isnan(h)
    if grouping == "system":
        counts = complete.sum(axis=1)
        rated = counts > 0
        x, y, z = (
            _merge_near_ties(np.where(complete, scores, 0.0).sum(axis=1)[rated] / counts[rated])
            for scores in (a, b, h)
        )
    else:
        x, y, z = a[complete], b[complete], h[complete]
    result = williams_test(
        pearson(x, z), pearson(y, z), pearson(x, y), int(x.size), alternative=alternative
    )
    return WilliamsTest(**{**asdict(result), "grouping": grouping})
