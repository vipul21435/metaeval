"""Ordinal agreement between a grader and human ratings, and how the grader uses the scale.

A grader here is a metric that scores on the same ordinal scale as the humans, for
example 1 to 5. Agreement is computed on scores rounded to the points of the scale;
means, spreads and the signed difference use the scores as given.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
from numpy.typing import ArrayLike

__all__ = [
    "ScaleAudit",
    "adjacent_agreement",
    "exact_agreement",
    "mean_signed_difference",
    "quadratic_weighted_kappa",
    "scale_audit",
    "to_scale",
]


def _paired(grader: ArrayLike, human: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """The outputs that have both scores, flattened."""
    g = np.asarray(grader, dtype=float)
    h = np.asarray(human, dtype=float)
    if g.shape != h.shape:
        raise ValueError(f"grader and human scores differ in shape: {g.shape} and {h.shape}")
    if np.isinf(g).any() or np.isinf(h).any():
        raise ValueError("scores must not be infinite")
    both = ~np.isnan(g) & ~np.isnan(h)
    return g[both], h[both]


def to_scale(values: ArrayLike, low: float, high: float, step: float = 1.0) -> np.ndarray:
    """Round scores to the nearest point of the scale, halves upwards, and clip to it."""
    if not (high > low and step > 0):
        raise ValueError("the scale needs high > low and a positive step")
    scores = np.asarray(values, dtype=float)
    return np.clip(np.floor((scores - low) / step + 0.5) * step + low, low, high)


def exact_agreement(grader: ArrayLike, human: ArrayLike) -> float:
    """Share of outputs on which the two scores are equal."""
    g, h = _paired(grader, human)
    return float(np.mean(g == h)) if g.size else math.nan


def adjacent_agreement(grader: ArrayLike, human: ArrayLike, step: float = 1.0) -> float:
    """Share of outputs on which the two scores differ by at most one step."""
    g, h = _paired(grader, human)
    return float(np.mean(np.abs(g - h) <= step * (1 + 1e-9))) if g.size else math.nan


def mean_signed_difference(grader: ArrayLike, human: ArrayLike) -> float:
    """Mean of grader minus human: positive when the grader is more generous."""
    g, h = _paired(grader, human)
    return float(np.mean(g - h)) if g.size else math.nan


def quadratic_weighted_kappa(grader: ArrayLike, human: ArrayLike) -> float:
    """Cohen's weighted kappa with quadratic weights.

    ``1 - sum(w * observed) / sum(w * expected)`` with ``w = (i - j)^2`` on the score
    values and the expected table built from the two marginal distributions. With these
    weights the sums reduce to ``1 - mean((g - h)^2) / (var(g) + var(h) + (mean(g) -
    mean(h))^2)``, which is what is computed. ``nan`` when both scores are the same
    constant, where the expected disagreement is zero.
    """
    g, h = _paired(grader, human)
    if g.size == 0:
        return math.nan
    expected = g.var() + h.var() + (g.mean() - h.mean()) ** 2
    if expected <= 0:
        return math.nan
    return float(1.0 - np.mean((g - h) ** 2) / expected)


def _tie_share(values: np.ndarray) -> float:
    """Share of pairs of outputs that have the same score."""
    n = values.size
    if n < 2:
        return math.nan
    counts = np.unique(values, return_counts=True)[1].astype(np.int64)
    return float((counts * (counts - 1) // 2).sum() / (n * (n - 1) // 2))


@dataclass(frozen=True)
class ScaleAudit:
    """Agreement of a grader with human scores and its use of the scale.

    ``distribution`` holds, for each point of the scale, the share of grader scores and
    of human scores that round to it. ``central_*`` is the share at the middle of the
    scale (the two middle points when their number is even), ``sd_ratio`` the grader's
    standard deviation over the humans' (below 1: the grader uses a narrower range),
    ``tie_share_*`` the share of pairs of outputs with equal scores and ``off_scale``
    the share of grader scores that are not a point of the scale.
    """

    n: int
    low: float
    high: float
    step: float
    exact: float
    adjacent: float
    quadratic_weighted_kappa: float
    mean_signed_difference: float
    mean_grader: float
    mean_human: float
    sd_grader: float
    sd_human: float
    sd_ratio: float
    min_grader: float
    max_grader: float
    min_human: float
    max_human: float
    distinct_grader: int
    distinct_human: int
    central_grader: float
    central_human: float
    tie_share_grader: float
    tie_share_human: float
    off_scale: float
    distribution: tuple[tuple[float, float, float], ...]

    def to_dict(self) -> dict[str, object]:
        fields = asdict(self)
        fields["distribution"] = [
            {"score": score, "grader": grader, "human": human}
            for score, grader, human in self.distribution
        ]
        return {
            key: None if isinstance(value, float) and math.isnan(value) else value
            for key, value in fields.items()
        }


def scale_audit(
    grader: ArrayLike,
    human: ArrayLike,
    *,
    low: float | None = None,
    high: float | None = None,
    step: float = 1.0,
) -> ScaleAudit:
    """Audit a grader against human scores on an ordinal scale from ``low`` to ``high``.

    The bounds default to the smallest and largest human score. Only outputs with both
    scores are used.
    """
    g, h = _paired(grader, human)
    if g.size == 0:
        raise ValueError("no output has both a grader and a human score")
    low = float(h.min()) if low is None else float(low)
    high = float(h.max()) if high is None else float(high)
    if not high > low:
        raise ValueError("the scale needs high > low; pass the bounds when humans use one value")
    on_g, on_h = to_scale(g, low, high, step), to_scale(h, low, high, step)
    points = low + step * np.arange(math.floor((high - low) / step + 1e-9) + 1)
    share_g = [float(np.mean(np.isclose(on_g, point))) for point in points]
    share_h = [float(np.mean(np.isclose(on_h, point))) for point in points]
    middle = [(len(points) - 1) // 2, len(points) // 2]
    sd_g, sd_h = float(g.std()), float(h.std())
    return ScaleAudit(
        n=int(g.size),
        low=low,
        high=high,
        step=float(step),
        exact=exact_agreement(on_g, on_h),
        adjacent=adjacent_agreement(on_g, on_h, step),
        quadratic_weighted_kappa=quadratic_weighted_kappa(on_g, on_h),
        mean_signed_difference=mean_signed_difference(g, h),
        mean_grader=float(g.mean()),
        mean_human=float(h.mean()),
        sd_grader=sd_g,
        sd_human=sd_h,
        sd_ratio=sd_g / sd_h if sd_h > 0 else math.nan,
        min_grader=float(g.min()),
        max_grader=float(g.max()),
        min_human=float(h.min()),
        max_human=float(h.max()),
        distinct_grader=int(np.unique(g).size),
        distinct_human=int(np.unique(h).size),
        central_grader=float(sum(share_g[index] for index in set(middle))),
        central_human=float(sum(share_h[index] for index in set(middle))),
        tie_share_grader=_tie_share(g),
        tie_share_human=_tie_share(h),
        off_scale=float(np.mean(~np.isclose(g, on_g))),
        distribution=tuple(
            (float(point), grader_share, human_share)
            for point, grader_share, human_share in zip(points, share_g, share_h, strict=True)
        ),
    )
