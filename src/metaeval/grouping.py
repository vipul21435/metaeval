"""Correlation between metric and human scores at a stated level of grouping.

Scores are arrays of shape ``(n_systems, n_items)``: entry ``[i, j]`` is the score of the
output of system ``i`` for item ``j``, and ``nan`` marks a missing score. The same pair of
arrays gives four different numbers:

``system``
    Average each system over its items, then correlate across systems.
``pooled``
    Correlate over all outputs at once, without grouping.
``by_item``
    Correlate across systems within each item, then average over items.
``by_system``
    Correlate across items within each system, then average over systems.

Only outputs that have both a metric and a human score are used, at every level.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
from numpy.typing import ArrayLike

from metaeval.coefficients import CALIBRATED, COEFFICIENTS, calibrate_ties, rowwise

__all__ = ["GROUPINGS", "STATISTICS", "Correlation", "correlate", "statistic"]

STATISTICS: tuple[str, ...] = (*COEFFICIENTS, CALIBRATED)
"""Every name accepted as ``coefficient``: the coefficients and calibrated pairwise accuracy."""

GROUPINGS: dict[str, str] = {
    "system": "system level: scores averaged over items, one point per system",
    "pooled": "item level, pooled: every output is one point, no grouping",
    "by_item": (
        "item level, grouped by item: coefficient across systems within each item, "
        "averaged over items"
    ),
    "by_system": (
        "item level, grouped by system: coefficient across items within each system, "
        "averaged over systems"
    ),
}
"""The groupings by name, each with the label printed next to its numbers."""

_MEAN_TIE_TOLERANCE = 1e-12


@dataclass(frozen=True)
class Correlation:
    """One coefficient at one grouping, with the counts needed to read it.

    ``n_groups`` is the number of groups that could be correlated (one for ``system`` and
    ``pooled``; for the grouped levels, the groups with at least two outputs that have
    both scores) and ``n_undefined`` the number of those in which the coefficient is
    undefined, typically because one side is constant. ``n_points`` counts what was
    correlated: systems at system level, outputs otherwise. ``epsilon`` is set for
    calibrated pairwise accuracy only: the metric difference up to which two outputs
    count as tied.
    """

    coefficient: str
    grouping: str
    value: float
    n_points: int
    n_groups: int
    n_undefined: int
    undefined: str
    epsilon: float | None = None

    @property
    def label(self) -> str:
        """The grouping in words."""
        return GROUPINGS[self.grouping]

    def to_dict(self) -> dict[str, object]:
        """Plain values for JSON, with an undefined value as ``None``."""
        fields = asdict(self)
        fields["value"] = None if math.isnan(self.value) else self.value
        fields["label"] = self.label
        if self.epsilon is None:
            del fields["epsilon"]
        return fields


def _merge_near_ties(values: np.ndarray) -> np.ndarray:
    """Give means that differ only by rounding the same float.

    Two systems with the same ratings in a different order can get means that differ
    in the last bits, which would hide a tie from the rank coefficients. Values are
    sorted and a value closer to its predecessor than ``1e-12`` of the largest magnitude
    takes its predecessor's value.
    """
    if values.size < 2:
        return values
    order = np.argsort(values, kind="stable")
    ordered = values[order]
    threshold = _MEAN_TIE_TOLERANCE * float(np.max(np.abs(ordered)))
    starts_cluster = np.concatenate(([True], np.diff(ordered) > threshold))
    first_of_cluster = np.flatnonzero(starts_cluster)
    merged = np.empty_like(values)
    merged[order] = ordered[first_of_cluster[np.cumsum(starts_cluster) - 1]]
    return merged


def _matrices(metric: ArrayLike, human: ArrayLike) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    m = np.asarray(metric, dtype=float)
    h = np.asarray(human, dtype=float)
    if m.ndim != 2 or h.ndim != 2:
        raise ValueError("scores must be two-dimensional: systems by items")
    if m.shape != h.shape:
        raise ValueError(f"metric and human scores differ in shape: {m.shape} and {h.shape}")
    if np.isinf(m).any() or np.isinf(h).any():
        raise ValueError("scores must not be infinite")
    return m, h, ~np.isnan(m) & ~np.isnan(h)


def correlate(
    metric: ArrayLike,
    human: ArrayLike,
    *,
    coefficient: str,
    grouping: str,
    undefined: str = "skip",
) -> Correlation:
    """Correlate metric scores with human scores at one grouping.

    Parameters
    ----------
    metric, human:
        Arrays of shape ``(n_systems, n_items)``; ``nan`` marks a missing score.
    coefficient:
        One of :data:`STATISTICS`: a key of :data:`metaeval.coefficients.COEFFICIENTS`, or
        ``"pairwise_accuracy_calibrated"`` for pairwise accuracy after tie calibration.
    grouping:
        A key of :data:`GROUPINGS`. There is no default: the number depends on it.
    undefined:
        What to do, at the grouped levels, with a group in which the coefficient is
        undefined: ``"skip"`` leaves it out of the average, ``"zero"`` counts it as 0.
    """
    if coefficient not in STATISTICS:
        raise ValueError(f"unknown coefficient {coefficient!r}; choose from {sorted(STATISTICS)}")
    if grouping not in GROUPINGS:
        raise ValueError(f"unknown grouping {grouping!r}; choose from {list(GROUPINGS)}")
    if undefined not in ("skip", "zero"):
        raise ValueError(f"undefined must be 'skip' or 'zero', got {undefined!r}")
    m, h, paired = _matrices(metric, human)
    epsilon = None
    if coefficient == CALIBRATED:
        threshold, value, _, n_groups, _ = calibrate_ties(*_rows(m, h, paired, grouping))
        n_points = _evaluate(m, h, paired, "pairwise_accuracy", grouping, undefined)[1]
        n_undefined = 0
        epsilon = None if math.isnan(threshold) else threshold
    else:
        value, n_points, n_groups, n_undefined = _evaluate(
            m, h, paired, coefficient, grouping, undefined
        )
    return Correlation(
        coefficient=coefficient,
        grouping=grouping,
        value=value,
        n_points=n_points,
        n_groups=n_groups,
        n_undefined=n_undefined,
        undefined=undefined,
        epsilon=epsilon,
    )


def _rows(
    m: np.ndarray, h: np.ndarray, paired: np.ndarray, grouping: str
) -> tuple[np.ndarray, np.ndarray]:
    """The groups of a grouping as rows, with ``nan`` where a score is missing."""
    if grouping == "system":
        counts = paired.sum(axis=1)
        rated = counts > 0
        x = _merge_near_ties(np.where(paired, m, 0.0).sum(axis=1)[rated] / counts[rated])
        y = _merge_near_ties(np.where(paired, h, 0.0).sum(axis=1)[rated] / counts[rated])
        return x[None, :], y[None, :]
    if grouping == "pooled":
        return m[paired][None, :], h[paired][None, :]
    x, y = np.where(paired, m, np.nan), np.where(paired, h, np.nan)
    return (x.T, y.T) if grouping == "by_item" else (x, y)


def _evaluate(
    m: np.ndarray,
    h: np.ndarray,
    paired: np.ndarray,
    coefficient: str,
    grouping: str,
    undefined: str,
) -> tuple[float, int, int, int]:
    """Value, points, groups and undefined groups for validated arrays."""
    if coefficient == CALIBRATED:
        _, value, _, n_groups, n_pairs = calibrate_ties(*_rows(m, h, paired, grouping))
        return value, n_pairs, n_groups, 0
    if grouping in ("system", "pooled"):
        if grouping == "system":
            counts = paired.sum(axis=1)
            rated = counts > 0
            x = _merge_near_ties(np.where(paired, m, 0.0).sum(axis=1)[rated] / counts[rated])
            y = _merge_near_ties(np.where(paired, h, 0.0).sum(axis=1)[rated] / counts[rated])
        else:
            x, y = m[paired], h[paired]
        value = COEFFICIENTS[coefficient](x, y)
        return value, int(x.size), 1, int(math.isnan(value))

    if grouping == "by_item":
        m, h, paired = m.T, h.T, paired.T
    sizes = paired.sum(axis=1)
    groups = sizes >= 2
    n_groups = int(groups.sum())
    if n_groups == 0:
        return math.nan, 0, 0, 0
    values = rowwise(
        coefficient, np.where(paired, m, np.nan)[groups], np.where(paired, h, np.nan)[groups]
    )
    defined = ~np.isnan(values)
    n_defined = int(defined.sum())
    if undefined == "zero":
        value = float(values[defined].sum() / n_groups)
    else:
        value = float(values[defined].mean()) if n_defined else math.nan
    return value, int(sizes[groups].sum()), n_groups, n_groups - n_defined


def statistic(
    metric: np.ndarray, human: np.ndarray, coefficient: str, grouping: str, undefined: str = "skip"
) -> float:
    """The value of :func:`correlate` alone, for arrays that are already validated.

    Used where the same statistic is computed many times, as in resampling.
    """
    paired = ~np.isnan(metric) & ~np.isnan(human)
    return _evaluate(metric, human, paired, coefficient, grouping, undefined)[0]
