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

from metaeval.coefficients import COEFFICIENTS

__all__ = ["GROUPINGS", "Correlation", "correlate"]

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
    correlated: systems at system level, outputs otherwise.
    """

    coefficient: str
    grouping: str
    value: float
    n_points: int
    n_groups: int
    n_undefined: int
    undefined: str

    @property
    def label(self) -> str:
        """The grouping in words."""
        return GROUPINGS[self.grouping]

    def to_dict(self) -> dict[str, object]:
        """Plain values for JSON, with an undefined value as ``None``."""
        fields = asdict(self)
        fields["value"] = None if math.isnan(self.value) else self.value
        fields["label"] = self.label
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
        A key of :data:`metaeval.coefficients.COEFFICIENTS`.
    grouping:
        A key of :data:`GROUPINGS`. There is no default: the number depends on it.
    undefined:
        What to do, at the grouped levels, with a group in which the coefficient is
        undefined: ``"skip"`` leaves it out of the average, ``"zero"`` counts it as 0.
    """
    if coefficient not in COEFFICIENTS:
        raise ValueError(f"unknown coefficient {coefficient!r}; choose from {sorted(COEFFICIENTS)}")
    if grouping not in GROUPINGS:
        raise ValueError(f"unknown grouping {grouping!r}; choose from {list(GROUPINGS)}")
    if undefined not in ("skip", "zero"):
        raise ValueError(f"undefined must be 'skip' or 'zero', got {undefined!r}")
    function = COEFFICIENTS[coefficient]
    m, h, paired = _matrices(metric, human)

    if grouping in ("system", "pooled"):
        if grouping == "system":
            counts = paired.sum(axis=1)
            rated = counts > 0
            x = _merge_near_ties(np.where(paired, m, 0.0).sum(axis=1)[rated] / counts[rated])
            y = _merge_near_ties(np.where(paired, h, 0.0).sum(axis=1)[rated] / counts[rated])
        else:
            x, y = m[paired], h[paired]
        value = function(x, y)
        return Correlation(
            coefficient=coefficient,
            grouping=grouping,
            value=value,
            n_points=int(x.size),
            n_groups=1,
            n_undefined=int(math.isnan(value)),
            undefined=undefined,
        )

    if grouping == "by_item":
        m, h, paired = m.T, h.T, paired.T
    values: list[float] = []
    n_points = n_groups = n_undefined = 0
    for row in range(m.shape[0]):
        keep = paired[row]
        size = int(keep.sum())
        if size < 2:
            continue
        n_groups += 1
        n_points += size
        value = function(m[row, keep], h[row, keep])
        if not math.isnan(value):
            values.append(value)
        else:
            n_undefined += 1
            if undefined == "zero":
                values.append(0.0)
    return Correlation(
        coefficient=coefficient,
        grouping=grouping,
        value=float(np.mean(values)) if values else math.nan,
        n_points=n_points,
        n_groups=n_groups,
        n_undefined=n_undefined,
        undefined=undefined,
    )
