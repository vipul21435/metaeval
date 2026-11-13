"""Human ceiling: how well one annotator tracks the others.

Each annotator in turn is left out and correlated with the mean rating of the remaining
annotators, exactly as a metric would be; the mean over annotators is the ceiling. To read
a metric against it on equal terms, the metric is correlated with the same leave-one-out
means (each built from one annotator fewer than the full panel) and averaged.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass

import numpy as np

from metaeval.grouping import GROUPINGS, STATISTICS, statistic

__all__ = ["Ceiling", "human_ceiling", "leave_one_out_means", "metric_against_rest"]


def leave_one_out_means(ratings: np.ndarray) -> np.ndarray:
    """For each annotator, the mean rating of the other annotators on every output.

    ``ratings`` has shape ``(annotators, systems, items)`` with ``nan`` for outputs an
    annotator did not rate; the result has the same shape, ``nan`` where nobody else
    rated the output.
    """
    present = ~np.isnan(ratings)
    totals = np.where(present, ratings, 0.0).sum(axis=0)
    counts = present.sum(axis=0)
    rest_totals = totals[None] - np.where(present, ratings, 0.0)
    rest_counts = counts[None] - present
    return np.divide(
        rest_totals, rest_counts, out=np.full(ratings.shape, np.nan), where=rest_counts > 0
    )


def _check(ratings: np.ndarray, coefficient: str, grouping: str) -> np.ndarray:
    ratings = np.asarray(ratings, dtype=float)
    if ratings.ndim != 3:
        raise ValueError("ratings must have shape (annotators, systems, items)")
    if ratings.shape[0] < 2:
        raise ValueError("the human ceiling needs at least two annotators")
    if coefficient not in STATISTICS:
        raise ValueError(f"unknown coefficient {coefficient!r}; choose from {sorted(STATISTICS)}")
    if grouping not in GROUPINGS:
        raise ValueError(f"unknown grouping {grouping!r}; choose from {list(GROUPINGS)}")
    return ratings


def _one_against_rest(
    ratings: np.ndarray, coefficient: str, grouping: str, undefined: str
) -> list[float]:
    rest = leave_one_out_means(ratings)
    return [
        statistic(ratings[index], rest[index], coefficient, grouping, undefined)
        for index in range(ratings.shape[0])
    ]


def _mean_defined(values: Sequence[float]) -> float:
    defined = [value for value in values if not math.isnan(value)]
    return float(np.mean(defined)) if defined else math.nan


@dataclass(frozen=True)
class Ceiling:
    """Leave-one-annotator-out agreement at one coefficient and grouping.

    ``value`` is the mean of ``per_annotator``, each annotator against the mean of the
    others; annotators for whom the coefficient is undefined are left out of the mean.
    """

    coefficient: str
    grouping: str
    value: float
    per_annotator: tuple[float, ...]

    def to_dict(self) -> dict[str, object]:
        fields = asdict(self)
        fields["value"] = None if math.isnan(self.value) else self.value
        fields["per_annotator"] = [None if math.isnan(v) else v for v in self.per_annotator]
        return fields


def human_ceiling(
    ratings: np.ndarray, *, coefficient: str, grouping: str, undefined: str = "skip"
) -> Ceiling:
    """Correlate each annotator with the mean of the others and average.

    ``ratings`` has shape ``(annotators, systems, items)``.
    """
    ratings = _check(ratings, coefficient, grouping)
    values = _one_against_rest(ratings, coefficient, grouping, undefined)
    return Ceiling(coefficient, grouping, _mean_defined(values), tuple(values))


def metric_against_rest(
    metric: np.ndarray,
    ratings: np.ndarray,
    *,
    coefficient: str,
    grouping: str,
    undefined: str = "skip",
) -> float:
    """Mean correlation of a metric with each leave-one-annotator-out mean rating.

    The references are the ones the ceiling uses, so the two numbers are comparable.
    """
    ratings = _check(ratings, coefficient, grouping)
    metric = np.asarray(metric, dtype=float)
    if metric.shape != ratings.shape[1:]:
        raise ValueError("metric and ratings differ in their systems-by-items shape")
    rest = leave_one_out_means(ratings)
    return _mean_defined(
        [
            statistic(metric, rest[index], coefficient, grouping, undefined)
            for index in range(ratings.shape[0])
        ]
    )
