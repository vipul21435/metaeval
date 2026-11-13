"""Correlation and agreement coefficients for two paired score vectors.

Every function takes two one-dimensional sequences of equal length holding finite
numbers, usually the metric scores ``x`` and the human scores ``y``, and returns a float.
A coefficient that is undefined for its input (fewer than two points, or a constant
vector where the formula divides by zero) is returned as ``nan`` instead of raising, so
that callers averaging over groups can count such groups and leave them out.

Two scores are tied when the stored floats are equal. Nothing is rounded here.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

__all__ = [
    "CALIBRATED",
    "COEFFICIENTS",
    "PairCounts",
    "calibrate_ties",
    "kendall_tau_b",
    "kendall_tau_c",
    "pair_counts",
    "pairwise_accuracy",
    "pearson",
    "rowwise",
    "spearman",
]


def _as_pair(x: ArrayLike, y: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """Validate two paired vectors and return them as float arrays."""
    a = np.asarray(x, dtype=float)
    b = np.asarray(y, dtype=float)
    if a.ndim != 1 or b.ndim != 1:
        raise ValueError("scores must be one-dimensional")
    if a.shape != b.shape:
        raise ValueError(f"scores must have the same length, got {a.size} and {b.size}")
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        raise ValueError("scores must be finite; drop missing values before calling")
    return a, b


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """Ranks starting at 1, tied values sharing the mean of the ranks they span."""
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    last_rank = np.cumsum(counts)
    return (last_rank - (counts - 1) / 2.0)[inverse]


def pearson(x: ArrayLike, y: ArrayLike) -> float:
    """Pearson product-moment correlation; ``nan`` if either vector is constant."""
    a, b = _as_pair(x, y)
    return _pearson(a, b)


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    # Test for a constant vector on the raw values: subtracting a rounded mean from a
    # constant vector can leave tiny non-zero residuals and a meaningless ratio.
    if a.size < 2 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return math.nan
    da = a - a.mean()
    db = b - b.mean()
    # Scale the deviations to a largest magnitude of 1 so that their squares neither
    # underflow nor overflow; the correlation does not depend on the scale.
    da = da / np.abs(da).max()
    db = db / np.abs(db).max()
    ratio = float(da @ db) / math.sqrt(float(da @ da) * float(db @ db))
    return min(1.0, max(-1.0, ratio))


def spearman(x: ArrayLike, y: ArrayLike) -> float:
    """Spearman rank correlation: Pearson correlation of the average ranks."""
    a, b = _as_pair(x, y)
    if a.size < 2:
        return math.nan
    return _pearson(_average_ranks(a), _average_ranks(b))


@dataclass(frozen=True)
class PairCounts:
    """How the ``n * (n - 1) / 2`` unordered pairs of points split up.

    ``tied_x`` and ``tied_y`` count pairs tied in that vector only; pairs tied in both
    vectors are counted in ``tied_both`` alone, so the five fields sum to ``total``.
    """

    n: int
    concordant: int
    discordant: int
    tied_x: int
    tied_y: int
    tied_both: int
    distinct_x: int
    distinct_y: int

    @property
    def total(self) -> int:
        return self.n * (self.n - 1) // 2


def _tied_pairs(group_sizes: np.ndarray) -> int:
    sizes = group_sizes.astype(np.int64)
    return int((sizes * (sizes - 1) // 2).sum())


def _count_inversions(codes: np.ndarray) -> int:
    """Number of index pairs ``i < j`` with ``codes[i] > codes[j]``.

    ``codes`` holds non-negative integers. This is a bottom-up merge sort in which every
    level is one stable sort of the whole array: the key is (block, value), each half
    of a block is already sorted, and stability keeps a left-half element ahead of an
    equal right-half one, so equal values never count as an inversion.
    """
    n = codes.size
    if n < 2:
        return 0
    values = codes.astype(np.int64)
    span = int(values.max()) + 1
    position = np.arange(n, dtype=np.int64)
    inversions = 0
    width = 1
    while width < n:
        block = position // (2 * width)
        order = np.argsort(block * span + values, kind="stable")
        from_right = (order // width) % 2 == 1
        rights_before = np.cumsum(from_right) - from_right
        # Left-half elements of the same block that sort at or before each position.
        lefts_before = position - rights_before - block * width
        # A block with a right half always has a full left half of `width` elements.
        inversions += int((width - lefts_before[from_right]).sum())
        values = values[order]
        width *= 2
    return inversions


def pair_counts(x: ArrayLike, y: ArrayLike) -> PairCounts:
    """Count concordant, discordant and tied pairs in ``O(n log^2 n)`` time."""
    a, b = _as_pair(x, y)
    return _pair_counts(a, b)


def _pair_counts(a: np.ndarray, b: np.ndarray) -> PairCounts:
    n = a.size
    if n == 0:
        return PairCounts(0, 0, 0, 0, 0, 0, 0, 0)
    # Dense integer codes keep both the order and the ties of the original values.
    _, x_codes, x_sizes = np.unique(a, return_inverse=True, return_counts=True)
    _, y_codes, y_sizes = np.unique(b, return_inverse=True, return_counts=True)
    x_codes = x_codes.astype(np.int64)
    y_codes = y_codes.astype(np.int64)
    tied_in_x = _tied_pairs(x_sizes)
    tied_in_y = _tied_pairs(y_sizes)
    joint_sizes = np.unique(x_codes * y_sizes.size + y_codes, return_counts=True)[1]
    tied_both = _tied_pairs(joint_sizes)
    # Sorted by x with ties in x ordered by y, a strict inversion in y is exactly a
    # discordant pair: pairs tied in x are in non-decreasing y order and pairs tied in
    # y are never strict inversions.
    order = np.lexsort((y_codes, x_codes))
    discordant = _count_inversions(y_codes[order])
    total = n * (n - 1) // 2
    concordant = total - tied_in_x - tied_in_y + tied_both - discordant
    return PairCounts(
        n=n,
        concordant=concordant,
        discordant=discordant,
        tied_x=tied_in_x - tied_both,
        tied_y=tied_in_y - tied_both,
        tied_both=tied_both,
        distinct_x=int(x_sizes.size),
        distinct_y=int(y_sizes.size),
    )


def kendall_tau_b(x: ArrayLike, y: ArrayLike) -> float:
    """Kendall's tau-b: ``(C - D) / sqrt((C + D + Tx) * (C + D + Ty))``.

    ``Tx`` and ``Ty`` are the pairs tied in one vector only. ``nan`` if either vector is
    constant, because every pair is then tied and the denominator is zero.
    """
    counts = pair_counts(x, y)
    untied = counts.concordant + counts.discordant
    product = (untied + counts.tied_x) * (untied + counts.tied_y)
    if product == 0:
        return math.nan
    return (counts.concordant - counts.discordant) / math.sqrt(product)


def kendall_tau_c(x: ArrayLike, y: ArrayLike) -> float:
    """Kendall's tau-c (Stuart): ``2 * (C - D) / (n^2 * (m - 1) / m)``.

    ``m`` is the smaller of the numbers of distinct values in the two vectors, so the
    coefficient can reach 1 on a rectangular table. ``nan`` if either vector is constant.
    """
    counts = pair_counts(x, y)
    m = min(counts.distinct_x, counts.distinct_y)
    if m < 2:
        return math.nan
    return 2.0 * (counts.concordant - counts.discordant) * m / (counts.n**2 * (m - 1))


def pairwise_accuracy(x: ArrayLike, y: ArrayLike) -> float:
    """Pairwise accuracy with ties: ``(C + Txy) / (C + D + Tx + Ty + Txy)``.

    The share of pairs that ``x`` orders the same way as ``y`` or ties where ``y`` ties.
    Unlike the tau variants it is defined for constant vectors; ``nan`` only when there
    are fewer than two points.
    """
    counts = pair_counts(x, y)
    if counts.total == 0:
        return math.nan
    return (counts.concordant + counts.tied_both) / counts.total


COEFFICIENTS: dict[str, Callable[[ArrayLike, ArrayLike], float]] = {
    "pearson": pearson,
    "spearman": spearman,
    "kendall_b": kendall_tau_b,
    "kendall_c": kendall_tau_c,
    "pairwise_accuracy": pairwise_accuracy,
}
"""Coefficient functions by the name used in the command line and in results."""

_BLOCK_ENTRIES = 4_000_000
"""Largest ``groups * n * n`` work array that :func:`rowwise` builds at once."""


def rowwise(name: str, x: ArrayLike, y: ArrayLike) -> np.ndarray:
    """Coefficient ``name`` for every row of two ``(groups, n)`` arrays at once.

    A cell that is ``nan`` in either array is missing and is left out of its row. The
    result has one value per row, ``nan`` where the coefficient is undefined, and agrees
    with calling the one-dimensional function on each row. Short rows are handled
    together by comparing all pairs of a row (``O(n^2)`` per row, vectorised over rows);
    rows too long for that are passed one at a time to the sort-based functions.
    """
    if name not in COEFFICIENTS:
        raise ValueError(f"unknown coefficient {name!r}; choose from {sorted(COEFFICIENTS)}")
    a = np.asarray(x, dtype=float)
    b = np.asarray(y, dtype=float)
    if a.ndim != 2 or a.shape != b.shape:
        raise ValueError("rowwise needs two arrays of the same two-dimensional shape")
    if np.isinf(a).any() or np.isinf(b).any():
        raise ValueError("scores must not be infinite")
    valid = ~np.isnan(a) & ~np.isnan(b)
    groups, n = a.shape
    result = np.full(groups, np.nan)
    if groups == 0 or n < 2:
        return result
    if n * n > _BLOCK_ENTRIES:
        function = COEFFICIENTS[name]
        for row in range(groups):
            keep = valid[row]
            result[row] = function(a[row, keep], b[row, keep])
        return result
    step = max(1, _BLOCK_ENTRIES // (n * n))
    for start in range(0, groups, step):
        rows = slice(start, start + step)
        result[rows] = _rowwise_block(name, a[rows], b[rows], valid[rows])
    return result


def _pearson_rows(a: np.ndarray, b: np.ndarray, valid: np.ndarray) -> np.ndarray:
    size = valid.sum(axis=1)
    count = np.maximum(size, 1)
    a = np.where(valid, a, 0.0)
    b = np.where(valid, b, 0.0)
    da = np.where(valid, a - (a.sum(axis=1) / count)[:, None], 0.0)
    db = np.where(valid, b - (b.sum(axis=1) / count)[:, None], 0.0)
    constant = (
        (size < 2)
        | (np.where(valid, a, -np.inf).max(axis=1) == np.where(valid, a, np.inf).min(axis=1))
        | (np.where(valid, b, -np.inf).max(axis=1) == np.where(valid, b, np.inf).min(axis=1))
    )
    scale_a = np.abs(da).max(axis=1)
    scale_b = np.abs(db).max(axis=1)
    da = da / np.where(scale_a > 0, scale_a, 1.0)[:, None]
    db = db / np.where(scale_b > 0, scale_b, 1.0)[:, None]
    denominator = np.sqrt((da * da).sum(axis=1) * (db * db).sum(axis=1))
    undefined = constant | (denominator == 0)
    ratio = (da * db).sum(axis=1) / np.where(undefined, 1.0, denominator)
    return np.where(undefined, np.nan, np.clip(ratio, -1.0, 1.0))


def _rowwise_block(name: str, a: np.ndarray, b: np.ndarray, valid: np.ndarray) -> np.ndarray:
    if name == "pearson":
        return _pearson_rows(a, b, valid)
    n = a.shape[1]
    a = np.where(valid, a, 0.0)
    b = np.where(valid, b, 0.0)
    # sign[g, i, j] is the sign of value i minus value j in row g.
    sign_x = np.sign(a[:, :, None] - a[:, None, :]).astype(np.int8)
    sign_y = np.sign(b[:, :, None] - b[:, None, :]).astype(np.int8)
    both = valid[:, :, None] & valid[:, None, :]
    if name == "spearman":
        return _pearson_rows(
            _ranks_from_signs(sign_x, both), _ranks_from_signs(sign_y, both), valid
        )

    pairs = both & np.triu(np.ones((n, n), dtype=bool), k=1)
    product = sign_x * sign_y
    concordant = (pairs & (product > 0)).sum(axis=(1, 2))
    discordant = (pairs & (product < 0)).sum(axis=(1, 2))
    tied_both = (pairs & (sign_x == 0) & (sign_y == 0)).sum(axis=(1, 2))
    size = valid.sum(axis=1)
    total = size * (size - 1) // 2
    difference = (concordant - discordant).astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        if name == "pairwise_accuracy":
            return np.where(total > 0, (concordant + tied_both) / total, np.nan)
        if name == "kendall_b":
            tied_x = (pairs & (sign_x == 0)).sum(axis=(1, 2)) - tied_both
            tied_y = (pairs & (sign_y == 0)).sum(axis=(1, 2)) - tied_both
            untied = concordant + discordant
            product_of_sums = (untied + tied_x).astype(float) * (untied + tied_y)
            return np.where(product_of_sums > 0, difference / np.sqrt(product_of_sums), np.nan)
        # kendall_c
        m = np.minimum(_distinct(sign_x, both, valid), _distinct(sign_y, both, valid))
        scale = size.astype(float) ** 2 * (m - 1)
        return np.where(m > 1, 2.0 * difference * m / np.where(m > 1, scale, 1.0), np.nan)


def _ranks_from_signs(sign: np.ndarray, both: np.ndarray) -> np.ndarray:
    """Average ranks within each row: values below, plus half of the tie group."""
    below = (both & (sign > 0)).sum(axis=2)
    equal = (both & (sign == 0)).sum(axis=2)
    return below + (equal + 1) / 2.0


def _distinct(sign: np.ndarray, both: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Number of distinct valid values in each row."""
    n = sign.shape[1]
    earlier = np.tril(np.ones((n, n), dtype=bool), k=-1)
    repeats = (both & (sign == 0) & earlier).any(axis=2)
    return (valid & ~repeats).sum(axis=1)


CALIBRATED = "pairwise_accuracy_calibrated"
"""Name of pairwise accuracy after tie calibration, accepted wherever a coefficient is."""

_PAIR_LIMIT = 30_000_000


def calibrate_ties(x: ArrayLike, y: ArrayLike) -> tuple[float, float, float, int, int]:
    """Tie calibration of Deutsch, Foster and Freitag (2023) for pairwise accuracy.

    ``x`` (metric) and ``y`` (human) are ``(groups, n)`` arrays with ``nan`` for missing
    cells. Two metric scores of a group count as tied when they differ by at most
    ``epsilon``; every absolute difference between two metric scores of a group, and
    zero, is a candidate, and the one that gives the highest pairwise accuracy, averaged
    over groups, is chosen (the smallest such value). One ``epsilon`` serves all groups.

    Returns ``(epsilon, accuracy, accuracy without calibration, groups, pairs)``, where
    groups are the rows with at least two points.
    """
    a = np.asarray(x, dtype=float)
    b = np.asarray(y, dtype=float)
    if a.ndim != 2 or a.shape != b.shape:
        raise ValueError("calibrate_ties needs two arrays of the same two-dimensional shape")
    groups, n = a.shape
    if groups * n * (n - 1) // 2 > _PAIR_LIMIT:
        raise ValueError(
            f"tie calibration compares all pairs within a group; {groups} group(s) of {n} "
            "outputs are too many pairs, use a grouped level"
        )
    valid = ~np.isnan(a) & ~np.isnan(b)
    first, second = np.triu_indices(n, k=1)
    in_pair = valid[:, first] & valid[:, second]
    per_group = in_pair.sum(axis=1)
    n_groups = int(np.count_nonzero(per_group))
    n_pairs = int(per_group.sum())
    if n_pairs == 0:
        return math.nan, math.nan, math.nan, 0, 0
    a = np.where(valid, a, 0.0)
    b = np.where(valid, b, 0.0)
    dx = (a[:, first] - a[:, second])[in_pair]
    dy = (b[:, first] - b[:, second])[in_pair]
    # A pair weighs 1 / (pairs of its group * number of groups), so that the sums below
    # are pairwise accuracies averaged over groups. When all groups have the same
    # number of pairs the sums are kept as integer counts and divided once at the end.
    sizes = per_group[per_group > 0]
    if (sizes == sizes[0]).all():
        weight = np.ones(n_pairs, dtype=np.int64)
        divisor = float(sizes[0]) * n_groups
    else:
        weight = np.broadcast_to(
            (1.0 / (np.maximum(per_group, 1) * n_groups))[:, None], in_pair.shape
        )
        weight = weight[in_pair]
        divisor = 1.0
    distance = np.abs(dx)
    order = np.argsort(distance, kind="stable")
    distance = distance[order]
    tied_human = np.cumsum((weight * (dy == 0))[order])
    concordant = np.cumsum((weight * (dx * dy > 0))[order])
    # With threshold e a pair is right when the humans tie it and its distance is at
    # most e, or when it is concordant and its distance is above e.
    last = np.flatnonzero(np.append(distance[1:] != distance[:-1], True))
    candidates = distance[last]
    accuracy = (tied_human[last] + (concordant[-1] - concordant[last])) / divisor
    if candidates[0] > 0:
        candidates = np.concatenate(([0.0], candidates))
        accuracy = np.concatenate(([concordant[-1] / divisor], accuracy))
    best = int(np.argmax(np.round(accuracy, 12)))
    return float(candidates[best]), float(accuracy[best]), float(accuracy[0]), n_groups, n_pairs
