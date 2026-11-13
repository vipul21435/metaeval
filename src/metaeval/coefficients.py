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
    "COEFFICIENTS",
    "PairCounts",
    "kendall_tau_b",
    "kendall_tau_c",
    "pair_counts",
    "pairwise_accuracy",
    "pearson",
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
