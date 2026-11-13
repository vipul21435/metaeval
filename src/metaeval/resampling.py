"""Uncertainty by resampling: bootstrap intervals and tests for a difference.

The score arrays have shape ``(n_systems, n_items)``. A bootstrap resample draws, with
replacement, the systems, the items, or both, and the statistic is recomputed on the
resampled arrays; a system or item drawn twice appears twice. These are the Boot-Systems,
Boot-Inputs and Boot-Both procedures of Deutsch, Dror and Roth (2021), with percentile
intervals, and the permutation test is their Perm-Systems, Perm-Inputs and Perm-Both.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass

import numpy as np
from numpy.typing import ArrayLike

from metaeval.coefficients import COEFFICIENTS
from metaeval.grouping import GROUPINGS, _matrices, statistic

__all__ = [
    "RESAMPLE",
    "Interval",
    "PairedBootstrap",
    "PermutationTest",
    "bootstrap",
    "paired_bootstrap",
    "percentile_interval",
    "permutation_test",
    "resample_statistic",
]

RESAMPLE: dict[str, str] = {
    "items": "items resampled, systems fixed",
    "systems": "systems resampled, items fixed",
    "both": "systems and items resampled",
}
"""What a resample draws, by name."""


def _check(coefficient: str, grouping: str, resample: str, n_resamples: int) -> None:
    if coefficient not in COEFFICIENTS:
        raise ValueError(f"unknown coefficient {coefficient!r}; choose from {sorted(COEFFICIENTS)}")
    if grouping not in GROUPINGS:
        raise ValueError(f"unknown grouping {grouping!r}; choose from {list(GROUPINGS)}")
    if resample not in RESAMPLE:
        raise ValueError(f"unknown resample {resample!r}; choose from {list(RESAMPLE)}")
    if n_resamples < 1:
        raise ValueError("n_resamples must be at least 1")


def _json(fields: dict[str, object]) -> dict[str, object]:
    """Replace ``nan`` by ``None`` so that the result is valid JSON."""
    return {
        key: None if isinstance(value, float) and math.isnan(value) else value
        for key, value in fields.items()
    }


def resample_statistic(
    function: Callable[..., float],
    arrays: Sequence[np.ndarray],
    *,
    resample: str = "both",
    n_resamples: int = 1000,
    seed: int = 0,
) -> np.ndarray:
    """Values of ``function(*arrays)`` on bootstrap resamples of the arrays.

    Every array has systems on its second-to-last axis and items on its last axis, and
    all arrays are resampled with the same draws. The result is reproducible for a given
    ``seed``.
    """
    if resample not in RESAMPLE:
        raise ValueError(f"unknown resample {resample!r}; choose from {list(RESAMPLE)}")
    n_systems, n_items = arrays[0].shape[-2:]
    rng = np.random.default_rng(seed)
    values = np.empty(n_resamples)
    for index in range(n_resamples):
        sample = list(arrays)
        if resample != "items":
            rows = rng.integers(0, n_systems, size=n_systems)
            sample = [array[..., rows, :] for array in sample]
        if resample != "systems":
            columns = rng.integers(0, n_items, size=n_items)
            sample = [array[..., columns] for array in sample]
        values[index] = function(*sample)
    return values


def percentile_interval(values: np.ndarray, confidence: float = 0.95) -> tuple[float, float, int]:
    """Percentile interval of resampled values, and how many of them were undefined."""
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1")
    defined = values[~np.isnan(values)]
    n_undefined = int(values.size - defined.size)
    if defined.size == 0:
        return math.nan, math.nan, n_undefined
    tail = (1.0 - confidence) / 2.0
    low, high = np.quantile(defined, [tail, 1.0 - tail])
    return float(low), float(high), n_undefined


@dataclass(frozen=True)
class Interval:
    """A coefficient with its bootstrap percentile interval.

    ``n_undefined`` counts the resamples in which the statistic was undefined; they are
    left out of the interval.
    """

    coefficient: str
    grouping: str
    value: float
    low: float
    high: float
    confidence: float
    resample: str
    n_resamples: int
    n_undefined: int
    seed: int

    def to_dict(self) -> dict[str, object]:
        return _json(asdict(self))


def bootstrap(
    metric: ArrayLike,
    human: ArrayLike,
    *,
    coefficient: str,
    grouping: str,
    resample: str = "both",
    n_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 0,
    undefined: str = "skip",
) -> Interval:
    """Bootstrap percentile interval for one coefficient at one grouping."""
    _check(coefficient, grouping, resample, n_resamples)
    m, h, _ = _matrices(metric, human)

    def function(a: np.ndarray, b: np.ndarray) -> float:
        return statistic(a, b, coefficient, grouping, undefined)

    values = resample_statistic(
        function, [m, h], resample=resample, n_resamples=n_resamples, seed=seed
    )
    low, high, n_undefined = percentile_interval(values, confidence)
    return Interval(
        coefficient=coefficient,
        grouping=grouping,
        value=function(m, h),
        low=low,
        high=high,
        confidence=confidence,
        resample=resample,
        n_resamples=n_resamples,
        n_undefined=n_undefined,
        seed=seed,
    )


@dataclass(frozen=True)
class PairedBootstrap:
    """The difference ``value_a - value_b`` of two metrics on the same resamples.

    ``low`` and ``high`` bound the difference; ``p_value`` is two-sided: twice the smaller
    share of resampled differences on either side of zero, with one added to the count
    and to the number of resamples.
    """

    coefficient: str
    grouping: str
    value_a: float
    value_b: float
    difference: float
    low: float
    high: float
    p_value: float
    confidence: float
    resample: str
    n_resamples: int
    n_undefined: int
    seed: int

    def to_dict(self) -> dict[str, object]:
        return _json(asdict(self))


def paired_bootstrap(
    metric_a: ArrayLike,
    metric_b: ArrayLike,
    human: ArrayLike,
    *,
    coefficient: str,
    grouping: str,
    resample: str = "both",
    n_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 0,
    undefined: str = "skip",
) -> PairedBootstrap:
    """Paired bootstrap for the difference between two metrics against the same humans."""
    _check(coefficient, grouping, resample, n_resamples)
    a, h, _ = _matrices(metric_a, human)
    b, _, _ = _matrices(metric_b, human)

    def difference(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
        return statistic(x, z, coefficient, grouping, undefined) - statistic(
            y, z, coefficient, grouping, undefined
        )

    values = resample_statistic(
        difference, [a, b, h], resample=resample, n_resamples=n_resamples, seed=seed
    )
    low, high, n_undefined = percentile_interval(values, confidence)
    defined = values[~np.isnan(values)]
    if defined.size:
        below = (np.count_nonzero(defined <= 0) + 1) / (defined.size + 1)
        above = (np.count_nonzero(defined >= 0) + 1) / (defined.size + 1)
        p_value = min(1.0, 2.0 * min(below, above))
    else:
        p_value = math.nan
    value_a = statistic(a, h, coefficient, grouping, undefined)
    value_b = statistic(b, h, coefficient, grouping, undefined)
    return PairedBootstrap(
        coefficient=coefficient,
        grouping=grouping,
        value_a=value_a,
        value_b=value_b,
        difference=value_a - value_b,
        low=low,
        high=high,
        p_value=p_value,
        confidence=confidence,
        resample=resample,
        n_resamples=n_resamples,
        n_undefined=n_undefined,
        seed=seed,
    )


@dataclass(frozen=True)
class PermutationTest:
    """Permutation test of ``value_a - value_b`` against no difference.

    ``p_value`` is the share of permutations whose difference is at least as extreme as
    the observed one (in absolute value when two-sided), with one added to the count
    and to the number of permutations.
    """

    coefficient: str
    grouping: str
    difference: float
    p_value: float
    alternative: str
    resample: str
    n_resamples: int
    n_undefined: int
    seed: int

    def to_dict(self) -> dict[str, object]:
        return _json(asdict(self))


def _standardize(scores: np.ndarray) -> np.ndarray:
    present = ~np.isnan(scores)
    if not present.any():
        return scores
    centred = scores - scores[present].mean()
    spread = centred[present].std()
    return centred / spread if spread > 0 else centred


def permutation_test(
    metric_a: ArrayLike,
    metric_b: ArrayLike,
    human: ArrayLike,
    *,
    coefficient: str,
    grouping: str,
    resample: str = "both",
    n_resamples: int = 1000,
    alternative: str = "two-sided",
    seed: int = 0,
    undefined: str = "skip",
) -> PermutationTest:
    """Permutation test for the difference between two metrics against the same humans.

    Both metrics are standardised, then their scores are exchanged with probability one
    half: for whole systems (``resample="systems"``), for whole items (``"items"``) or
    output by output (``"both"``). ``alternative`` is ``"two-sided"`` or ``"greater"``
    (metric a correlates more than metric b).
    """
    _check(coefficient, grouping, resample, n_resamples)
    if alternative not in ("two-sided", "greater"):
        raise ValueError("alternative must be 'two-sided' or 'greater'")
    a, h, _ = _matrices(metric_a, human)
    b, _, _ = _matrices(metric_b, human)
    a, b = _standardize(a), _standardize(b)

    def difference(x: np.ndarray, y: np.ndarray) -> float:
        return statistic(x, h, coefficient, grouping, undefined) - statistic(
            y, h, coefficient, grouping, undefined
        )

    observed = difference(a, b)
    shape = {"both": a.shape, "systems": (a.shape[0], 1), "items": (1, a.shape[1])}[resample]
    rng = np.random.default_rng(seed)
    values = np.empty(n_resamples)
    for index in range(n_resamples):
        swap = np.broadcast_to(rng.random(shape) < 0.5, a.shape)
        values[index] = difference(np.where(swap, b, a), np.where(swap, a, b))
    defined = values[~np.isnan(values)]
    if math.isnan(observed) or defined.size == 0:
        p_value = math.nan
    else:
        # A small tolerance keeps permutations that reproduce the observed value, up to
        # rounding, on the "at least as extreme" side.
        slack = 1e-12
        if alternative == "two-sided":
            extreme = np.count_nonzero(np.abs(defined) >= abs(observed) - slack)
        else:
            extreme = np.count_nonzero(defined >= observed - slack)
        p_value = (extreme + 1) / (defined.size + 1)
    return PermutationTest(
        coefficient=coefficient,
        grouping=grouping,
        difference=observed,
        p_value=p_value,
        alternative=alternative,
        resample=resample,
        n_resamples=n_resamples,
        n_undefined=int(values.size - defined.size),
        seed=seed,
    )
