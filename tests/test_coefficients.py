"""Coefficients: hand-worked cases, scipy as an independent reference, and properties."""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest
from scipy import stats

from metaeval.coefficients import (
    COEFFICIENTS,
    PairCounts,
    _average_ranks,
    _count_inversions,
    kendall_tau_b,
    kendall_tau_c,
    pair_counts,
    pairwise_accuracy,
    pearson,
    spearman,
)

# Six points worked by hand. Listing the 15 pairs (i, j) with i < j:
#   concordant:      (0,3) (0,4) (0,5) (2,4) (2,5)                -> C   = 5
#   discordant:      (1,2)                                        -> D   = 1
#   tied in x only:  (0,1) (2,3)                                  -> Tx  = 2
#   tied in y only:  (0,2) (1,3) (1,4) (1,5) (3,4) (3,5)          -> Ty  = 6
#   tied in both:    (4,5)                                        -> Txy = 1
HAND_X = [1, 1, 2, 2, 3, 3]
HAND_Y = [1, 2, 1, 2, 2, 2]


def brute_force_counts(x, y) -> tuple[int, int, int, int, int]:
    """Count the five kinds of pair by looking at every pair."""
    concordant = discordant = tied_x = tied_y = tied_both = 0
    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            if dx == 0 and dy == 0:
                tied_both += 1
            elif dx == 0:
                tied_x += 1
            elif dy == 0:
                tied_y += 1
            elif dx * dy > 0:
                concordant += 1
            else:
                discordant += 1
    return concordant, discordant, tied_x, tied_y, tied_both


def as_tuple(counts: PairCounts) -> tuple[int, int, int, int, int]:
    return (
        counts.concordant,
        counts.discordant,
        counts.tied_x,
        counts.tied_y,
        counts.tied_both,
    )


def random_cases(seed: int, count: int):
    """Paired vectors with heavy ties, a few levels each, plus some continuous ones."""
    rng = np.random.default_rng(seed)
    for _ in range(count):
        n = int(rng.integers(2, 120))
        levels_x = int(rng.integers(1, 7))
        levels_y = int(rng.integers(1, 7))
        latent = rng.normal(size=n)
        if rng.random() < 0.2:
            x = latent + rng.normal(size=n)
        else:
            x = np.floor(rng.random(n) * levels_x + latent * 0.5)
        y = np.floor(rng.random(n) * levels_y + latent * 0.5)
        if rng.random() < 0.05:
            x = np.full(n, 2.0)
        if rng.random() < 0.05:
            y = np.full(n, -1.0)
        yield x, y


def scipy_value(function, x, y, **kwargs) -> float:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(function(x, y, **kwargs).statistic)


def assert_same(ours: float, reference: float) -> None:
    if math.isnan(reference):
        assert math.isnan(ours)
    else:
        assert ours == pytest.approx(reference, abs=1e-12)


def test_hand_worked_pair_counts():
    counts = pair_counts(HAND_X, HAND_Y)
    assert as_tuple(counts) == (5, 1, 2, 6, 1)
    assert counts.n == 6
    assert counts.total == 15
    assert counts.distinct_x == 3
    assert counts.distinct_y == 2


def test_hand_worked_tau_b():
    # (5 - 1) / sqrt((5 + 1 + 2) * (5 + 1 + 6)) = 4 / sqrt(96)
    assert kendall_tau_b(HAND_X, HAND_Y) == pytest.approx(4 / math.sqrt(96))


def test_hand_worked_tau_c():
    # m = min(3, 2) = 2, so 2 * (5 - 1) / (6^2 * (2 - 1) / 2) = 8 / 18
    assert kendall_tau_c(HAND_X, HAND_Y) == pytest.approx(4 / 9)


def test_tau_c_reaches_one_on_a_rectangular_table():
    # Three levels of x against two of y, in perfect order. Ties are unavoidable, so
    # tau-b stays below 1, while tau-c is scaled by m = min(3, 2) = 2 and reaches 1.
    x = [1, 1, 2, 3, 3, 3]
    y = [1, 1, 1, 2, 2, 2]
    # concordant: each of the points 0, 1, 2 against each of 3, 4, 5      -> C   = 9
    # tied in y only: (0,2) (1,2)                                         -> Ty  = 2
    # tied in both: (0,1) (3,4) (3,5) (4,5)                               -> Txy = 4
    assert as_tuple(pair_counts(x, y)) == (9, 0, 0, 2, 4)
    assert kendall_tau_c(x, y) == pytest.approx(2 * 9 / (36 * 0.5))
    assert kendall_tau_c(x, y) == pytest.approx(1.0)
    assert kendall_tau_b(x, y) == pytest.approx(9 / math.sqrt(9 * 11))


def test_tau_c_on_a_table_with_a_discordant_cell():
    x = [1, 1, 2, 2, 3, 3]
    y = [1, 1, 1, 2, 2, 2]
    # concordant: 0 and 1 against 3, 4, 5 (6 pairs), then (2,4) (2,5)     -> C   = 8
    # tied in x only: (2,3)                                               -> Tx  = 1
    # tied in y only: (0,2) (1,2) (3,4) (3,5)                             -> Ty  = 4
    # tied in both: (0,1) (4,5)                                           -> Txy = 2
    assert as_tuple(pair_counts(x, y)) == (8, 0, 1, 4, 2)
    assert kendall_tau_c(x, y) == pytest.approx(2 * 8 / (36 * 0.5))
    assert kendall_tau_b(x, y) == pytest.approx(8 / math.sqrt(9 * 12))


def test_hand_worked_pairwise_accuracy():
    # (C + Txy) / 15 = (5 + 1) / 15
    assert pairwise_accuracy(HAND_X, HAND_Y) == pytest.approx(0.4)


def test_pairwise_accuracy_rewards_a_predicted_tie():
    human = [1, 2, 2, 3]
    # The metric ties the two outputs the humans tied: all six pairs are right.
    assert pairwise_accuracy([0.1, 0.2, 0.2, 0.9], human) == 1.0
    # Same order but the tie is broken: five concordant pairs, one tied in human only.
    assert pairwise_accuracy([0.1, 0.2, 0.3, 0.9], human) == pytest.approx(5 / 6)
    assert kendall_tau_b([0.1, 0.2, 0.3, 0.9], human) == pytest.approx(5 / math.sqrt(30))


def test_pairwise_accuracy_is_defined_for_constant_vectors():
    assert pairwise_accuracy([3, 3, 3], [1, 2, 3]) == 0.0
    assert pairwise_accuracy([3, 3, 3], [2, 2, 2]) == 1.0
    assert pairwise_accuracy([1, 2, 3], [2, 2, 2]) == 0.0


@pytest.mark.parametrize("name", ["pearson", "spearman", "kendall_b", "kendall_c"])
def test_constant_vector_is_undefined(name):
    function = COEFFICIENTS[name]
    assert math.isnan(function([1, 1, 1, 1], [1, 2, 3, 4]))
    assert math.isnan(function([1, 2, 3, 4], [5, 5, 5, 5]))
    assert math.isnan(function([2, 2], [2, 2]))


@pytest.mark.parametrize("name", sorted(COEFFICIENTS))
def test_fewer_than_two_points_is_undefined(name):
    function = COEFFICIENTS[name]
    assert math.isnan(function([], []))
    assert math.isnan(function([1.0], [2.0]))


def test_constant_vector_with_inexact_mean_is_undefined():
    # (0.1 + 0.1 + 0.1) / 3 is not 0.1 in floating point; the residuals must not be
    # mistaken for variation.
    assert math.isnan(pearson([0.1, 0.1, 0.1], [1.0, 2.0, 3.0]))
    assert math.isnan(spearman([0.1, 0.1, 0.1], [1.0, 2.0, 3.0]))


def test_pearson_does_not_depend_on_the_scale():
    # The squared deviations of the first vector underflow to zero and those of the
    # second overflow if they are not rescaled first.
    expected = pearson([1, 2, 4], [1, 2, 3])
    assert pearson([1e-200, 2e-200, 4e-200], [1, 2, 3]) == pytest.approx(expected)
    assert pearson([1e200, 2e200, 4e200], [1, 2, 3]) == pytest.approx(expected)
    assert pearson([1e-200, 2e-200, 4e-200], [1e200, 2e200, 3e200]) == pytest.approx(expected)


@pytest.mark.parametrize("name", sorted(COEFFICIENTS))
def test_invalid_input_raises(name):
    function = COEFFICIENTS[name]
    with pytest.raises(ValueError, match="same length"):
        function([1, 2, 3], [1, 2])
    with pytest.raises(ValueError, match="one-dimensional"):
        function([[1, 2], [3, 4]], [[1, 2], [3, 4]])
    with pytest.raises(ValueError, match="finite"):
        function([1.0, float("nan"), 3.0], [1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="finite"):
        function([1.0, 2.0, 3.0], [1.0, float("inf"), 3.0])


def test_average_ranks_match_scipy():
    rng = np.random.default_rng(7)
    for _ in range(50):
        values = rng.integers(0, 6, size=int(rng.integers(1, 40))).astype(float)
        np.testing.assert_allclose(_average_ranks(values), stats.rankdata(values))


def test_count_inversions_matches_brute_force():
    rng = np.random.default_rng(11)
    assert _count_inversions(np.array([], dtype=np.int64)) == 0
    assert _count_inversions(np.array([4])) == 0
    for n in range(2, 70):
        for levels in (1, 2, 3, n, 5 * n):
            codes = rng.integers(0, levels, size=n)
            expected = sum(1 for i in range(n) for j in range(i + 1, n) if codes[i] > codes[j])
            assert _count_inversions(codes) == expected
    assert _count_inversions(np.arange(50)) == 0
    assert _count_inversions(np.arange(50)[::-1].copy()) == 50 * 49 // 2


def test_pair_counts_match_brute_force():
    for x, y in random_cases(seed=3, count=150):
        counts = pair_counts(x, y)
        assert as_tuple(counts) == brute_force_counts(x, y)
        assert sum(as_tuple(counts)) == counts.total
        assert counts.distinct_x == len(set(x.tolist()))
        assert counts.distinct_y == len(set(y.tolist()))


def test_pearson_matches_scipy():
    for x, y in random_cases(seed=21, count=200):
        assert_same(pearson(x, y), scipy_value(stats.pearsonr, x, y))


def test_spearman_matches_scipy():
    for x, y in random_cases(seed=22, count=200):
        assert_same(spearman(x, y), scipy_value(stats.spearmanr, x, y))


def test_kendall_tau_b_matches_scipy():
    for x, y in random_cases(seed=23, count=200):
        assert_same(kendall_tau_b(x, y), scipy_value(stats.kendalltau, x, y, variant="b"))


def test_kendall_tau_c_matches_scipy():
    for x, y in random_cases(seed=24, count=200):
        assert_same(kendall_tau_c(x, y), scipy_value(stats.kendalltau, x, y, variant="c"))


def test_kendall_matches_scipy_on_a_long_vector():
    rng = np.random.default_rng(5)
    n = 20_000
    latent = rng.normal(size=n)
    x = np.round(latent + rng.normal(size=n), 1)
    y = np.clip(np.round(3 + latent), 1, 5)
    assert_same(kendall_tau_b(x, y), scipy_value(stats.kendalltau, x, y, variant="b"))
    assert_same(kendall_tau_c(x, y), scipy_value(stats.kendalltau, x, y, variant="c"))


def test_perfect_agreement_and_reversal():
    x = [0.2, 0.5, 0.7, 1.1, 1.3]
    y = [1, 2, 3, 4, 5]
    reversed_y = y[::-1]
    for name in ("spearman", "kendall_b", "kendall_c"):
        assert COEFFICIENTS[name](x, y) == pytest.approx(1.0)
        assert COEFFICIENTS[name](x, reversed_y) == pytest.approx(-1.0)
    assert pearson(y, [2 * value + 1 for value in y]) == pytest.approx(1.0)
    assert pairwise_accuracy(x, y) == 1.0
    assert pairwise_accuracy(x, reversed_y) == 0.0


def test_coefficients_are_symmetric_in_their_arguments():
    for x, y in random_cases(seed=31, count=60):
        for name, function in COEFFICIENTS.items():
            forward, backward = function(x, y), function(y, x)
            if math.isnan(forward):
                assert math.isnan(backward), name
            else:
                assert forward == pytest.approx(backward, abs=1e-12), name


def test_rank_coefficients_ignore_monotone_transforms():
    for x, y in random_cases(seed=32, count=60):
        stretched = np.exp(x / 3.0)
        for name in ("spearman", "kendall_b", "kendall_c", "pairwise_accuracy"):
            before, after = COEFFICIENTS[name](x, y), COEFFICIENTS[name](stretched, y)
            if math.isnan(before):
                assert math.isnan(after), name
            else:
                assert before == pytest.approx(after, abs=1e-12), name


def test_negating_one_vector_flips_the_sign():
    for x, y in random_cases(seed=33, count=60):
        for name in ("pearson", "spearman", "kendall_b", "kendall_c"):
            value, flipped = COEFFICIENTS[name](x, y), COEFFICIENTS[name](-x, y)
            if math.isnan(value):
                assert math.isnan(flipped), name
            else:
                assert value == pytest.approx(-flipped, abs=1e-12), name


def test_values_stay_in_range():
    for x, y in random_cases(seed=34, count=100):
        for name in ("pearson", "spearman", "kendall_b", "kendall_c"):
            value = COEFFICIENTS[name](x, y)
            assert math.isnan(value) or -1.0 <= value <= 1.0
        assert 0.0 <= pairwise_accuracy(x, y) <= 1.0
