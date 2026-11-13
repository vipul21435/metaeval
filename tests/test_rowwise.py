"""The batched coefficients agree with the one-dimensional ones, row by row."""

from __future__ import annotations

import math

import numpy as np
import pytest

from metaeval import coefficients
from metaeval.coefficients import COEFFICIENTS, rowwise


def random_rows(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    groups, n = int(rng.integers(1, 7)), int(rng.integers(2, 14))
    x = np.floor(rng.normal(size=(groups, n)) * 2.0)
    y = np.floor(rng.normal(size=(groups, n)) * 1.5)
    x[rng.random((groups, n)) < 0.15] = np.nan
    y[rng.random((groups, n)) < 0.15] = np.nan
    if rng.random() < 0.15:
        x[0] = 1.0
    if rng.random() < 0.15:
        y[-1, : n // 2] = np.nan
    return x, y


def assert_rows_match(name: str, x: np.ndarray, y: np.ndarray) -> None:
    values = rowwise(name, x, y)
    assert values.shape == (x.shape[0],)
    for row in range(x.shape[0]):
        keep = ~np.isnan(x[row]) & ~np.isnan(y[row])
        expected = COEFFICIENTS[name](x[row, keep], y[row, keep])
        assert values[row] == pytest.approx(expected, abs=1e-12, nan_ok=True), (name, row)


@pytest.mark.parametrize("name", sorted(COEFFICIENTS))
def test_rowwise_matches_the_one_dimensional_functions(name):
    rng = np.random.default_rng(41)
    for _ in range(150):
        assert_rows_match(name, *random_rows(rng))


@pytest.mark.parametrize("name", sorted(COEFFICIENTS))
@pytest.mark.parametrize("entries", [1, 60, 400])
def test_rowwise_is_the_same_in_small_blocks_and_row_by_row(name, entries, monkeypatch):
    # A budget of 1 entry sends every row to the sort-based functions; 60 and 400
    # split the rows into blocks of different sizes.
    rng = np.random.default_rng(42)
    cases = [random_rows(rng) for _ in range(40)]
    expected = [rowwise(name, x, y) for x, y in cases]
    monkeypatch.setattr(coefficients, "_BLOCK_ENTRIES", entries)
    for (x, y), reference in zip(cases, expected, strict=True):
        np.testing.assert_allclose(rowwise(name, x, y), reference, atol=1e-12, equal_nan=True)


def test_rowwise_edge_cases():
    assert rowwise("pearson", np.empty((0, 5)), np.empty((0, 5))).shape == (0,)
    assert np.isnan(rowwise("kendall_b", np.ones((3, 1)), np.ones((3, 1)))).all()
    nan = math.nan
    x = np.array([[1.0, 2.0, 3.0], [1.0, nan, nan], [nan, nan, nan]])
    y = np.array([[1.0, 2.0, 3.0], [2.0, 1.0, nan], [nan, nan, nan]])
    for name in COEFFICIENTS:
        values = rowwise(name, x, y)
        assert values[0] == pytest.approx(1.0)
        assert np.isnan(values[1:]).all()
    with pytest.raises(ValueError, match="unknown coefficient"):
        rowwise("tau", x, y)
    with pytest.raises(ValueError, match="two-dimensional"):
        rowwise("pearson", x[0], y[0])
    with pytest.raises(ValueError, match="two-dimensional"):
        rowwise("pearson", x, y[:, :2])
    with pytest.raises(ValueError, match="infinite"):
        rowwise("pearson", np.where(x == 3.0, np.inf, x), y)
