"""metaeval: how well does a metric or an automatic grader track human ratings."""

from __future__ import annotations

from metaeval.coefficients import (
    COEFFICIENTS,
    PairCounts,
    kendall_tau_b,
    kendall_tau_c,
    pair_counts,
    pairwise_accuracy,
    pearson,
    spearman,
)
from metaeval.grouping import GROUPINGS, Correlation, correlate
from metaeval.table import ScoreFormatError, ScoreTable, load_scores

__version__ = "0.1.0.dev0"

__all__ = [
    "COEFFICIENTS",
    "GROUPINGS",
    "Correlation",
    "PairCounts",
    "ScoreFormatError",
    "ScoreTable",
    "__version__",
    "correlate",
    "kendall_tau_b",
    "kendall_tau_c",
    "load_scores",
    "pair_counts",
    "pairwise_accuracy",
    "pearson",
    "spearman",
]
