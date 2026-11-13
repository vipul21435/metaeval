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

__version__ = "0.1.0.dev0"

__all__ = [
    "COEFFICIENTS",
    "PairCounts",
    "__version__",
    "kendall_tau_b",
    "kendall_tau_c",
    "pair_counts",
    "pairwise_accuracy",
    "pearson",
    "spearman",
]
