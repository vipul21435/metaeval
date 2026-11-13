"""metaeval: how well does a metric or an automatic grader track human ratings."""

from __future__ import annotations

from metaeval.agreement import (
    ScaleAudit,
    adjacent_agreement,
    exact_agreement,
    mean_signed_difference,
    quadratic_weighted_kappa,
    scale_audit,
)
from metaeval.ceiling import Ceiling, human_ceiling, metric_against_rest
from metaeval.coefficients import (
    CALIBRATED,
    COEFFICIENTS,
    PairCounts,
    calibrate_ties,
    kendall_tau_b,
    kendall_tau_c,
    pair_counts,
    pairwise_accuracy,
    pearson,
    rowwise,
    spearman,
)
from metaeval.grouping import GROUPINGS, STATISTICS, Correlation, correlate
from metaeval.resampling import (
    PERMUTE,
    PERMUTE_BY_GROUPING,
    RESAMPLE,
    Interval,
    PairedBootstrap,
    PermutationTest,
    bootstrap,
    paired_bootstrap,
    permutation_test,
    resample_statistic,
)
from metaeval.significance import WilliamsTest, williams, williams_test
from metaeval.table import ScoreFormatError, ScoreTable, load_scores

__version__ = "0.1.0"

__all__ = [
    "CALIBRATED",
    "COEFFICIENTS",
    "GROUPINGS",
    "PERMUTE",
    "PERMUTE_BY_GROUPING",
    "RESAMPLE",
    "STATISTICS",
    "Ceiling",
    "Correlation",
    "Interval",
    "PairCounts",
    "PairedBootstrap",
    "PermutationTest",
    "ScaleAudit",
    "ScoreFormatError",
    "ScoreTable",
    "WilliamsTest",
    "__version__",
    "adjacent_agreement",
    "bootstrap",
    "calibrate_ties",
    "correlate",
    "exact_agreement",
    "human_ceiling",
    "kendall_tau_b",
    "kendall_tau_c",
    "load_scores",
    "mean_signed_difference",
    "metric_against_rest",
    "pair_counts",
    "paired_bootstrap",
    "pairwise_accuracy",
    "pearson",
    "permutation_test",
    "quadratic_weighted_kappa",
    "resample_statistic",
    "rowwise",
    "scale_audit",
    "spearman",
    "williams",
    "williams_test",
]
