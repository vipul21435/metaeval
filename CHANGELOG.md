# Changelog

## 0.1.0 (2026-11-13)

First release.

- Coefficients: Pearson, Spearman, Kendall's tau-b and tau-c, pairwise accuracy with ties, and pairwise
  accuracy after tie calibration.
- Groupings: system level, pooled, grouped by item and grouped by system; every result names its grouping.
- Uncertainty: bootstrap percentile intervals over items, systems or both.
- Differences between two metrics: paired bootstrap, permutation test, Williams test.
- Ordinal agreement of a grader with human ratings (quadratic weighted kappa, exact and adjacent agreement,
  mean signed difference) and an audit of how it uses the scale.
- Human ceiling by leave-one-annotator-out.
- Commands `correlate`, `compare`, `scale`, `ceiling` and `report`, in Markdown and JSON.
- Input: long CSV or JSONL and nested JSONL, with several annotators and rated dimensions.

Not included: charts.
