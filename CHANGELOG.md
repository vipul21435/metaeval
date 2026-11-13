# Changelog

## 0.1.1 (2026-11-13)

Fixes after an independent review of 0.1.0.

- The permutation test no longer exchanges single outputs by default. That variant declares two equally good
  metrics different far too often when a metric has an error of its own per system or per item (up to 66 % of
  simulated tables at a nominal 5 %). The unit now follows the grouping: whole systems for `system` and
  `by_item`, whole items for `by_system`, both in turn for `pooled` with the larger p value. `--permute` and
  the `unit` argument choose another unit; `permutation_test` no longer takes `resample`, and its result names
  the `unit`.
- `examples/simulate_tests.py` measures how often each test rejects on simulated tables.
- The README states what each test assumes, that the paired bootstrap has little power, how the kappa weights
  relate to sklearn's `labels`, and that `bigram_f1` is not ROUGE-2; the SummEval comparison is rewritten.
- An annotator without any rating is no longer counted in the header of the output.

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
