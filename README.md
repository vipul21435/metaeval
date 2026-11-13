# metaeval

Correlate automatic metric or grader scores with human ratings, with the grouping stated on every number.

## Why this is hard

"The metric correlates 0.6 with human judgement" is not one number. The same metric on the same data gives
different correlations at system level (average each system, then correlate), over all outputs pooled, within
each item across systems, and within each system across items, and the differences can be large enough to change
which of two metrics looks better. Ties add a second choice: ordinal human ratings tie often, and the variants of
Kendall's tau treat ties differently. A reported correlation is hard to interpret unless both choices are stated.

## Status

This is an early version. What works today:

- Pearson, Spearman, Kendall's tau-b and tau-c, and pairwise accuracy with ties, implemented in numpy and
  tested against scipy and hand-worked cases.
- Four groupings (`system`, `pooled`, `by_item`, `by_system`); every result carries its grouping, the number of
  points and groups, and the number of groups where the coefficient was undefined.
- Reading long CSV or JSONL and nested JSONL score files, with several annotators and rated dimensions.
- `metaeval correlate` with Markdown and JSON output.

Not implemented yet: tie calibration for pairwise accuracy, bootstrap intervals, significance tests for the
difference between two metrics, ordinal agreement and the scale-usage audit, the human ceiling, the `compare`,
`scale`, `ceiling` and `report` commands, charts, and results on real public data. Until intervals exist, the
numbers this tool prints are point estimates and should be read as such.

## Install

```
pip install git+https://github.com/vipul21435/metaeval
```

Python 3.10 or later; the only runtime dependency is numpy.

## Quickstart

`examples/toy_scores.jsonl` is a small synthetic table (4 systems, 6 items, 2 annotators) written by
`examples/make_toy_scores.py`. `metric_a` follows the quality of the system and ignores the item; `metric_b`
follows the difficulty of the item and ignores the system.

```
$ metaeval correlate examples/toy_scores.jsonl
# Correlation with human ratings

- scores: examples/toy_scores.jsonl (4 systems, 6 items)
- human score: mean of 2 annotators (ann_1, ann_2), dimension 'overall'

| metric | grouping | pearson | spearman | kendall_b | kendall_c | pairwise_accuracy | points | groups |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| metric_a | system | 0.976 | 1.000 | 1.000 | 1.000 | 1.000 | 4 | 1 |
| metric_a | pooled | 0.562 | 0.534 | 0.394 | 0.402 | 0.634 | 24 | 1 |
| metric_a | by_item | 0.596 | 0.620 | 0.575 | 0.576 | 0.722 | 24 | 6 |
| metric_a | by_system | 0.308 | 0.261 | 0.190 | 0.192 | 0.550 | 24 | 4 |
| metric_b | system | 0.471 | 0.800 | 0.667 | 0.667 | 0.833 | 4 | 1 |
| metric_b | pooled | 0.671 | 0.678 | 0.538 | 0.551 | 0.707 | 24 | 1 |
| metric_b | by_item | 0.125 | 0.087 | 0.122 | 0.118 | 0.500 | 24 | 6 |
| metric_b | by_system | 0.797 | 0.763 | 0.676 | 0.683 | 0.783 | 24 | 4 |

Groupings:

- system: system level: scores averaged over items, one point per system
- pooled: item level, pooled: every output is one point, no grouping
- by_item: item level, grouped by item: coefficient across systems within each item, averaged over items
- by_system: item level, grouped by system: coefficient across items within each system, averaged over systems

points: systems at system level, otherwise outputs that have both scores. groups: groups with at least two such outputs.
```

On this toy table the pooled tau-b prefers `metric_b` (0.538 against 0.394), while within items, where systems
are compared on the same input, `metric_a` is ahead (0.575 against 0.122). With 4 systems and 6 items none of
these numbers is precise; the example only shows that the grouping has to be named.

Options: `--metric`, `--coefficient` and `--grouping` (each repeatable) restrict the table, `--dimension` and
`--annotators 'expert_*'` choose the human score, `--undefined zero` counts undefined groups as 0 instead of
leaving them out, and `--format json` prints the same results as JSON.

From Python:

```python
from metaeval import correlate, load_scores

table = load_scores("examples/toy_scores.jsonl")
human = table.human()  # mean over annotators, shape (systems, items)
for grouping in ("system", "by_item", "by_system"):
    result = correlate(table.metric("metric_a"), human, coefficient="kendall_b", grouping=grouping)
    print(f"{result.grouping:<9} {result.value:.3f}  groups={result.n_groups}")
```

```
system    1.000  groups=1
by_item   0.575  groups=6
by_system 0.190  groups=4
```

`correlate` has no default coefficient or grouping; both have to be named.

## Input

Scores are absolute numbers, one per output (an item and a system), not pairwise verdicts. Two record layouts
are read, and both can be mixed in one file.

Long records, as JSONL or as a CSV file with these columns (`dimension` is optional and applies to human rows):

```
item,system,source,name,dimension,score
doc_1,sys_a,human,ann_1,fluency,4
doc_1,sys_a,human,ann_2,fluency,5
doc_1,sys_a,metric,overlap,,0.31
```

Nested JSONL, one output per line; `human` maps annotators to scores, or dimensions to such maps:

```
{"item": "doc_1", "system": "sys_a", "human": {"ann_1": 4, "ann_2": 5}, "metrics": {"overlap": 0.31}}
{"item": "doc_2", "system": "sys_a", "human": {"fluency": {"ann_1": 2}}, "metrics": {"overlap": 0.12}}
```

A missing, empty or NaN score leaves that cell empty. The human score of an output is the mean over the chosen
annotators who rated it. A second score for the same cell is an error, reported with its line number.

## How it works

For a metric and a human score over `n` outputs, the `n(n-1)/2` pairs of outputs are counted as concordant
(`C`), discordant (`D`), tied in the metric only (`Tm`), tied in the human score only (`Th`) or tied in both
(`Tmh`). Two scores are tied when they are equal as floats.

- tau-b = `(C - D) / sqrt((C + D + Tm) * (C + D + Th))`
- tau-c = `2 * (C - D) / (n^2 * (k - 1) / k)`, with `k` the smaller number of distinct values of the two scores
- pairwise accuracy = `(C + Tmh) / (C + D + Tm + Th + Tmh)`

The tau formulas are the ones documented for
[`scipy.stats.kendalltau`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kendalltau.html),
which the tests use as the reference. Pairwise accuracy is the `acc_eq` of Deutsch, Foster and Freitag,
[Ties Matter: Meta-Evaluating Modern Metrics with Pairwise Accuracy and Tie Calibration](https://arxiv.org/abs/2305.14324)
(2023): a metric gets credit for a pair it orders like the humans and for a pair it ties where the humans tie.
Their tie calibration step, which lets a continuous metric predict ties, is not implemented here yet. Spearman
is the Pearson correlation of average ranks. The pair counts take `O(n log^2 n)` time.

The groupings follow the same literature. With `x[i, j]` the score of system `i` on item `j`:

- `system`: average `x[i, :]` over items, then one coefficient across systems. This is the system-level
  correlation of Deutsch, Dror and Roth,
  [A Statistical Analysis of Summarization Evaluation Metrics using Resampling Methods](https://arxiv.org/abs/2104.00054)
  (2021).
- `by_item`: one coefficient across systems for each item, averaged over items; their summary-level
  correlation, and "group-by-item" in Deutsch, Foster and Freitag.
- `by_system`: one coefficient across items for each system, averaged over systems ("group-by-system").
- `pooled`: one coefficient over all outputs ("no-grouping").

Three details that change numbers:

- Only outputs that have both a metric and a human score are used, at every level, system means included.
- A group where the coefficient is undefined (for Pearson, Spearman and the tau variants: one side is constant)
  is left out of the average by default, and the result reports how many groups that was. Deutsch, Foster and
  Freitag note that dropping such groups is what is done in practice and that it can favour a metric that ties
  a whole group. Pairwise accuracy is defined for every group with two outputs.
- System means that differ by less than `1e-12` of the largest mean are treated as equal, because float sums of
  the same ratings in a different order can differ in the last bits and would hide a tie.

## Limitations

- No uncertainty yet: every number is a point estimate (see Status).
- Ties are exact float equality, so a metric with continuous scores almost never predicts a tie and is at a
  disadvantage under pairwise accuracy until tie calibration is added.
- The grouped levels call the coefficient once per group in Python, which is slow for tens of thousands of
  groups.
- Scores must be absolute numbers; pairwise preference data is out of scope.

## Related tools

- [judgebias](https://github.com/vipul21435/judgebias) audits LLM judges that give pairwise verdicts (position
  and length bias, agreement with humans). metaeval covers absolute scores instead.
- [rubricops](https://github.com/vipul21435/rubricops) computes inter-rater agreement coefficients (Cohen's and
  Fleiss' kappa, Krippendorff's alpha) alongside rubric scoring.

## Licence

MIT. See [LICENSE](LICENSE).

Built by Vipul Raj Jha · https://vipul21435.github.io
