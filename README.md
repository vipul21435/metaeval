# metaeval

Correlate automatic metric or grader scores with human ratings, with the grouping stated and an interval on
every number, and test whether one metric is really better than another.

## Why this is hard

"The metric correlates 0.6 with human judgement" is not one number. The same metric on the same data gives
different correlations at system level (average each system, then correlate), over all outputs pooled, within
each item across systems, and within each system across items, and the differences can be large enough to change
which of two metrics looks better. Ties add a second choice: ordinal human ratings tie often, and the variants of
Kendall's tau treat ties differently. With a few dozen systems the system-level number also has a wide interval,
and two metrics scored on the same outputs are not independent, so comparing their two numbers by eye says
little. A reported correlation is hard to interpret unless the grouping, the tie handling and the uncertainty
are all stated.

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
$ metaeval correlate examples/toy_scores.jsonl --coefficient kendall_b --coefficient pearson
# Correlation with human ratings

- scores: examples/toy_scores.jsonl (4 systems, 6 items)
- human score: mean of 2 annotators (ann_1, ann_2), dimension 'overall'
- intervals: 95% bootstrap percentile, 1000 resamples, systems and items resampled, seed 0

| metric | grouping | kendall_b | pearson | points | groups |
| --- | --- | ---: | ---: | ---: | ---: |
| metric_a | system | 1.000 [0.473, 1.000] | 0.976 [0.784, 1.000] | 4 | 1 |
| metric_a | pooled | 0.394 [-0.134, 0.821] | 0.562 [0.092, 0.899] | 24 | 1 |
| metric_a | by_item | 0.575 [0.000, 1.000] | 0.596 [0.000, 1.000] | 24 | 6 |
| metric_a | by_system | 0.190 [-0.537, 0.773] | 0.308 [-0.530, 0.863] | 24 | 4 |
| metric_b | system | 0.667 [-1.000, 1.000] | 0.471 [-1.000, 1.000] | 4 | 1 |
| metric_b | pooled | 0.538 [0.076, 0.903] | 0.671 [0.162, 0.939] | 24 | 1 |
| metric_b | by_item | 0.122 [-0.667, 1.000] | 0.125 [-0.805, 1.000] | 24 | 6 |
| metric_b | by_system | 0.676 [0.204, 1.000] | 0.797 [0.338, 0.998] | 24 | 4 |

Groupings:

- system: system level: scores averaged over items, one point per system
- pooled: item level, pooled: every output is one point, no grouping
- by_item: item level, grouped by item: coefficient across systems within each item, averaged over items
- by_system: item level, grouped by system: coefficient across items within each system, averaged over systems

points: systems at system level, otherwise outputs that have both scores. groups: groups with at least two such outputs.
```

On this toy table the pooled tau-b prefers `metric_b` (0.538 against 0.394), while within items, where systems
are compared on the same input, `metric_a` is ahead (0.575 against 0.122). The intervals show how little 4
systems and 6 items can say, and the test agrees:

```
$ metaeval compare examples/toy_scores.jsonl --grouping by_item
...
| metric a | metric b | grouping | coefficient | a | b | a - b | p bootstrap | p permutation | p Williams |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| metric_a | metric_b | by_item | kendall_b | 0.575 | 0.122 | 0.453 [-0.333, 1.333] | 0.307 | 0.225 | - |
```

The five commands share their options:

| command | what it prints |
| --- | --- |
| `metaeval correlate` | each coefficient at each grouping, with bootstrap intervals |
| `metaeval compare` | the difference between every two metrics: interval, paired bootstrap, permutation and Williams tests |
| `metaeval scale` | ordinal agreement of a grader with the human score and how it uses the rating scale |
| `metaeval ceiling` | each annotator against the mean of the others, and the metrics against the same references |
| `metaeval report` | all of the above in one document (`--grader NAME` adds the scale audit for that metric) |

`--metric`, `--coefficient` and `--grouping` (each repeatable) restrict the output; `--dimension` and
`--annotators 'expert_*'` choose the human score; `--resamples N`, `--resample items|systems|both`,
`--confidence` and `--seed` control the intervals (`--resamples 0` prints point estimates only);
`--undefined zero` counts undefined groups as 0 instead of leaving them out; `--permute` chooses what the
permutation test exchanges (see below); `--format json` prints the same results as JSON and `--output FILE`
writes them to a file. `--coefficient pairwise_accuracy_calibrated` adds
pairwise accuracy after tie calibration.

From Python (`examples/quickstart.py`):

```python
from pathlib import Path

import numpy as np

from metaeval import (
    bootstrap,
    correlate,
    human_ceiling,
    load_scores,
    paired_bootstrap,
    permutation_test,
)

table = load_scores(Path(__file__).with_name("toy_scores.jsonl"))
human = table.human()  # mean over annotators, shape (systems, items)
metric_a, metric_b = table.metric("metric_a"), table.metric("metric_b")

print("Kendall tau-b of metric_a with the human score, by grouping")
for grouping in ("system", "pooled", "by_item", "by_system"):
    point = correlate(metric_a, human, coefficient="kendall_b", grouping=grouping)
    interval = bootstrap(
        metric_a, human, coefficient="kendall_b", grouping=grouping, n_resamples=500, seed=0
    )
    bounds = f"[{interval.low:.3f}, {interval.high:.3f}]"
    print(f"  {grouping:<9} {point.value:6.3f}  95% interval {bounds}")

settings = dict(coefficient="kendall_b", grouping="by_item", n_resamples=500, seed=0)
paired = paired_bootstrap(metric_a, metric_b, human, **settings)
permutation = permutation_test(metric_a, metric_b, human, **settings)
print("metric_a minus metric_b within items")
print(f"  difference {paired.difference:.3f}  interval [{paired.low:.3f}, {paired.high:.3f}]")
print(f"  p = {paired.p_value:.3f} (paired bootstrap)")
print(f"  p = {permutation.p_value:.3f} (permutation, whole {permutation.unit} exchanged)")

ratings = np.stack([table.ratings["overall"][name] for name in table.annotators()])
ceiling = human_ceiling(ratings, coefficient="kendall_b", grouping="by_item")
print(f"human ceiling within items: {ceiling.value:.3f}")
```

```
Kendall tau-b of metric_a with the human score, by grouping
  system     1.000  95% interval [0.200, 1.000]
  pooled     0.394  95% interval [-0.188, 0.781]
  by_item    0.575  95% interval [-0.137, 1.000]
  by_system  0.190  95% interval [-0.500, 0.741]
metric_a minus metric_b within items
  difference 0.453  interval [-0.333, 1.333]
  p = 0.350 (paired bootstrap)
  p = 0.214 (permutation, whole systems exchanged)
human ceiling within items: 0.783
```

`correlate` and the resampling functions have no default coefficient or grouping; both have to be named.

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

### Coefficients

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
Spearman is the Pearson correlation of average ranks. Pair counts take `O(n log^2 n)` time for one long vector;
the grouped levels compare all pairs of a group, vectorised over groups.

Tie calibration, from the same paper, lets a continuous metric predict ties: two metric scores of a group count
as tied when they differ by at most `eps`. Every absolute difference between two metric scores of a group is a
candidate, and the `eps` with the highest pairwise accuracy (averaged over groups; one `eps` for all groups; the
smallest in case of equal accuracy) is chosen. It is recomputed on every bootstrap resample. The threshold is
fitted on the data it is reported on, as in the paper, and is printed next to the number.

### Groupings

With `x[i, j]` the score of system `i` on item `j`:

- `system`: average `x[i, :]` over items, then one coefficient across systems. This is the system-level
  correlation of Deutsch, Dror and Roth,
  [A Statistical Analysis of Summarization Evaluation Metrics using Resampling Methods](https://arxiv.org/abs/2104.00054)
  (2021).
- `by_item`: one coefficient across systems for each item, averaged over items; their summary-level
  correlation, and "group-by-item" in Deutsch, Foster and Freitag.
- `by_system`: one coefficient across items for each system, averaged over systems ("group-by-system").
- `pooled`: one coefficient over all outputs ("no-grouping").

Shared tasks report at more than one of these levels; the
[WMT23 metrics shared task](https://aclanthology.org/2023.wmt-1.51/) (Freitag et al., 2023) evaluates metrics at
the system and at the segment level. Three details that change numbers:

- Only outputs that have both a metric and a human score are used, at every level, system means included.
- A group where the coefficient is undefined (for Pearson, Spearman and the tau variants: one side is constant)
  is left out of the average by default, and the result reports how many groups that was. Deutsch, Foster and
  Freitag note that dropping such groups is what is done in practice and that it can favour a metric that ties
  a whole group. Pairwise accuracy is defined for every group with two outputs.
- System means that differ by less than `1e-12` of the largest mean are treated as equal, because float sums of
  the same ratings in a different order can differ in the last bits and would hide a tie.

### Intervals and tests

Intervals are bootstrap percentile intervals. A resample draws, with replacement, the items, the systems, or
both (the default), and the whole statistic is recomputed; these are Boot-Inputs, Boot-Systems and Boot-Both of
Deutsch, Dror and Roth, who recommend Boot-Both. Resamples in which the statistic is undefined are left out and
counted in the JSON output.

`compare` reports three tests of the difference `a - b` between two metrics against the same human score:

- Paired bootstrap: both metrics on the same resamples; the interval is the percentile interval of the
  difference, and the two-sided p value is twice the smaller share of resampled differences on either side of
  zero (one added to the count and to the number of resamples). This test is conservative and has less power
  than the others (next section): a large p value from it is weak evidence that two metrics are equal.
- Permutation test: both metrics are standardised and their scores exchanged with probability one half; the
  p value is the share of permutations with a difference at least as large in absolute value. What is
  exchanged, whole systems, whole items or single outputs, decides what the test assumes; see the next section.
- Williams test, for Pearson correlations at `system` or `pooled` level:
  `t = (r_a - r_b) * sqrt((n - 1) * (1 + r_ab)) / sqrt(2 * K * (n - 1) / (n - 3) + (r_a + r_b)^2 / 4 * (1 - r_ab)^3)`
  with `K = 1 - r_a^2 - r_b^2 - r_ab^2 + 2 * r_a * r_b * r_ab`, on `n - 3` degrees of freedom, where `r_ab` is
  the correlation between the two metrics. Graham and Baldwin,
  [Testing for Significance of Increased Correlation with Human Judgment](https://aclanthology.org/D14-1020/)
  (2014), use it to compare metrics because two metrics scored on the same data are not independent.
  The implementation reproduces the worked example of the
  [cocor paper](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0121945) (Diedenhofen and
  Musch, 2015: `t = -2.7743`, `df = 288`, `p = 0.0059`). Deutsch, Dror and Roth point out that it assumes normal
  scores and has low power on summarization data.

### What the permutation test exchanges

Deutsch, Dror and Roth define the permutation test with whole systems exchanged (Perm-Systems), whole items
(Perm-Inputs) or single outputs (Perm-Both) and recommend Perm-Both because it had the highest power in their
experiments. Their paper measures power; it does not report how often the variants reject two metrics that
are equally good.

The null hypothesis of the test is that the two metrics' scores can be swapped, unit by unit, without changing
how the pair behaves. For single outputs that holds when a metric's errors are independent from one output to
the next. A metric usually also has an error of its own for a whole system (it likes or dislikes everything
one system writes) or for a whole item. Swapping single outputs then mixes the two metrics inside every system,
the permuted differences come out too small, and two equally good metrics are declared different. A small p
value from that variant can mean no more than that the metrics make different kinds of error.

`metaeval` therefore exchanges the unit the coefficient is computed across: whole systems for `system` and
`by_item`, whole items for `by_system`, and for `pooled` both in turn, reporting the larger p value.
`--permute systems`, `items` or `outputs` overrides this; `outputs` is Perm-Both.

`examples/simulate_tests.py` measures how often each test rejects at the 5 % level on simulated tables of 16
systems by 30 items (80 tables per cell, 99 permutations or resamples, seed 0). In the set-ups `output`,
`system` and `item` the two metrics are equally good, so about 0.05 is the right answer; they have independent
noise per output, and in `system` and `item` also an error of their own per system or per item. In the last
two set-ups metric b really is worse and the rate is the power of the test. The Williams column is always
the test on Pearson correlations. With Pearson (`python examples/simulate_tests.py --coefficient pearson`):

| set-up | grouping | permute systems | permute items | permute outputs | default unit | paired bootstrap | Williams |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| output | system | 0.062 | 0.013 | 0.037 | 0.062 (systems) | 0.000 | 0.025 |
| output | pooled | 0.037 | 0.025 | 0.000 | 0.000 (systems and items) | 0.000 | 0.000 |
| output | by_item | 0.037 | 0.037 | 0.037 | 0.037 (systems) | 0.000 | - |
| output | by_system | 0.037 | 0.000 | 0.013 | 0.000 (items) | 0.000 | - |
| system | system | 0.050 | 0.713 | 0.662 | 0.050 (systems) | 0.013 | 0.013 |
| system | pooled | 0.037 | 0.425 | 0.375 | 0.037 (systems and items) | 0.025 | 0.400 |
| system | by_item | 0.050 | 0.475 | 0.450 | 0.050 (systems) | 0.013 | - |
| system | by_system | 0.037 | 0.013 | 0.000 | 0.013 (items) | 0.000 | - |
| item | system | 0.037 | 0.037 | 0.013 | 0.037 (systems) | 0.000 | 0.025 |
| item | pooled | 0.287 | 0.037 | 0.275 | 0.037 (systems and items) | 0.013 | 0.287 |
| item | by_item | 0.013 | 0.037 | 0.000 | 0.013 (systems) | 0.000 | - |
| item | by_system | 0.300 | 0.037 | 0.250 | 0.037 (items) | 0.013 | - |
| b is worse | system | 1.000 | 1.000 | 1.000 | 1.000 (systems) | 1.000 | 1.000 |
| b is worse | pooled | 1.000 | 1.000 | 1.000 | 1.000 (systems and items) | 1.000 | 1.000 |
| b is worse | by_item | 1.000 | 1.000 | 1.000 | 1.000 (systems) | 1.000 | - |
| b is worse | by_system | 1.000 | 1.000 | 1.000 | 1.000 (items) | 1.000 | - |
| b errs per system | system | 0.825 | 1.000 | 1.000 | 0.825 (systems) | 0.700 | 0.988 |
| b errs per system | pooled | 0.350 | 0.575 | 0.613 | 0.338 (systems and items) | 0.100 | 0.613 |
| b errs per system | by_item | 0.163 | 0.512 | 0.475 | 0.163 (systems) | 0.087 | - |
| b errs per system | by_system | 0.062 | 0.013 | 0.050 | 0.013 (items) | 0.000 | - |

With Kendall's tau-b at system level
(`python examples/simulate_tests.py --coefficient kendall_b --grouping system`):

| set-up | grouping | permute systems | permute items | permute outputs | default unit | paired bootstrap | Williams |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| output | system | 0.025 | 0.025 | 0.037 | 0.025 (systems) | 0.000 | 0.025 |
| system | system | 0.037 | 0.537 | 0.463 | 0.037 (systems) | 0.000 | 0.013 |
| item | system | 0.000 | 0.000 | 0.000 | 0.000 (systems) | 0.000 | 0.025 |
| b is worse | system | 0.975 | 1.000 | 1.000 | 0.975 (systems) | 0.738 | 1.000 |
| b errs per system | system | 0.550 | 0.900 | 0.925 | 0.550 (systems) | 0.050 | 0.988 |

With 80 tables a rate near 0.05 has a standard error of about 0.02.

- With independent noise per output every variant stays near or below 5 %.
- With an error per system, exchanging single outputs rejects 66.2 % of equally good pairs at system level
  (46.3 % with tau-b), 45.0 % within items and 37.5 % pooled; with an error per item it rejects 27.5 % pooled
  and 25.0 % within systems. Exchanging the wrong whole unit fails in the same way.
- The default unit stays between 0.0 % and 6.2 % in all twelve cells where the metrics are equally good. It
  finds the worse metric in every table of `b is worse` with Pearson (97.5 % with tau-b at system level), and
  in `b errs per system` in 82.5 % of tables at system level with Pearson and 55.0 % with tau-b.
- The Williams test keeps its level at system level, where it also has the most power, and rejects 40.0 % and
  28.7 % of equally good pairs at the pooled level, where outputs of one system or item are not independent.
- The paired bootstrap rejects between 0.0 % and 2.5 % of equally good pairs, well below the nominal 5 %, and
  finds a worse metric less often than the permutation test. In `b errs per system` at system level it rejects
  5.0 % against 55.0 % with tau-b and 70.0 % against 82.5 % with Pearson, and pooled 10.0 % against 33.8 %; in
  `b is worse` with tau-b, 73.8 % against 97.5 %. Deutsch, Dror and Roth report a power near 0 for Boot-Both at
  system level in their experiments.

### Ordinal agreement and scale use

`scale` treats a metric as a grader on the human rating scale. Scores are rounded to the points of the scale
(halves upwards) for exact agreement, adjacent agreement (within one step) and quadratic weighted kappa
(Cohen, 1968, "Weighted kappa: Nominal scale agreement with provision for scaled disagreement or partial
credit", Psychological Bulletin 70(4), 213-220; formula as on the
[Wikipedia page for Cohen's kappa](https://en.wikipedia.org/wiki/Cohen%27s_kappa) with weights `(i - j)^2`).
The weights are taken on the score values, so two points of the scale are as far apart as their values say even
when a point between them is never used.
[`sklearn.metrics.cohen_kappa_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.cohen_kappa_score.html)
gives the same number only when the points of the scale are passed as `labels`; with its default it indexes the
values that occur and can differ when a point is unused. The tests compare against it with `labels` set.
The mean signed difference is grader minus human on the scores as given. The audit also prints the distribution
over the scale for grader and humans, the share of scores at the centre of the scale, the ratio of standard
deviations (range restriction), the share of tied pairs of outputs, and the number of distinct values used.
A grader is often summarised by one correlation (G-Eval, Liu et al.,
[G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment](https://arxiv.org/abs/2303.16634), 2023,
reports a Spearman correlation of 0.514 with human judgements on summarization); the audit shows what a
grader does with the scale besides. metaeval does not call any model: it reads scores that already exist.

### Human ceiling

`ceiling` leaves each annotator out in turn and correlates that annotator with the mean rating of the others,
exactly as a metric would be scored; the mean over annotators is the ceiling. Each metric is then scored against
the same leave-one-out means and averaged, so that metric and annotators are compared against references built
from the same number of people.

## Results on SummEval

SummEval (Fabbri et al., [SummEval: Re-evaluating Summarization Evaluation](https://arxiv.org/abs/2007.12626))
has ratings by three experts and five crowd workers on coherence, consistency, fluency and relevance for 1,600
summaries: 16 systems on 100 news articles.

Data and licence. The [SummEval repository](https://github.com/Yale-LILY/SummEval) is under the MIT licence;
its README states no separate licence for the annotation file it links, so this repository contains only the
script that downloads and converts that file, and the numbers below. No article, summary or rating is committed.
The annotation file that the repository lists as also carrying the paper's metric scores no longer downloads
(checked on 2026-11-13), so `examples/summeval_scores.py` computes its own reference-overlap scores from the
summaries and their 11 references: `unigram_f1`, `bigram_f1` and `lcs_f1` (best F1 over the references, in the
style of ROUGE-1, ROUGE-2 and ROUGE-L but with plain lower-cased word tokens and no stemming) and `length`
(tokens in the summary). They are not the ROUGE scores of the SummEval paper. `crowd_<dimension>` is the mean of
the five crowd ratings.

All numbers below come from these commands, with the default 1,000 resamples of systems and items and seed 0:

```
python examples/summeval_scores.py --out summeval_scores.jsonl
metaeval correlate summeval_scores.jsonl --dimension relevance --annotators 'expert_*' \
    --metric unigram_f1 --metric bigram_f1 --metric lcs_f1 --metric length \
    --coefficient kendall_b --coefficient pearson
```

### The grouping moves the number

Correlation with the mean expert rating of relevance, 95 % intervals:

| metric | grouping | kendall_b | pearson | points | groups |
| --- | --- | ---: | ---: | ---: | ---: |
| unigram_f1 | system | 0.617 [0.079, 0.874] | 0.670 [0.273, 0.908] | 16 | 1 |
| unigram_f1 | pooled | 0.233 [0.155, 0.309] | 0.339 [0.226, 0.441] | 1600 | 1 |
| unigram_f1 | by_item | 0.205 [0.111, 0.302] | 0.298 [0.177, 0.404] | 1600 | 100 |
| unigram_f1 | by_system | 0.204 [0.137, 0.278] | 0.288 [0.190, 0.386] | 1600 | 16 |
| bigram_f1 | system | 0.517 [-0.089, 0.807] | 0.651 [0.097, 0.892] | 16 | 1 |
| bigram_f1 | pooled | 0.160 [0.081, 0.242] | 0.227 [0.118, 0.335] | 1600 | 1 |
| bigram_f1 | by_item | 0.153 [0.056, 0.249] | 0.231 [0.091, 0.338] | 1600 | 100 |
| bigram_f1 | by_system | 0.137 [0.065, 0.208] | 0.182 [0.087, 0.279] | 1600 | 16 |
| lcs_f1 | system | 0.467 [-0.069, 0.780] | 0.699 [0.171, 0.899] | 16 | 1 |
| lcs_f1 | pooled | 0.170 [0.088, 0.244] | 0.245 [0.136, 0.344] | 1600 | 1 |
| lcs_f1 | by_item | 0.166 [0.077, 0.254] | 0.241 [0.126, 0.338] | 1600 | 100 |
| lcs_f1 | by_system | 0.136 [0.061, 0.210] | 0.194 [0.092, 0.293] | 1600 | 16 |
| length | system | 0.233 [-0.255, 0.696] | 0.239 [-0.309, 0.752] | 16 | 1 |
| length | pooled | 0.178 [0.067, 0.289] | 0.260 [0.126, 0.397] | 1600 | 1 |
| length | by_item | 0.106 [-0.022, 0.242] | 0.149 [-0.006, 0.292] | 1600 | 100 |
| length | by_system | 0.191 [0.101, 0.276] | 0.283 [0.165, 0.387] | 1600 | 16 |

For `unigram_f1`, tau-b is 0.617 at system level and 0.205 within items: the same metric, the same ratings,
a factor of three. The system-level interval, from 16 systems, runs from 0.079 to 0.874. For `bigram_f1` it is
[-0.089, 0.807]. Deutsch, Dror and Roth report [-.09, .84] for ROUGE-2 F1 on these relevance ratings, also
Kendall's tau at system level with Boot-Both; `bigram_f1` is not ROUGE-2 (no stemming, other tokens, best
reference instead of the toolkit's handling of several), so the closeness is only indicative.
`length` is the weakest at system level and within items, yet within systems (0.191) it is close to
`unigram_f1` (0.204): longer summaries of one system tend to be rated more relevant.

The other dimensions, for `unigram_f1` (`metaeval correlate summeval_scores.jsonl --dimension <dimension>
--annotators 'expert_*' --metric unigram_f1 --coefficient kendall_b`):

| dimension | grouping | kendall_b of unigram_f1 |
| --- | --- | ---: |
| coherence | system | 0.383 [-0.143, 0.732] |
| coherence | pooled | 0.146 [0.076, 0.208] |
| coherence | by_item | 0.129 [0.040, 0.208] |
| coherence | by_system | 0.140 [0.069, 0.210] |
| consistency | system | 0.283 [-0.250, 0.699] |
| consistency | pooled | 0.130 [-0.007, 0.254] |
| consistency | by_item | 0.133 [-0.067, 0.303] [96/100] |
| consistency | by_system | 0.045 [-0.030, 0.119] |
| fluency | system | 0.427 [-0.036, 0.764] |
| fluency | pooled | 0.093 [-0.011, 0.194] |
| fluency | by_item | 0.108 [-0.018, 0.235] [98/100] |
| fluency | by_system | 0.041 [-0.033, 0.118] |

`[96/100]` means that for 4 articles all 16 summaries have the same mean expert rating of consistency; tau-b
is undefined there and those items are left out.

### Which differences are significant

```
metaeval compare summeval_scores.jsonl --dimension relevance --annotators 'expert_*' \
    --metric unigram_f1 --metric bigram_f1 --metric lcs_f1 --metric length \
    --coefficient kendall_b --grouping system --grouping by_item
```

| metric a | metric b | grouping | coefficient | a | b | a - b | p bootstrap | p permutation | p Williams |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| unigram_f1 | bigram_f1 | system | kendall_b | 0.617 | 0.517 | 0.100 [-0.202, 0.444] | 0.585 | 0.587 | - |
| unigram_f1 | bigram_f1 | by_item | kendall_b | 0.205 | 0.153 | 0.052 [-0.002, 0.113] | 0.058 | 0.013 | - |
| unigram_f1 | lcs_f1 | system | kendall_b | 0.617 | 0.467 | 0.150 [-0.230, 0.562] | 0.571 | 0.302 | - |
| unigram_f1 | lcs_f1 | by_item | kendall_b | 0.205 | 0.166 | 0.040 [-0.022, 0.107] | 0.228 | 0.029 | - |
| unigram_f1 | length | system | kendall_b | 0.617 | 0.233 | 0.383 [-0.474, 1.018] | 0.386 | 0.312 | - |
| unigram_f1 | length | by_item | kendall_b | 0.205 | 0.106 | 0.099 [-0.102, 0.284] | 0.300 | 0.221 | - |
| bigram_f1 | lcs_f1 | system | kendall_b | 0.517 | 0.467 | 0.050 [-0.296, 0.400] | 0.973 | 0.589 | - |
| bigram_f1 | lcs_f1 | by_item | kendall_b | 0.153 | 0.166 | -0.012 [-0.072, 0.048] | 0.645 | 0.318 | - |
| bigram_f1 | length | system | kendall_b | 0.517 | 0.233 | 0.283 [-0.673, 0.947] | 0.525 | 0.455 | - |
| bigram_f1 | length | by_item | kendall_b | 0.153 | 0.106 | 0.047 [-0.162, 0.240] | 0.651 | 0.576 | - |
| lcs_f1 | length | system | kendall_b | 0.467 | 0.233 | 0.233 [-0.667, 0.930] | 0.595 | 0.540 | - |
| lcs_f1 | length | by_item | kendall_b | 0.166 | 0.106 | 0.060 [-0.146, 0.252] | 0.563 | 0.445 | - |

The permutation test exchanges whole systems at both levels. At system level no test separates any two of
the four metrics: the permutation p values run from 0.302 to 0.589 and every bootstrap interval for a
difference includes zero. The bootstrap column adds little evidence: that test rarely rejects even when a
metric is worse (see the simulation), so "none of the six differences is significant" by it is a weak
statement. With 16 systems the data cannot tell unigram overlap from summary length (0.617
against 0.233, p = 0.312). Within items the permutation test separates `unigram_f1` from `bigram_f1`
(p = 0.013) and from `lcs_f1` (p = 0.029), and nothing else; `unigram_f1` against `length`, the largest
difference there (0.099), has p = 0.221. Twelve tests were run without a correction for multiple comparisons;
a Bonferroni threshold would be 0.05 / 12 = 0.004, so the two p values below 0.05 are suggestive at most.

Version 0.1.0 exchanged single outputs and printed p = 0.002 and 0.004 for `unigram_f1` against `length`.
As the simulation above shows, that variant rejects equally good metrics far too often when the two metrics
err differently from system to system, as an overlap score and a length count can be expected to; those p
values should not be used.

The same comparison on the system-level Pearson correlations, where the Williams test applies (`--metric
unigram_f1 --metric bigram_f1 --metric length --coefficient pearson --grouping system`):

| metric a | metric b | grouping | coefficient | a | b | a - b | p bootstrap | p permutation | p Williams |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| unigram_f1 | bigram_f1 | system | pearson | 0.670 | 0.651 | 0.019 [-0.082, 0.272] | 0.635 | 0.806 | 0.672 |
| unigram_f1 | length | system | pearson | 0.670 | 0.239 | 0.431 [-0.296, 1.141] | 0.216 | 0.231 | 0.156 |
| bigram_f1 | length | system | pearson | 0.651 | 0.239 | 0.412 [-0.488, 1.141] | 0.288 | 0.269 | 0.207 |

None of the three tests rejects with 16 systems: for `unigram_f1` against `length` the permutation p is 0.231,
the Williams p 0.156 and the bootstrap p 0.216.

### Human ceiling

```
metaeval ceiling summeval_scores.jsonl --dimension relevance --annotators 'expert_*' \
    --metric unigram_f1 --metric crowd_relevance
```

| grouping | coefficient | scores | against the mean of the other annotators |
| --- | --- | --- | ---: |
| system | kendall_b | human ceiling | 0.769 [0.504, 0.899] |
| system | kendall_b | expert_1 | 0.795 |
| system | kendall_b | expert_2 | 0.733 |
| system | kendall_b | expert_3 | 0.778 |
| system | kendall_b | unigram_f1 | 0.602 [0.079, 0.842] |
| system | kendall_b | crowd_relevance | -0.133 [-0.460, 0.424] |
| pooled | kendall_b | human ceiling | 0.462 [0.383, 0.519] |
| pooled | kendall_b | expert_1 | 0.476 |
| pooled | kendall_b | expert_2 | 0.482 |
| pooled | kendall_b | expert_3 | 0.426 |
| pooled | kendall_b | unigram_f1 | 0.224 [0.151, 0.298] |
| pooled | kendall_b | crowd_relevance | 0.065 [-0.002, 0.128] |
| by_item | kendall_b | human ceiling | 0.413 [0.326, 0.479] |
| by_item | kendall_b | expert_1 | 0.416 |
| by_item | kendall_b | expert_2 | 0.435 |
| by_item | kendall_b | expert_3 | 0.388 |
| by_item | kendall_b | unigram_f1 | 0.201 [0.109, 0.293] |
| by_item | kendall_b | crowd_relevance | 0.025 [-0.052, 0.094] |
| by_system | kendall_b | human ceiling | 0.419 [0.351, 0.478] |
| by_system | kendall_b | expert_1 | 0.436 |
| by_system | kendall_b | expert_2 | 0.434 |
| by_system | kendall_b | expert_3 | 0.386 |
| by_system | kendall_b | unigram_f1 | 0.196 [0.134, 0.267] |
| by_system | kendall_b | crowd_relevance | 0.081 [0.011, 0.140] |

One expert against the other two reaches tau-b 0.769 [0.504, 0.899] at system level and 0.413 [0.326, 0.479]
within items. Against the same references `unigram_f1` reaches 0.602 and 0.201: about half of the human ceiling
within items. The mean crowd rating barely tracks the experts on relevance: 0.025 within items, 0.081 within
systems, -0.133 at system level.

### The crowd as a grader

```
metaeval scale summeval_scores.jsonl --dimension relevance --annotators 'expert_*' \
    --metric crowd_relevance --scale 1 5
```

| grader | exact | adjacent | quadratic weighted kappa | mean signed difference | sd ratio | outputs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| crowd_relevance | 0.404 [0.352, 0.461] | 0.854 [0.805, 0.896] | 0.057 [-0.030, 0.134] | 0.176 [-0.002, 0.378] | 0.840 [0.730, 0.992] | 1600 |

| scores | distinct values | mean | sd | min | max | at centre | tied pairs | off scale |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| human | 13 | 3.777 | 0.797 | 1.000 | 5.000 | 0.226 | 0.129 | - |
| crowd_relevance | 17 | 3.953 | 0.670 | 1.000 | 5.000 | 0.168 | 0.169 | 0.393 |

| scores | 1 | 2 | 3 | 4 | 5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| crowd_relevance | 0.003 | 0.028 | 0.168 | 0.623 | 0.180 |
| human | 0.007 | 0.075 | 0.226 | 0.523 | 0.169 |

After rounding both to the 1 to 5 scale, the mean crowd rating is within one point of the mean expert rating on
85 % of the summaries, and the quadratic weighted kappa is 0.057 [-0.030, 0.134]: both mostly say 4. The crowd uses the lower end less (3.1 % at 1 or 2
against 8.2 %) and has a narrower spread (ratio 0.840). Adjacent agreement alone would have looked reassuring.

### Ties

On consistency the mean expert rating rounds to 5 for 86.7 % of the summaries and 67.1 % of all pairs of
summaries are tied (`metaeval scale summeval_scores.jsonl --dimension consistency --annotators 'expert_*'
--metric crowd_consistency --scale 1 5 --resamples 0`, human row), so a continuous metric, which never ties,
does badly on pairwise accuracy:

```
metaeval correlate summeval_scores.jsonl --dimension consistency --annotators 'expert_*' \
    --metric unigram_f1 --coefficient kendall_b --coefficient pairwise_accuracy \
    --coefficient pairwise_accuracy_calibrated --grouping system --grouping by_item --grouping by_system
```

| metric | grouping | kendall_b | pairwise_accuracy | pairwise_accuracy_calibrated | points | groups |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| unigram_f1 | system | 0.283 [-0.250, 0.699] | 0.642 [0.408, 0.858] | 0.642 [0.408, 0.858] (eps 0) | 16 | 1 |
| unigram_f1 | by_item | 0.133 [-0.067, 0.303] [96/100] | 0.208 [0.156, 0.388] | 0.675 [0.559, 0.851] (eps 0.2673) | 1600 | 100 |
| unigram_f1 | by_system | 0.045 [-0.030, 0.119] | 0.157 [0.091, 0.256] | 0.718 [0.576, 0.850] (eps 0.5676) | 1600 | 16 |

Within items the accuracy rises from 0.208 to 0.675 with a threshold of 0.267, and within systems from 0.157 to
0.718 with 0.568. Thresholds that large on a 0 to 1 score tie most pairs, so the calibrated number here mostly
reflects how many pairs the experts tie, not skill at ordering summaries; tau-b on the same data is 0.133 and
0.045. At system level, where the mean ratings rarely tie, calibration changes nothing.

## Limitations

- No chart output: results are tables and JSON only.
- The real-data results use overlap scores recomputed by a plain script, not the SummEval paper's metric scores,
  and only one seed (0); the intervals carry Monte Carlo error from 1,000 resamples.
- Percentile intervals only; no BCa or studentised intervals. With few systems, many system resamples repeat a
  system, which adds ties, and a resample can leave the statistic undefined (counted, then left out).
- The paired bootstrap is conservative and has little power: in the simulation above it rejected at most
  2.5 % of equally good pairs at a nominal 5 % and found worse metrics less often than the permutation test.
  Its p value is read off the resampled differences without studentising.
- No correction for multiple comparisons is applied when several pairs of metrics are tested.
- The Williams test is offered at system and pooled level only and assumes normal scores. At pooled level it
  ignores that outputs of one system or item are dependent: in the simulation above it rejected 40.0 % and
  28.7 % of equally good pairs there. Read it at system level.
- The permutation test exchanges whole systems or items, so with n of them there are 2^n patterns and p cannot
  fall below 2^(1-n): with fewer than 6 systems a two-sided p below 0.05 is impossible. Its level was checked
  by simulation for an error per system or per item, one at a time, not for both at once.
- Tie calibration fits its threshold on the data it reports on and compares all pairs of a group, so the pooled
  level needs memory quadratic in the number of outputs (it refuses above 30 million pairs); intervals for the
  calibrated pooled accuracy are slow.
- Ties are exact float equality, apart from the system means described above.
- Agreement statistics pool all outputs and are computed against the mean rating rounded to the scale, not
  against individual annotators. The ceiling uses the mean of the remaining annotators as reference.
- Scores must be absolute numbers; pairwise preference data is out of scope.
- A bootstrap run on SummEval-sized data takes seconds to a minute per command; nothing is parallelised.

## Related tools

- [judgebias](https://github.com/vipul21435/judgebias) audits LLM judges that give pairwise verdicts (position
  and length bias, agreement with humans). metaeval covers absolute scores instead.
- [rubricops](https://github.com/vipul21435/rubricops) computes inter-rater agreement coefficients (Cohen's and
  Fleiss' kappa, Krippendorff's alpha) alongside rubric scoring; use it for agreement among annotators beyond
  the leave-one-out ceiling here.
- [ppieval](https://github.com/vipul21435/ppieval): combines a few human labels with many automatic judgments and keeps the interval valid.
- [labelfuse](https://github.com/vipul21435/labelfuse): aggregates noisy labels from several annotators into one label with a probability, and audits the annotators.
- [rankstable](https://github.com/vipul21435/rankstable): how stable a leaderboard's order is to the averaging rule and the choice of benchmarks.

## Licence

MIT. See [LICENSE](LICENSE).

Built by Vipul Raj Jha · https://vipul21435.github.io
