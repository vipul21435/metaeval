"""Correlate, compare and bound two metrics on the toy table, from Python.

Run from the repository root:

    python examples/quickstart.py
"""

from __future__ import annotations

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
print(f"  p = {paired.p_value:.3f} (paired bootstrap), {permutation.p_value:.3f} (permutation)")

ratings = np.stack([table.ratings["overall"][name] for name in table.annotators()])
ceiling = human_ceiling(ratings, coefficient="kendall_b", grouping="by_item")
print(f"human ceiling within items: {ceiling.value:.3f}")
