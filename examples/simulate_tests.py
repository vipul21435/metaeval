"""How often do the tests of `metaeval compare` reject, on simulated score tables?

Two metrics are scored against the same human ratings on tables of 16 systems by 30
items. In the first three set-ups the metrics are equally good, so a test at the 5 %
level should reject in about 5 % of the tables; they differ in where a metric's own
error sits:

- ``output``: independent noise on every output;
- ``system``: in addition an error of its own for each system (the metric likes or
  dislikes all outputs of a system);
- ``item``: in addition an error of its own for each item.

In the last two set-ups metric b really is worse and the rejection rate is the power of
the test: in ``b is worse`` it carries much more of all three errors, in ``b errs per
system`` only a moderate error per system that metric a does not have.

For every grouping the script prints the rejection rate of the permutation test with
whole systems, whole items and single outputs exchanged, of the unit `metaeval` picks
by default, of the paired bootstrap, and of the Williams test (Pearson, system and
pooled level only). Run from the repository root; this takes a few minutes:

    python examples/simulate_tests.py --coefficient pearson
    python examples/simulate_tests.py --coefficient kendall_b --grouping system
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from metaeval import GROUPINGS, paired_bootstrap, permutation_test, williams
from metaeval.resampling import PERMUTE, PERMUTE_BY_GROUPING

SYSTEMS, ITEMS = 16, 30
# (output noise, error per system, error per item) of metric a and metric b.
SETUPS = {
    "output": ((1.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
    "system": ((1.0, 1.0, 0.0), (1.0, 1.0, 0.0)),
    "item": ((1.0, 0.0, 1.0), (1.0, 0.0, 1.0)),
    "b is worse": ((0.5, 0.0, 0.0), (2.0, 1.5, 1.5)),
    "b errs per system": ((1.0, 0.0, 0.0), (1.0, 0.7, 0.0)),
}


def table(rng: np.random.Generator, setup: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Human ratings and two metrics for one simulated table."""
    shape = (SYSTEMS, ITEMS)
    quality = rng.normal(size=(SYSTEMS, 1)) + rng.normal(size=(1, ITEMS)) + rng.normal(size=shape)
    human = quality + 0.5 * rng.normal(size=shape)
    metrics = []
    for noise, per_system, per_item in SETUPS[setup]:
        metrics.append(
            quality
            + noise * rng.normal(size=shape)
            + per_system * rng.normal(size=(SYSTEMS, 1))
            + per_item * rng.normal(size=(1, ITEMS))
        )
    return metrics[0], metrics[1], human


def rejection_rates(
    setup: str,
    grouping: str,
    *,
    replications: int,
    resamples: int,
    coefficient: str = "kendall_b",
    level: float = 0.05,
    seed: int = 0,
) -> dict[str, float]:
    """Share of simulated tables in which each test gives ``p <= level``."""
    rng = np.random.default_rng([seed, list(SETUPS).index(setup)])
    rejected = dict.fromkeys([*PERMUTE, "default", "bootstrap", "williams"], 0)
    for replication in range(replications):
        a, b, human = table(rng, setup)
        common = dict(coefficient=coefficient, grouping=grouping, n_resamples=resamples)
        p_values = {
            unit: permutation_test(a, b, human, unit=unit, seed=replication, **common).p_value
            for unit in PERMUTE
        }
        for unit, p_value in p_values.items():
            rejected[unit] += p_value <= level
        rejected["default"] += max(p_values[u] for u in PERMUTE_BY_GROUPING[grouping]) <= level
        paired = paired_bootstrap(a, b, human, seed=replication, **common)
        rejected["bootstrap"] += paired.p_value <= level
        if grouping in ("system", "pooled"):
            rejected["williams"] += williams(a, b, human, grouping=grouping).p_value <= level
    rates = {name: count / replications for name, count in rejected.items()}
    if grouping not in ("system", "pooled"):
        rates["williams"] = float("nan")
    return rates


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--replications", type=int, default=80, help="simulated tables per cell")
    parser.add_argument("--resamples", type=int, default=99, help="permutations and resamples")
    parser.add_argument(
        "--coefficient", default="kendall_b", help="coefficient (default: kendall_b)"
    )
    parser.add_argument(
        "--grouping",
        action="append",
        choices=list(GROUPINGS),
        help="grouping to simulate; repeat for several (default: all)",
    )
    parser.add_argument(
        "--setup",
        action="append",
        choices=list(SETUPS),
        help="set-up to simulate; repeat for several (default: all)",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    print(
        f"Rejection rate at the 5 % level, {args.coefficient}, {SYSTEMS} systems by {ITEMS} items, "
        f"{args.replications} tables per cell, {args.resamples} permutations or resamples, "
        f"seed {args.seed}."
    )
    print()
    print(
        "| set-up | grouping | permute systems | permute items | permute outputs "
        "| default unit | paired bootstrap | Williams |"
    )
    print("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for setup in args.setup or SETUPS:
        for grouping in args.grouping or GROUPINGS:
            rates = rejection_rates(
                setup,
                grouping,
                replications=args.replications,
                resamples=args.resamples,
                coefficient=args.coefficient,
                seed=args.seed,
            )
            cells = [
                "-" if np.isnan(rates[name]) else f"{rates[name]:.3f}"
                for name in ("systems", "items", "outputs", "default", "bootstrap", "williams")
            ]
            default = " and ".join(PERMUTE_BY_GROUPING[grouping])
            print(
                f"| {setup} | {grouping} | "
                + " | ".join(cells[:3])
                + f" | {cells[3]} ({default}) | "
                + " | ".join(cells[4:])
                + " |"
            )
            sys.stdout.flush()


if __name__ == "__main__":
    main()
