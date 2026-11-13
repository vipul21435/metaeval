"""Command line: ``metaeval correlate``."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Sequence

from metaeval import __version__
from metaeval.coefficients import COEFFICIENTS
from metaeval.grouping import GROUPINGS, Correlation, correlate
from metaeval.table import ScoreFormatError, ScoreTable, load_scores

__all__ = ["build_parser", "main"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="metaeval",
        description="How well does a metric or an automatic grader track human ratings.",
    )
    parser.add_argument("--version", action="version", version=f"metaeval {__version__}")
    commands = parser.add_subparsers(dest="command", required=True, metavar="command")

    correlate_parser = commands.add_parser(
        "correlate",
        help="correlate metric scores with human ratings at each grouping",
        description=(
            "Correlate metric scores with the mean human rating. Every number is printed "
            "with its grouping: " + "; ".join(GROUPINGS.values()) + "."
        ),
    )
    correlate_parser.add_argument("scores", help="JSONL or CSV file of scores")
    correlate_parser.add_argument(
        "--metric",
        action="append",
        metavar="NAME",
        help="metric to report; repeat for several (default: every metric in the file)",
    )
    correlate_parser.add_argument(
        "--dimension",
        metavar="NAME",
        help="rated dimension to use; needed when the file has several",
    )
    correlate_parser.add_argument(
        "--annotators",
        action="append",
        metavar="PATTERN",
        help="annotator name or shell-style pattern such as 'expert_*'; repeat for several "
        "(default: all annotators)",
    )
    correlate_parser.add_argument(
        "--coefficient",
        action="append",
        choices=sorted(COEFFICIENTS),
        help="coefficient to report; repeat for several (default: all)",
    )
    correlate_parser.add_argument(
        "--grouping",
        action="append",
        choices=list(GROUPINGS),
        help="grouping to report; repeat for several (default: all)",
    )
    correlate_parser.add_argument(
        "--undefined",
        choices=("skip", "zero"),
        default="skip",
        help="a group where the coefficient is undefined is left out of the average (skip, "
        "the default) or counted as 0 (zero)",
    )
    correlate_parser.add_argument(
        "--format", choices=("md", "json"), default="md", help="output format (default: md)"
    )
    correlate_parser.add_argument(
        "--input-format",
        choices=("jsonl", "csv"),
        help="input format (default: csv for a .csv file, otherwise jsonl)",
    )
    correlate_parser.set_defaults(run=_run_correlate)
    return parser


def _ordered(chosen: Sequence[str] | None, known: Sequence[str]) -> list[str]:
    """The chosen names without repeats, or all known names."""
    return list(dict.fromkeys(chosen)) if chosen else list(known)


def _run_correlate(args: argparse.Namespace) -> str:
    table = load_scores(args.scores, args.input_format)
    metrics = _ordered(args.metric, list(table.metrics))
    if not metrics:
        raise KeyError("the file has no metric scores")
    coefficients = _ordered(args.coefficient, list(COEFFICIENTS))
    groupings = _ordered(args.grouping, list(GROUPINGS))
    dimension = args.dimension
    annotators = table.annotators(dimension, args.annotators)
    human = table.human(dimension, annotators)
    if dimension is None:
        dimension = table.dimensions[0]

    results: list[tuple[str, Correlation]] = []
    for metric in metrics:
        scores = table.metric(metric)
        for grouping in groupings:
            for coefficient in coefficients:
                result = correlate(
                    scores,
                    human,
                    coefficient=coefficient,
                    grouping=grouping,
                    undefined=args.undefined,
                )
                results.append((metric, result))

    if args.format == "json":
        return _correlate_json(args, table, dimension, annotators, groupings, results)
    return _correlate_markdown(args, table, dimension, annotators, coefficients, groupings, results)


def _correlate_json(
    args: argparse.Namespace,
    table: ScoreTable,
    dimension: str,
    annotators: Sequence[str],
    groupings: Sequence[str],
    results: Sequence[tuple[str, Correlation]],
) -> str:
    document = {
        "scores": args.scores,
        "systems": len(table.systems),
        "items": len(table.items),
        "human": {"dimension": dimension, "annotators": list(annotators), "aggregate": "mean"},
        "undefined": args.undefined,
        "groupings": {name: GROUPINGS[name] for name in groupings},
        "results": [{"metric": metric, **result.to_dict()} for metric, result in results],
    }
    return json.dumps(document, indent=2, allow_nan=False)


def _cell(result: Correlation) -> str:
    """A value to three decimals, with the defined share of groups when some are not."""
    if math.isnan(result.value):
        return "n/a"
    text = f"{result.value:.3f}"
    if float(text) == 0.0:
        text = "0.000"
    if result.n_undefined and result.n_groups > 1:
        text += f" [{result.n_groups - result.n_undefined}/{result.n_groups}]"
    return text


def _correlate_markdown(
    args: argparse.Namespace,
    table: ScoreTable,
    dimension: str,
    annotators: Sequence[str],
    coefficients: Sequence[str],
    groupings: Sequence[str],
    results: Sequence[tuple[str, Correlation]],
) -> str:
    by_row: dict[tuple[str, str], dict[str, Correlation]] = {}
    for metric, result in results:
        by_row.setdefault((metric, result.grouping), {})[result.coefficient] = result

    plural = "s" if len(annotators) != 1 else ""
    lines = [
        "# Correlation with human ratings",
        "",
        f"- scores: {args.scores} ({len(table.systems)} systems, {len(table.items)} items)",
        f"- human score: mean of {len(annotators)} annotator{plural} "
        f"({', '.join(annotators)}), dimension {dimension!r}",
        "",
        "| metric | grouping | " + " | ".join(coefficients) + " | points | groups |",
        "| --- | --- | " + " | ".join("---:" for _ in coefficients) + " | ---: | ---: |",
    ]
    any_undefined = False
    for (metric, grouping), row in by_row.items():
        first = next(iter(row.values()))
        cells = [_cell(row[coefficient]) for coefficient in coefficients]
        any_undefined = any_undefined or any(result.n_undefined for result in row.values())
        lines.append(
            f"| {metric} | {grouping} | "
            + " | ".join(cells)
            + f" | {first.n_points} | {first.n_groups} |"
        )
    lines += ["", "Groupings:", ""]
    lines += [f"- {name}: {GROUPINGS[name]}" for name in groupings]
    lines += [
        "",
        "points: systems at system level, otherwise outputs that have both scores. "
        "groups: groups with at least two such outputs.",
    ]
    if any_undefined:
        treatment = (
            "left out of the average" if args.undefined == "skip" else "counted as 0 in the average"
        )
        lines += [
            "",
            "n/a: the coefficient is undefined, usually because one side is constant. "
            f"[k/n]: defined in k of n groups; the other groups are {treatment}.",
        ]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command line; returns the exit status."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        output = args.run(args)
    except (ScoreFormatError, OSError) as error:
        print(f"metaeval: error: {error}", file=sys.stderr)
        return 2
    except KeyError as error:
        print(f"metaeval: error: {error.args[0]}", file=sys.stderr)
        return 2
    print(output)
    return 0
