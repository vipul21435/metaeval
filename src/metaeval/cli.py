"""Command line: ``metaeval correlate``, ``compare``, ``scale``, ``ceiling`` and ``report``."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from metaeval import __version__
from metaeval.agreement import scale_audit
from metaeval.ceiling import human_ceiling, metric_against_rest
from metaeval.coefficients import COEFFICIENTS
from metaeval.grouping import GROUPINGS, STATISTICS, correlate
from metaeval.resampling import (
    PERMUTE,
    PERMUTE_BY_GROUPING,
    RESAMPLE,
    bootstrap,
    paired_bootstrap,
    percentile_interval,
    permutation_test,
    resample_statistic,
)
from metaeval.significance import williams
from metaeval.table import ScoreFormatError, ScoreTable, load_scores

__all__ = ["build_parser", "main"]

_COMMANDS = {
    "correlate": "correlate metric scores with human ratings at each grouping, with intervals",
    "compare": "test whether two metrics differ in their correlation with human ratings",
    "scale": "ordinal agreement of a grader with human ratings and its use of the scale",
    "ceiling": "human ceiling: each annotator against the mean of the others",
    "report": "all of the above in one document",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="metaeval",
        description="How well does a metric or an automatic grader track human ratings.",
    )
    parser.add_argument("--version", action="version", version=f"metaeval {__version__}")
    commands = parser.add_subparsers(dest="command", required=True, metavar="command")
    for name, summary in _COMMANDS.items():
        sub = commands.add_parser(
            name,
            help=summary,
            description=summary[0].upper()
            + summary[1:]
            + ". Every number is printed with its grouping: "
            + "; ".join(GROUPINGS.values())
            + ".",
        )
        sub.add_argument("scores", help="JSONL or CSV file of scores")
        sub.add_argument(
            "--metric",
            action="append",
            metavar="NAME",
            help="metric to use; repeat for several (default: every metric in the file)",
        )
        sub.add_argument(
            "--dimension", metavar="NAME", help="rated dimension; needed when the file has several"
        )
        sub.add_argument(
            "--annotators",
            action="append",
            metavar="PATTERN",
            help="annotator name or shell-style pattern such as 'expert_*'; repeat for several "
            "(default: all annotators)",
        )
        sub.add_argument(
            "--resamples",
            type=int,
            default=1000,
            metavar="N",
            help="bootstrap resamples for the intervals; 0 prints point estimates only "
            "(default: 1000)",
        )
        sub.add_argument(
            "--resample",
            choices=list(RESAMPLE),
            default="both",
            help="what a resample draws: items, systems or both (default: both)",
        )
        sub.add_argument(
            "--confidence",
            type=float,
            default=0.95,
            metavar="P",
            help="interval level (default: 0.95)",
        )
        sub.add_argument("--seed", type=int, default=0, help="seed of the resampling (default: 0)")
        sub.add_argument(
            "--format", choices=("md", "json"), default="md", help="output format (default: md)"
        )
        sub.add_argument(
            "--input-format",
            choices=("jsonl", "csv"),
            help="input format (default: csv for a .csv file, otherwise jsonl)",
        )
        sub.add_argument("--output", metavar="FILE", help="write the result to FILE")
        if name != "scale":
            default = "all five coefficients" if name == "correlate" else "kendall_b"
            sub.add_argument(
                "--coefficient",
                action="append",
                choices=list(STATISTICS),
                help=f"coefficient; repeat for several (default: {default})",
            )
            sub.add_argument(
                "--grouping",
                action="append",
                choices=list(GROUPINGS),
                help="grouping; repeat for several (default: all)",
            )
            sub.add_argument(
                "--undefined",
                choices=("skip", "zero"),
                default="skip",
                help="a group where the coefficient is undefined is left out of the average "
                "(skip, the default) or counted as 0 (zero)",
            )
        if name in ("compare", "report"):
            sub.add_argument(
                "--permutations",
                type=int,
                metavar="N",
                help="permutations for the permutation test; 0 skips it (default: as --resamples)",
            )
            sub.add_argument(
                "--permute",
                choices=("auto", *PERMUTE),
                default="auto",
                help="what the permutation test exchanges between two metrics: whole systems, "
                "whole items or single outputs. auto (the default) follows the grouping: systems "
                "for system and by_item, items for by_system, and for pooled both in turn with "
                "the larger p value. Exchanging single outputs rejects too often when a metric "
                "has an error of its own per system or item",
            )
        if name in ("scale", "report"):
            sub.add_argument(
                "--scale",
                nargs=2,
                type=float,
                metavar=("LOW", "HIGH"),
                help="ends of the rating scale (default: lowest and highest human rating)",
            )
            sub.add_argument(
                "--step", type=float, default=1.0, help="distance between scale points (default: 1)"
            )
        if name == "report":
            sub.add_argument(
                "--grader",
                action="append",
                metavar="NAME",
                help="metric that scores on the human scale and gets the scale audit; "
                "repeat for several (default: none)",
            )
        sub.set_defaults(run=_RUNNERS[name])
    return parser


@dataclass
class _Context:
    args: argparse.Namespace
    table: ScoreTable
    dimension: str
    annotators: tuple[str, ...]
    unrated: tuple[str, ...]
    human: np.ndarray
    ratings: np.ndarray
    metrics: list[str]

    @property
    def intervals(self) -> bool:
        return self.args.resamples > 0

    def resampling(self) -> dict[str, object]:
        args = self.args
        return dict(
            resample=args.resample,
            n_resamples=args.resamples,
            confidence=args.confidence,
            seed=args.seed,
        )

    def header(self, title: str) -> list[str]:
        args = self.args
        plural = "s" if len(self.annotators) != 1 else ""
        lines = [
            f"# {title}",
            "",
            f"- scores: {args.scores} ({len(self.table.systems)} systems, "
            f"{len(self.table.items)} items)",
            f"- human score: mean of {len(self.annotators)} annotator{plural} "
            f"({', '.join(self.annotators)}), dimension {self.dimension!r}"
            + (f"; no ratings from {', '.join(self.unrated)}" if self.unrated else ""),
        ]
        if self.intervals:
            lines.append(
                f"- intervals: {args.confidence:.0%} bootstrap percentile, {args.resamples} "
                f"resamples, {RESAMPLE[args.resample]}, seed {args.seed}"
            )
        else:
            lines.append("- intervals: none, point estimates only")
        return lines

    def describe(self) -> dict[str, object]:
        args = self.args
        human: dict[str, object] = {
            "dimension": self.dimension,
            "annotators": list(self.annotators),
            "aggregate": "mean",
        }
        if self.unrated:
            human["annotators_without_ratings"] = list(self.unrated)
        return {
            "scores": args.scores,
            "systems": len(self.table.systems),
            "items": len(self.table.items),
            "human": human,
            "resampling": self.resampling() if self.intervals else None,
        }


def _ordered(chosen: Sequence[str] | None, known: Sequence[str]) -> list[str]:
    """The chosen names without repeats, or all known names."""
    return list(dict.fromkeys(chosen)) if chosen else list(known)


def _context(args: argparse.Namespace) -> _Context:
    if args.resamples < 0:
        raise ValueError("--resamples must not be negative")
    if not 0.0 < args.confidence < 1.0:
        raise ValueError("--confidence must be between 0 and 1")
    table = load_scores(args.scores, args.input_format)
    metrics = _ordered(args.metric, list(table.metrics))
    if not metrics:
        raise KeyError("the file has no metric scores")
    for metric in metrics:
        table.metric(metric)
    chosen = table.annotators(args.dimension, args.annotators)
    dimension = args.dimension if args.dimension is not None else table.dimensions[0]
    # An annotator can be named in the file without a single rating in this dimension;
    # such an annotator adds nothing to the mean and is reported apart.
    unrated = tuple(name for name in chosen if np.isnan(table.ratings[dimension][name]).all())
    annotators = tuple(name for name in chosen if name not in unrated)
    if not annotators:
        raise KeyError("the chosen annotators gave no ratings")
    return _Context(
        args=args,
        table=table,
        dimension=dimension,
        annotators=annotators,
        unrated=unrated,
        human=table.human(dimension, annotators),
        ratings=np.stack([table.ratings[dimension][name] for name in annotators]),
        metrics=metrics,
    )


def _number(value: float | None, digits: int = 3) -> str:
    if value is None or math.isnan(value):
        return "n/a"
    text = f"{value:.{digits}f}"
    return text.lstrip("-") if float(text) == 0.0 else text


def _range(low: float, high: float) -> str:
    return f" [{_number(low)}, {_number(high)}]"


def _p(value: float | None) -> str:
    if value is None or math.isnan(value):
        return "n/a"
    return "<0.001" if value < 0.001 else f"{value:.3f}"


def _groupings_legend(groupings: Sequence[str]) -> list[str]:
    return ["", "Groupings:", "", *[f"- {name}: {GROUPINGS[name]}" for name in groupings]]


# correlate ------------------------------------------------------------------------------


def _section_correlate(ctx: _Context) -> tuple[list[str], list[dict[str, object]]]:
    args = ctx.args
    coefficients = _ordered(args.coefficient, list(COEFFICIENTS))
    groupings = _ordered(args.grouping, list(GROUPINGS))
    entries: list[dict[str, object]] = []
    lines = [
        "| metric | grouping | " + " | ".join(coefficients) + " | points | groups |",
        "| --- | --- | " + " | ".join("---:" for _ in coefficients) + " | ---: | ---: |",
    ]
    any_undefined = False
    for metric in ctx.metrics:
        scores = ctx.table.metric(metric)
        for grouping in groupings:
            cells = []
            for coefficient in coefficients:
                result = correlate(
                    scores,
                    ctx.human,
                    coefficient=coefficient,
                    grouping=grouping,
                    undefined=args.undefined,
                )
                entry: dict[str, object] = {"metric": metric, **result.to_dict()}
                text = _number(result.value)
                if ctx.intervals and not math.isnan(result.value):
                    interval = bootstrap(
                        scores,
                        ctx.human,
                        coefficient=coefficient,
                        grouping=grouping,
                        undefined=args.undefined,
                        **ctx.resampling(),
                    )
                    entry["interval"] = {
                        "low": interval.to_dict()["low"],
                        "high": interval.to_dict()["high"],
                        "undefined_resamples": interval.n_undefined,
                    }
                    text += _range(interval.low, interval.high)
                if result.epsilon is not None:
                    text += f" (eps {result.epsilon:.4g})"
                if result.n_undefined and result.n_groups > 1 and not math.isnan(result.value):
                    any_undefined = True
                    text += f" [{result.n_groups - result.n_undefined}/{result.n_groups}]"
                any_undefined = any_undefined or math.isnan(result.value)
                cells.append(text)
                entries.append(entry)
            lines.append(
                f"| {metric} | {grouping} | "
                + " | ".join(cells)
                + f" | {result.n_points} | {result.n_groups} |"
            )
    lines += _groupings_legend(groupings)
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
    if any(coefficient not in COEFFICIENTS for coefficient in coefficients):
        lines += [
            "",
            "eps: tie calibration threshold; metric scores closer than this count as tied.",
        ]
    return lines, entries


# compare --------------------------------------------------------------------------------


def _section_compare(ctx: _Context) -> tuple[list[str], list[dict[str, object]]]:
    args = ctx.args
    if len(ctx.metrics) < 2:
        raise KeyError("comparing needs at least two metrics")
    coefficients = _ordered(args.coefficient, ["kendall_b"])
    groupings = _ordered(args.grouping, list(GROUPINGS))
    permutations = args.resamples if args.permutations is None else args.permutations
    entries: list[dict[str, object]] = []
    lines = [
        "| metric a | metric b | grouping | coefficient | a | b | a - b | p bootstrap "
        "| p permutation | p Williams |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name_a, name_b in itertools.combinations(ctx.metrics, 2):
        a, b = ctx.table.metric(name_a), ctx.table.metric(name_b)
        for grouping in groupings:
            for coefficient in coefficients:
                common = dict(coefficient=coefficient, grouping=grouping, undefined=args.undefined)
                value_a = correlate(a, ctx.human, **common).value
                value_b = correlate(b, ctx.human, **common).value
                entry: dict[str, object] = {
                    "metric_a": name_a,
                    "metric_b": name_b,
                    "coefficient": coefficient,
                    "grouping": grouping,
                    "label": GROUPINGS[grouping],
                    "value_a": None if math.isnan(value_a) else value_a,
                    "value_b": None if math.isnan(value_b) else value_b,
                    "difference": None if math.isnan(value_a - value_b) else value_a - value_b,
                }
                difference = _number(value_a - value_b)
                p_bootstrap = p_permutation = p_williams = None
                if ctx.intervals:
                    paired = paired_bootstrap(a, b, ctx.human, **common, **ctx.resampling())
                    difference += _range(paired.low, paired.high)
                    p_bootstrap = paired.p_value
                    entry["paired_bootstrap"] = paired.to_dict()
                if permutations > 0:
                    permutation = permutation_test(
                        a,
                        b,
                        ctx.human,
                        **common,
                        unit=args.permute,
                        n_resamples=permutations,
                        seed=args.seed,
                    )
                    p_permutation = permutation.p_value
                    entry["permutation"] = permutation.to_dict()
                if coefficient == "pearson" and grouping in ("system", "pooled"):
                    test = williams(a, b, ctx.human, grouping=grouping)
                    p_williams = test.p_value
                    entry["williams"] = test.to_dict()
                entries.append(entry)
                lines.append(
                    f"| {name_a} | {name_b} | {grouping} | {coefficient} | {_number(value_a)} "
                    f"| {_number(value_b)} | {difference} | {_p(p_bootstrap)} "
                    f"| {_p(p_permutation)} | {_p(p_williams) if p_williams is not None else '-'} |"
                )
    lines += _groupings_legend(groupings)
    lines += [
        "",
        "p values are two-sided. p bootstrap: paired bootstrap of a - b on the same resamples. "
        f"p permutation: scores of a and b exchanged ({_permuted(groupings, args.permute)}), "
        f"{permutations} permutations. p Williams: Williams test for dependent Pearson "
        "correlations, at system or pooled level only.",
    ]
    return lines, entries


def _permuted(groupings: Sequence[str], unit: str) -> str:
    """What the permutation test exchanged, in words, for the groupings shown."""
    if unit != "auto":
        return PERMUTE[unit]
    parts = []
    for grouping in groupings:
        units = PERMUTE_BY_GROUPING[grouping]
        what = " and ".join(f"whole {name}" for name in units)
        if len(units) > 1:
            what += " in turn, larger p kept"
        parts.append(f"{grouping}: {what}")
    return "; ".join(parts)


# scale ----------------------------------------------------------------------------------

_AGREEMENT = ("exact", "adjacent", "quadratic_weighted_kappa", "mean_signed_difference", "sd_ratio")


def _section_scale(
    ctx: _Context, graders: Sequence[str]
) -> tuple[list[str], list[dict[str, object]]]:
    args = ctx.args
    present = ctx.ratings[~np.isnan(ctx.ratings)]
    if present.size == 0:
        raise KeyError("the chosen annotators gave no ratings")
    low, high = args.scale if args.scale else (float(present.min()), float(present.max()))
    entries: list[dict[str, object]] = []
    agreement = [
        "| grader | exact | adjacent | quadratic weighted kappa | mean signed difference "
        "| sd ratio | outputs |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    usage = [
        "| scores | distinct values | mean | sd | min | max | at centre | tied pairs | off scale |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    distribution: list[str] = []
    human_row_done = False
    for grader in graders:
        scores = ctx.table.metric(grader)
        audit = scale_audit(scores, ctx.human, low=low, high=high, step=args.step)
        entry: dict[str, object] = {"grader": grader, **audit.to_dict()}
        cells = [_number(getattr(audit, name)) for name in _AGREEMENT]
        if ctx.intervals:
            intervals = {}
            for index, name in enumerate(_AGREEMENT):

                def one(g: np.ndarray, h: np.ndarray, name: str = name) -> float:
                    try:
                        return getattr(scale_audit(g, h, low=low, high=high, step=args.step), name)
                    except ValueError:
                        return math.nan

                values = resample_statistic(
                    one,
                    [scores, ctx.human],
                    resample=args.resample,
                    n_resamples=args.resamples,
                    seed=args.seed,
                )
                bound_low, bound_high, _ = percentile_interval(values, args.confidence)
                intervals[name] = {"low": bound_low, "high": bound_high}
                cells[index] += _range(bound_low, bound_high)
            entry["intervals"] = intervals
        agreement.append(f"| {grader} | " + " | ".join(cells) + f" | {audit.n} |")
        if not human_row_done:
            usage.append(
                f"| human | {audit.distinct_human} | {_number(audit.mean_human)} "
                f"| {_number(audit.sd_human)} | {_number(audit.min_human)} "
                f"| {_number(audit.max_human)} | {_number(audit.central_human)} "
                f"| {_number(audit.tie_share_human)} | - |"
            )
            human_row_done = True
        usage.append(
            f"| {grader} | {audit.distinct_grader} | {_number(audit.mean_grader)} "
            f"| {_number(audit.sd_grader)} | {_number(audit.min_grader)} "
            f"| {_number(audit.max_grader)} | {_number(audit.central_grader)} "
            f"| {_number(audit.tie_share_grader)} | {_number(audit.off_scale)} |"
        )
        points = " | ".join(f"{score:g}" for score, _, _ in audit.distribution)
        distribution += [
            "",
            f"Share of scores at each point of the scale, {grader} against human:",
            "",
            f"| scores | {points} |",
            "| --- | " + " | ".join("---:" for _ in audit.distribution) + " |",
            f"| {grader} | " + " | ".join(_number(g) for _, g, _ in audit.distribution) + " |",
            "| human | " + " | ".join(_number(h) for _, _, h in audit.distribution) + " |",
        ]
        entries.append(entry)
    lines = [
        f"Scale: {low:g} to {high:g} in steps of {args.step:g}. Agreement is computed on scores "
        "rounded to the scale, over all outputs pooled; the mean signed difference is grader "
        "minus human on the scores as given.",
        "",
        *agreement,
        "",
        *usage,
        *distribution,
        "",
        "sd ratio: standard deviation of the grader over that of the human score; below 1 the "
        "grader uses a narrower range. at centre: share of scores at the middle of the scale. "
        "tied pairs: share of pairs of outputs with the same score. off scale: share of grader "
        "scores that are not a point of the scale.",
    ]
    return lines, entries


# ceiling --------------------------------------------------------------------------------


def _section_ceiling(ctx: _Context) -> tuple[list[str], list[dict[str, object]]]:
    args = ctx.args
    if len(ctx.annotators) < 2:
        raise KeyError("the human ceiling needs at least two annotators")
    coefficients = _ordered(args.coefficient, ["kendall_b"])
    groupings = _ordered(args.grouping, list(GROUPINGS))
    entries: list[dict[str, object]] = []
    lines = [
        "| grouping | coefficient | scores | against the mean of the other annotators |",
        "| --- | --- | --- | ---: |",
    ]
    for grouping in groupings:
        for coefficient in coefficients:
            common = dict(coefficient=coefficient, grouping=grouping, undefined=args.undefined)
            ceiling = human_ceiling(ctx.ratings, **common)
            entry: dict[str, object] = {
                **ceiling.to_dict(),
                "label": GROUPINGS[grouping],
                "annotators": list(ctx.annotators),
                "metrics": {},
            }
            text = _number(ceiling.value)
            if ctx.intervals:
                values = resample_statistic(
                    lambda r, common=common: human_ceiling(r, **common).value,
                    [ctx.ratings],
                    resample=args.resample,
                    n_resamples=args.resamples,
                    seed=args.seed,
                )
                low, high, _ = percentile_interval(values, args.confidence)
                entry["interval"] = {"low": low, "high": high}
                text += _range(low, high)
            lines.append(f"| {grouping} | {coefficient} | human ceiling | {text} |")
            for name, value in zip(ctx.annotators, ceiling.per_annotator, strict=True):
                lines.append(f"| {grouping} | {coefficient} | {name} | {_number(value)} |")
            for metric in ctx.metrics:
                scores = ctx.table.metric(metric)
                value = metric_against_rest(scores, ctx.ratings, **common)
                text = _number(value)
                metric_entry: dict[str, object] = {"value": None if math.isnan(value) else value}
                if ctx.intervals:
                    values = resample_statistic(
                        lambda m, r, common=common: metric_against_rest(m, r, **common),
                        [scores, ctx.ratings],
                        resample=args.resample,
                        n_resamples=args.resamples,
                        seed=args.seed,
                    )
                    low, high, _ = percentile_interval(values, args.confidence)
                    metric_entry["interval"] = {"low": low, "high": high}
                    text += _range(low, high)
                entry["metrics"][metric] = metric_entry
                lines.append(f"| {grouping} | {coefficient} | {metric} | {text} |")
            entries.append(entry)
    lines += _groupings_legend(groupings)
    lines += [
        "",
        "human ceiling: mean over annotators of each annotator against the mean rating of the "
        "others. A metric is scored against the same leave-one-out means and averaged, so it "
        "is compared with references built from one annotator fewer than the full panel.",
    ]
    return lines, entries


# commands -------------------------------------------------------------------------------


def _document(ctx: _Context, title: str, sections: dict[str, tuple[list[str], object]]) -> str:
    if ctx.args.format == "json":
        document = {**ctx.describe(), **{key: data for key, (_, data) in sections.items()}}
        return json.dumps(_clean(document), indent=2, allow_nan=False)
    lines = ctx.header(title)
    for heading, (section, _) in sections.items():
        if len(sections) > 1:
            lines += ["", f"## {_HEADINGS[heading]}"]
        lines += ["", *section]
    return "\n".join(lines)


_HEADINGS = {
    "results": "Correlation",
    "comparisons": "Differences between metrics",
    "ceiling": "Human ceiling",
    "scale": "Scale use and ordinal agreement",
}


def _clean(value: object) -> object:
    """Replace ``nan`` by ``None`` anywhere in nested JSON data."""
    if isinstance(value, dict):
        return {key: _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def _run_correlate(args: argparse.Namespace) -> str:
    ctx = _context(args)
    return _document(ctx, "Correlation with human ratings", {"results": _section_correlate(ctx)})


def _run_compare(args: argparse.Namespace) -> str:
    ctx = _context(args)
    sections = {"comparisons": _section_compare(ctx)}
    return _document(ctx, "Differences between metrics", sections)


def _run_scale(args: argparse.Namespace) -> str:
    ctx = _context(args)
    sections = {"scale": _section_scale(ctx, ctx.metrics)}
    return _document(ctx, "Scale use and ordinal agreement", sections)


def _run_ceiling(args: argparse.Namespace) -> str:
    ctx = _context(args)
    return _document(ctx, "Human ceiling", {"ceiling": _section_ceiling(ctx)})


def _run_report(args: argparse.Namespace) -> str:
    ctx = _context(args)
    sections: dict[str, tuple[list[str], object]] = {"results": _section_correlate(ctx)}
    if len(ctx.metrics) >= 2:
        sections["comparisons"] = _section_compare(ctx)
    if len(ctx.annotators) >= 2:
        sections["ceiling"] = _section_ceiling(ctx)
    if args.grader:
        graders = _ordered(args.grader, [])
        for grader in graders:
            ctx.table.metric(grader)
        sections["scale"] = _section_scale(ctx, graders)
    return _document(ctx, "Metric against human ratings", sections)


_RUNNERS = {
    "correlate": _run_correlate,
    "compare": _run_compare,
    "scale": _run_scale,
    "ceiling": _run_ceiling,
    "report": _run_report,
}


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command line; returns the exit status."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        output = args.run(args)
    except (ScoreFormatError, OSError, ValueError) as error:
        print(f"metaeval: error: {error}", file=sys.stderr)
        return 2
    except KeyError as error:
        print(f"metaeval: error: {error.args[0]}", file=sys.stderr)
        return 2
    if args.output:
        Path(args.output).write_text(output + "\n", encoding="utf-8")
    else:
        print(output)
    return 0
