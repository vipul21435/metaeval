"""Score tables: metric scores and human ratings laid out as systems by items.

A table is read from long records, one score per record::

    {"item": "doc_1", "system": "sys_a", "source": "human", "name": "ann_1", "score": 4}
    {"item": "doc_1", "system": "sys_a", "source": "metric", "name": "overlap", "score": 0.31}

(the same fields as columns of a CSV file, with an optional ``dimension`` on human
rows when people rated several qualities), or from nested JSONL records, one output per
record::

    {"item": "doc_1", "system": "sys_a",
     "human": {"ann_1": 4, "ann_2": 5}, "metrics": {"overlap": 0.31}}

where ``human`` may also map a dimension to its annotators. Scores are absolute numbers;
a missing, empty or NaN score leaves the cell empty.
"""

from __future__ import annotations

import csv
import fnmatch
import json
import math
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

__all__ = ["DEFAULT_DIMENSION", "ScoreFormatError", "ScoreTable", "load_scores", "read_records"]

DEFAULT_DIMENSION = "overall"
"""Dimension given to human ratings that do not name one."""

_LONG_FIELDS = ("item", "system", "source", "name", "score")


class ScoreFormatError(ValueError):
    """A record that cannot be read as scores; the message names the record."""


@dataclass(frozen=True)
class ScoreTable:
    """Metric scores and human ratings as arrays of shape ``(n_systems, n_items)``.

    Row ``i`` is a system and column ``j`` an item, so ``array[i, j]`` is the score of the
    output of system ``i`` for item ``j``. ``nan`` marks a missing score.
    """

    systems: tuple[str, ...]
    items: tuple[str, ...]
    metrics: dict[str, np.ndarray]
    ratings: dict[str, dict[str, np.ndarray]]

    @property
    def shape(self) -> tuple[int, int]:
        return len(self.systems), len(self.items)

    @property
    def dimensions(self) -> tuple[str, ...]:
        """Rated dimensions in the order they first appear."""
        return tuple(self.ratings)

    def metric(self, name: str) -> np.ndarray:
        """Scores of one metric."""
        try:
            return self.metrics[name]
        except KeyError:
            known = ", ".join(self.metrics) or "none"
            raise KeyError(f"unknown metric {name!r}; the table has: {known}") from None

    def annotators(
        self, dimension: str | None = None, select: Sequence[str] | None = None
    ) -> tuple[str, ...]:
        """Annotators of a dimension, optionally those matching names or patterns.

        ``select`` holds annotator names or shell-style patterns such as ``expert_*``;
        every entry has to match at least one annotator.
        """
        names = tuple(self.ratings[self._dimension(dimension)])
        if select is None:
            return names
        chosen: list[str] = []
        for pattern in select:
            matched = [name for name in names if fnmatch.fnmatchcase(name, pattern)]
            if not matched:
                raise KeyError(
                    f"no annotator matches {pattern!r}; the table has: {', '.join(names)}"
                )
            chosen.extend(name for name in matched if name not in chosen)
        return tuple(chosen)

    def human(
        self, dimension: str | None = None, annotators: Sequence[str] | None = None
    ) -> np.ndarray:
        """Mean rating over annotators for each output.

        The mean is taken over the annotators that rated the output; an output none of
        them rated is ``nan``. ``dimension`` may be left out when the table has one.
        """
        dimension = self._dimension(dimension)
        chosen = self.annotators(dimension, annotators)
        stacked = np.stack([self.ratings[dimension][name] for name in chosen])
        present = np.isfinite(stacked)
        counts = present.sum(axis=0)
        totals = np.where(present, stacked, 0.0).sum(axis=0)
        return np.divide(totals, counts, out=np.full(self.shape, np.nan), where=counts > 0)

    def _dimension(self, dimension: str | None) -> str:
        if not self.ratings:
            raise KeyError("the table has no human ratings")
        if dimension is None:
            if len(self.ratings) > 1:
                raise KeyError(
                    "the table has several dimensions, choose one of: " + ", ".join(self.ratings)
                )
            return next(iter(self.ratings))
        if dimension not in self.ratings:
            raise KeyError(
                f"unknown dimension {dimension!r}; the table has: {', '.join(self.ratings)}"
            )
        return dimension

    @classmethod
    def from_records(
        cls, records: Iterable[Mapping[str, object] | tuple[int, Mapping[str, object]]]
    ) -> ScoreTable:
        """Build a table from long or nested records (see the module docstring).

        Records may be given as ``(number, record)`` pairs; the number is then used in
        error messages, otherwise records are numbered from 1.
        """
        builder = _Builder()
        for position, entry in enumerate(records, start=1):
            if isinstance(entry, tuple):
                number, record = entry
            else:
                number, record = position, entry
            builder.add(record, f"record {number}")
        return builder.build()


class _Builder:
    def __init__(self) -> None:
        self.systems: dict[str, int] = {}
        self.items: dict[str, int] = {}
        self.metrics: dict[str, dict[tuple[int, int], float]] = {}
        self.ratings: dict[str, dict[str, dict[tuple[int, int], float]]] = {}

    def add(self, record: Mapping[str, object], where: str) -> None:
        if not isinstance(record, Mapping):
            raise ScoreFormatError(f"{where}: expected an object with fields, got {record!r}")
        cell = (
            self.systems.setdefault(_identifier(record, "system", where), len(self.systems)),
            self.items.setdefault(_identifier(record, "item", where), len(self.items)),
        )
        if "source" in record:
            self._add_long(record, cell, where)
        elif "human" in record or "metrics" in record:
            self._add_nested(record, cell, where)
        else:
            raise ScoreFormatError(
                f"{where}: expected either source/name/score fields or human/metrics fields"
            )

    def _add_long(self, record: Mapping[str, object], cell: tuple[int, int], where: str) -> None:
        source = str(record["source"]).strip().lower()
        name = _identifier(record, "name", where)
        dimension = str(record.get("dimension") or "").strip()
        if "score" not in record:
            raise ScoreFormatError(f"{where}: missing field 'score'")
        score = _score(record["score"], where)
        if source == "metric":
            if dimension:
                raise ScoreFormatError(
                    f"{where}: a metric row takes no dimension; put it in the metric name"
                )
            self._set(self.metrics.setdefault(name, {}), cell, score, f"metric {name!r}", where)
        elif source == "human":
            self._add_rating(dimension or DEFAULT_DIMENSION, name, cell, score, where)
        else:
            raise ScoreFormatError(
                f"{where}: source must be 'human' or 'metric', got {record['source']!r}"
            )

    def _add_nested(self, record: Mapping[str, object], cell: tuple[int, int], where: str) -> None:
        metrics = record.get("metrics") or {}
        human = record.get("human") or {}
        if not isinstance(metrics, Mapping) or not isinstance(human, Mapping):
            raise ScoreFormatError(f"{where}: 'human' and 'metrics' must map names to scores")
        for name, value in metrics.items():
            score = _score(value, where)
            self._set(
                self.metrics.setdefault(str(name), {}), cell, score, f"metric {name!r}", where
            )
        by_dimension = [isinstance(value, Mapping) for value in human.values()]
        if any(by_dimension) and not all(by_dimension):
            raise ScoreFormatError(
                f"{where}: 'human' must map annotators to scores, or dimensions to such maps, "
                "not a mix of both"
            )
        if any(by_dimension):
            for dimension, by_annotator in human.items():
                for annotator, value in by_annotator.items():
                    self._add_rating(
                        str(dimension), str(annotator), cell, _score(value, where), where
                    )
        else:
            for annotator, value in human.items():
                self._add_rating(
                    DEFAULT_DIMENSION, str(annotator), cell, _score(value, where), where
                )

    def _add_rating(
        self, dimension: str, annotator: str, cell: tuple[int, int], score: float | None, where: str
    ) -> None:
        cells = self.ratings.setdefault(dimension, {}).setdefault(annotator, {})
        self._set(cells, cell, score, f"annotator {annotator!r} on {dimension!r}", where)

    @staticmethod
    def _set(
        cells: dict[tuple[int, int], float],
        cell: tuple[int, int],
        score: float | None,
        what: str,
        where: str,
    ) -> None:
        if score is None:
            return
        if cell in cells:
            raise ScoreFormatError(f"{where}: second score of {what} for the same item and system")
        cells[cell] = score

    def build(self) -> ScoreTable:
        if not self.metrics and not self.ratings:
            raise ScoreFormatError("no scores found")
        shape = (len(self.systems), len(self.items))

        def to_array(cells: dict[tuple[int, int], float]) -> np.ndarray:
            array = np.full(shape, np.nan)
            for (row, column), score in cells.items():
                array[row, column] = score
            array.setflags(write=False)
            return array

        return ScoreTable(
            systems=tuple(self.systems),
            items=tuple(self.items),
            metrics={name: to_array(cells) for name, cells in self.metrics.items()},
            ratings={
                dimension: {name: to_array(cells) for name, cells in annotators.items()}
                for dimension, annotators in self.ratings.items()
            },
        )


def _identifier(record: Mapping[str, object], field: str, where: str) -> str:
    value = record.get(field)
    text = "" if value is None else str(value).strip()
    if not text:
        raise ScoreFormatError(f"{where}: missing field {field!r}")
    return text


def _score(value: object, where: str) -> float | None:
    """A finite float, or ``None`` for a missing score."""
    if value is None:
        return None
    if isinstance(value, bool):
        raise ScoreFormatError(f"{where}: score must be a number, got {value!r}")
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            number = float(text)
        except ValueError:
            raise ScoreFormatError(f"{where}: score must be a number, got {value!r}") from None
    elif isinstance(value, (int, float)):
        number = float(value)
    else:
        raise ScoreFormatError(f"{where}: score must be a number, got {value!r}")
    if math.isnan(number):
        return None
    if math.isinf(number):
        raise ScoreFormatError(f"{where}: score must be finite, got {value!r}")
    return number


def read_records(
    path: str | Path, fmt: str | None = None
) -> Iterator[tuple[int, Mapping[str, object]]]:
    """Yield ``(line number, record)`` from a JSONL or CSV file.

    ``fmt`` is ``"jsonl"`` or ``"csv"``; by default it follows the file extension, with
    anything other than ``.csv`` read as JSONL.
    """
    path = Path(path)
    if fmt is None:
        fmt = "csv" if path.suffix.lower() == ".csv" else "jsonl"
    if fmt == "csv":
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            missing = [field for field in _LONG_FIELDS if field not in (reader.fieldnames or [])]
            if missing:
                raise ScoreFormatError(f"{path.name}: CSV header lacks: {', '.join(missing)}")
            for row in reader:
                yield reader.line_num, row
    elif fmt == "jsonl":
        with path.open(encoding="utf-8-sig") as handle:
            for number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ScoreFormatError(f"{path.name} line {number}: {error.msg}") from None
                yield number, record
    else:
        raise ValueError(f"unknown format {fmt!r}; use 'jsonl' or 'csv'")


def load_scores(path: str | Path, fmt: str | None = None) -> ScoreTable:
    """Read a JSONL or CSV file of scores into a :class:`ScoreTable`."""
    builder = _Builder()
    name = Path(path).name
    for number, record in read_records(path, fmt):
        builder.add(record, f"{name} line {number}")
    return builder.build()
