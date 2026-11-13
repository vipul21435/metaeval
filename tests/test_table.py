"""Score tables: long and nested records, CSV and JSONL files, and format errors."""

from __future__ import annotations

import json

import numpy as np
import pytest

from metaeval.table import DEFAULT_DIMENSION, ScoreFormatError, ScoreTable, load_scores

NAN = float("nan")

LONG_RECORDS = [
    {"item": "doc_1", "system": "sys_a", "source": "human", "name": "ann_1", "score": 4},
    {"item": "doc_1", "system": "sys_a", "source": "human", "name": "ann_2", "score": 5},
    {"item": "doc_1", "system": "sys_a", "source": "metric", "name": "overlap", "score": 0.31},
    {"item": "doc_1", "system": "sys_b", "source": "human", "name": "ann_1", "score": 2},
    {"item": "doc_1", "system": "sys_b", "source": "metric", "name": "overlap", "score": 0.12},
    {"item": "doc_2", "system": "sys_a", "source": "human", "name": "ann_2", "score": 3},
    {"item": "doc_2", "system": "sys_a", "source": "metric", "name": "overlap", "score": 0.25},
    {"item": "doc_2", "system": "sys_b", "source": "metric", "name": "length", "score": 40},
]

NESTED_RECORDS = [
    {
        "item": "doc_1",
        "system": "sys_a",
        "human": {"ann_1": 4, "ann_2": 5},
        "metrics": {"overlap": 0.31},
    },
    {"item": "doc_1", "system": "sys_b", "human": {"ann_1": 2}, "metrics": {"overlap": 0.12}},
    {"item": "doc_2", "system": "sys_a", "human": {"ann_2": 3}, "metrics": {"overlap": 0.25}},
    {"item": "doc_2", "system": "sys_b", "metrics": {"length": 40}},
]


def check_small_table(table: ScoreTable) -> None:
    assert table.systems == ("sys_a", "sys_b")
    assert table.items == ("doc_1", "doc_2")
    assert table.shape == (2, 2)
    assert table.dimensions == (DEFAULT_DIMENSION,)
    assert table.annotators() == ("ann_1", "ann_2")
    assert list(table.metrics) == ["overlap", "length"]
    np.testing.assert_array_equal(table.metric("overlap"), [[0.31, 0.25], [0.12, NAN]])
    np.testing.assert_array_equal(table.metric("length"), [[NAN, NAN], [NAN, 40.0]])
    np.testing.assert_array_equal(table.ratings["overall"]["ann_1"], [[4.0, NAN], [2.0, NAN]])
    np.testing.assert_array_equal(table.ratings["overall"]["ann_2"], [[5.0, 3.0], [NAN, NAN]])
    # Mean over the annotators that rated the output; nan where nobody did.
    np.testing.assert_array_equal(table.human(), [[4.5, 3.0], [2.0, NAN]])
    np.testing.assert_array_equal(table.human(annotators=["ann_1"]), [[4.0, NAN], [2.0, NAN]])


def test_long_records():
    check_small_table(ScoreTable.from_records(LONG_RECORDS))


def test_nested_records():
    check_small_table(ScoreTable.from_records(NESTED_RECORDS))


def test_long_and_nested_records_can_be_mixed():
    check_small_table(ScoreTable.from_records(LONG_RECORDS[:5] + NESTED_RECORDS[2:]))


def test_jsonl_file(tmp_path):
    path = tmp_path / "scores.jsonl"
    lines = [json.dumps(record) for record in NESTED_RECORDS]
    path.write_text("\n".join([lines[0], "", *lines[1:]]) + "\n", encoding="utf-8")
    check_small_table(load_scores(path))


def test_csv_file(tmp_path):
    path = tmp_path / "scores.csv"
    rows = ["item,system,source,name,score"]
    rows += [
        f"{r['item']},{r['system']},{r['source']},{r['name']},{r['score']}" for r in LONG_RECORDS
    ]
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    check_small_table(load_scores(path))


def test_csv_with_dimension_column_and_blank_scores(tmp_path):
    path = tmp_path / "scores.csv"
    path.write_text(
        "item,system,source,name,dimension,score\n"
        "doc_1,sys_a,human,ann_1,fluency,4\n"
        "doc_1,sys_a,human,ann_1,relevance,2\n"
        "doc_1,sys_a,human,ann_2,fluency,\n"
        "doc_1,sys_a,human,ann_2,relevance,nan\n"
        "doc_1,sys_a,metric,overlap,,0.5\n",
        encoding="utf-8",
    )
    table = load_scores(path)
    assert table.dimensions == ("fluency", "relevance")
    np.testing.assert_array_equal(table.human("fluency"), [[4.0]])
    np.testing.assert_array_equal(table.human("relevance"), [[2.0]])
    np.testing.assert_array_equal(table.ratings["fluency"]["ann_2"], [[NAN]])
    np.testing.assert_array_equal(table.metric("overlap"), [[0.5]])


def test_format_can_be_forced(tmp_path):
    path = tmp_path / "scores.txt"
    path.write_text("item,system,source,name,score\nd,s,metric,m,1\n", encoding="utf-8")
    assert load_scores(path, fmt="csv").metric("m")[0, 0] == 1.0
    with pytest.raises(ScoreFormatError, match=r"scores\.txt line 1"):
        load_scores(path)
    with pytest.raises(ValueError, match="unknown format"):
        load_scores(path, fmt="tsv")


def test_nested_human_by_dimension():
    table = ScoreTable.from_records(
        [
            {
                "item": 1,
                "system": "s",
                "human": {"coherence": {"expert_1": 4, "expert_2": 2}, "fluency": {"expert_1": 5}},
            },
            {
                "item": 2,
                "system": "s",
                "human": {"coherence": {"expert_1": 1, "crowd_1": 3}},
            },
        ]
    )
    assert table.items == ("1", "2")
    assert table.dimensions == ("coherence", "fluency")
    assert table.annotators("coherence") == ("expert_1", "expert_2", "crowd_1")
    assert table.annotators("coherence", ["expert_*"]) == ("expert_1", "expert_2")
    assert table.annotators("coherence", ["crowd_1", "expert_*", "*_1"]) == (
        "crowd_1",
        "expert_1",
        "expert_2",
    )
    np.testing.assert_array_equal(table.human("coherence"), [[3.0, 2.0]])
    np.testing.assert_array_equal(table.human("coherence", ["expert_*"]), [[3.0, 1.0]])
    np.testing.assert_array_equal(table.human("fluency"), [[5.0, NAN]])
    with pytest.raises(KeyError, match="several dimensions"):
        table.human()
    with pytest.raises(KeyError, match="unknown dimension"):
        table.human("relevance")
    with pytest.raises(KeyError, match="no annotator matches"):
        table.human("coherence", ["judge_*"])


def test_unknown_metric_and_missing_ratings():
    table = ScoreTable.from_records([{"item": "d", "system": "s", "metrics": {"m": 1.0}}])
    with pytest.raises(KeyError, match="unknown metric 'x'; the table has: m"):
        table.metric("x")
    with pytest.raises(KeyError, match="no human ratings"):
        table.human()
    assert table.dimensions == ()


def test_arrays_are_read_only():
    table = ScoreTable.from_records(NESTED_RECORDS)
    with pytest.raises(ValueError, match="read-only"):
        table.metric("overlap")[0, 0] = 1.0


@pytest.mark.parametrize(
    ("record", "message"),
    [
        ({"system": "s", "source": "metric", "name": "m", "score": 1}, "missing field 'item'"),
        ({"item": "d", "source": "metric", "name": "m", "score": 1}, "missing field 'system'"),
        ({"item": " ", "system": "s", "metrics": {"m": 1}}, "missing field 'item'"),
        ({"item": "d", "system": "s", "source": "metric", "score": 1}, "missing field 'name'"),
        ({"item": "d", "system": "s", "source": "metric", "name": "m"}, "missing field 'score'"),
        (
            {"item": "d", "system": "s", "source": "judge", "name": "m", "score": 1},
            "source must be",
        ),
        ({"item": "d", "system": "s", "score": 1}, "expected either"),
        ({"item": "d", "system": "s", "metrics": {"m": "high"}}, "score must be a number"),
        ({"item": "d", "system": "s", "metrics": {"m": True}}, "score must be a number"),
        ({"item": "d", "system": "s", "metrics": {"m": [1]}}, "score must be a number"),
        ({"item": "d", "system": "s", "metrics": {"m": float("inf")}}, "score must be finite"),
        ({"item": "d", "system": "s", "metrics": [1, 2]}, "must map names to scores"),
        ({"item": "d", "system": "s", "human": {"a": 1, "dim": {"b": 2}}}, "not a mix"),
        (
            {
                "item": "d",
                "system": "s",
                "source": "metric",
                "name": "m",
                "dimension": "fluency",
                "score": 1,
            },
            "takes no dimension",
        ),
        (["d", "s", 1], "expected an object"),
    ],
)
def test_malformed_record(record, message):
    with pytest.raises(ScoreFormatError, match=message):
        ScoreTable.from_records([record])


def test_error_names_the_record():
    good = {"item": "d", "system": "s", "metrics": {"m": 1}}
    with pytest.raises(ScoreFormatError, match=r"^record 2: second score of metric 'm'"):
        ScoreTable.from_records([good, good])
    with pytest.raises(ScoreFormatError, match=r"^record 7: second score of annotator 'a'"):
        ScoreTable.from_records(
            [
                (3, {"item": "d", "system": "s", "human": {"a": 1}}),
                (7, {"item": "d", "system": "s", "source": "human", "name": "a", "score": 2}),
            ]
        )


def test_file_errors_name_the_line(tmp_path):
    path = tmp_path / "broken.jsonl"
    path.write_text('{"item": "d", "system": "s", "metrics": {"m": 1}}\n{"item": \n', "utf-8")
    with pytest.raises(ScoreFormatError, match=r"broken\.jsonl line 2"):
        load_scores(path)
    path.write_text('{"item": "d", "system": "s", "metrics": {"m": "x"}}\n', "utf-8")
    with pytest.raises(ScoreFormatError, match=r"broken\.jsonl line 1: score must be a number"):
        load_scores(path)
    csv_path = tmp_path / "broken.csv"
    csv_path.write_text("item,system,name,score\nd,s,m,1\n", "utf-8")
    with pytest.raises(ScoreFormatError, match=r"broken\.csv: CSV header lacks: source"):
        load_scores(csv_path)
    csv_path.write_text("item,system,source,name,score\nd,s,metric,m,1\nd,s,metric,m,2\n", "utf-8")
    with pytest.raises(ScoreFormatError, match=r"broken\.csv line 3: second score"):
        load_scores(csv_path)


def test_empty_input():
    with pytest.raises(ScoreFormatError, match="no scores found"):
        ScoreTable.from_records([])


def test_missing_scores_leave_cells_empty():
    table = ScoreTable.from_records(
        [
            {"item": "d1", "system": "s", "human": {"a": None, "b": 3}, "metrics": {"m": None}},
            {"item": "d2", "system": "s", "human": {"a": "", "b": "4"}, "metrics": {"m": " 2.5 "}},
        ]
    )
    np.testing.assert_array_equal(table.metric("m"), [[NAN, 2.5]])
    np.testing.assert_array_equal(table.human(), [[3.0, 4.0]])
    np.testing.assert_array_equal(table.human(annotators=["a"]), [[NAN, NAN]])
