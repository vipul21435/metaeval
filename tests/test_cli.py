"""The correlate command: Markdown and JSON output, selection options and errors."""

from __future__ import annotations

import json
import runpy
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from metaeval import __version__
from metaeval.cli import main
from metaeval.coefficients import COEFFICIENTS
from metaeval.grouping import GROUPINGS, correlate

EXAMPLE = Path(__file__).resolve().parent.parent / "examples" / "toy_scores.jsonl"

# Three systems by four items; see test_grouping.py for the closed-form values.
HUMAN = np.array([[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6]], dtype=float)
METRIC = np.array([[4, 3, 2, 1], [5, 4, 3, 2], [6, 5, 4, 3]], dtype=float)


def write_scores(path: Path, human: np.ndarray, **metrics: np.ndarray) -> Path:
    """Write a nested JSONL file with one annotator and the given metric arrays."""
    lines = []
    for system in range(human.shape[0]):
        for item in range(human.shape[1]):
            record = {
                "item": f"doc_{item + 1}",
                "system": f"sys_{system + 1}",
                "human": {"ann_1": human[system, item]},
                "metrics": {name: scores[system, item] for name, scores in metrics.items()},
            }
            lines.append(json.dumps(record))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


@pytest.fixture
def grid(tmp_path) -> Path:
    return write_scores(tmp_path / "grid.jsonl", HUMAN, follows=METRIC, copies=HUMAN)


def run(capsys, *arguments: str) -> tuple[int, str, str]:
    status = main([str(argument) for argument in arguments])
    captured = capsys.readouterr()
    return status, captured.out, captured.err


def table_rows(markdown: str) -> list[list[str]]:
    rows = [line for line in markdown.splitlines() if line.startswith("|")]
    return [[cell.strip() for cell in row.strip("|").split("|")] for row in rows]


def test_markdown_has_every_grouping_labelled(capsys, grid):
    status, out, err = run(capsys, "correlate", grid)
    assert status == 0
    assert err == ""
    assert out.startswith("# Correlation with human ratings\n")
    assert f"- scores: {grid} (3 systems, 4 items)" in out
    assert "- human score: mean of 1 annotator (ann_1), dimension 'overall'" in out
    for name, label in GROUPINGS.items():
        assert f"- {name}: {label}" in out
    rows = table_rows(out)
    assert rows[0] == ["metric", "grouping", *COEFFICIENTS, "points", "groups"]
    assert [row[:2] for row in rows[2:]] == [
        [metric, grouping] for metric in ("follows", "copies") for grouping in GROUPINGS
    ]
    by_key = {(row[0], row[1]): dict(zip(rows[0], row, strict=True)) for row in rows[2:]}
    assert by_key["follows", "system"]["kendall_b"] == "1.000"
    assert by_key["follows", "system"]["points"] == "3"
    assert by_key["follows", "pooled"]["pearson"] == f"{-7 / 23:.3f}"
    assert by_key["follows", "pooled"]["kendall_b"] == f"{-14 / 58:.3f}"
    assert by_key["follows", "pooled"]["pairwise_accuracy"] == f"{18 / 66:.3f}"
    assert by_key["follows", "by_item"]["spearman"] == "1.000"
    assert by_key["follows", "by_item"]["groups"] == "4"
    assert by_key["follows", "by_system"]["kendall_c"] == "-1.000"
    assert by_key["follows", "by_system"]["pairwise_accuracy"] == "0.000"
    assert by_key["follows", "by_system"]["points"] == "12"
    # A metric equal to the human score: every coefficient is 1 except tau-c, which
    # stays below 1 when the tied values are unevenly spread. The 12 pooled human
    # scores hold 8 tied pairs, so C = 66 - 8 = 58 and tau-c = 2 * 58 * 6 / (144 * 5).
    copies = by_key["copies", "pooled"]
    for name in ("pearson", "spearman", "kendall_b", "pairwise_accuracy"):
        assert copies[name] == "1.000"
    assert copies["kendall_c"] == f"{2 * 58 * 6 / (144 * 5):.3f}" == "0.967"
    assert "n/a" not in out


def test_json_matches_the_library(capsys, grid):
    status, out, _ = run(capsys, "correlate", grid, "--format", "json")
    assert status == 0
    document = json.loads(out)
    assert document["scores"] == str(grid)
    assert (document["systems"], document["items"]) == (3, 4)
    assert document["human"] == {
        "dimension": "overall",
        "annotators": ["ann_1"],
        "aggregate": "mean",
    }
    assert document["undefined"] == "skip"
    assert document["groupings"] == GROUPINGS
    assert len(document["results"]) == 2 * len(GROUPINGS) * len(COEFFICIENTS)
    for entry in document["results"]:
        scores = METRIC if entry["metric"] == "follows" else HUMAN
        expected = correlate(
            scores, HUMAN, coefficient=entry["coefficient"], grouping=entry["grouping"]
        )
        assert entry == {"metric": entry["metric"], **expected.to_dict()}
        assert entry["label"] == GROUPINGS[entry["grouping"]]


def test_selection_options(capsys, grid):
    status, out, _ = run(
        capsys,
        "correlate",
        grid,
        "--metric",
        "follows",
        "--coefficient",
        "kendall_b",
        "--coefficient",
        "pearson",
        "--coefficient",
        "kendall_b",
        "--grouping",
        "by_system",
    )
    assert status == 0
    rows = table_rows(out)
    assert rows[0] == ["metric", "grouping", "kendall_b", "pearson", "points", "groups"]
    assert rows[2:] == [["follows", "by_system", "-1.000", "-1.000", "12", "3"]]
    assert "- by_system:" in out
    assert "- pooled:" not in out


def test_undefined_groups_are_marked(capsys, tmp_path):
    # On item 2 the humans tie all three outputs, so tau-b is undefined there.
    human = np.array([[1, 2], [2, 2], [3, 2]], dtype=float)
    metric = np.array([[0.1, 0.5], [0.2, 0.4], [0.3, 0.6]])
    path = write_scores(tmp_path / "ties.jsonl", human, m=metric, flat=np.full((3, 2), 0.5))
    arguments = ["correlate", path, "--grouping", "by_item", "--grouping", "pooled"]
    status, out, _ = run(capsys, *arguments)
    assert status == 0
    rows = {(row[0], row[1]): row for row in table_rows(out)[2:]}
    header = table_rows(out)[0]
    by_item = dict(zip(header, rows["m", "by_item"], strict=True))
    assert by_item["kendall_b"] == "1.000 [1/2]"
    assert by_item["pairwise_accuracy"] == "0.500"
    assert dict(zip(header, rows["flat", "pooled"], strict=True))["pearson"] == "n/a"
    assert dict(zip(header, rows["flat", "by_item"], strict=True))["kendall_b"] == "n/a"
    assert "[k/n]: defined in k of n groups; the other groups are left out of the average." in out

    status, out, _ = run(capsys, *arguments, "--undefined", "zero")
    rows = {(row[0], row[1]): row for row in table_rows(out)[2:]}
    assert dict(zip(header, rows["m", "by_item"], strict=True))["kendall_b"] == "0.500 [1/2]"
    assert dict(zip(header, rows["flat", "by_item"], strict=True))["kendall_b"] == "0.000 [0/2]"
    assert "the other groups are counted as 0 in the average." in out

    status, out, _ = run(capsys, *arguments, "--format", "json")
    values = {
        (entry["metric"], entry["grouping"], entry["coefficient"]): entry
        for entry in json.loads(out)["results"]
    }
    assert values["flat", "pooled", "pearson"]["value"] is None
    assert values["m", "by_item", "kendall_b"]["n_undefined"] == 1


def test_dimension_and_annotator_selection(capsys, tmp_path):
    path = tmp_path / "dimensions.jsonl"
    records = []
    for system in range(3):
        for item in range(4):
            records.append(
                {
                    "item": item,
                    "system": system,
                    "human": {
                        "fluency": {"expert_1": HUMAN[system, item], "crowd_1": 3},
                        "relevance": {"expert_1": 7 - HUMAN[system, item]},
                    },
                    "metrics": {"m": HUMAN[system, item]},
                }
            )
    path.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")

    status, _, err = run(capsys, "correlate", path)
    assert status == 2
    assert (
        err
        == "metaeval: error: the table has several dimensions, choose one of: fluency, relevance\n"
    )

    arguments = ["correlate", path, "--grouping", "pooled", "--coefficient", "pearson"]
    status, out, _ = run(capsys, *arguments, "--dimension", "relevance")
    assert status == 0
    assert "mean of 1 annotator (expert_1), dimension 'relevance'" in out
    assert table_rows(out)[2] == ["m", "pooled", "-1.000", "12", "1"]

    status, out, _ = run(capsys, *arguments, "--dimension", "fluency", "--annotators", "expert_*")
    assert "mean of 1 annotator (expert_1), dimension 'fluency'" in out
    assert table_rows(out)[2] == ["m", "pooled", "1.000", "12", "1"]

    status, out, _ = run(capsys, *arguments, "--dimension", "fluency")
    assert "mean of 2 annotators (expert_1, crowd_1), dimension 'fluency'" in out

    status, _, err = run(capsys, *arguments, "--dimension", "fluency", "--annotators", "judge")
    assert status == 2
    assert (
        err == "metaeval: error: no annotator matches 'judge'; the table has: expert_1, crowd_1\n"
    )


def test_csv_input(capsys, tmp_path):
    rows = ["item,system,source,name,score"]
    for system in range(3):
        for item in range(4):
            rows.append(f"d{item},s{system},human,ann,{HUMAN[system, item]}")
            rows.append(f"d{item},s{system},metric,m,{METRIC[system, item]}")
    path = tmp_path / "scores.csv"
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    arguments = ["--grouping", "pooled", "--coefficient", "kendall_b"]
    status, out, _ = run(capsys, "correlate", path, *arguments)
    assert status == 0
    assert table_rows(out)[2] == ["m", "pooled", f"{-14 / 58:.3f}", "12", "1"]

    renamed = tmp_path / "scores.data"
    renamed.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    status, out, _ = run(capsys, "correlate", renamed, "--input-format", "csv", *arguments)
    assert status == 0
    assert table_rows(out)[2] == ["m", "pooled", f"{-14 / 58:.3f}", "12", "1"]


def test_errors_exit_with_status_two(capsys, grid, tmp_path):
    status, out, err = run(capsys, "correlate", tmp_path / "absent.jsonl")
    assert (status, out) == (2, "")
    assert err.startswith("metaeval: error: ")
    assert "absent.jsonl" in err

    status, _, err = run(capsys, "correlate", grid, "--metric", "bleu")
    assert status == 2
    assert err == "metaeval: error: unknown metric 'bleu'; the table has: follows, copies\n"

    broken = tmp_path / "broken.jsonl"
    broken.write_text('{"item": "d", "system": "s", "metrics": {"m": "x"}}\n', encoding="utf-8")
    status, _, err = run(capsys, "correlate", broken)
    assert status == 2
    assert err == "metaeval: error: broken.jsonl line 1: score must be a number, got 'x'\n"

    only_human = tmp_path / "human.jsonl"
    only_human.write_text('{"item": "d", "system": "s", "human": {"a": 1}}\n', encoding="utf-8")
    status, _, err = run(capsys, "correlate", only_human)
    assert status == 2
    assert err == "metaeval: error: the file has no metric scores\n"

    only_metric = tmp_path / "metric.jsonl"
    only_metric.write_text('{"item": "d", "system": "s", "metrics": {"m": 1}}\n', encoding="utf-8")
    status, _, err = run(capsys, "correlate", only_metric)
    assert status == 2
    assert err == "metaeval: error: the table has no human ratings\n"


def test_argparse_rejects_unknown_choices(capsys, grid):
    with pytest.raises(SystemExit) as raised:
        main(["correlate", str(grid), "--coefficient", "tau"])
    assert raised.value.code == 2
    assert "invalid choice: 'tau'" in capsys.readouterr().err
    with pytest.raises(SystemExit) as raised:
        main([])
    assert raised.value.code == 2


def test_version(capsys):
    with pytest.raises(SystemExit) as raised:
        main(["--version"])
    assert raised.value.code == 0
    assert capsys.readouterr().out == f"metaeval {__version__}\n"


def test_python_dash_m_runs_the_command(capsys, grid, monkeypatch):
    arguments = ["correlate", str(grid), "--grouping", "system", "--coefficient", "pearson"]
    monkeypatch.setattr(sys, "argv", ["metaeval", *arguments])
    with pytest.raises(SystemExit) as raised:
        runpy.run_module("metaeval", run_name="__main__")
    assert raised.value.code == 0
    assert table_rows(capsys.readouterr().out)[2] == ["follows", "system", "1.000", "3", "1"]


def test_module_entry_point_on_the_example_file():
    completed = subprocess.run(
        [sys.executable, "-m", "metaeval", "correlate", str(EXAMPLE), "--format", "json"],
        capture_output=True,
        text=True,
        check=True,
    )
    document = json.loads(completed.stdout)
    assert (document["systems"], document["items"]) == (4, 6)
    assert document["human"]["annotators"] == ["ann_1", "ann_2"]
    assert {entry["metric"] for entry in document["results"]} == {"metric_a", "metric_b"}
    assert all(entry["n_undefined"] == 0 for entry in document["results"])
