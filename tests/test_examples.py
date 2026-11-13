"""The SummEval conversion script: overlap scores by hand and the record layout."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from metaeval import ScoreTable

SCRIPT = Path(__file__).resolve().parent.parent / "examples" / "summeval_scores.py"
spec = importlib.util.spec_from_file_location("summeval_scores", SCRIPT)
summeval = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summeval)


def test_tokens_and_overlap_scores_by_hand():
    candidate = summeval.tokens("The cat sat on the mat.")
    reference = summeval.tokens("the cat is on THE mat")
    assert candidate == ["the", "cat", "sat", "on", "the", "mat"]
    # Unigrams: the(2) cat on mat overlap = 5 of 6 and 6.
    assert summeval.ngram_f1(candidate, reference, 1) == pytest.approx(5 / 6)
    # Bigrams in common: "the cat", "on the", "the mat" = 3 of 5 and 5.
    assert summeval.ngram_f1(candidate, reference, 2) == pytest.approx(3 / 5)
    # Longest common subsequence: the cat on the mat.
    assert summeval.lcs_length(candidate, reference) == 5
    assert summeval.lcs_f1(candidate, reference) == pytest.approx(5 / 6)
    assert summeval.lcs_length(list("abcbdab"), list("bdcaba")) == 4
    assert summeval.lcs_length([], reference) == 0
    assert summeval.ngram_f1(["a"], ["b"], 1) == 0.0
    assert summeval.ngram_f1(["a"], ["a"], 2) == 0.0


def test_converted_record_loads_as_scores():
    row = {
        "id": "doc-1",
        "model_id": "M1",
        "decoded": "the cat sat",
        "references": ["a cat sat down", "the dog sat"],
        "expert_annotations": [{d: 3 + k for d in summeval.DIMENSIONS} for k in range(3)],
        "turker_annotations": [{d: 2 for d in summeval.DIMENSIONS} for _ in range(5)],
    }
    record = summeval.convert(row)
    assert set(record) == {"item", "system", "human", "metrics"}
    assert "decoded" not in str(record) and "cat" not in str(record)
    assert record["metrics"]["length"] == 3
    # Unigrams: "the dog sat" shares 2 of 3 and 3 tokens. Bigrams: "a cat sat down"
    # shares "cat sat", 1 of 2 and 3.
    assert record["metrics"]["unigram_f1"] == pytest.approx(2 / 3)
    assert record["metrics"]["bigram_f1"] == pytest.approx(2 * (1 / 2) * (1 / 3) / (1 / 2 + 1 / 3))
    assert record["metrics"]["crowd_fluency"] == 2.0
    table = ScoreTable.from_records([record])
    assert table.dimensions == summeval.DIMENSIONS
    assert table.annotators("relevance", ["expert_*"]) == ("expert_1", "expert_2", "expert_3")
    assert table.human("relevance", ["expert_*"])[0, 0] == 4.0
    assert table.human("relevance", ["crowd_*"])[0, 0] == 2.0
