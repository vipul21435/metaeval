"""Turn the SummEval human annotations into a score file for metaeval.

SummEval (Fabbri et al., 2021) holds ratings by three experts and five crowd workers on
coherence, consistency, fluency and relevance for 1,600 summaries: 16 systems on 100
news articles. This script downloads the annotation file linked from the SummEval
repository (about 6 MB) and writes one record per summary with

- the eight ratings of each dimension, as annotators ``expert_1`` to ``expert_3`` and
  ``crowd_1`` to ``crowd_5``;
- reference-overlap scores computed here from the summary and its 11 references:
  ``unigram_f1``, ``bigram_f1`` and ``lcs_f1`` (best F1 over the references, in the
  style of ROUGE-1, ROUGE-2 and ROUGE-L but with plain lower-cased word tokens and no
  stemming, so they are not the scores of the ROUGE toolkit), and ``length``, the
  number of tokens in the summary;
- ``crowd_<dimension>``, the mean of the five crowd ratings, so that the crowd can be
  read as a grader against the experts.

No article, summary or reference text is written. Usage:

    python examples/summeval_scores.py --out summeval_scores.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import urllib.request
from collections import Counter
from pathlib import Path

import numpy as np

URL = (
    "https://storage.googleapis.com/sfr-summarization-repo-research/model_annotations.aligned.jsonl"
)
DIMENSIONS = ("coherence", "consistency", "fluency", "relevance")
TOKEN = re.compile(r"[a-z0-9]+")


def tokens(text: str) -> list[str]:
    return TOKEN.findall(text.lower())


def f1(overlap: int, candidate: int, reference: int) -> float:
    if overlap == 0 or candidate == 0 or reference == 0:
        return 0.0
    precision, recall = overlap / candidate, overlap / reference
    return 2 * precision * recall / (precision + recall)


def ngram_f1(candidate: list[str], reference: list[str], n: int) -> float:
    left = Counter(zip(*(candidate[k:] for k in range(n)), strict=False))
    right = Counter(zip(*(reference[k:] for k in range(n)), strict=False))
    return f1(sum((left & right).values()), sum(left.values()), sum(right.values()))


def lcs_length(candidate: list[str], reference: list[str]) -> int:
    """Length of the longest common subsequence, one vectorised row at a time."""
    if not candidate or not reference:
        return 0
    codes = {token: code for code, token in enumerate(dict.fromkeys(candidate + reference))}
    right = np.array([codes[token] for token in reference])
    row = np.zeros(len(reference) + 1, dtype=np.int64)
    for token in candidate:
        best = np.maximum(row[1:], row[:-1] + (right == codes[token]))
        row[1:] = np.maximum.accumulate(best)
    return int(row[-1])


def lcs_f1(candidate: list[str], reference: list[str]) -> float:
    return f1(lcs_length(candidate, reference), len(candidate), len(reference))


def convert(row: dict) -> dict:
    candidate = tokens(row["decoded"])
    references = [tokens(reference) for reference in row["references"]]
    human: dict[str, dict[str, int]] = {}
    metrics = {
        "unigram_f1": max(ngram_f1(candidate, reference, 1) for reference in references),
        "bigram_f1": max(ngram_f1(candidate, reference, 2) for reference in references),
        "lcs_f1": max(lcs_f1(candidate, reference) for reference in references),
        "length": len(candidate),
    }
    for dimension in DIMENSIONS:
        ratings = {
            f"expert_{k}": rating[dimension]
            for k, rating in enumerate(row["expert_annotations"], start=1)
        }
        crowd = [rating[dimension] for rating in row["turker_annotations"]]
        ratings.update({f"crowd_{k}": rating for k, rating in enumerate(crowd, start=1)})
        human[dimension] = ratings
        metrics[f"crowd_{dimension}"] = sum(crowd) / len(crowd)
    return {"item": row["id"], "system": row["model_id"], "human": human, "metrics": metrics}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default="summeval_scores.jsonl", help="score file to write")
    parser.add_argument(
        "--raw",
        default="model_annotations.aligned.jsonl",
        help="where the downloaded annotation file is kept (downloaded if absent)",
    )
    args = parser.parse_args()
    raw = Path(args.raw)
    if not raw.exists():
        print(f"downloading {URL}")
        urllib.request.urlretrieve(URL, raw)
    data = raw.read_bytes()
    print(f"{raw.name}: {len(data)} bytes, sha256 {hashlib.sha256(data).hexdigest()}")
    rows = [json.loads(line) for line in data.decode("utf-8").splitlines() if line.strip()]
    records = [convert(row) for row in rows]
    Path(args.out).write_text(
        "\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8"
    )
    systems = {record["system"] for record in records}
    items = {record["item"] for record in records}
    print(
        f"wrote {len(records)} summaries, {len(systems)} systems, {len(items)} items to {args.out}"
    )


if __name__ == "__main__":
    main()
