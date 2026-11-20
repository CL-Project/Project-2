"""Tests for MIRACL-hi loading, pooling and metrics (S33), offline and tiny."""

from __future__ import annotations

import gzip
import json

import numpy as np
import pandas as pd
import pytest

from src.tasks import miracl
from src.tasks.miracl import MiraclPassage, build_pool, metric_rows, per_item_frame
from src.tasks.parallel import RetrievalResult


def _corpus() -> list[MiraclPassage]:
    return [MiraclPassage(docid=f"d{index}", text=f"passage number {index}") for index in range(20)]


def _topics() -> dict[str, str]:
    return {"q1": "पहला प्रश्न", "q2": "दूसरा प्रश्न"}


def _qrels() -> dict[str, list[str]]:
    return {"q1": ["d0", "d5"], "q2": ["d10"]}


def test_read_topics(tmp_path):
    path = tmp_path / "topics.tsv"
    path.write_text("q1\tपहला प्रश्न\nq2\tदूसरा प्रश्न\n", encoding="utf-8")
    assert miracl.read_topics(path) == {"q1": "पहला प्रश्न", "q2": "दूसरा प्रश्न"}


def test_read_topics_rejects_duplicates(tmp_path):
    path = tmp_path / "topics.tsv"
    path.write_text("q1\ta\nq1\tb\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate query id"):
        miracl.read_topics(path)


def test_read_qrels_keeps_only_relevant_grades(tmp_path):
    path = tmp_path / "qrels.tsv"
    path.write_text(
        "q1\tQ0\td0\t1\nq1\tQ0\td1\t0\nq2\tQ0\td2\t2\nq1\tQ0\td0\t1\n",
        encoding="utf-8",
    )
    assert miracl.read_qrels(path) == {"q1": ["d0"], "q2": ["d2"]}


def test_read_qrels_rejects_bad_columns(tmp_path):
    path = tmp_path / "qrels.tsv"
    path.write_text("q1\tQ0\td0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="four tab-separated fields"):
        miracl.read_qrels(path)


def test_iter_corpus_reads_every_gzipped_shard(tmp_path):
    for shard, rows in (
        ("docs-0.jsonl.gz", [{"docid": "a", "text": "पहला"}, {"docid": "b", "text": "दूसरा"}]),
        ("docs-1.jsonl.gz", [{"docid": "c", "text": "तीसरा"}]),
    ):
        with gzip.open(tmp_path / shard, "wt", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    passages = list(miracl.iter_corpus(tmp_path))
    assert [passage.docid for passage in passages] == ["a", "b", "c"]
    assert passages[0].text == "पहला"


def test_build_pool_includes_every_relevant_and_the_requested_random_count():
    pool = build_pool(_topics(), _qrels(), _corpus(), max_random_passages=5, seed=42)

    assert pool.n_queries == 2
    assert pool.n_relevant_passages == 3
    assert pool.n_random_passages == 5
    assert pool.n_candidates == 8
    assert pool.corpus_size == 20
    relevant = {pool.passage_ids[gold] for golds in pool.gold_indices for gold in golds}
    assert relevant == {"d0", "d5", "d10"}
    assert len(set(pool.passage_ids)) == pool.n_candidates


def test_build_pool_is_deterministic():
    first = build_pool(_topics(), _qrels(), _corpus(), max_random_passages=5, seed=42)
    second = build_pool(_topics(), _qrels(), _corpus(), max_random_passages=5, seed=42)
    assert first.passage_ids == second.passage_ids


def test_build_pool_uses_every_passage_when_the_corpus_is_small():
    pool = build_pool(_topics(), _qrels(), _corpus(), max_random_passages=100, seed=42)
    assert pool.n_random_passages == 17
    assert pool.n_candidates == 20


def test_build_pool_rejects_a_missing_relevant_passage():
    qrels = {"q1": ["missing"], "q2": ["d10"]}
    with pytest.raises(ValueError, match="missing from the corpus"):
        build_pool(_topics(), qrels, _corpus(), max_random_passages=5)


def test_build_pool_rejects_a_query_without_gold():
    qrels = {"q1": ["d0"]}
    with pytest.raises(ValueError, match="no relevant passage"):
        build_pool(_topics(), qrels, _corpus(), max_random_passages=5)


def test_per_item_frame_keeps_the_true_reciprocal_rank():
    pool = build_pool(_topics(), _qrels(), _corpus(), max_random_passages=0, seed=42)
    result = RetrievalResult(
        ranks=np.array([2, 15], dtype=np.int64),
        best_index=np.zeros(2, dtype=np.int64),
        best_score=np.ones(2, dtype=np.float32),
        gold_score=np.ones(2, dtype=np.float32),
        n_candidates=pool.n_candidates,
    )
    frame = per_item_frame(pool, result)
    assert list(frame.columns) == [
        "query_id",
        "gold_ids",
        "rank_of_best_gold",
        "top1",
        "top5",
        "rr",
        "n_candidates",
    ]
    np.testing.assert_allclose(frame["rr"].to_numpy(), [0.5, 1.0 / 15.0])
    assert frame["top5"].tolist() == [True, False]


def test_metric_rows_cap_the_reciprocal_rank_at_ten():
    """A rank of 15 contributes 0 to MRR@10 but a rank of 8 contributes 1/8."""
    frame = pd.DataFrame({"rank_of_best_gold": [1, 2, 8, 15]})
    rows = {row["metric"]: row for row in metric_rows(frame, system="s", setting="hi-hi", n_boot=50)}
    expected_mrr = (1 + 0.5 + 0.125 + 0.0) / 4
    assert rows["MRR@10"]["value"] == pytest.approx(expected_mrr)
    assert rows["Recall@10"]["value"] == pytest.approx(0.75)
    assert rows["MRR@10"]["n_items"] == 4
    assert rows["MRR@10"]["ci_low"] <= expected_mrr <= rows["MRR@10"]["ci_high"]


def test_capped_reciprocal_rank_rejects_zero():
    with pytest.raises(ValueError):
        miracl.capped_reciprocal_rank(np.array([0, 1]))
