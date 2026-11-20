"""Tests for the T3 evaluation paths in ``scripts/run_eval.py`` (S29 to S31).

Only the parts that need no model weights are exercised: the monolingual guard
on ``bm25``, the ``hi-en`` guard on ``bm25_mt``, and the BM25 scoring path on a
tiny synthetic pool.  The encoder and static paths are covered by the module
tests they delegate to.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.tasks import xquad
from src.tasks.xquad import PoolParagraph, T3Pool

ROOT = Path(__file__).resolve().parents[1]


def _load_run_eval():
    """Import ``scripts/run_eval.py`` as a module (its ``--help`` is tested separately)."""
    path = ROOT / "scripts" / "run_eval.py"
    spec = importlib.util.spec_from_file_location("run_eval_module", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _pool(query_lang: str, doc_lang: str) -> T3Pool:
    """Two documents in ``doc_lang``; two queries whose gold is the first document."""
    words = {
        "hi": ["नदी", "पर्वत", "शहर", "गाँव", "पुस्तक", "जल"],
        "en": ["river", "mountain", "city", "village", "book", "water"],
    }
    docs = [
        " ".join(words[doc_lang][:3] * 4),
        " ".join(words[doc_lang][3:] * 4),
    ]
    return T3Pool(
        setting=f"{query_lang}-{doc_lang}",
        query_lang=query_lang,
        doc_lang=doc_lang,
        query_ids=["q1", "q2"],
        query_texts=[docs[0], docs[1]],
        candidate_ids=["xq-000", "xq-001"],
        candidate_texts=docs,
        gold_indices=[[0], [1]],
    )


def _args(system: str):
    return SimpleNamespace(system=system, force=False)


def test_bm25_rejects_cross_lingual_settings():
    module = _load_run_eval()
    with pytest.raises(SystemExit, match="monolingual"):
        module.run_t3_bm25(_args("bm25"), "hi-en", _pool("hi", "en"), force=False)


def test_bm25_mt_rejects_other_settings():
    module = _load_run_eval()
    with pytest.raises(SystemExit, match="hi-en"):
        module.run_t3_bm25(_args("bm25_mt"), "en-en", _pool("en", "en"), force=False)


def test_bm25_scores_a_monolingual_pool():
    """The matching document ranks first, so both queries are Recall@1 hits."""
    module = _load_run_eval()
    frame, notes = module.run_t3_bm25(_args("bm25"), "en-en", _pool("en", "en"), force=False)

    assert notes["system_kind"] == "monolingual_bm25"
    assert list(frame.columns) == [
        "query_id",
        "gold_ids",
        "rank_of_best_gold",
        "top1",
        "top5",
        "rr",
        "n_candidates",
    ]
    np.testing.assert_array_equal(frame["rank_of_best_gold"].to_numpy(), [1, 1])
    assert frame["gold_ids"].tolist() == [["xq-000"], ["xq-001"]]
    assert frame["n_candidates"].tolist() == [2, 2]


def test_t3_system_lists_are_registered():
    module = _load_run_eval()
    assert "t3_xquad" in module.TASK_RUNNERS
    for system in ("bm25", "bm25_mt", "labse", "me5", "muril", "w2v", "ft", "ft_ref"):
        assert system in module.KNOWN_T3_SYSTEMS


def test_t3_metrics_match_per_item_columns():
    """The metric names in ``configs/tasks.yaml`` must exist as per-item columns."""
    from src.results_log import PER_ITEM_COLUMNS

    for column in xquad.T3_METRICS.values():
        assert column in PER_ITEM_COLUMNS


def test_pool_gold_pids_helper():
    pool = [
        PoolParagraph(pid="xq-000", text="a", is_gold_candidate=True),
        PoolParagraph(pid="dist-000", text="b", is_gold_candidate=False),
        PoolParagraph(pid="xq-001", text="c", is_gold_candidate=True),
    ]
    assert xquad.gold_pids(pool) == ["xq-000", "xq-001"]
