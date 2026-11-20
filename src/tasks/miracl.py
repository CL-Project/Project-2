"""T3b: monolingual Hindi retrieval on MIRACL-hi dev (S33, optional).

This is the larger Hindi-side check the proposal promises.  The task is
**monolingual** (Hindi query, Hindi passage), so the aligned static systems take
part without any Procrustes mapping: ``w2v`` and ``ft`` embed both sides in the
same Hindi space.

Pool design (PLAN S33): every passage judged relevant to any dev query, plus a
random sample of about 20,000 further passages from the Hindi corpus.  The full
corpus is about half a million short passages, so sampling is stated explicitly
rather than silently searched in full.

Metrics are MRR@10 and Recall@10, the standard MIRACL cut-offs.  ``MRR@10``
gives a query whose relevant passages all rank beyond 10 a reciprocal rank of 0,
so it is not the same as the plain MRR used on T3.
"""

from __future__ import annotations

import gzip
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import pandas as pd

from src.tasks.parallel import RetrievalResult

LOGGER = logging.getLogger("miracl")

MRR_CUTOFF = 10
RECALL_CUTOFF = 10


@dataclass(frozen=True)
class MiraclPassage:
    """One passage of the Hindi corpus."""

    docid: str
    text: str


@dataclass(frozen=True)
class MiraclQuery:
    """One dev query and its judged-relevant passage ids."""

    qid: str
    text: str
    relevant: tuple[str, ...]


@dataclass(frozen=True)
class MiraclPool:
    """Queries, passages and gold indices for the monolingual Hindi task."""

    query_ids: list[str]
    query_texts: list[str]
    passage_ids: list[str]
    passage_texts: list[str]
    gold_indices: list[list[int]]
    n_relevant_passages: int
    n_random_passages: int
    corpus_size: int

    @property
    def n_queries(self) -> int:
        """Number of dev queries."""
        return len(self.query_ids)

    @property
    def n_candidates(self) -> int:
        """Size of the passage pool."""
        return len(self.passage_ids)


def read_topics(path: Path | str) -> dict[str, str]:
    """Read a MIRACL topics TSV (``qid<TAB>query``) into an ordered mapping."""
    topics: dict[str, str] = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.rstrip("\n")
            if not stripped:
                continue
            parts = stripped.split("\t")
            if len(parts) != 2:
                raise ValueError(f"{path}:{line_number}: expected two tab-separated fields, got {len(parts)}")
            qid, text = parts
            if qid in topics:
                raise ValueError(f"{path}:{line_number}: duplicate query id {qid!r}")
            topics[qid] = text.strip()
    if not topics:
        raise ValueError(f"{path} contains no queries")
    return topics


def read_qrels(path: Path | str) -> dict[str, list[str]]:
    """Read a MIRACL qrels TSV and return ``qid -> [relevant passage ids]``.

    Columns are ``qid, iteration, docid, relevance``; only relevance >= 1 is
    kept, and a passage judged relevant more than once is listed once.
    """
    relevant: dict[str, list[str]] = {}
    seen: dict[str, set[str]] = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.rstrip("\n")
            if not stripped:
                continue
            parts = stripped.split("\t")
            if len(parts) != 4:
                raise ValueError(f"{path}:{line_number}: expected four tab-separated fields, got {len(parts)}")
            qid, _, docid, grade = parts
            if int(grade) < 1:
                continue
            bucket = seen.setdefault(qid, set())
            if docid in bucket:
                continue
            bucket.add(docid)
            relevant.setdefault(qid, []).append(docid)
    if not relevant:
        raise ValueError(f"{path} contains no relevant judgments")
    return relevant


def iter_corpus(directory: Path | str) -> Iterator[MiraclPassage]:
    """Yield every passage of the gzipped MIRACL corpus shards."""
    paths = sorted(Path(directory).glob("*.jsonl.gz"))
    if not paths:
        raise FileNotFoundError(f"no *.jsonl.gz corpus shards under {directory}")
    for path in paths:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                record = json.loads(stripped)
                docid = str(record.get("docid", ""))
                text = record.get("text")
                if not docid or not isinstance(text, str) or not text.strip():
                    raise ValueError(f"{path}:{line_number}: passage without a docid or text")
                yield MiraclPassage(docid=docid, text=text)


def sample_passages(
    passages: Sequence[MiraclPassage],
    *,
    count: int,
    seed: int,
) -> list[MiraclPassage]:
    """Return ``count`` passages sampled without replacement in a deterministic order."""
    rng = np.random.default_rng(seed)
    if len(passages) <= count:
        return list(passages)
    order = rng.permutation(len(passages))[:count]
    return [passages[int(index)] for index in sorted(order)]


def build_pool(
    topics: dict[str, str],
    qrels: dict[str, list[str]],
    corpus: Sequence[MiraclPassage],
    *,
    max_random_passages: int,
    seed: int = 42,
) -> MiraclPool:
    """Build the monolingual Hindi pool of relevant passages plus random ones.

    Args:
        topics: ``qid -> query text``.
        qrels: ``qid -> [relevant passage ids]``.
        corpus: every passage of the Hindi corpus.
        max_random_passages: how many further passages to sample.
        seed: sampling seed (the global one from ``configs/``).

    A query with no relevant passage in the corpus is a data error and raises:
    silently dropping it would flatter every system.
    """
    by_docid = {passage.docid: passage for passage in corpus}
    if len(by_docid) != len(corpus):
        raise ValueError("the corpus has duplicate docids")

    relevant_ids: list[str] = []
    seen: set[str] = set()
    missing: list[str] = []
    for qid in topics:
        for docid in qrels.get(qid, []):
            if docid not in by_docid:
                missing.append(docid)
                continue
            if docid not in seen:
                seen.add(docid)
                relevant_ids.append(docid)
    if missing:
        raise ValueError(
            f"{len(missing)} judged-relevant passages are missing from the corpus, e.g. {missing[:3]}"
        )

    relevant_set = set(relevant_ids)
    remaining = [passage for passage in corpus if passage.docid not in relevant_set]
    sampled = sample_passages(remaining, count=int(max_random_passages), seed=seed)

    pool_passages = [by_docid[docid] for docid in relevant_ids] + sampled
    position = {passage.docid: index for index, passage in enumerate(pool_passages)}
    if len(position) != len(pool_passages):
        raise ValueError("the pool has duplicate passage ids")

    query_ids: list[str] = []
    query_texts: list[str] = []
    gold_indices: list[list[int]] = []
    for qid, text in topics.items():
        gold = sorted({position[docid] for docid in qrels.get(qid, []) if docid in position})
        if not gold:
            raise ValueError(f"query {qid!r} has no relevant passage in the pool")
        query_ids.append(qid)
        query_texts.append(text)
        gold_indices.append(gold)

    LOGGER.info(
        "MIRACL-hi: %d queries, %d relevant passages, %d random passages "
        "(%d of %d corpus passages searched)",
        len(query_ids),
        len(relevant_ids),
        len(sampled),
        len(pool_passages),
        len(corpus),
    )
    return MiraclPool(
        query_ids=query_ids,
        query_texts=query_texts,
        passage_ids=[passage.docid for passage in pool_passages],
        passage_texts=[passage.text for passage in pool_passages],
        gold_indices=gold_indices,
        n_relevant_passages=len(relevant_ids),
        n_random_passages=len(sampled),
        corpus_size=len(corpus),
    )


def capped_reciprocal_rank(ranks: np.ndarray, cutoff: int = MRR_CUTOFF) -> np.ndarray:
    """Return ``1/rank`` for ranks within ``cutoff`` and 0 beyond it."""
    values = np.asarray(ranks, dtype=np.int64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("ranks must be a non-empty 1-D array")
    if np.any(values < 1):
        raise ValueError("ranks are 1-based and must be >= 1")
    return np.where(values <= cutoff, 1.0 / values, 0.0)


def per_item_frame(pool: MiraclPool, result: RetrievalResult) -> pd.DataFrame:
    """Return the per-item table for one MIRACL run.

    The columns are the canonical ones from ``src/results_log.py``: ``rr`` is the
    true reciprocal rank and ``rank_of_best_gold`` the true rank.  Both MIRACL
    metrics are derived from ``rank_of_best_gold`` when they are scored, so the
    per-item file keeps one meaning across tasks and a paired bootstrap can be
    recomputed from it later.
    """
    if len(result) != pool.n_queries:
        raise ValueError(f"result has {len(result)} rows but the pool has {pool.n_queries} queries")
    ranks = np.asarray(result.ranks, dtype=np.int64)
    return pd.DataFrame(
        {
            "query_id": pool.query_ids,
            "gold_ids": [[pool.passage_ids[index] for index in gold] for gold in pool.gold_indices],
            "rank_of_best_gold": ranks,
            "top1": ranks <= 1,
            "top5": ranks <= 5,
            "rr": 1.0 / ranks,
            "n_candidates": np.full(pool.n_queries, result.n_candidates, dtype=np.int64),
        }
    )


def metric_rows(
    frame: pd.DataFrame,
    *,
    system: str,
    setting: str,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
    task: str = "t3b_miracl",
) -> list[dict]:
    """Return ``MRR@10`` and ``Recall@10`` rows with bootstrap CIs.

    Both are computed from ``rank_of_best_gold``, so they need no extra per-item
    column: ``MRR@10`` averages the rank capped at 10 and ``Recall@10`` is the
    fraction of queries whose best relevant passage is within the top 10.
    """
    from src.metrics import bootstrap_ci

    ranks = pd.to_numeric(frame["rank_of_best_gold"], errors="raise").to_numpy(dtype=np.int64)
    values = {
        "MRR@10": capped_reciprocal_rank(ranks, MRR_CUTOFF),
        "Recall@10": (ranks <= RECALL_CUTOFF).astype(float),
    }
    rows: list[dict] = []
    for metric, scores in values.items():
        value, low, high = bootstrap_ci(scores, n=n_boot, alpha=alpha, seed=seed)
        rows.append(
            {
                "system": system,
                "task": task,
                "setting": setting,
                "metric": metric,
                "value": value,
                "ci_low": low,
                "ci_high": high,
                "n_items": int(len(frame)),
                "n_excluded": 0,
            }
        )
    return rows
