"""BM25 ranking baselines (S17 for T2, S29 for T3).

``rank_bm25.BM25Okapi`` with the documented defaults (``k1=1.5``, ``b=0.75``),
tokens from ``src.textproc.tokenize``, no stemming and no stopword list.  BM25 is
lexical, so it runs on the *same* candidate text for every query: a query is
compared with candidates in one language only and cross-lingual settings go
through machine translation first (``bm25_mt``).

The ranker is returned through the same :class:`~src.tasks.parallel.RetrievalResult`
shape as cosine retrieval so per-item logging and metrics are shared.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import yaml
from rank_bm25 import BM25Okapi

from src.retrieval import rank_of_gold
from src.tasks.parallel import RetrievalResult
from src.textproc import tokenize

REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS_CONFIG_PATH = REPO_ROOT / "configs" / "models.yaml"

LOGGER = logging.getLogger("rankers")


def bm25_settings(config: dict[str, Any] | None = None) -> dict[str, float]:
    """Return ``{"k1": ..., "b": ...}`` from ``configs/models.yaml``."""
    if config is None:
        with MODELS_CONFIG_PATH.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
    settings = dict(config["baselines"]["bm25"])
    return {"k1": float(settings["k1"]), "b": float(settings["b"])}


def bm25_scores(
    queries: Sequence[str],
    candidates: Sequence[str],
    *,
    lang: str,
    k1: float = 1.5,
    b: float = 0.75,
    chunk: int = 512,
) -> np.ndarray:
    """Return a ``(n_queries, n_candidates)`` BM25 score matrix in float32.

    Both sides are tokenized with the shared tokenizer (``AGENTS.md`` §10.1), so
    Devanagari words are never split at a matra.  An all-zero query scores zero
    against every candidate rather than raising.
    """
    if not queries or not candidates:
        raise ValueError("queries and candidates must not be empty")
    if chunk < 1:
        raise ValueError("chunk must be >= 1")

    tokenized_candidates = [tokenize(text, lang) for text in candidates]
    model = BM25Okapi(tokenized_candidates, k1=k1, b=b)

    tokenized_queries = [tokenize(text, lang) for text in queries]
    scores = np.empty((len(queries), len(candidates)), dtype=np.float32)
    for start in range(0, len(queries), chunk):
        stop = min(start + chunk, len(queries))
        for offset, tokens in enumerate(tokenized_queries[start:stop]):
            row = np.asarray(model.get_scores(tokens), dtype=np.float32)
            if row.shape != (len(candidates),):
                raise RuntimeError(f"BM25 returned {row.shape} scores for {len(candidates)} candidates")
            scores[start + offset] = row
    return scores


def bm25_rank(
    queries: Sequence[str],
    candidates: Sequence[str],
    gold_indices: Sequence[Sequence[int]],
    *,
    lang: str,
    config: dict[str, Any] | None = None,
    chunk: int = 512,
) -> RetrievalResult:
    """Rank ``candidates`` for each query with BM25 and compute per-item outcomes.

    Ties are handled exactly like the cosine path: :func:`rank_of_gold` counts
    the candidates strictly above the best gold score, so an exact tie for first
    place ranks 1.
    """
    if len(queries) != len(gold_indices):
        raise ValueError(f"{len(queries)} queries but {len(gold_indices)} gold entries")
    settings = bm25_settings(config)
    matrix = bm25_scores(queries, candidates, lang=lang, k1=settings["k1"], b=settings["b"], chunk=chunk)

    ranks = np.empty(len(queries), dtype=np.int64)
    best_index = np.empty(len(queries), dtype=np.int64)
    best_score = np.empty(len(queries), dtype=np.float32)
    gold_score = np.empty(len(queries), dtype=np.float32)
    for index, gold in enumerate(gold_indices):
        row = matrix[index]
        ranks[index] = rank_of_gold(row, gold)
        best_index[index] = int(np.argmax(row))
        best_score[index] = float(row.max())
        gold_score[index] = float(row[list(gold)].max())
    return RetrievalResult(
        ranks=ranks,
        best_index=best_index,
        best_score=best_score,
        gold_score=gold_score,
        n_candidates=len(candidates),
    )
