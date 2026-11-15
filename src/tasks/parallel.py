"""T2: parallel sentence retrieval on FLORES-200 (S15 to S17).

The pool is FLORES ``dev`` plus ``devtest`` (997 + 1,012 = 2,009 pairs).  For a
direction ``query_lang-candidate_lang`` the queries are all sentences in the
query language and the candidates are all sentences in the other language; the
gold of query *i* is the candidate taken from the same pair.  Chunks of FLORES
never share a sentence, so each query has one gold, but the implementation
groups candidates by identical normalized text so a duplicated sentence can
never produce a spurious rank.

Scoring is plain cosine on L2-normalized vectors (the plan specifies cosine for
T2).  Per-item rows follow the schema in ``src/results_log.py``; metrics are
Acc@1 (``top1``), MRR (``rr``) and Recall@5 (``top5``).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from src.retrieval import rank_of_gold
from src.textproc import normalize

T2_METRICS = {"Acc@1": "top1", "MRR": "rr", "Recall@5": "top5"}
DEFAULT_CHUNK = 512


@dataclass(frozen=True)
class SentencePair:
    """One FLORES pair: a Hindi sentence and its English translation."""

    pair_id: str
    split: str
    index: int
    hi: str
    en: str


@dataclass(frozen=True)
class ParallelPool:
    """Queries, candidates and gold indices for one direction."""

    setting: str
    query_lang: str
    candidate_lang: str
    query_ids: list[str]
    query_texts: list[str]
    candidate_ids: list[str]
    candidate_texts: list[str]
    gold_indices: list[list[int]]

    @property
    def n_queries(self) -> int:
        """Number of queries."""
        return len(self.query_ids)

    @property
    def n_candidates(self) -> int:
        """Number of candidates."""
        return len(self.candidate_ids)


@dataclass
class RetrievalResult:
    """Per-query outcome of a retrieval run."""

    ranks: np.ndarray
    best_index: np.ndarray
    best_score: np.ndarray
    gold_score: np.ndarray
    n_candidates: int

    def __len__(self) -> int:
        return int(self.ranks.size)


def parse_setting(setting: str) -> tuple[str, str]:
    """Split ``"hi-en"`` into ``("hi", "en")``, validating the language codes."""
    parts = setting.split("-")
    if len(parts) != 2 or any(part not in {"hi", "en"} for part in parts):
        raise ValueError(f"invalid setting {setting!r}; expected 'hi-en' or 'en-hi'")
    return parts[0], parts[1]


def read_flores_split(path: Path) -> list[str]:
    """Read one FLORES file: one sentence per line, blank lines dropped."""
    lines = [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines()]
    sentences = [line for line in lines if line]
    if not sentences:
        raise ValueError(f"{path} contains no sentences")
    return sentences


def load_flores_pairs(
    raw_dir: Path | str,
    *,
    splits: Sequence[str] = ("dev", "devtest"),
    languages: dict[str, str] | None = None,
) -> list[SentencePair]:
    """Load pooled FLORES pairs, asserting equal lengths per split.

    Args:
        raw_dir: ``data/raw/flores``.
        splits: split names, in the order they should be pooled.
        languages: mapping ``{"hi": "hin_Deva", "en": "eng_Latn"}``.

    Returns:
        Pairs in split order, with ids ``"<split>:<index>"`` (index 1-based, so
        ids are stable when the pooled set is extended).
    """
    codes = {"hi": "hin_Deva", "en": "eng_Latn"} if languages is None else dict(languages)
    missing = {"hi", "en"} - set(codes)
    if missing:
        raise ValueError(f"languages must define {sorted(missing)}")
    root = Path(raw_dir)

    pairs: list[SentencePair] = []
    for split in splits:
        hindi = read_flores_split(root / split / f"{codes['hi']}.{split}")
        english = read_flores_split(root / split / f"{codes['en']}.{split}")
        if len(hindi) != len(english):
            raise ValueError(
                f"{split}: {len(hindi)} Hindi lines but {len(english)} English lines; refusing to zip"
            )
        for offset, (hi, en) in enumerate(zip(hindi, english)):
            pairs.append(SentencePair(pair_id=f"{split}:{offset + 1:04d}", split=split, index=offset + 1, hi=hi, en=en))
    if not pairs:
        raise ValueError("no FLORES pairs loaded")
    return pairs


def build_pool(pairs: Iterable[SentencePair], setting: str) -> ParallelPool:
    """Build the query/candidate pool for one direction.

    Candidate ids follow the pair order; the gold of query *i* is every candidate
    whose normalized text equals that of the pair *i* sentence in the candidate
    language, so duplicate sentences are handled with multiple gold ids.
    """
    query_lang, candidate_lang = parse_setting(setting)
    items = list(pairs)
    if not items:
        raise ValueError("pairs must not be empty")

    def text_of(pair: SentencePair, lang: str) -> str:
        return pair.hi if lang == "hi" else pair.en

    candidate_texts = [text_of(pair, candidate_lang) for pair in items]
    position_by_key: dict[str, list[int]] = {}
    for position, text in enumerate(candidate_texts):
        position_by_key.setdefault(normalize(text), []).append(position)

    gold_indices: list[list[int]] = []
    for pair in items:
        key = normalize(text_of(pair, candidate_lang))
        gold_indices.append(position_by_key[key])
    for query_index, gold in enumerate(gold_indices):
        if not gold:
            raise RuntimeError(f"query {query_index} has no gold candidate; pool construction is broken")

    return ParallelPool(
        setting=setting,
        query_lang=query_lang,
        candidate_lang=candidate_lang,
        query_ids=[pair.pair_id for pair in items],
        query_texts=[text_of(pair, query_lang) for pair in items],
        candidate_ids=[pair.pair_id for pair in items],
        candidate_texts=candidate_texts,
        gold_indices=gold_indices,
    )


def retrieve_by_cosine(
    queries: np.ndarray,
    candidates: np.ndarray,
    gold_indices: Sequence[Sequence[int]],
    *,
    chunk: int = DEFAULT_CHUNK,
) -> RetrievalResult:
    """Rank every query against every candidate by cosine similarity.

    Vectors must already be L2-normalized and row-aligned with the pool.  The
    score matrix is built in row chunks, so the implementation also works when
    the candidate pool is large.
    """
    query_matrix = np.asarray(queries, dtype=np.float32)
    candidate_matrix = np.asarray(candidates, dtype=np.float32)
    if query_matrix.ndim != 2 or candidate_matrix.ndim != 2:
        raise ValueError("queries and candidates must be 2-D")
    if query_matrix.shape[1] != candidate_matrix.shape[1]:
        raise ValueError("queries and candidates must have the same dimension")
    if query_matrix.shape[0] != len(gold_indices):
        raise ValueError(
            f"{query_matrix.shape[0]} query vectors but {len(gold_indices)} gold entries"
        )
    if chunk < 1:
        raise ValueError("chunk must be >= 1")

    n_queries = query_matrix.shape[0]
    n_candidates = candidate_matrix.shape[0]
    ranks = np.empty(n_queries, dtype=np.int64)
    best_index = np.empty(n_queries, dtype=np.int64)
    best_score = np.empty(n_queries, dtype=np.float32)
    gold_score = np.empty(n_queries, dtype=np.float32)

    for start in range(0, n_queries, chunk):
        stop = min(start + chunk, n_queries)
        block = query_matrix[start:stop] @ candidate_matrix.T
        for offset, gold in enumerate(gold_indices[start:stop]):
            row = block[offset]
            ranks[start + offset] = rank_of_gold(row, gold)
            best_index[start + offset] = int(np.argmax(row))
            best_score[start + offset] = float(row.max())
            gold_score[start + offset] = float(row[list(gold)].max())
    return RetrievalResult(
        ranks=ranks,
        best_index=best_index,
        best_score=best_score,
        gold_score=gold_score,
        n_candidates=n_candidates,
    )


def per_item_frame(pool: ParallelPool, result: RetrievalResult) -> pd.DataFrame:
    """Return the per-item table for one run (schema in ``src/results_log.py``)."""
    if len(result) != pool.n_queries:
        raise ValueError(f"result has {len(result)} rows but the pool has {pool.n_queries} queries")
    return pd.DataFrame(
        {
            "query_id": pool.query_ids,
            "gold_ids": [[pool.candidate_ids[index] for index in gold] for gold in pool.gold_indices],
            "rank_of_best_gold": result.ranks,
            "top1": result.ranks <= 1,
            "top5": result.ranks <= 5,
            "rr": 1.0 / result.ranks,
            "n_candidates": np.full(pool.n_queries, result.n_candidates, dtype=np.int64),
        }
    )
