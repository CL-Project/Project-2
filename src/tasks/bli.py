"""T1: bilingual lexicon induction on the MUSE dictionaries (S22, S23, S24).

Queries are the unique source words of the MUSE **test** split
(``<pair>.5000-6500.txt``); the gold of a query is the set of its translations.
The candidate pool is the shared target vocabulary built in S20, identical for
every system (``AGENTS.md`` §9.2).

Two retrieval modes are scored:

* **CSLS** with ``k = 10`` — the primary metric (Conneau et al., 2018).
* **cosine** — secondary.

``r_S(y)`` (the candidate-side hubness term) is the expensive part: it needs the
cosine of every candidate against every source candidate.  It is computed once
per system and direction and can be cached, because it does not depend on the
queries.

A query belongs to the **common query set** only when it is in the shared source
candidate list *and* has at least one gold translation in the shared target
candidate list; excluded queries are counted, never silently dropped
(``AGENTS.md`` §7).
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from src.retrieval import hubness_mean, rank_of_gold
from src.tasks.parallel import RetrievalResult

LOGGER = logging.getLogger("bli")

T1_METRICS = {"P@1": "top1", "P@5": "top5"}
DEFAULT_CSLS_K = 10
DEFAULT_CHUNK = 64


@dataclass(frozen=True)
class BLIPool:
    """Queries, shared candidate pool and gold indices for one direction."""

    setting: str
    source_lang: str
    target_lang: str
    query_words: list[str]
    candidate_words: list[str]
    gold_indices: list[list[int]]
    n_excluded: int
    n_total: int

    @property
    def n_queries(self) -> int:
        """Number of queries on the common set."""
        return len(self.query_words)

    @property
    def n_candidates(self) -> int:
        """Size of the shared candidate pool."""
        return len(self.candidate_words)


def read_dictionary(path: Path | str) -> dict[str, set[str]]:
    """Read a MUSE dictionary into ``source word -> {target words}``.

    The MUSE files are ``source<TAB>target`` per line.  A source word may appear
    on several lines with different translations; all of them are gold.
    """
    entries: dict[str, set[str]] = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.rstrip("\n")
            if not stripped:
                continue
            parts = stripped.split("\t")
            if len(parts) != 2:
                raise ValueError(f"{path}:{line_number}: expected two tab-separated fields, got {len(parts)}")
            source, target = (part.strip() for part in parts)
            if not source or not target:
                continue
            entries.setdefault(source, set()).add(target)
    if not entries:
        raise ValueError(f"{path} contains no translation pairs")
    return entries


def parse_setting(setting: str) -> tuple[str, str]:
    """Split ``"hi-en"`` into ``("hi", "en")``."""
    parts = setting.split("-")
    if len(parts) != 2 or any(part not in {"hi", "en"} for part in parts):
        raise ValueError(f"invalid setting {setting!r}; expected 'hi-en' or 'en-hi'")
    return parts[0], parts[1]


def build_pool(
    dictionary: Mapping[str, Sequence[str]],
    *,
    setting: str,
    source_candidates: Sequence[str],
    target_candidates: Sequence[str],
) -> BLIPool:
    """Build the common query set against the shared candidate pool.

    Args:
        dictionary: ``source word -> gold target words``.
        setting: ``source_lang-target_lang``.
        source_candidates: shared source candidate list (words the systems embed).
        target_candidates: shared target candidate list (retrieval pool).

    Returns:
        A :class:`BLIPool`.  ``n_excluded`` counts dictionary source words that
        are OOV for the shared source candidates or have no gold left in the
        target pool.
    """
    source_lang, target_lang = parse_setting(setting)
    source_available = set(source_candidates)
    target_index = {word: position for position, word in enumerate(target_candidates)}

    query_words: list[str] = []
    gold_indices: list[list[int]] = []
    excluded = 0
    for source_word in sorted(dictionary):
        if source_word not in source_available:
            excluded += 1
            continue
        gold = sorted({target_index[word] for word in dictionary[source_word] if word in target_index})
        if not gold:
            excluded += 1
            continue
        query_words.append(source_word)
        gold_indices.append(gold)

    LOGGER.info(
        "%s: %d dictionary entries, %d queries on the common set, %d excluded (%d source candidates, %d target candidates)",
        setting,
        len(dictionary),
        len(query_words),
        excluded,
        len(source_candidates),
        len(target_candidates),
    )
    if not query_words:
        raise RuntimeError(f"{setting}: no query survived the common-set filter")
    return BLIPool(
        setting=setting,
        source_lang=source_lang,
        target_lang=target_lang,
        query_words=query_words,
        candidate_words=list(target_candidates),
        gold_indices=gold_indices,
        n_excluded=excluded,
        n_total=len(dictionary),
    )


def _check_matrices(queries: np.ndarray, candidates: np.ndarray, gold_indices: Sequence[Sequence[int]]) -> None:
    """Validate shapes before scoring."""
    if queries.ndim != 2 or candidates.ndim != 2:
        raise ValueError("queries and candidates must be 2-D")
    if queries.shape[1] != candidates.shape[1]:
        raise ValueError("queries and candidates must have the same dimension")
    if queries.shape[0] != len(gold_indices):
        raise ValueError(f"{queries.shape[0]} query vectors but {len(gold_indices)} gold entries")


def rank_cosine(
    queries: np.ndarray,
    candidates: np.ndarray,
    gold_indices: Sequence[Sequence[int]],
    *,
    chunk: int = DEFAULT_CHUNK,
) -> RetrievalResult:
    """Rank candidates for every query by cosine and return exact gold ranks."""
    query_matrix = np.asarray(queries, dtype=np.float32)
    candidate_matrix = np.asarray(candidates, dtype=np.float32)
    _check_matrices(query_matrix, candidate_matrix, gold_indices)

    n_queries = query_matrix.shape[0]
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
    return RetrievalResult(ranks, best_index, best_score, gold_score, candidate_matrix.shape[0])


def rank_csls(
    queries: np.ndarray,
    candidates: np.ndarray,
    gold_indices: Sequence[Sequence[int]],
    source: np.ndarray,
    *,
    csls_k: int = DEFAULT_CSLS_K,
    chunk: int = DEFAULT_CHUNK,
    hubness: np.ndarray | None = None,
) -> RetrievalResult:
    """Rank candidates for every query by CSLS and return exact gold ranks.

    ``source`` is the mapped source-candidate matrix used for ``r_S(y)``; pass a
    precomputed ``hubness`` vector to reuse it across query sets (S24 does this,
    because the five cross-validation folds share the same candidate pool).
    """
    query_matrix = np.asarray(queries, dtype=np.float32)
    candidate_matrix = np.asarray(candidates, dtype=np.float32)
    _check_matrices(query_matrix, candidate_matrix, gold_indices)

    if hubness is None:
        r_source = hubness_mean(source, candidate_matrix, csls_k)
    else:
        r_source = np.asarray(hubness, dtype=np.float32)
        if r_source.shape != (candidate_matrix.shape[0],):
            raise ValueError(f"hubness has shape {r_source.shape}, expected ({candidate_matrix.shape[0]},)")

    neighbour_k = min(csls_k, candidate_matrix.shape[0])
    n_queries = query_matrix.shape[0]
    ranks = np.empty(n_queries, dtype=np.int64)
    best_index = np.empty(n_queries, dtype=np.int64)
    best_score = np.empty(n_queries, dtype=np.float32)
    gold_score = np.empty(n_queries, dtype=np.float32)

    for start in range(0, n_queries, chunk):
        stop = min(start + chunk, n_queries)
        cosine = query_matrix[start:stop] @ candidate_matrix.T
        # r_T(x) is the mean of the k largest raw cosines; it is constant per row
        # and therefore never changes the ranking, but it is added back so the
        # returned scores are true CSLS scores.
        top_cosines = np.partition(-cosine, neighbour_k - 1, axis=1)[:, :neighbour_k]
        r_query = (-top_cosines).mean(axis=1)
        adjusted = 2.0 * cosine - r_source[None, :] - r_query[:, None]
        for offset, gold in enumerate(gold_indices[start:stop]):
            row = adjusted[offset]
            ranks[start + offset] = rank_of_gold(row, gold)
            best_index[start + offset] = int(np.argmax(row))
            best_score[start + offset] = float(row.max())
            gold_score[start + offset] = float(row[list(gold)].max())
    return RetrievalResult(ranks, best_index, best_score, gold_score, candidate_matrix.shape[0])


def per_item_frame(pool: BLIPool, result: RetrievalResult) -> pd.DataFrame:
    """Return the per-item table for one T1 run (schema in ``src/results_log.py``)."""
    if len(result) != pool.n_queries:
        raise ValueError(f"result has {len(result)} rows but the pool has {pool.n_queries} queries")
    return pd.DataFrame(
        {
            "query_id": pool.query_words,
            "gold_ids": [[pool.candidate_words[index] for index in gold] for gold in pool.gold_indices],
            "rank_of_best_gold": result.ranks,
            "top1": result.ranks <= 1,
            "top5": result.ranks <= 5,
            "rr": 1.0 / result.ranks,
            "n_candidates": np.full(pool.n_queries, pool.n_candidates, dtype=np.int64),
        }
    )


def coverage_per_item_frame(
    dictionary: Mapping[str, Sequence[str]],
    target_candidates: Sequence[str],
    pool: "BLIPool",
    result: RetrievalResult,
) -> pd.DataFrame:
    """Return per-item rows over the **full** test dictionary, not the common set.

    The common-set metrics compare systems on the queries every system can
    embed.  This secondary view keeps every dictionary entry and marks an
    excluded query with rank ``n_candidates + 1``, i.e. "scored as wrong"
    (PLAN S22).  Items that could not be scored therefore have an empty
    ``gold_ids`` list and a rank beyond the pool, so a paired bootstrap over the
    full set remains well defined.
    """
    target_index = {word: position for position, word in enumerate(target_candidates)}
    position_in_pool = {word: index for index, word in enumerate(pool.query_words)}
    sentinel = pool.n_candidates + 1

    query_ids: list[str] = []
    gold_ids: list[list[str]] = []
    ranks: list[int] = []
    for source_word in sorted(dictionary):
        query_ids.append(source_word)
        gold_ids.append([word for word in sorted(dictionary[source_word]) if word in target_index])
        index = position_in_pool.get(source_word)
        ranks.append(int(result.ranks[index]) if index is not None else sentinel)

    rank_array = np.asarray(ranks, dtype=np.int64)
    return pd.DataFrame(
        {
            "query_id": query_ids,
            "gold_ids": gold_ids,
            "rank_of_best_gold": rank_array,
            "top1": rank_array <= 1,
            "top5": rank_array <= 5,
            "rr": 1.0 / rank_array,
            "n_candidates": np.full(len(query_ids), pool.n_candidates, dtype=np.int64),
        }
    )


def hubness_cache_path(
    cache_dir: Path | str,
    system: str,
    setting: str,
    n_candidates: int,
    csls_k: int,
) -> Path:
    """Return the cache path of the ``r_S`` vector for one system and pool."""
    key = f"{system}__{setting}__n{n_candidates}__k{csls_k}"
    return Path(cache_dir) / f"{key}.npy"


def load_hubness(path: Path | str) -> np.ndarray | None:
    """Load a cached ``r_S`` vector, or ``None`` when it does not exist."""
    candidate = Path(path)
    if not candidate.exists():
        return None
    return np.load(candidate)


def save_hubness(path: Path | str, values: np.ndarray) -> None:
    """Persist an ``r_S`` vector."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.save(target, np.asarray(values, dtype=np.float32))


def pool_digest(pool: BLIPool) -> str:
    """Return a short digest of the candidate pool (cache-safety check)."""
    digest = hashlib.sha256()
    for word in pool.candidate_words:
        digest.update(word.encode("utf-8"))
        digest.update(b"\x1e")
    return digest.hexdigest()[:16]


def fold_assignment(words: Sequence[str], *, folds: int = 5, seed: int = 42) -> dict[str, int]:
    """Assign each word to one cross-validation fold, deterministically.

    ``PLAN`` S24 uses five folds over the 5,000 MUSE **train** source words: four
    folds fit the Procrustes mapping, the held-out fold is scored.  The seed is
    the global one from ``configs/tasks.yaml``.
    """
    if folds < 2:
        raise ValueError("folds must be >= 2")
    ordered = sorted(words)
    if len(ordered) < folds:
        raise ValueError(f"cannot split {len(ordered)} words into {folds} folds")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(ordered))
    return {ordered[int(index)]: position % folds for position, index in enumerate(order)}
