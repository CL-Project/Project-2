"""Run a (system, task) combination from ``configs/`` and log the results.

``--task t2_flores`` covers the FLORES pool (S15 to S17, plus the static SIF
systems from S25); ``--task t1_bli`` covers MUSE word-level induction (S22 to
S24); ``--task t3_xquad`` covers Hindi question to paragraph retrieval (S29 to
S31).  Every task writes ``results/results.csv`` and
``results/per_item/<task>__<system>__<setting>.csv``.

Examples::

    uv run python scripts/run_eval.py --task t2_flores --system labse
    uv run python scripts/run_eval.py --task t1_bli --system ft --settings hi-en
    uv run python scripts/run_eval.py --task t3_xquad --system labse
    uv run python scripts/run_eval.py --task t3_xquad --system bm25 --settings en-en
    uv run python scripts/run_eval.py --task t1_bli --system labse --cvfreq

The script is library-thin on purpose: all scoring lives in ``src/tasks/`` and
all metric aggregation in ``src/results_log.py``.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any, Callable

import yaml
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src import results_log
from src.embeddings import align, encoders, sentence_vectors, static
from src.seeding import set_global_seed
from src.tasks import bli, miracl, parallel, xquad

DATA_CONFIG_PATH = REPO_ROOT / "configs" / "data.yaml"
TASKS_CONFIG_PATH = REPO_ROOT / "configs" / "tasks.yaml"

LOGGER = logging.getLogger("run_eval")

ENCODER_SYSTEMS = ("labse", "me5", "muril")
STATIC_SYSTEMS = ("w2v", "ft", "ft_ref")
KNOWN_T2_SYSTEMS = ENCODER_SYSTEMS + STATIC_SYSTEMS + ("bm25_mt",)
KNOWN_T1_SYSTEMS = STATIC_SYSTEMS + ENCODER_SYSTEMS
KNOWN_T3_SYSTEMS = ENCODER_SYSTEMS + STATIC_SYSTEMS + ("bm25", "bm25_mt")

MUSE_DIR = REPO_ROOT / "data" / "raw" / "muse"
MIRACL_POOL_CACHE = REPO_ROOT / "data" / "vectors" / "miracl_pool.jsonl"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
CSLS_CACHE = REPO_ROOT / "data" / "vectors" / "csls"
PC_CACHE = REPO_ROOT / "data" / "vectors" / "sif_pc"
FASTTEXT_RAW = REPO_ROOT / "data" / "raw" / "fasttext"
PRETRAINED_FILES = {"hi": "wiki.hi.align.vec", "en": "wiki.en.align.vec"}
SIF_FIT_SENTENCES = 100_000
SIF_FIT_PARAGRAPHS = 20_000


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML config file."""
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def flores_pairs_for_task(data: dict[str, Any]) -> list[parallel.SentencePair]:
    """Load the pooled FLORES pairs using ``configs/data.yaml``."""
    flores = data["flores"]
    return parallel.load_flores_pairs(
        REPO_ROOT / flores["dir"],
        splits=tuple(flores["splits"]),
        languages=dict(flores["languages"]),
    )


def _encode_side(system: str, texts: list[str], lang: str, *, use_cache: bool) -> tuple[Any, Any]:
    """Encode one side of the pool and return ``(vectors, stats)``."""
    return encoders.encode(
        system,
        texts,
        "sentence",
        dataset="flores",
        lang=lang,
        use_cache=use_cache,
        return_stats=True,
    )


def run_t2_encoder(
    system: str,
    pool: parallel.ParallelPool,
    *,
    use_cache: bool = True,
) -> tuple[Any, dict[str, Any]]:
    """Encode and score one T2 direction with a pretrained encoder.

    Returns the per-item frame and the run notes (truncation counts); the
    caller writes both the per-item file and the metric rows.
    """
    query_vectors, query_stats = _encode_side(
        system, pool.query_texts, pool.query_lang, use_cache=use_cache
    )
    candidate_vectors, candidate_stats = _encode_side(
        system, pool.candidate_texts, pool.candidate_lang, use_cache=use_cache
    )
    if query_vectors.shape[0] != pool.n_queries:
        raise RuntimeError("query vector count does not match the pool")
    if candidate_vectors.shape[0] != pool.n_candidates:
        raise RuntimeError("candidate vector count does not match the pool")

    result = parallel.retrieve_by_cosine(query_vectors, candidate_vectors, pool.gold_indices)
    frame = parallel.per_item_frame(pool, result)
    notes = {
        "query_truncated": query_stats.n_truncated,
        "candidate_truncated": candidate_stats.n_truncated,
        "max_len": query_stats.max_len,
    }
    return frame, notes


def run_t2_bm25_mt(
    setting: str,
    pool: parallel.ParallelPool,
    *,
    force: bool,
) -> tuple[Any, dict[str, Any]]:
    """Score one T2 direction with BM25 over machine-translated queries (S17).

    Only ``hi-en`` is meaningful: Hindi queries are translated to English and
    matched against the English candidate sentences.  Other settings raise so a
    meaningless run cannot be logged by accident.
    """
    if setting != "hi-en":
        raise SystemExit("bm25_mt on T2 is only defined for the hi-en direction")

    from src.mt import translate_cached
    from src.rankers import bm25_rank

    processed = REPO_ROOT / "data" / "processed"
    cache_path = processed / "flores_hi2en.tsv"
    translations = translate_cached(pool.query_texts, cache_path, force=force)
    if len(translations) != pool.n_queries:
        raise RuntimeError(f"{len(translations)} translations for {pool.n_queries} queries")

    result = bm25_rank(translations, pool.candidate_texts, pool.gold_indices, lang="en")
    frame = parallel.per_item_frame(pool, result)
    return frame, {"translation_cache": str(cache_path.relative_to(REPO_ROOT))}


def run_t2(
    args: argparse.Namespace,
    data: dict[str, Any],
    tasks: dict[str, Any],
) -> list[dict[str, Any]]:
    """Run every requested T2 direction for one system and log the results."""
    pairs = flores_pairs_for_task(data)
    settings = args.settings or list(tasks["t2_parallel"]["directions"])
    rows: list[dict[str, Any]] = []
    n_boot = int(tasks.get("bootstrap_samples", 1000))
    seed = int(tasks.get("seed", 42))

    for setting in settings:
        pool = parallel.build_pool(pairs, setting)
        tag = f"{setting}_avg" if args.ablation else setting
        LOGGER.info("T2 %s %s: %d queries, %d candidates", args.system, tag, pool.n_queries, pool.n_candidates)

        with results_log.timed() as timing:
            if args.system in ENCODER_SYSTEMS:
                frame, notes = run_t2_encoder(
                    args.system, pool, use_cache=not args.force
                )
            elif args.system == "bm25_mt":
                frame, notes = run_t2_bm25_mt(setting, pool, force=args.force)
            elif args.system in STATIC_SYSTEMS:
                mode = "average" if args.ablation else "sif"
                frame, notes = run_t2_static(args, setting, pool, mode=mode)
            else:
                raise SystemExit(f"unknown T2 system {args.system!r}; known: {KNOWN_T2_SYSTEMS}")

        per_item_path = results_log.write_per_item(
            "t2_flores", args.system, tag, frame, results_dir=args.results_dir
        )
        LOGGER.info("wrote %s (%d rows)", per_item_path, len(frame))

        metric_rows = results_log.summarize_per_item(
            frame,
            args.system,
            "t2_flores",
            tag,
            parallel.T2_METRICS,
            n_boot=n_boot,
            seed=seed,
        )
        note = " ".join(f"{key}={value}" for key, value in sorted(notes.items()))
        for row in metric_rows:
            row.update(
                {
                    "runtime_s": round(timing.runtime_s, 2),
                    "peak_mem_mb": round(timing.peak_mem_mb, 1),
                    "notes": note,
                }
            )
            results_log.append_result(row, results_dir=args.results_dir)
        rows.extend(metric_rows)
        LOGGER.info(
            "T2 %s %s: Acc@1=%.4f MRR=%.4f Recall@5=%.4f (%.1fs)",
            args.system,
            tag,
            metric_rows[0]["value"],
            metric_rows[1]["value"],
            metric_rows[2]["value"],
            timing.runtime_s,
        )
    return rows


def run_t1_cvfreq(args: argparse.Namespace, tasks: dict[str, Any]) -> list[dict[str, Any]]:
    """Run S24: five-fold cross-validation over the 5,000 MUSE train words.

    The test split only contains words ranked 5,000 to 6,500, so H2's
    "frequent words" clause needs a separate, genuinely frequent band.  For each
    fold the mapping is fitted on the other four folds and the held-out fold is
    scored with the S22 protocol; the five held-out sets are pooled into one
    ``hi-en_cvfreq`` per-item file.
    """
    import numpy as np
    from src.retrieval import hubness_mean

    setting = "hi-en"
    source_lang, target_lang = bli.parse_setting(setting)
    dictionary = bli.read_dictionary(MUSE_DIR / f"{setting}.0-5000.txt")
    source_candidates = load_candidates(source_lang)
    pool = bli.build_pool(
        dictionary,
        setting=setting,
        source_candidates=source_candidates,
        target_candidates=load_candidates(target_lang),
    )
    folds = int(tasks["t1_bli"].get("cv_folds", 5))
    seed = int(tasks.get("seed", 42))
    n_boot = int(tasks.get("bootstrap_samples", 1000))
    csls_k = int(static.load_models_config()["alignment"]["csls_k"])
    fold_of = bli.fold_assignment(sorted(dictionary), folds=folds, seed=seed)

    n_queries = pool.n_queries
    ranks_csls = np.empty(n_queries, dtype=np.int64)
    ranks_cosine = np.empty(n_queries, dtype=np.int64)
    best_index = np.empty(n_queries, dtype=np.int64)
    best_score = np.empty(n_queries, dtype=np.float32)
    gold_score = np.empty(n_queries, dtype=np.float32)

    fold_indices = {fold: [i for i, word in enumerate(pool.query_words) if fold_of.get(word) == fold] for fold in range(folds)}
    empty_folds = [fold for fold, members in fold_indices.items() if not members]
    if empty_folds:
        raise RuntimeError(f"folds {empty_folds} have no queries; the fold split is wrong")

    notes: dict[str, Any] = {"cv_folds": folds}

    with results_log.timed() as timing:
        if args.system in STATIC_SYSTEMS and args.system != "ft_ref":
            source_vectors = static_vectors(args.system, source_lang)
            target_vectors = static_vectors(args.system, target_lang)
            candidates = align.l2_normalize(static.vector_matrix(target_vectors, pool.candidate_words))
            for fold in range(folds):
                train_pairs = [
                    (source, target)
                    for source in sorted(dictionary)
                    if fold_of.get(source) != fold
                    for target in dictionary[source]
                ]
                mapping, n_used = align.fit_mapping(source_vectors, target_vectors, train_pairs)
                source_matrix = align.apply_mapping(static.vector_matrix(source_vectors, source_candidates), mapping)
                members = fold_indices[fold]
                queries = align.apply_mapping(
                    static.vector_matrix(source_vectors, [pool.query_words[index] for index in members]), mapping
                )
                golds = [pool.gold_indices[index] for index in members]
                hubness_path = bli.hubness_cache_path(
                    CSLS_CACHE, f"{args.system}_cvfreq_f{fold}", setting, pool.n_candidates, csls_k
                )
                hubness = None if args.force else bli.load_hubness(hubness_path)
                if hubness is None:
                    LOGGER.info("fold %d: computing r_S for %s", fold, args.system)
                    hubness = hubness_mean(source_matrix, candidates, csls_k)
                    bli.save_hubness(hubness_path, hubness)
                csls_result = bli.rank_csls(queries, candidates, golds, source=source_matrix, csls_k=csls_k, hubness=hubness)
                cosine_result = bli.rank_cosine(queries, candidates, golds)
                ranks_csls[members] = csls_result.ranks
                ranks_cosine[members] = cosine_result.ranks
                best_index[members] = csls_result.best_index
                best_score[members] = csls_result.best_score
                gold_score[members] = csls_result.gold_score
                LOGGER.info("fold %d: %d queries, %d seed pairs, P@1=%.4f", fold, len(members), n_used, float(np.mean(csls_result.ranks <= 1)))
        else:
            if args.system == "ft_ref":
                source_vectors = static_vectors("ft_ref", source_lang)
                target_vectors = static_vectors("ft_ref", target_lang)
                queries = align.l2_normalize(static.vector_matrix(source_vectors, pool.query_words))
                source_matrix = align.l2_normalize(static.vector_matrix(source_vectors, source_candidates))
                candidates = align.l2_normalize(static.vector_matrix(target_vectors, pool.candidate_words))
                notes["mapping"] = "none (pretrained aligned)"
            elif args.system in ENCODER_SYSTEMS:
                source_matrix = encoders.encode(
                    args.system,
                    source_candidates,
                    "word",
                    dataset="candidates",
                    lang=source_lang,
                    use_cache=not args.force,
                )
                index = {word: position for position, word in enumerate(source_candidates)}
                queries = source_matrix[[index[word] for word in pool.query_words]]
                candidates = encoders.encode(
                    args.system, pool.candidate_words, "word", dataset="candidates", lang=target_lang, use_cache=not args.force
                )
                notes["mapping"] = "none (encoder)"
            else:
                raise SystemExit(f"unknown T1 system {args.system!r}; known: {KNOWN_T1_SYSTEMS}")
            hubness_path = bli.hubness_cache_path(CSLS_CACHE, f"{args.system}_cvfreq", setting, pool.n_candidates, csls_k)
            hubness = None if args.force else bli.load_hubness(hubness_path)
            if hubness is None:
                hubness = hubness_mean(source_matrix, candidates, csls_k)
                bli.save_hubness(hubness_path, hubness)
            csls_result = bli.rank_csls(queries, candidates, pool.gold_indices, source=source_matrix, csls_k=csls_k, hubness=hubness)
            cosine_result = bli.rank_cosine(queries, candidates, pool.gold_indices)
            ranks_csls[:] = csls_result.ranks
            ranks_cosine[:] = cosine_result.ranks
            best_index[:] = csls_result.best_index
            best_score[:] = csls_result.best_score
            gold_score[:] = csls_result.gold_score

    tag = f"{setting}_cvfreq"
    rows: list[dict[str, Any]] = []
    for suffix, ranks in (("", ranks_csls), ("_cos", ranks_cosine)):
        result = parallel.RetrievalResult(ranks, best_index, best_score, gold_score, pool.n_candidates)
        frame = bli.per_item_frame(pool, result)
        path = results_log.write_per_item("t1_bli", args.system, f"{tag}{suffix}", frame, results_dir=args.results_dir)
        LOGGER.info("wrote %s (%d rows)", path, len(frame))
        metric_rows = results_log.summarize_per_item(
            frame,
            args.system,
            "t1_bli",
            f"{tag}{suffix}",
            bli.T1_METRICS,
            n_boot=n_boot,
            seed=seed,
            n_excluded=pool.n_excluded,
        )
        note = " ".join(f"{key}={value}" for key, value in sorted(notes.items()))
        note = f"{note} csls_k={csls_k} retrieval={'csls' if not suffix else 'cosine'}"
        for row in metric_rows:
            row.update({"runtime_s": round(timing.runtime_s, 2), "peak_mem_mb": round(timing.peak_mem_mb, 1), "notes": note})
            results_log.append_result(row, results_dir=args.results_dir)
        rows.extend(metric_rows)
        LOGGER.info(
            "T1 cvfreq %s%s: P@1=%.4f P@5=%.4f",
            args.system,
            suffix,
            metric_rows[0]["value"],
            metric_rows[1]["value"],
        )
    return rows


def load_candidates(lang: str) -> list[str]:
    """Load the shared candidate vocabulary built in S20."""
    path = PROCESSED_DIR / f"candidates_{lang}.txt"
    if not path.exists():
        raise SystemExit(f"{path} does not exist; run scripts/build_candidates.py first")
    words = [line.rstrip("\n") for line in path.open("r", encoding="utf-8") if line.strip()]
    if not words:
        raise SystemExit(f"{path} is empty")
    return words


def static_vectors(system: str, lang: str, *, with_subwords: bool = False):
    """Load the word vectors of a static system (``ft_ref`` uses the pretrained file)."""
    if system == "ft_ref":
        limit = int(static.load_models_config()["static"]["vocab_cap"])
        return static.load_pretrained(FASTTEXT_RAW / PRETRAINED_FILES[lang], limit=limit)
    return static.load(system, lang, with_subwords=with_subwords)


def direction_mapping(system: str, setting: str, source_lang: str, target_lang: str):
    """Fit the direction's Procrustes mapping from the MUSE train pairs."""
    source_vectors = static_vectors(system, source_lang)
    target_vectors = static_vectors(system, target_lang)
    pairs = align.read_pairs(MUSE_DIR / f"{setting}.0-5000.txt")
    mapping, n_used = align.fit_mapping(source_vectors, target_vectors, pairs)
    if not align.is_orthogonal(mapping):
        raise RuntimeError("Procrustes returned a non-orthogonal mapping")
    LOGGER.info("%s %s: mapping fitted on %d seed pairs", system, setting, n_used)
    return mapping, n_used


def t1_vectors(args: argparse.Namespace, setting: str, pool: bli.BLIPool):
    """Return ``(query, candidate, source, notes)`` matrices for one T1 run."""
    source_lang, target_lang = bli.parse_setting(setting)
    source_candidates = load_candidates(source_lang)

    if args.system in STATIC_SYSTEMS:
        source_vectors = static_vectors(args.system, source_lang)
        target_vectors = static_vectors(args.system, target_lang)
        if args.system == "ft_ref":
            query = align.l2_normalize(static.vector_matrix(source_vectors, pool.query_words))
            source = align.l2_normalize(static.vector_matrix(source_vectors, source_candidates))
            notes = {"mapping": "none (pretrained aligned)"}
        else:
            mapping, n_used = direction_mapping(args.system, setting, source_lang, target_lang)
            query = align.apply_mapping(static.vector_matrix(source_vectors, pool.query_words), mapping)
            source = align.apply_mapping(static.vector_matrix(source_vectors, source_candidates), mapping)
            notes = {"mapping_pairs": n_used}
        candidates = align.l2_normalize(static.vector_matrix(target_vectors, pool.candidate_words))
        return query, candidates, source, notes

    if args.system in ENCODER_SYSTEMS:
        source_matrix, source_stats = encoders.encode(
            args.system,
            source_candidates,
            "word",
            dataset="candidates",
            lang=source_lang,
            use_cache=not args.force,
            return_stats=True,
        )
        candidate_matrix, candidate_stats = encoders.encode(
            args.system,
            pool.candidate_words,
            "word",
            dataset="candidates",
            lang=target_lang,
            use_cache=not args.force,
            return_stats=True,
        )
        index = {word: position for position, word in enumerate(source_candidates)}
        query = source_matrix[[index[word] for word in pool.query_words]]
        notes = {
            "source_truncated": source_stats.n_truncated,
            "candidate_truncated": candidate_stats.n_truncated,
            "max_len": source_stats.max_len,
        }
        return query, candidate_matrix, source_matrix, notes

    raise SystemExit(f"unknown T1 system {args.system!r}; known: {KNOWN_T1_SYSTEMS}")


def run_t1_coverage_only(args: argparse.Namespace, tasks: dict[str, Any]) -> list[dict[str, Any]]:
    """Backfill the coverage-aware view (S22 secondary metric) from a logged CSLS run.

    Reads ``results/per_item/t1_bli__<system>__<setting>.csv`` and expands it to
    the **full** test dictionary, scoring the queries outside the common set as
    wrong.  No retrieval is repeated, so this cannot change a primary number.
    """
    directions = args.settings or list(tasks["t1_bli"]["directions"])
    n_boot = int(tasks.get("bootstrap_samples", 1000))
    seed = int(tasks.get("seed", 42))
    csls_k = int(static.load_models_config()["alignment"]["csls_k"])
    rows: list[dict[str, Any]] = []

    for setting in directions:
        source_lang, target_lang = bli.parse_setting(setting)
        dictionary = bli.read_dictionary(MUSE_DIR / f"{setting}.5000-6500.txt")
        target_candidates = load_candidates(target_lang)
        pool = bli.build_pool(
            dictionary,
            setting=setting,
            source_candidates=load_candidates(source_lang),
            target_candidates=target_candidates,
        )
        source_frame = pd.read_csv(
            Path(args.results_dir) / "per_item" / f"t1_bli__{args.system}__{setting}.csv",
            keep_default_na=False,
        )
        if len(source_frame) != pool.n_queries:
            raise SystemExit(
                f"{setting}: the logged file has {len(source_frame)} rows but the pool has "
                f"{pool.n_queries}; re-run the primary evaluation first"
            )
        ranks = pd.to_numeric(source_frame["rank_of_best_gold"], errors="raise").to_numpy(dtype=np.int64)
        result = parallel.RetrievalResult(
            ranks=ranks,
            best_index=np.zeros_like(ranks),
            best_score=np.zeros_like(ranks, dtype=np.float32),
            gold_score=np.zeros_like(ranks, dtype=np.float32),
            n_candidates=pool.n_candidates,
        )

        with results_log.timed() as timing:
            frame = bli.coverage_per_item_frame(dictionary, target_candidates, pool, result)

        tag = f"{setting}_cov"
        path = results_log.write_per_item("t1_bli", args.system, tag, frame, results_dir=args.results_dir)
        LOGGER.info("wrote %s (%d rows)", path, len(frame))
        metric_rows = results_log.summarize_per_item(
            frame, args.system, "t1_bli", tag, bli.T1_METRICS, n_boot=n_boot, seed=seed, n_excluded=0
        )
        note = (
            f"csls_k={csls_k} retrieval=csls coverage=full_test_dictionary oov_counted_wrong=1 "
            f"backfilled_from={setting}"
        )
        for row in metric_rows:
            row.update(
                {"runtime_s": round(timing.runtime_s, 2), "peak_mem_mb": round(timing.peak_mem_mb, 1), "notes": note}
            )
            results_log.append_result(row, results_dir=args.results_dir)
        rows.extend(metric_rows)
        LOGGER.info(
            "T1 coverage %s %s: P@1=%.4f P@5=%.4f", args.system, tag, metric_rows[0]["value"], metric_rows[1]["value"]
        )
    return rows


def run_t1(
    args: argparse.Namespace,
    data: dict[str, Any],
    tasks: dict[str, Any],
) -> list[dict[str, Any]]:
    """Run T1 for one system over the requested directions and log the results."""
    from src.retrieval import hubness_mean

    if getattr(args, "cvfreq", False):
        return run_t1_cvfreq(args, tasks)
    if getattr(args, "coverage_only", False):
        return run_t1_coverage_only(args, tasks)

    directions = args.settings or list(tasks["t1_bli"]["directions"])
    n_boot = int(tasks.get("bootstrap_samples", 1000))
    seed = int(tasks.get("seed", 42))
    csls_k = int(static.load_models_config()["alignment"]["csls_k"])
    rows: list[dict[str, Any]] = []

    for setting in directions:
        source_lang, target_lang = bli.parse_setting(setting)
        dictionary = bli.read_dictionary(MUSE_DIR / f"{setting}.5000-6500.txt")
        pool = bli.build_pool(
            dictionary,
            setting=setting,
            source_candidates=load_candidates(source_lang),
            target_candidates=load_candidates(target_lang),
        )

        with results_log.timed() as timing:
            query, candidates, source, notes = t1_vectors(args, setting, pool)
            hubness_path = bli.hubness_cache_path(CSLS_CACHE, args.system, setting, pool.n_candidates, csls_k)
            hubness = None if args.force else bli.load_hubness(hubness_path)
            if hubness is None:
                LOGGER.info("computing r_S for %s %s (%d x %d)", args.system, setting, source.shape[0], candidates.shape[0])
                hubness = hubness_mean(source, candidates, csls_k)
                bli.save_hubness(hubness_path, hubness)
            csls_result = bli.rank_csls(
                query, candidates, pool.gold_indices, source=source, csls_k=csls_k, hubness=hubness
            )
            cosine_result = bli.rank_cosine(query, candidates, pool.gold_indices)

        coverage_frame = bli.coverage_per_item_frame(
            dictionary, pool.candidate_words, pool, csls_result
        )
        runs = (
            (setting, bli.per_item_frame(pool, csls_result), pool.n_excluded),
            (f"{setting}_cos", bli.per_item_frame(pool, cosine_result), pool.n_excluded),
            # Secondary view (PLAN S22): the full test dictionary, with the
            # queries outside the common set scored as wrong.
            (f"{setting}_cov", coverage_frame, 0),
        )
        for tag, frame, n_excluded in runs:
            path = results_log.write_per_item("t1_bli", args.system, tag, frame, results_dir=args.results_dir)
            LOGGER.info("wrote %s (%d rows)", path, len(frame))
            metric_rows = results_log.summarize_per_item(
                frame,
                args.system,
                "t1_bli",
                tag,
                bli.T1_METRICS,
                n_boot=n_boot,
                seed=seed,
                n_excluded=n_excluded,
            )
            note = " ".join(f"{key}={value}" for key, value in sorted(notes.items()))
            retrieval = "cosine" if tag.endswith("_cos") else "csls"
            note = f"{note} csls_k={csls_k} retrieval={retrieval}"
            if tag.endswith("_cov"):
                note = f"{note} coverage=full_test_dictionary oov_counted_wrong=1"
            for row in metric_rows:
                row.update(
                    {
                        "runtime_s": round(timing.runtime_s, 2),
                        "peak_mem_mb": round(timing.peak_mem_mb, 1),
                        "notes": note,
                    }
                )
                results_log.append_result(row, results_dir=args.results_dir)
            rows.extend(metric_rows)
            LOGGER.info(
                "T1 %s %s (%s): P@1=%.4f P@5=%.4f (%.1fs)",
                args.system,
                tag,
                note,
                metric_rows[0]["value"],
                metric_rows[1]["value"],
                timing.runtime_s,
            )
    return rows


def _vector_table(vectors, mapping, words: list[str]) -> dict[str, Any]:
    """Return a ``word -> vector`` table for ``words`` present in ``vectors``.

    The vectors are mapped in bulk rather than one at a time, which matters when
    the table holds hundreds of thousands of words.
    """
    import numpy as np

    present = [word for word in words if word in vectors]
    if not present:
        return {}
    matrix = np.vstack([np.asarray(vectors[word], dtype=np.float32) for word in present])
    if mapping is not None:
        matrix = align.apply_mapping(matrix, mapping)
    return {word: matrix[position] for position, word in enumerate(present)}


def _cached_fit_sentences(lang: str, *, limit: int = SIF_FIT_SENTENCES, seed: int = 42) -> list[str]:
    """Return the SIF fitting sentences, sampling and caching them on first use."""
    cache = PC_CACHE / f"fit_sentences_{lang}.txt"
    if cache.exists():
        return [line.rstrip("\n") for line in cache.open("r", encoding="utf-8") if line.strip()]
    sentences = sentence_vectors.sample_fit_sentences(
        PROCESSED_DIR / f"wiki.{lang}.para.tsv", lang, shard="fit", limit=limit, seed=seed
    )
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text("".join(f"{sentence}\n" for sentence in sentences), encoding="utf-8")
    return sentences


def _sif_pc(system: str, lang: str, mapping, *, seed: int = 42):
    """Return the first principal component for ``(system, lang)``, cached to disk.

    The component is fitted on ``fit``-shard Wikipedia sentences composed in the
    same space the evaluation uses (mapped into the target space when the
    language is the source side), never on evaluation data.
    """
    import numpy as np
    from src.textproc import tokenize

    tag = "mapped" if mapping is not None else "plain"
    cache = PC_CACHE / f"{system}__{lang}__{tag}.npy"
    if cache.exists():
        return np.load(cache)

    vectors = static_vectors(system, lang)
    counts, total = sentence_vectors.read_word_counts(PROCESSED_DIR / f"wordcounts_{lang}.tsv")
    sentences = _cached_fit_sentences(lang, seed=seed)
    tokens = [tokenize(sentence, lang) for sentence in sentences]
    needed = sorted({token for row in tokens for token in row})
    table = _vector_table(vectors, mapping, needed)
    LOGGER.info("%s %s: fitting SIF component on %d sentences (%d/%d words in vocab)", system, lang, len(tokens), len(table), len(needed))
    composed, _ = sentence_vectors.compose_sif(
        tokens, table, counts, total, dim=int(vectors.vector_size), pc=None
    )
    component = sentence_vectors.fit_pc(composed)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, component)
    return component


def sif_matrix(system: str, lang: str, texts: list[str], mapping, *, mode: str = "sif"):
    """Compose SIF (or plain-average) vectors for ``texts`` in ``lang``."""
    from src.textproc import tokenize

    vectors = static_vectors(system, lang)
    counts, total = sentence_vectors.read_word_counts(PROCESSED_DIR / f"wordcounts_{lang}.tsv")
    tokens = [tokenize(text, lang) for text in texts]
    needed = sorted({token for row in tokens for token in row})
    table = _vector_table(vectors, mapping, needed)
    if mode == "average":
        return sentence_vectors.compose_average(tokens, table, dim=int(vectors.vector_size))
    component = _sif_pc(system, lang, mapping)
    return sentence_vectors.compose_sif(
        tokens, table, counts, total, dim=int(vectors.vector_size), pc=component
    )


def run_t2_static(args: argparse.Namespace, setting: str, pool: parallel.ParallelPool, *, mode: str = "sif"):
    """Compose sentence vectors for a static system and score one T2 direction."""
    from src.tasks.parallel import retrieve_by_cosine

    source_lang, target_lang = parallel.parse_setting(setting)
    if args.system == "ft_ref":
        source_mapping = target_mapping = None
        notes = {"mapping": "none (pretrained aligned)"}
    else:
        mapping, n_used = direction_mapping(args.system, setting, source_lang, target_lang)
        source_mapping, target_mapping = mapping, None
        notes = {"mapping_pairs": n_used}

    query_vectors, query_stats = sif_matrix(args.system, source_lang, pool.query_texts, source_mapping, mode=mode)
    candidate_vectors, candidate_stats = sif_matrix(
        args.system, target_lang, pool.candidate_texts, target_mapping, mode=mode
    )
    result = retrieve_by_cosine(query_vectors, candidate_vectors, pool.gold_indices)
    frame = parallel.per_item_frame(pool, result)
    notes.update(
        {
            "mode": mode,
            "query_oov_rate": round(query_stats.oov_rate, 4),
            "candidate_oov_rate": round(candidate_stats.oov_rate, 4),
            "query_empty": query_stats.n_empty_items,
            "candidate_empty": candidate_stats.n_empty_items,
        }
    )
    return frame, notes


def _cached_fit_paragraphs(lang: str, *, limit: int = SIF_FIT_PARAGRAPHS, seed: int = 42) -> list[str]:
    """Return the paragraph-level SIF fitting texts, caching them on first use."""
    cache = PC_CACHE / f"fit_paragraphs_{lang}.txt"
    if cache.exists():
        return [line.rstrip("\n") for line in cache.open("r", encoding="utf-8") if line.strip()]
    paragraphs = sentence_vectors.sample_fit_paragraphs(
        PROCESSED_DIR / f"wiki.{lang}.para.tsv", shard="fit", limit=limit, seed=seed
    )
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text("".join(f"{paragraph}\n" for paragraph in paragraphs), encoding="utf-8")
    return paragraphs


def _paragraph_pc(system: str, lang: str, mapping, *, seed: int = 42):
    """Return the **paragraph-level** first principal component, cached to disk.

    T3 composes whole paragraphs, so the component is fitted on 20,000 held-out
    ``fit``-shard paragraphs (PLAN S25, D03), not on sentences.  It is fitted in
    the space the evaluation uses, i.e. after the Procrustes mapping when the
    language is the source side.
    """
    from src.textproc import tokenize

    tag = "mapped" if mapping is not None else "plain"
    cache = PC_CACHE / f"{system}__{lang}__paragraph__{tag}.npy"
    if cache.exists():
        return np.load(cache)

    vectors = static_vectors(system, lang)
    counts, total = sentence_vectors.read_word_counts(PROCESSED_DIR / f"wordcounts_{lang}.tsv")
    paragraphs = _cached_fit_paragraphs(lang, seed=seed)
    tokens = [tokenize(paragraph, lang) for paragraph in paragraphs]
    needed = sorted({token for row in tokens for token in row})
    table = _vector_table(vectors, mapping, needed)
    LOGGER.info(
        "%s %s: fitting paragraph SIF component on %d paragraphs (%d/%d words in vocab)",
        system,
        lang,
        len(tokens),
        len(table),
        len(needed),
    )
    composed, _ = sentence_vectors.compose_sif(tokens, table, counts, total, dim=int(vectors.vector_size), pc=None)
    component = sentence_vectors.fit_pc(composed)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, component)
    return component


def t3_paragraph_matrix(system: str, lang: str, texts: list[str], mapping, *, mode: str = "sif"):
    """Compose SIF (or plain-average) paragraph vectors for ``texts`` in ``lang``."""
    from src.textproc import tokenize

    vectors = static_vectors(system, lang)
    counts, total = sentence_vectors.read_word_counts(PROCESSED_DIR / f"wordcounts_{lang}.tsv")
    tokens = [tokenize(text, lang) for text in texts]
    needed = sorted({token for row in tokens for token in row})
    table = _vector_table(vectors, mapping, needed)
    if mode == "average":
        return sentence_vectors.compose_average(tokens, table, dim=int(vectors.vector_size))
    component = _paragraph_pc(system, lang, mapping)
    return sentence_vectors.compose_sif(tokens, table, counts, total, dim=int(vectors.vector_size), pc=component)


def t3_queries_and_pools(
    args: argparse.Namespace, data: dict[str, Any]
) -> tuple[xquad.XQuadData, dict[str, list[xquad.PoolParagraph]]]:
    """Load the aligned XQuAD tables and the paragraph pools."""
    settings = data["xquad"]
    xquad_data = xquad.load_xquad(
        REPO_ROOT / settings["dir"],
        expected_questions=int(settings["expected_questions"]),
        expected_paragraphs=int(settings["expected_paragraphs"]),
    )
    return xquad_data, xquad.load_pools(PROCESSED_DIR)


def run_t3_encoder(
    args: argparse.Namespace, setting: str, pool: xquad.T3Pool
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Encode questions and the document pool for one T3 setting (S30)."""
    query_kind = "query"
    query_vectors, query_stats = encoders.encode(
        args.system,
        pool.query_texts,
        query_kind,
        dataset="t3_xquad",
        lang=pool.query_lang,
        kind_tag="query",
        use_cache=not args.force,
        return_stats=True,
    )
    candidate_vectors, candidate_stats = encoders.encode(
        args.system,
        pool.candidate_texts,
        "passage",
        dataset="t3_xquad",
        lang=pool.doc_lang,
        kind_tag="passage",
        use_cache=not args.force,
        return_stats=True,
    )
    result = parallel.retrieve_by_cosine(query_vectors, candidate_vectors, pool.gold_indices)
    frame = xquad.per_item_frame(pool, result)
    notes = {
        "query_truncated": query_stats.n_truncated,
        "candidate_truncated": candidate_stats.n_truncated,
        "candidate_truncated_fraction": round(candidate_stats.truncated_fraction, 4),
        "max_len": candidate_stats.max_len,
    }
    return frame, notes


def run_t3_bm25(
    args: argparse.Namespace, setting: str, pool: xquad.T3Pool, *, force: bool
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Score one T3 setting with BM25 (S29).

    ``bm25`` is monolingual and therefore only defined for ``en-en`` and
    ``hi-hi``.  ``bm25_mt`` translates the Hindi questions into English and
    matches them against the English pool, so it is defined only for ``hi-en``.
    The other combinations raise rather than log a meaningless row.
    """
    from src.rankers import bm25_rank

    query_lang, doc_lang = pool.query_lang, pool.doc_lang
    if args.system == "bm25":
        if query_lang != doc_lang:
            raise SystemExit(
                f"bm25 is monolingual; {setting!r} needs cross-lingual retrieval (use bm25_mt for hi-en)"
            )
        queries = pool.query_texts
        notes: dict[str, Any] = {"system_kind": "monolingual_bm25"}
    elif args.system == "bm25_mt":
        if (query_lang, doc_lang) != ("hi", "en"):
            raise SystemExit("bm25_mt on T3 is only defined for the hi-en setting")
        from src.mt import translate_cached

        cache_path = PROCESSED_DIR / "xquad_hi2en.tsv"
        queries = translate_cached(pool.query_texts, cache_path, force=force)
        notes = {"system_kind": "bm25_over_mt", "translation_cache": str(cache_path.relative_to(REPO_ROOT))}
    else:
        raise SystemExit(f"unknown T3 BM25 system {args.system!r}")

    result = bm25_rank(queries, pool.candidate_texts, pool.gold_indices, lang=doc_lang)
    frame = xquad.per_item_frame(pool, result)
    return frame, notes


def run_t3_static(
    args: argparse.Namespace, setting: str, pool: xquad.T3Pool, *, mode: str = "sif"
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Compose paragraph vectors for a static system and score one T3 setting (S31).

    Hindi paragraphs are mapped into the English space with the direction's
    Procrustes mapping (``hi-en``) and the reverse mapping is used for ``en-hi``.
    Same-language settings use no mapping: both sides already live in one space.
    """
    query_lang, doc_lang = pool.query_lang, pool.doc_lang

    if query_lang == doc_lang:
        query_mapping = doc_mapping = None
        notes: dict[str, Any] = {"mapping": "none (same language)"}
    elif args.system == "ft_ref":
        query_mapping = doc_mapping = None
        notes = {"mapping": "none (pretrained aligned)"}
    else:
        mapping, n_used = direction_mapping(args.system, setting, query_lang, doc_lang)
        query_mapping = mapping
        doc_mapping = None
        notes = {"mapping_pairs": n_used}

    query_vectors, query_stats = t3_paragraph_matrix(
        args.system, query_lang, pool.query_texts, query_mapping, mode=mode
    )
    candidate_vectors, candidate_stats = t3_paragraph_matrix(
        args.system, doc_lang, pool.candidate_texts, doc_mapping, mode=mode
    )
    result = parallel.retrieve_by_cosine(query_vectors, candidate_vectors, pool.gold_indices)
    frame = xquad.per_item_frame(pool, result)
    notes.update(
        {
            "mode": mode,
            "query_oov_rate": round(query_stats.oov_rate, 4),
            "candidate_oov_rate": round(candidate_stats.oov_rate, 4),
            "query_empty": query_stats.n_empty_items,
            "candidate_empty": candidate_stats.n_empty_items,
        }
    )
    return frame, notes


def run_t3(
    args: argparse.Namespace,
    data: dict[str, Any],
    tasks: dict[str, Any],
) -> list[dict[str, Any]]:
    """Run every requested T3 setting for one system and log the results."""
    if args.system not in KNOWN_T3_SYSTEMS:
        raise SystemExit(f"unknown T3 system {args.system!r}; known: {KNOWN_T3_SYSTEMS}")
    if args.system == "bm25" and args.settings is None:
        raise SystemExit("bm25 is monolingual; pass --settings en-en hi-hi explicitly")

    xquad_data, pools = t3_queries_and_pools(args, data)
    settings = args.settings or list(tasks["t3_xquad"]["settings"])
    n_boot = int(tasks.get("bootstrap_samples", 1000))
    seed = int(tasks.get("seed", 42))
    rows: list[dict[str, Any]] = []

    for setting in settings:
        pool = xquad.build_pool(xquad_data, setting=setting, pools=pools)
        tag = f"{setting}_avg" if args.ablation else setting
        LOGGER.info("T3 %s %s: %d queries, %d candidates", args.system, tag, pool.n_queries, pool.n_candidates)

        with results_log.timed() as timing:
            if args.system in ENCODER_SYSTEMS:
                frame, notes = run_t3_encoder(args, setting, pool)
            elif args.system in ("bm25", "bm25_mt"):
                frame, notes = run_t3_bm25(args, setting, pool, force=args.force)
            else:
                mode = "average" if args.ablation else "sif"
                frame, notes = run_t3_static(args, setting, pool, mode=mode)

        per_item_path = results_log.write_per_item("t3_xquad", args.system, tag, frame, results_dir=args.results_dir)
        LOGGER.info("wrote %s (%d rows)", per_item_path, len(frame))

        metric_rows = results_log.summarize_per_item(
            frame,
            args.system,
            "t3_xquad",
            tag,
            xquad.T3_METRICS,
            n_boot=n_boot,
            seed=seed,
        )
        note = " ".join(f"{key}={value}" for key, value in sorted(notes.items()))
        note = f"{note} pool={pool.n_candidates} gold=240"
        for row in metric_rows:
            row.update(
                {
                    "runtime_s": round(timing.runtime_s, 2),
                    "peak_mem_mb": round(timing.peak_mem_mb, 1),
                    "notes": note,
                }
            )
            results_log.append_result(row, results_dir=args.results_dir)
        rows.extend(metric_rows)
        LOGGER.info(
            "T3 %s %s: Recall@1=%.4f Recall@5=%.4f MRR=%.4f (%.1fs)",
            args.system,
            tag,
            metric_rows[0]["value"],
            metric_rows[1]["value"],
            metric_rows[2]["value"],
            timing.runtime_s,
        )
    return rows


def load_miracl_pool(data: dict[str, Any], tasks: dict[str, Any], *, force: bool = False) -> miracl.MiraclPool:
    """Load (or build and cache) the MIRACL-hi dev pool (S33).

    Building the pool reads the whole Hindi corpus, so the result is cached to a
    JSONL file keyed by the pool parameters; a mismatch rebuilds instead of
    silently reusing a pool of a different size.
    """
    import json

    settings = data["miracl"]
    pool_settings = tasks.get("t3b_miracl", {}).get("pools", {})
    max_random = int(pool_settings.get("max_random_passages", settings["max_random_passages"]))
    seed = int(pool_settings.get("seed", tasks.get("seed", 42)))
    language = settings["language"]
    split = settings["split"]

    if MIRACL_POOL_CACHE.exists() and not force:
        payload = json.loads(MIRACL_POOL_CACHE.read_text(encoding="utf-8"))
        meta = payload.get("meta", {})
        if (
            meta.get("max_random_passages") == max_random
            and meta.get("seed") == seed
            and meta.get("language") == language
            and meta.get("split") == split
        ):
            LOGGER.info(
                "MIRACL pool cache: %d queries, %d passages", len(payload["query_ids"]), len(payload["passage_ids"])
            )
            return miracl.MiraclPool(
                query_ids=payload["query_ids"],
                query_texts=payload["query_texts"],
                passage_ids=payload["passage_ids"],
                passage_texts=payload["passage_texts"],
                gold_indices=payload["gold_indices"],
                n_relevant_passages=int(meta["n_relevant_passages"]),
                n_random_passages=int(meta["n_random_passages"]),
                corpus_size=int(meta["corpus_size"]),
            )
        LOGGER.info("MIRACL pool cache does not match the current settings; rebuilding")

    directory = REPO_ROOT / settings["dir"]
    topics = miracl.read_topics(directory / f"topics.miracl-v1.0-{language}-{split}.tsv")
    qrels = miracl.read_qrels(directory / f"qrels.miracl-v1.0-{language}-{split}.tsv")
    corpus = list(miracl.iter_corpus(directory / "corpus"))
    LOGGER.info("MIRACL-hi: %d queries, %d corpus passages", len(topics), len(corpus))
    pool = miracl.build_pool(topics, qrels, corpus, max_random_passages=max_random, seed=seed)

    payload = {
        "meta": {
            "language": language,
            "split": split,
            "max_random_passages": max_random,
            "seed": seed,
            "n_relevant_passages": pool.n_relevant_passages,
            "n_random_passages": pool.n_random_passages,
            "corpus_size": pool.corpus_size,
        },
        "query_ids": pool.query_ids,
        "query_texts": pool.query_texts,
        "passage_ids": pool.passage_ids,
        "passage_texts": pool.passage_texts,
        "gold_indices": pool.gold_indices,
    }
    MIRACL_POOL_CACHE.parent.mkdir(parents=True, exist_ok=True)
    MIRACL_POOL_CACHE.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")
    LOGGER.info("wrote %s", MIRACL_POOL_CACHE)
    return pool


def run_t3b_miracl(
    args: argparse.Namespace,
    data: dict[str, Any],
    tasks: dict[str, Any],
) -> list[dict[str, Any]]:
    """Run T3b: monolingual Hindi retrieval on MIRACL-hi dev (S33).

    Every system is scored on the same pool; there is no alignment because both
    sides are Hindi.  Encoders encode queries with ``kind="query"`` and passages
    with ``kind="passage"``; the static systems compose paragraph SIF vectors
    with **no** mapping; ``bm25`` is monolingual by nature.
    """
    if args.system not in KNOWN_T3_SYSTEMS:
        raise SystemExit(f"unknown T3b system {args.system!r}; known: {KNOWN_T3_SYSTEMS}")
    if args.system == "bm25_mt":
        raise SystemExit("bm25_mt is not defined on T3b: MIRACL-hi is monolingual Hindi")

    setting = "hi-hi"
    n_boot = int(tasks.get("bootstrap_samples", 1000))
    seed = int(tasks.get("seed", 42))
    pool = load_miracl_pool(data, tasks, force=args.force)
    LOGGER.info(
        "T3b %s: %d queries, %d passages (%d relevant, %d random of %d corpus passages)",
        args.system,
        pool.n_queries,
        pool.n_candidates,
        pool.n_relevant_passages,
        pool.n_random_passages,
        pool.corpus_size,
    )

    with results_log.timed() as timing:
        if args.system in ENCODER_SYSTEMS:
            query_vectors, query_stats = encoders.encode(
                args.system,
                pool.query_texts,
                "query",
                dataset="miracl_hi",
                lang="hi",
                kind_tag="query",
                use_cache=not args.force,
                return_stats=True,
            )
            passage_vectors, passage_stats = encoders.encode(
                args.system,
                pool.passage_texts,
                "passage",
                dataset="miracl_hi",
                lang="hi",
                kind_tag="passage",
                use_cache=not args.force,
                return_stats=True,
            )
            result = parallel.retrieve_by_cosine(query_vectors, passage_vectors, pool.gold_indices)
            notes = {
                "query_truncated": query_stats.n_truncated,
                "passage_truncated": passage_stats.n_truncated,
                "passage_truncated_fraction": round(passage_stats.truncated_fraction, 4),
                "max_len": passage_stats.max_len,
            }
        elif args.system == "bm25":
            from src.rankers import bm25_rank

            result = bm25_rank(pool.query_texts, pool.passage_texts, pool.gold_indices, lang="hi")
            notes = {"system_kind": "monolingual_bm25"}
        else:
            mode = "average" if args.ablation else "sif"
            query_vectors, query_stats = t3_paragraph_matrix(args.system, "hi", pool.query_texts, None, mode=mode)
            passage_vectors, passage_stats = t3_paragraph_matrix(
                args.system, "hi", pool.passage_texts, None, mode=mode
            )
            result = parallel.retrieve_by_cosine(query_vectors, passage_vectors, pool.gold_indices)
            notes = {
                "mode": mode,
                "mapping": "none (monolingual)",
                "query_oov_rate": round(query_stats.oov_rate, 4),
                "passage_oov_rate": round(passage_stats.oov_rate, 4),
            }

    frame = miracl.per_item_frame(pool, result)
    tag = f"{setting}_avg" if args.ablation else setting
    per_item_path = results_log.write_per_item("t3b_miracl", args.system, tag, frame, results_dir=args.results_dir)
    LOGGER.info("wrote %s (%d rows)", per_item_path, len(frame))

    metric_rows = miracl.metric_rows(frame, system=args.system, setting=tag, n_boot=n_boot, seed=seed)
    note = " ".join(f"{key}={value}" for key, value in sorted(notes.items()))
    note = (
        f"{note} pool={pool.n_candidates} relevant={pool.n_relevant_passages} "
        f"random={pool.n_random_passages} corpus={pool.corpus_size} mrr_cutoff=10"
    )
    for row in metric_rows:
        row.update(
            {
                "runtime_s": round(timing.runtime_s, 2),
                "peak_mem_mb": round(timing.peak_mem_mb, 1),
                "notes": note,
            }
        )
        results_log.append_result(row, results_dir=args.results_dir)
    LOGGER.info(
        "T3b %s %s: MRR@10=%.4f Recall@10=%.4f (%.1fs)",
        args.system,
        tag,
        metric_rows[0]["value"],
        metric_rows[1]["value"],
        timing.runtime_s,
    )
    return metric_rows


TASK_RUNNERS: dict[str, Callable[[argparse.Namespace, dict[str, Any], dict[str, Any]], list[dict[str, Any]]]] = {
    "t1_bli": run_t1,
    "t2_flores": run_t2,
    "t3_xquad": run_t3,
    "t3b_miracl": run_t3b_miracl,
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", required=True, choices=sorted(TASK_RUNNERS), help="task id")
    parser.add_argument(
        "--system",
        required=True,
        help=(
            "system id (t1_bli: w2v, ft, ft_ref, labse, me5, muril; t2_flores adds bm25_mt; "
            "t3_xquad adds bm25; t3b_miracl: w2v, ft, ft_ref, labse, me5, muril, bm25)"
        ),
    )
    parser.add_argument(
        "--settings",
        nargs="+",
        default=None,
        help="directions/settings to run, e.g. hi-en en-hi (default: all configured)",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=results_log.DEFAULT_RESULTS_DIR,
        help="directory holding results.csv and per_item/",
    )
    parser.add_argument("--force", action="store_true", help="ignore cached translations and embeddings")
    parser.add_argument(
        "--ablation",
        action="store_true",
        help="static T2 only: use plain average instead of SIF (logged under <setting>_avg)",
    )
    parser.add_argument(
        "--cvfreq",
        action="store_true",
        help="T1 only: run the five-fold cross-validated frequent band (S24, setting hi-en_cvfreq)",
    )
    parser.add_argument(
        "--coverage-only",
        action="store_true",
        help="T1 only: backfill the full-dictionary coverage view (<setting>_cov) from a logged CSLS run",
    )
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the requested task for one system."""
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s %(message)s")
    seed = set_global_seed()

    data = load_yaml(DATA_CONFIG_PATH)
    tasks = load_yaml(TASKS_CONFIG_PATH)
    started = time.perf_counter()
    rows = TASK_RUNNERS[args.task](args, data, tasks)
    LOGGER.info(
        "logged %d metric rows for %s/%s in %.1fs (seed %d)",
        len(rows),
        args.task,
        args.system,
        time.perf_counter() - started,
        seed,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
