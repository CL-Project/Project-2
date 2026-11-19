"""Run a (system, task) combination from ``configs/`` and log the results.

``--task t2_flores`` is implemented (S15 to S17): it pools FLORES dev + devtest,
encodes or scores the pool, and writes ``results/results.csv`` plus
``results/per_item/<task>__<system>__<setting>.csv``.  Later phases add T1 and
T3 to :data:`TASK_RUNNERS`; every task keeps the same logging contract.

Examples::

    uv run python scripts/run_eval.py --task t2_flores --system labse
    uv run python scripts/run_eval.py --task t2_flores --system me5 --settings hi-en
    uv run python scripts/run_eval.py --task t2_flores --system bm25_mt

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

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src import results_log
from src.embeddings import encoders
from src.seeding import set_global_seed
from src.tasks import parallel

DATA_CONFIG_PATH = REPO_ROOT / "configs" / "data.yaml"
TASKS_CONFIG_PATH = REPO_ROOT / "configs" / "tasks.yaml"

LOGGER = logging.getLogger("run_eval")

ENCODER_SYSTEMS = ("labse", "me5", "muril")
KNOWN_T2_SYSTEMS = ENCODER_SYSTEMS + ("bm25_mt",)


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
        LOGGER.info("T2 %s %s: %d queries, %d candidates", args.system, setting, pool.n_queries, pool.n_candidates)

        with results_log.timed() as timing:
            if args.system in ENCODER_SYSTEMS:
                frame, notes = run_t2_encoder(
                    args.system, pool, use_cache=not args.force
                )
            elif args.system == "bm25_mt":
                frame, notes = run_t2_bm25_mt(setting, pool, force=args.force)
            else:
                raise SystemExit(f"unknown T2 system {args.system!r}; known: {KNOWN_T2_SYSTEMS}")

        per_item_path = results_log.write_per_item(
            "t2_flores", args.system, setting, frame, results_dir=args.results_dir
        )
        LOGGER.info("wrote %s (%d rows)", per_item_path, len(frame))

        metric_rows = results_log.summarize_per_item(
            frame,
            args.system,
            "t2_flores",
            setting,
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
            setting,
            metric_rows[0]["value"],
            metric_rows[1]["value"],
            metric_rows[2]["value"],
            timing.runtime_s,
        )
    return rows


TASK_RUNNERS: dict[str, Callable[[argparse.Namespace, dict[str, Any], dict[str, Any]], list[dict[str, Any]]]] = {
    "t2_flores": run_t2,
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", required=True, choices=sorted(TASK_RUNNERS), help="task id")
    parser.add_argument("--system", required=True, help="system id (labse, me5, muril, bm25_mt)")
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
