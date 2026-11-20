"""Build the T3 paragraph pools: XQuAD gold paragraphs plus Wikipedia distractors (S28).

For each language the pool is

    240 XQuAD paragraphs (``xq-NNN``)  +  10,000 Wikipedia distractors (``dist-NNN``)

and is written to ``data/processed/t3_pool_<lang>.jsonl``.  The distractors come
from the **`distractor` shard only**, which is disjoint from the static training
shard by article id (PLAN S12, D04), so no distractor is text the static vectors
were trained on.

Distractor rules (PLAN S28, D05):

* paragraphs of the `distractor` shard, 40 to 250 whitespace tokens;
* deduplicated on normalized text, sampled without replacement with seed 42;
* English distractors sharing an 8-token shingle with **any** XQuAD English
  paragraph are dropped: XQuAD is built from SQuAD v1.1 Wikipedia articles, which
  may also be in the dump.

The length distribution of gold versus distractor paragraphs is printed and
written to ``data/processed/t3_pool_stats.json``.  A large mismatch would let
paragraph length act as a shortcut, so it is recorded rather than hidden.

Examples::

    uv run python scripts/build_t3_pool.py
    uv run python scripts/build_t3_pool.py --force
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.tasks import xquad
from src.tasks.xquad import PoolParagraph
from src.textproc import is_devanagari_ratio, normalize

CONFIG_PATH = REPO_ROOT / "configs" / "data.yaml"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
RAW_XQUAD = REPO_ROOT / "data" / "raw" / "xquad"
STATS_PATH = PROCESSED_DIR / "t3_pool_stats.json"

LOGGER = logging.getLogger("build_t3_pool")

LANGUAGE_CODES = ("hi", "en")


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    """Load ``configs/data.yaml``."""
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def read_distractor_candidates(
    para_path: Path,
    *,
    lang: str,
    min_tokens: int,
    max_tokens: int,
    min_devanagari_ratio: float,
) -> tuple[list[str], dict[str, int]]:
    """Read the paragraph table and return the usable distractor texts.

    Returns:
        ``(texts, counters)``.  ``texts`` are deduplicated on normalized text and
        preserve file order, so a later seeded sample is reproducible.
    """
    counters = Counter()
    texts: list[str] = []
    seen: set[str] = set()
    with Path(para_path).open("r", encoding="utf-8") as handle:
        for line in handle:
            fields = line.rstrip("\n").split("\t", 4)
            if len(fields) != 5:
                counters["malformed_rows"] += 1
                continue
            shard, _, _, _, text = fields
            if shard != "distractor":
                continue
            counters["distractor_rows"] += 1
            length = len(text.split())
            if length < min_tokens or length > max_tokens:
                counters["dropped_length"] += 1
                continue
            if lang == "hi" and is_devanagari_ratio(text) < min_devanagari_ratio:
                counters["dropped_script"] += 1
                continue
            key = normalize(text)
            if key in seen:
                counters["dropped_duplicate"] += 1
                continue
            seen.add(key)
            texts.append(key)
            counters["kept"] += 1
    return texts, dict(counters)


def sample_distractors(
    texts: list[str],
    *,
    count: int,
    seed: int,
    reference_shingles: set[tuple[str, ...]] | None = None,
    shingle_size: int = 8,
) -> tuple[list[str], dict[str, int]]:
    """Sample ``count`` distractor texts without replacement after the overlap filter.

    Args:
        texts: deduplicated candidate texts.
        count: how many distractors are wanted.
        seed: sampling seed (the global one from ``configs/``).
        reference_shingles: shingles to reject against; ``None`` disables the filter.
        shingle_size: shingle width.

    Returns:
        ``(sampled, counters)``.  ``counters["shortfall"]`` is positive when the
        pool could not supply ``count`` distractors; the caller logs it instead of
        silently producing a smaller pool.
    """
    from src.textproc import tokenize

    counters = Counter()
    eligible = texts
    if reference_shingles is not None:
        eligible = [
            text
            for text in texts
            if not xquad.overlaps_reference(
                tokenize(text, "en"), reference_shingles, size=shingle_size
            )
        ]
        counters["dropped_overlap"] = len(texts) - len(eligible)

    rng = np.random.default_rng(seed)
    if len(eligible) <= count:
        sampled = list(eligible)
        counters["shortfall"] = count - len(sampled)
    else:
        order = rng.permutation(len(eligible))[:count]
        sampled = [eligible[int(index)] for index in sorted(order)]
        counters["shortfall"] = 0
    counters["eligible"] = len(eligible)
    counters["sampled"] = len(sampled)
    return sampled, dict(counters)


def length_summary(lengths: list[int]) -> dict[str, float]:
    """Return count/mean/median/quartiles for a list of paragraph lengths."""
    if not lengths:
        return {"n": 0}
    values = np.asarray(lengths, dtype=np.float64)
    return {
        "n": int(values.size),
        "mean": round(float(values.mean()), 2),
        "median": round(float(np.median(values)), 2),
        "p10": round(float(np.percentile(values, 10)), 2),
        "p90": round(float(np.percentile(values, 90)), 2),
        "min": int(values.min()),
        "max": int(values.max()),
    }


def build_language_pool(
    config: dict[str, Any],
    data: xquad.XQuadData,
    *,
    lang: str,
    force: bool,
) -> dict[str, Any]:
    """Build and write one language's pool, returning its statistics."""
    settings = config["t3_distractors"]
    target_count = int(settings["n_per_language"])
    seed = int(settings["seed"])
    shingle_size = int(settings["overlap_shingle"])
    destination = PROCESSED_DIR / f"t3_pool_{lang}.jsonl"

    para_path = PROCESSED_DIR / f"wiki.{lang}.para.tsv"
    if not para_path.exists():
        raise SystemExit(f"{para_path} is missing; run scripts/preprocess_wiki.py --lang {lang}")

    if destination.exists() and not force:
        paragraphs = xquad.read_pool(destination)
        LOGGER.info("skip %s (%d paragraphs); pass --force to rebuild", destination.name, len(paragraphs))
        gold = [paragraph for paragraph in paragraphs if paragraph.is_gold_candidate]
        distractors = [paragraph for paragraph in paragraphs if not paragraph.is_gold_candidate]
        return {
            "n_gold": len(gold),
            "n_distractors": len(distractors),
            "n_total": len(paragraphs),
            "gold_lengths": length_summary([len(paragraph.text.split()) for paragraph in gold]),
            "distractor_lengths": length_summary([len(paragraph.text.split()) for paragraph in distractors]),
            "reused": True,
        }

    started = time.perf_counter()
    candidates, counters = read_distractor_candidates(
        para_path,
        lang=lang,
        min_tokens=int(settings["min_tokens"]),
        max_tokens=int(settings["max_tokens"]),
        min_devanagari_ratio=float(config["wikipedia"]["min_devanagari_ratio_hi"]),
    )
    LOGGER.info("%s: %d usable distractor paragraphs (%s)", lang, len(candidates), counters)

    reference = None
    if lang == "en":
        reference = xquad.reference_shingles(
            [paragraph.text_en for paragraph in data.paragraphs], "en", size=shingle_size
        )
        LOGGER.info("en: %d XQuAD reference shingles of width %d", len(reference), shingle_size)

    sampled, sample_counters = sample_distractors(
        candidates,
        count=target_count,
        seed=seed,
        reference_shingles=reference,
        shingle_size=shingle_size,
    )
    if sample_counters["shortfall"]:
        LOGGER.warning(
            "%s: only %d eligible distractors, %d short of %d; the pool is smaller than planned",
            lang,
            sample_counters["sampled"],
            sample_counters["shortfall"],
            target_count,
        )

    gold = [
        PoolParagraph(
            pid=paragraph.pid,
            text=paragraph.text_hi if lang == "hi" else paragraph.text_en,
            is_gold_candidate=True,
        )
        for paragraph in data.paragraphs
    ]
    distractors = [
        PoolParagraph(pid=f"dist-{index:05d}", text=text, is_gold_candidate=False)
        for index, text in enumerate(sampled)
    ]
    paragraphs = gold + distractors
    xquad.write_pool(destination, paragraphs)
    elapsed = round(time.perf_counter() - started, 2)
    LOGGER.info(
        "%s: wrote %s (%d gold + %d distractors, %.1fs)",
        lang,
        destination.name,
        len(gold),
        len(distractors),
        elapsed,
    )

    return {
        "n_gold": len(gold),
        "n_distractors": len(distractors),
        "n_total": len(paragraphs),
        "eligible_distractors": sample_counters["eligible"],
        "dropped_overlap": sample_counters.get("dropped_overlap", 0),
        "shortfall": sample_counters["shortfall"],
        "candidate_counters": counters,
        "gold_lengths": length_summary([len(paragraph.text.split()) for paragraph in gold]),
        "distractor_lengths": length_summary([len(paragraph.text.split()) for paragraph in distractors]),
        "runtime_s": elapsed,
        "reused": False,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lang", choices=LANGUAGE_CODES + ("all",), default="all", help="language (default: all)")
    parser.add_argument("--force", action="store_true", help="rebuild a pool that already exists")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Build the requested pools and write their statistics."""
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s %(message)s")
    config = load_config()
    settings = config["xquad"]
    data = xquad.load_xquad(
        RAW_XQUAD,
        expected_questions=int(settings["expected_questions"]),
        expected_paragraphs=int(settings["expected_paragraphs"]),
    )

    languages = list(LANGUAGE_CODES) if args.lang == "all" else [args.lang]
    stats: dict[str, Any] = {}
    if STATS_PATH.exists():
        try:
            stats = json.loads(STATS_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            stats = {}
    for lang in languages:
        stats[lang] = build_language_pool(config, data, lang=lang, force=args.force)
        LOGGER.info(
            "%s: gold length %s vs distractor length %s",
            lang,
            stats[lang]["gold_lengths"],
            stats[lang]["distractor_lengths"],
        )

    stats["settings"] = {
        "n_per_language": int(config["t3_distractors"]["n_per_language"]),
        "seed": int(config["t3_distractors"]["seed"]),
        "min_tokens": int(config["t3_distractors"]["min_tokens"]),
        "max_tokens": int(config["t3_distractors"]["max_tokens"]),
        "overlap_shingle": int(config["t3_distractors"]["overlap_shingle"]),
        "source_shard": config["t3_distractors"]["source_shard"],
    }
    STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATS_PATH.write_text(json.dumps(stats, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    LOGGER.info("wrote %s", STATS_PATH)
    return 0


if __name__ == "__main__":
    sys.exit(main())
