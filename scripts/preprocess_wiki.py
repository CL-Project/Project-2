"""Build the Wikipedia corpora used for training, SIF fitting and T3 distractors (S12).

Inputs are the raw Wikipedia parquet shards (Hindi) and the streamed English
config, both from ``wikimedia/wikipedia``.  Outputs under ``data/processed/``:

``wiki.<lang>.para.tsv``       ``shard, article_id, title, para_idx, text`` for all shards
``wiki.<lang>.sent.txt``       cleaned ``train``-shard sentences, one per line
``wiki.<lang>.tok.txt``        space-joined tokens of those sentences (>= 3 tokens)
``wordcounts_<lang>.tsv``      ``token<TAB>count`` from the ``train`` token file

Design decisions (``docs/DECISIONS.md`` D04):

* Every article is assigned to ``train`` / ``distractor`` / ``fit`` by a
  deterministic hash of its title, so the three shards are disjoint by
  construction and a rerun picks the same assignment.
* Hindi paragraphs whose Devanagari letter ratio is below the configured
  threshold are dropped (English and template residue).
* Any training sentence that exactly matches a FLORES or XQuAD sentence of the
  same language, after normalization, is removed.  Counts are reported in
  ``docs/DATA.md`` by S13.
* ``train`` sizes are matched: Hindi is processed fully, its token count is
  recorded, and English articles are streamed in a seeded random order until
  the English token count reaches that total (``AGENTS.md`` §4).

The script is idempotent: an existing output is kept unless ``--force`` is
passed.  Hindi and English can be run separately (``--lang``), and English
requires the Hindi token count from a previous Hindi run.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import logging
import os
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.textproc import is_devanagari_ratio, normalize, sentence_split, tokenize

CONFIG_PATH = REPO_ROOT / "configs" / "data.yaml"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
SHARD_NAMES = ("train", "distractor", "fit")
MIN_TOKENS = 3
MIN_PARAGRAPH_CHARS = 40

LOGGER = logging.getLogger("preprocess_wiki")


@dataclass
class CorpusStats:
    """Counters reported in ``docs/DATA.md`` and the run log."""

    articles_seen: int = 0
    articles_kept: int = 0
    shard_articles: Counter = field(default_factory=Counter)
    paragraphs_seen: int = 0
    paragraphs_kept: int = 0
    paragraphs_dropped_short: int = 0
    paragraphs_dropped_script: int = 0
    sentences_seen: int = 0
    sentences_kept: int = 0
    sentences_dropped_short: int = 0
    sentences_dropped_contamination: int = 0
    tokens: int = 0
    sentences_contamination_examples: list[str] = field(default_factory=list)
    elapsed_s: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        """Return the counters as a JSON-serializable dict."""
        payload = {
            "articles_seen": self.articles_seen,
            "articles_kept": self.articles_kept,
            "shard_articles": dict(self.shard_articles),
            "paragraphs_seen": self.paragraphs_seen,
            "paragraphs_kept": self.paragraphs_kept,
            "paragraphs_dropped_short": self.paragraphs_dropped_short,
            "paragraphs_dropped_script": self.paragraphs_dropped_script,
            "sentences_seen": self.sentences_seen,
            "sentences_kept": self.sentences_kept,
            "sentences_dropped_short": self.sentences_dropped_short,
            "sentences_dropped_contamination": self.sentences_dropped_contamination,
            "tokens": self.tokens,
            "contamination_examples": self.sentences_contamination_examples[:20],
            "elapsed_s": round(self.elapsed_s, 1),
        }
        return payload


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    """Load ``configs/data.yaml``."""
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def shard_for_title(title: str, shard_probabilities: dict[str, float]) -> str:
    """Return the shard of ``title`` from a deterministic hash (disjoint by construction).

    The hash is SHA-256 of the normalized title, so the assignment is stable
    across runs, machines and Python versions (unlike ``hash()``).
    """
    digest = hashlib.sha256(normalize(title).encode("utf-8")).digest()
    fraction = int.from_bytes(digest[:8], "big") / 2**64
    cumulative = 0.0
    for name in SHARD_NAMES:
        cumulative += float(shard_probabilities[name])
        if fraction < cumulative:
            return name
    return SHARD_NAMES[-1]


def paragraphs_from_text(text: str, min_chars: int = MIN_PARAGRAPH_CHARS) -> list[str]:
    """Split article text into cleaned paragraph strings.

    Blank-line-separated blocks are paragraphs; within a block, lines shorter
    than ``min_chars`` are dropped (headings, list residue, captions).
    """
    blocks: list[str] = []
    for block in text.split("\n\n"):
        lines = [
            line.strip()
            for line in block.splitlines()
            if len(line.strip()) >= min_chars and not line.lstrip().startswith("==")
        ]
        if lines:
            blocks.append(" ".join(lines))
    return blocks


def sentence_key(text: str, language: str) -> str:
    """Return a punctuation- and spacing-insensitive key for exact-match filtering.

    Wikipedia sentences reach the filter after :func:`sentence_split`, which
    removes the terminating danda or period, while FLORES and XQuAD keep theirs.
    Comparing normalized token sequences (``src.textproc.tokenize``) makes the
    match independent of that difference and of repeated whitespace.
    """
    return " ".join(tokenize(text, language))


def contamination_set(config: dict[str, Any], language: str) -> set[str]:
    """Return keys of evaluation sentences that must not appear in training.

    Covers FLORES dev/devtest sentences and XQuAD contexts, questions and answers
    of the same language (``AGENTS.md`` §10.4).  XQuAD contexts are split into
    sentences so that a Wikipedia sentence matching part of a context is caught.
    Missing files are skipped with a warning so the script still runs before
    every dataset is downloaded.
    """
    texts: set[str] = set()

    def add(text: str) -> None:
        for sentence in sentence_split(text, language):
            key = sentence_key(sentence, language)
            # Only keys that a training sentence could actually take: shorter
            # ones are dropped by MIN_TOKENS anyway, and single-token XQuAD
            # answers ("A", "S") would otherwise inflate the removed count.
            if len(key.split()) >= MIN_TOKENS:
                texts.add(key)

    flores = config["flores"]
    flores_dir = REPO_ROOT / flores["dir"]
    code = flores["languages"][language]
    for split in flores["splits"]:
        path = flores_dir / split / f"{code}.{split}"
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    add(line)
        else:
            LOGGER.warning("FLORES file missing, not filtering against it: %s", path.name)

    xquad_dir = REPO_ROOT / config["xquad"]["dir"]
    xquad_path = xquad_dir / f"xquad.{language}.jsonl"
    if xquad_path.exists():
        for line in xquad_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            add(record["context"])
            add(record["question"])
            for answer in record.get("answers", []):
                if isinstance(answer, dict) and answer.get("text"):
                    add(answer["text"])
    else:
        LOGGER.warning("XQuAD file missing, not filtering against it: %s", xquad_path.name)
    return texts


def iter_hindi_articles(config: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """Yield Hindi articles from the downloaded parquet shards.

    Uses ``datasets`` on local parquet files so no network access is needed.
    """
    from datasets import load_dataset

    directory = REPO_ROOT / config["wikipedia"]["dir"] / f"{config['wikipedia']['dump_date']}.hi"
    files = sorted(directory.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(
            f"no Hindi parquet files under {directory}; run scripts/download_data.py --groups wikipedia"
        )
    dataset = load_dataset("parquet", data_files=[str(path) for path in files], split="train")
    for record in dataset:
        yield {"id": record.get("id", ""), "title": record["title"], "text": record["text"]}


def iter_english_articles(config: dict[str, Any], seed: int) -> Iterator[dict[str, Any]]:
    """Yield English articles streamed from the Hub in a seeded shuffled order.

    English is never downloaded in full (``AGENTS.md`` §4).  ``shuffle`` uses
    the configured seed and a bounded buffer, which keeps memory flat while
    randomizing article order for the equal-size stop rule.
    """
    from datasets import load_dataset

    settings = config["wikipedia"]
    config_name = f"{settings['dump_date']}.en"
    dataset = load_dataset("wikimedia/wikipedia", config_name, split="train", streaming=True)
    dataset = dataset.shuffle(seed=seed, buffer_size=10_000)
    try:
        for record in dataset:
            yield {"id": record.get("id", ""), "title": record["title"], "text": record["text"]}
    finally:
        del dataset
        gc.collect()


def write_paragraph_rows(
    handle: Any,
    language: str,
    article: dict[str, Any],
    shard: str,
    stats: CorpusStats,
    config: dict[str, Any],
) -> list[tuple[int, str]]:
    """Write one article's paragraphs to the TSV and return ``(para_idx, text)``.

    Paragraphs are split on blank lines; each paragraph is tab-free so the TSV
    stays parseable.
    """
    threshold = float(config["wikipedia"]["min_devanagari_ratio_hi"])
    kept: list[tuple[int, str]] = []
    for index, paragraph in enumerate(paragraphs_from_text(article["text"])):
        stats.paragraphs_seen += 1
        cleaned = normalize(paragraph).replace("\t", " ")
        if language == "hi" and is_devanagari_ratio(cleaned) < threshold:
            stats.paragraphs_dropped_script += 1
            continue
        if len(cleaned) < MIN_PARAGRAPH_CHARS:
            stats.paragraphs_dropped_short += 1
            continue
        if not cleaned:
            stats.paragraphs_dropped_short += 1
            continue
        kept.append((index, cleaned))
        stats.paragraphs_kept += 1
        handle.write(f"{shard}\t{article['id']}\t{article['title'].replace(chr(9), ' ')}\t{index}\t{cleaned}\n")
    return kept


def process_article(
    language: str,
    article: dict[str, Any],
    shard: str,
    stats: CorpusStats,
    config: dict[str, Any],
    para_handle: Any,
    sent_handles: dict[str, Any],
    contamination: set[str],
    token_target: int | None,
) -> int:
    """Process one article and return the number of ``train`` tokens it added.

    Paragraph rows are written for every shard.  Sentences, tokens and word
    counts are produced for the ``train`` shard only, stopping early once
    ``token_target`` is reached (the English equal-size rule).
    """
    kept = write_paragraph_rows(para_handle, language, article, shard, stats, config)
    if shard != "train" or not kept:
        return 0

    added_tokens = 0
    for _, paragraph in kept:
        if token_target is not None and stats.tokens >= token_target:
            break
        for sentence in sentence_split(paragraph, language):
            stats.sentences_seen += 1
            normalized = normalize(sentence)
            tokens = tokenize(normalized, language)
            if sentence_key(sentence, language) in contamination:
                stats.sentences_dropped_contamination += 1
                if len(stats.sentences_contamination_examples) < 20:
                    stats.sentences_contamination_examples.append(normalized)
                continue
            if len(tokens) < MIN_TOKENS:
                stats.sentences_dropped_short += 1
                continue
            sent_handles["sent"].write(normalized + "\n")
            sent_handles["tok"].write(" ".join(tokens) + "\n")
            stats.sentences_kept += 1
            stats.tokens += len(tokens)
            added_tokens += len(tokens)
    return added_tokens


def word_counts(token_path: Path) -> Counter:
    """Return token counts from a ``wiki.<lang>.tok.txt`` file."""
    counts: Counter = Counter()
    with token_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            counts.update(line.split())
    return counts


def write_word_counts(counts: Counter, path: Path) -> None:
    """Write ``token<TAB>count`` sorted by descending count, then token."""
    with path.open("w", encoding="utf-8") as handle:
        for token, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
            handle.write(f"{token}\t{count}\n")


def read_token_target(path: Path) -> int | None:
    """Return the ``train`` token count recorded by a previous run, or ``None``.

    ``None`` means the output cannot be trusted as complete: the stats file is
    missing, empty or not a JSON object with a ``tokens`` field.
    """
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8") or "null")
    if not isinstance(payload, dict):
        return None
    tokens = payload.get("tokens")
    return int(tokens) if isinstance(tokens, int) and tokens > 0 else None


def output_paths(language: str) -> dict[str, Path]:
    """Return the output paths for one language."""
    return {
        "para": PROCESSED_DIR / f"wiki.{language}.para.tsv",
        "sent": PROCESSED_DIR / f"wiki.{language}.sent.txt",
        "tok": PROCESSED_DIR / f"wiki.{language}.tok.txt",
        "counts": PROCESSED_DIR / f"wordcounts_{language}.tsv",
        "stats": PROCESSED_DIR / f"wiki.{language}.stats.json",
    }


def process_language(
    language: str,
    config: dict[str, Any],
    *,
    force: bool,
    token_target: int | None,
    limit_articles: int | None,
    log_every: int,
) -> CorpusStats:
    """Run the full preprocessing pipeline for one language and write all outputs."""
    paths = output_paths(language)
    existing = read_token_target(paths["stats"])
    if paths["tok"].exists() and existing is not None and not force:
        LOGGER.info(
            "%s: output exists (%d tokens); skipping (pass --force to rebuild)",
            language,
            existing,
        )
        return CorpusStats(tokens=existing)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    shard_probabilities = config["wikipedia"]["shards"]
    seed = int(config["t3_distractors"]["seed"])
    contamination = contamination_set(config, language)
    LOGGER.info("%s: %d evaluation sentences excluded from training", language, len(contamination))

    stats = CorpusStats()
    started = time.perf_counter()
    articles = (
        iter_hindi_articles(config) if language == "hi" else iter_english_articles(config, seed)
    )

    with paths["para"].open("w", encoding="utf-8") as para_handle, paths["sent"].open(
        "w", encoding="utf-8"
    ) as sent_handle, paths["tok"].open("w", encoding="utf-8") as tok_handle:
        sent_handles = {"sent": sent_handle, "tok": tok_handle}
        for article in articles:
            if limit_articles is not None and stats.articles_seen >= limit_articles:
                break
            if token_target is not None and stats.tokens >= token_target:
                break
            stats.articles_seen += 1
            title = article.get("title") or ""
            if not title.strip():
                continue
            shard = shard_for_title(title, shard_probabilities)
            stats.articles_kept += 1
            stats.shard_articles[shard] += 1
            process_article(
                language,
                article,
                shard,
                stats,
                config,
                para_handle,
                sent_handles,
                contamination,
                token_target,
            )
            if log_every and stats.articles_seen % log_every == 0:
                LOGGER.info(
                    "%s: %d articles, %d sentences, %d tokens",
                    language,
                    stats.articles_seen,
                    stats.sentences_kept,
                    stats.tokens,
                )

    stats.elapsed_s = time.perf_counter() - started
    counts = word_counts(paths["tok"])
    write_word_counts(counts, paths["counts"])
    paths["stats"].write_text(json.dumps(stats.as_dict(), indent=2) + "\n", encoding="utf-8")
    LOGGER.info(
        "%s: %d articles, %d sentences, %d tokens, %d types in %.0fs",
        language,
        stats.articles_seen,
        stats.sentences_kept,
        stats.tokens,
        len(counts),
        stats.elapsed_s,
    )
    return stats


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lang", choices=["hi", "en", "both"], default="both")
    parser.add_argument("--force", action="store_true", help="rebuild outputs that already exist")
    parser.add_argument(
        "--limit-articles",
        type=int,
        default=None,
        help="process at most N articles per language (smoke tests only)",
    )
    parser.add_argument("--log-every", type=int, default=5000, help="progress log interval in articles")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Process the requested languages, Hindi first so English can match its size."""
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s %(message)s")
    config = load_config()
    seed = int(config["t3_distractors"]["seed"])

    if args.lang in {"hi", "both"}:
        hindi = process_language(
            "hi",
            config,
            force=args.force,
            token_target=None,
            limit_articles=args.limit_articles,
            log_every=args.log_every,
        )
        hindi_tokens = hindi.tokens or read_token_target(output_paths("hi")["stats"])
        if hindi_tokens is None:
            raise SystemExit("Hindi token count unknown; run the Hindi pass first")
        LOGGER.info("Hindi train tokens: %d", hindi_tokens)
    else:
        hindi_tokens = read_token_target(output_paths("hi")["stats"])
        if hindi_tokens is None:
            raise SystemExit("run --lang hi first: English is capped at the Hindi token count")
        LOGGER.info("using Hindi token target: %d", hindi_tokens)

    if args.lang in {"en", "both"}:
        english = process_language(
            "en",
            config,
            force=args.force,
            token_target=hindi_tokens,
            limit_articles=args.limit_articles,
            log_every=args.log_every,
        )
        LOGGER.info(
            "English train tokens: %d (%.2f%% of Hindi)",
            english.tokens,
            100.0 * english.tokens / max(1, hindi_tokens),
        )
    LOGGER.debug("seed %d used for the English stream order", seed)
    return 0


def run() -> int:
    """Run :func:`main` and exit without the ``datasets`` streaming shutdown crash.

    ``datasets`` streaming leaves a pyarrow reader thread that races with
    interpreter finalization on this stack (Python 3.12 + pyarrow), producing
    ``Fatal Python error: PyGILState_Release: thread state ... must be current``
    *after* every output has already been written and flushed.  All work is
    complete at that point, so the process exits directly with its status code.
    """
    status = 0
    try:
        status = main()
    finally:
        logging.shutdown()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(status)


if __name__ == "__main__":
    run()
