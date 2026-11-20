"""Build the shared candidate vocabularies for T1 (S20).

For each language the candidate list is the top 200,000 Wikipedia words by count,
restricted to tokens containing a letter (and, for Hindi, at least one Devanagari
letter), then intersected with the vocabularies of every static system so that
all systems are scored on *the same* candidate pool (``AGENTS.md`` §9.2).

Output: ``data/processed/candidates_<lang>.txt``, one word per line; the line
number is the rank.  A JSON sidecar records how many words each system lacks.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import regex

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.embeddings import static

LOGGER = logging.getLogger("build_candidates")

PROCESSED = REPO_ROOT / "data" / "processed"
FASTTEXT_RAW = REPO_ROOT / "data" / "raw" / "fasttext"

_HAS_LETTER = regex.compile(r"\p{L}")
_HAS_DEVANAGARI = regex.compile(r"(?V1)[\p{Devanagari}&&\p{L}]")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lang", choices=["hi", "en"], default=None, help="language to build (default: both)")
    parser.add_argument("--limit", type=int, default=None, help="override the configured vocab_cap")
    parser.add_argument("--force", action="store_true", help="rebuild even when the output exists")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args(argv)


def read_word_counts(path: Path) -> list[tuple[str, int]]:
    """Read ``wordcounts_<lang>.tsv`` sorted by descending count."""
    counts: list[tuple[str, int]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.rstrip("\n")
            if not stripped:
                continue
            parts = stripped.split("\t")
            if len(parts) != 2:
                raise ValueError(f"{path}:{line_number}: expected two columns, got {len(parts)}")
            word, count = parts
            counts.append((word, int(count)))
    counts.sort(key=lambda item: (-item[1], item[0]))
    return counts


def keep_word(word: str, lang: str) -> bool:
    """Return whether ``word`` is a candidate: it must contain a letter."""
    if not _HAS_LETTER.search(word):
        return False
    if lang == "hi" and not _HAS_DEVANAGARI.search(word):
        return False
    return True


def build(
    lang: str,
    *,
    limit: int,
    force: bool = False,
    processed: Path = PROCESSED,
    fasttext_raw: Path = FASTTEXT_RAW,
) -> dict:
    """Build ``candidates_<lang>.txt`` and return a coverage summary."""
    output = processed / f"candidates_{lang}.txt"
    sidecar = processed / f"candidates_{lang}.json"
    if output.exists() and not force:
        LOGGER.info("skip %s (already built)", output)
        return json.loads(sidecar.read_text(encoding="utf-8")) if sidecar.exists() else {"path": str(output)}

    counts = read_word_counts(processed / f"wordcounts_{lang}.tsv")
    filtered = [(word, count) for word, count in counts if keep_word(word, lang)]
    top = filtered[:limit]
    LOGGER.info("%s: %d tokens, %d keep a letter, taking top %d", lang, len(counts), len(filtered), len(top))

    vocabs: dict[str, set[str]] = {}
    for system in static.SYSTEMS:
        vocabs[system] = static.vocabulary(static.load(system, lang))
    ref_name = "wiki.hi.align.vec" if lang == "hi" else "wiki.en.align.vec"
    ref = static.load_pretrained(fasttext_raw / ref_name, limit=limit)
    vocabs["ft_ref"] = static.vocabulary(ref)

    shared = [word for word, _ in top if all(word in vocab for vocab in vocabs.values())]
    LOGGER.info("%s: %d words after intersecting with %s", lang, len(shared), sorted(vocabs))

    output.write_text("".join(f"{word}\n" for word in shared), encoding="utf-8")
    summary = {
        "lang": lang,
        "tokens": len(counts),
        "with_letter": len(filtered),
        "top": len(top),
        "candidates": len(shared),
        "system_vocab": {name: len(vocab) for name, vocab in vocabs.items()},
        "missing_from_shared": {
            system: sum(1 for word, _ in top if word not in vocab) for system, vocab in vocabs.items()
        },
        "path": str(output.relative_to(REPO_ROOT)) if output.is_relative_to(REPO_ROOT) else str(output),
    }
    sidecar.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    """Build the requested candidate lists."""
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s %(message)s")
    limit = args.limit if args.limit is not None else int(static.load_models_config()["static"]["vocab_cap"])
    languages = [args.lang] if args.lang else list(static.LANGUAGES)

    summaries = [build(lang, limit=limit, force=args.force) for lang in languages]
    print(json.dumps(summaries, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
