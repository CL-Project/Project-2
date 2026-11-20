"""Train the static word vectors (S19): ``--system {w2v,ft} --lang {hi,en}``.

Four runs in total, always sequential: a fastText run needs a few GB of RAM per
language, so training two at once on a 16 GB laptop is not safe (``AGENTS.md``
§4).  Run this inside tmux under ``systemd-inhibit``::

    systemd-inhibit --what=idle:sleep:handle-lid-switch \\
        uv run python scripts/train_static.py --all

The script is idempotent: an existing model is skipped unless ``--force`` is
passed, so an interrupted run can simply be started again.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.embeddings import static
from src.seeding import set_global_seed

LOGGER = logging.getLogger("train_static")

SANITY_WORDS = {
    "hi": ["भारत", "दिल्ली", "पानी", "सरकार", "फ़िल्म", "किताब", "साल", "शहर", "माँ", "सड़क"],
    "en": ["india", "delhi", "water", "government", "film", "book", "year", "city", "mother", "road"],
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--system", choices=sorted(static.SYSTEMS), default=None, help="static family to train")
    parser.add_argument("--lang", choices=sorted(static.LANGUAGES), default=None, help="language to train")
    parser.add_argument("--all", action="store_true", help="train all four (system, language) combinations")
    parser.add_argument("--force", action="store_true", help="retrain even when the model file exists")
    parser.add_argument("--sanity", action="store_true", help="print nearest neighbours after training")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args(argv)


def resolve_jobs(args: argparse.Namespace) -> list[tuple[str, str]]:
    """Return the ``(system, lang)`` pairs to train, in a deterministic order."""
    if args.all:
        return [(system, lang) for system in static.SYSTEMS for lang in static.LANGUAGES]
    if args.system is None or args.lang is None:
        raise SystemExit("pass --all, or both --system and --lang")
    return [(args.system, args.lang)]


def main(argv: list[str] | None = None) -> int:
    """Train the requested models and print a summary."""
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s %(message)s")
    seed = set_global_seed()
    config = static.load_models_config()

    summaries = []
    for system, lang in resolve_jobs(args):
        summary = static.train(system, lang, seed=seed, config=config, force=args.force)
        summaries.append(summary)
        LOGGER.info(
            "%s.%s: vocab=%s runtime=%ss",
            system,
            lang,
            summary.get("vocab_size"),
            summary.get("runtime_s"),
        )
        if args.sanity:
            vectors = static.load(system, lang)
            neighbours = static.nearest_neighbours(vectors, SANITY_WORDS[lang])
            LOGGER.info("sanity neighbours for %s.%s: %s", system, lang, json.dumps(neighbours, ensure_ascii=False))

    print(json.dumps(summaries, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
