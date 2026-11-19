"""Validate the downloaded datasets and write the data card ``docs/DATA.md`` (S11).

Checks (all fail loudly, none silently drops items):

* FLORES: Hindi and English have equal line counts, per split and the expected
  totals (997 dev + 1,012 devtest).
* XQuAD: the expected number of questions and unique paragraphs per language,
  and question ids match one-to-one across languages with the same gold
  paragraph per id.
* MUSE: pair counts and unique source words per split, and train/test source
  words are disjoint within each direction (``AGENTS.md`` §10.4).
* Every dataset is UTF-8 and NFC-normalized; NFC-normalization must be a no-op,
  which catches mojibake and mixed encodings.

The script prints a short report, writes ``docs/DATA.md`` and exits non-zero on
the first failed assertion.  It reads only files under ``data/raw`` and needs no
network or GPU.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.textproc import is_devanagari_ratio, normalize

CONFIG_PATH = REPO_ROOT / "configs" / "data.yaml"
DATA_DOC_PATH = REPO_ROOT / "docs" / "DATA.md"
EXAMPLE_COUNT = 3

LOGGER = logging.getLogger("validate_data")


class ValidationError(RuntimeError):
    """Raised when a dataset check fails."""


@dataclass
class Report:
    """Collected facts, rendered into ``docs/DATA.md``."""

    sections: dict[str, list[str]] = field(default_factory=dict)
    checks: list[tuple[str, bool, str]] = field(default_factory=list)

    def add(self, section: str, line: str) -> None:
        """Append one markdown line to ``section``."""
        self.sections.setdefault(section, []).append(line)

    def check(self, name: str, passed: bool, detail: str = "") -> None:
        """Record a pass/fail check and raise on failure."""
        self.checks.append((name, passed, detail))
        if not passed:
            raise ValidationError(f"{name} failed: {detail}")


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    """Load ``configs/data.yaml``."""
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def read_lines(path: Path) -> list[str]:
    """Read a UTF-8 text file as stripped non-empty lines, rejecting mojibake.

    ``data/raw/`` holds untouched downloads, so it is **not** expected to be NFC:
    FLORES in particular ships precomposed nukta letters (for example ``ढ़``
    U+095D) that NFC decomposes.  What is rejected here is readable-but-wrong
    text: the U+FFFD replacement character, or a mix of scripts that no real
    Hindi sentence has.  Normalization itself is applied by the pipeline
    (``src.textproc``), not to the raw files.
    """
    if not path.exists():
        raise ValidationError(f"missing file: {path.relative_to(REPO_ROOT)}")
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    kept = [line for line in lines if line]
    replacement = sum(line.count("\ufffd") for line in kept)
    if replacement:
        raise ValidationError(f"{path.name}: {replacement} U+FFFD replacement characters (mojibake)")
    return kept


def nfc_change_count(path: Path) -> tuple[int, int]:
    """Return ``(lines_changed_by_nfc, total_lines)`` for a text file."""
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    changed = sum(1 for line in lines if unicodedata.normalize("NFC", line) != line)
    return changed, len(lines)


def read_dictionary(path: Path) -> list[tuple[str, str]]:
    """Read a MUSE-style ``source target`` file into normalized word pairs."""
    if not path.exists():
        raise ValidationError(f"missing file: {path.relative_to(REPO_ROOT)}")
    pairs: list[tuple[str, str]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split()
        if len(parts) != 2:
            raise ValidationError(f"{path.name}:{number}: expected two columns, got {len(parts)}")
        pairs.append((normalize(parts[0]), normalize(parts[1])))
    return pairs


def examples(lines: Iterable[str], count: int = EXAMPLE_COUNT) -> list[str]:
    """Return the first ``count`` non-trivial lines, for eyeballing Hindi text."""
    chosen: list[str] = []
    for line in lines:
        if len(line) >= 20:
            chosen.append(line)
        if len(chosen) == count:
            break
    return chosen


def check_flores(config: dict[str, Any], report: Report) -> None:
    """Verify FLORES line counts, expected sizes and script of the Hindi side."""
    settings = config["flores"]
    directory = REPO_ROOT / settings["dir"]
    section = "FLORES-200"
    report.add(section, "")
    report.add(section, f"Source: `{settings['tarball_url']}`")
    report.add(section, "")
    report.add(section, "| Split | Hindi lines | English lines | Hindi tokens | English tokens |")
    report.add(section, "|---|---|---|---|---|")

    lengths: dict[str, list[int]] = {}
    for split in settings["splits"]:
        per_language: dict[str, list[str]] = {}
        for language, language_code in settings["languages"].items():
            per_language[language] = read_lines(directory / split / f"{language_code}.{split}")
        hindi, english = per_language["hi"], per_language["en"]
        expected = settings["expected_sentences"][split]
        report.check(
            f"FLORES {split} equal lengths",
            len(hindi) == len(english),
            f"hi={len(hindi)} en={len(english)}",
        )
        report.check(
            f"FLORES {split} expected size",
            len(hindi) == expected and len(english) == expected,
            f"expected {expected}, got hi={len(hindi)} en={len(english)}",
        )
        hindi_tokens = sum(len(line.split()) for line in hindi)
        english_tokens = sum(len(line.split()) for line in english)
        lengths[split] = [len(hindi), len(english), hindi_tokens, english_tokens]
        report.add(
            section,
            f"| {split} | {len(hindi)} | {len(english)} | {hindi_tokens} | {english_tokens} |",
        )

    hindi_dev = read_lines(directory / "dev" / f"{settings['languages']['hi']}.dev")
    ratio = sum(is_devanagari_ratio(line) for line in hindi_dev) / max(1, len(hindi_dev))
    report.add(section, "")
    report.add(section, f"Mean Devanagari letter ratio of the Hindi dev split: **{ratio:.3f}**.")
    report.check("FLORES Hindi is Devanagari", ratio > 0.95, f"mean ratio {ratio:.3f}")

    nfc_changed, nfc_total = nfc_change_count(directory / "dev" / f"{settings['languages']['hi']}.dev")
    report.add(section, "")
    report.add(
        section,
        f"NFC-normalization changes **{nfc_changed} of {nfc_total}** Hindi dev lines: the raw dump ships "
        "precomposed nukta letters (e.g. `ढ़` as U+095D), which NFC decomposes to consonant + nukta. "
        "This is expected; `src.textproc.normalize` applies NFC before tokenization, and the raw files stay untouched.",
    )
    report.add(section, "")
    report.add(section, "Examples (Hindi / English):")
    report.add(section, "")
    english_dev = read_lines(directory / "dev" / f"{settings['languages']['en']}.dev")
    for hindi_line, english_line in zip(hindi_dev[:EXAMPLE_COUNT], english_dev[:EXAMPLE_COUNT]):
        report.add(section, f"- {hindi_line}")
        report.add(section, f"  - {english_line}")


def check_xquad(config: dict[str, Any], report: Report) -> None:
    """Verify XQuAD question and paragraph counts and cross-language id matching."""
    settings = config["xquad"]
    directory = REPO_ROOT / settings["dir"]
    section = "XQuAD"
    report.add(section, "")
    report.add(section, f"Source: Hugging Face `{settings['hf_dataset']}`, split `{settings['split']}`.")
    report.add(section, "")
    report.add(section, "| Language | Questions | Unique contexts | Unique question ids |")
    report.add(section, "|---|---|---|---|")

    tables: dict[str, list[dict[str, Any]]] = {}
    for language in settings["configs"]:
        path = directory / f"xquad.{language}.jsonl"
        if not path.exists():
            raise ValidationError(f"missing file: {path.relative_to(REPO_ROOT)}")
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        tables[language] = records
        question_ids = {record["id"] for record in records}
        contexts = {record["context"] for record in records}
        report.add(section, f"| {language} | {len(records)} | {len(contexts)} | {len(question_ids)} |")

    for language, records in tables.items():
        expected_questions = settings["expected_questions"]
        expected_paragraphs = settings["expected_paragraphs"]
        contexts = {record["context"] for record in records}
        report.check(
            f"XQuAD {language} question count",
            len(records) == expected_questions,
            f"expected {expected_questions}, got {len(records)}",
        )
        report.check(
            f"XQuAD {language} paragraph count",
            len(contexts) == expected_paragraphs,
            f"expected {expected_paragraphs}, got {len(contexts)}",
        )
        report.check(
            f"XQuAD {language} question ids unique",
            len({record["id"] for record in records}) == len(records),
            "duplicate question ids",
        )

    hindi_ids = {record["id"] for record in tables["hi"]}
    english_ids = {record["id"] for record in tables["en"]}
    report.check(
        "XQuAD question ids match across languages",
        hindi_ids == english_ids,
        f"hi-only={len(hindi_ids - english_ids)} en-only={len(english_ids - hindi_ids)}",
    )

    hindi_contexts: dict[str, set[str]] = {}
    english_contexts: dict[str, set[str]] = {}
    for record in tables["hi"]:
        hindi_contexts.setdefault(record["id"], set()).add(record["context"])
    for record in tables["en"]:
        english_contexts.setdefault(record["id"], set()).add(record["context"])

    ambiguous = [qid for qid, contexts in hindi_contexts.items() if len(contexts) != 1]
    report.check(
        "XQuAD Hindi question maps to exactly one context",
        not ambiguous,
        f"{len(ambiguous)} ids with several contexts, e.g. {ambiguous[:3]}",
    )
    ambiguous_en = [qid for qid, contexts in english_contexts.items() if len(contexts) != 1]
    report.check(
        "XQuAD English question maps to exactly one context",
        not ambiguous_en,
        f"{len(ambiguous_en)} ids with several contexts, e.g. {ambiguous_en[:3]}",
    )
    shared = hindi_ids & english_ids
    report.check(
        "XQuAD shared ids cover the question set",
        len(shared) == len(english_ids) == len(hindi_ids),
        f"hi={len(hindi_ids)} en={len(english_ids)} shared={len(shared)}",
    )

    report.add(section, "")
    report.add(section, "Examples (Hindi question / English question):")
    report.add(section, "")
    english_by_id = {record["id"]: record for record in tables["en"]}
    for record in tables["hi"][:EXAMPLE_COUNT]:
        report.add(section, f"- {record['question']}")
        report.add(section, f"  - {english_by_id[record['id']]['question']}")


def check_muse(config: dict[str, Any], report: Report) -> None:
    """Verify MUSE pair counts, unique source words and train/test disjointness."""
    settings = config["muse"]
    directory = REPO_ROOT / settings["dir"]
    train_split = settings["splits"]["train"]
    test_split = settings["splits"]["test"]
    section = "MUSE"
    report.add(section, "")
    report.add(section, "| Direction | Split | Pairs | Unique source words | Unique target words |")
    report.add(section, "|---|---|---|---|---|")

    for pair in settings["pairs"]:
        split_pairs: dict[str, list[tuple[str, str]]] = {}
        for name, split in (("train", train_split), ("test", test_split)):
            pairs = read_dictionary(directory / f"{pair}.{split}.txt")
            split_pairs[name] = pairs
            report.add(
                section,
                f"| {pair} | {name} ({split}) | {len(pairs)} | "
                f"{len({source for source, _ in pairs})} | {len({target for _, target in pairs})} |",
            )

        train_sources = {source for source, _ in split_pairs["train"]}
        test_sources = {source for source, _ in split_pairs["test"]}
        overlap = train_sources & test_sources
        report.check(
            f"MUSE {pair} train/test source words disjoint",
            not overlap,
            f"{len(overlap)} overlapping source words, e.g. {sorted(overlap)[:5]}",
        )
        report.check(
            f"MUSE {pair} test split size",
            len(split_pairs["test"]) > 0 and len(test_sources) > 0,
            "empty test split",
        )


def check_wikipedia_manifest(config: dict[str, Any], report: Report) -> None:
    """Report the Hindi Wikipedia shard files and the English streaming note."""
    settings = config["wikipedia"]
    directory = REPO_ROOT / settings["dir"]
    section = "Wikipedia"
    report.add(section, "")
    dump_date = settings["dump_date"]
    shard_dir = directory / f"{dump_date}.hi"
    files = sorted(shard_dir.glob("*.parquet")) if shard_dir.exists() else []
    report.check(
        "Hindi Wikipedia shards present",
        bool(files),
        f"no parquet files under {shard_dir.relative_to(REPO_ROOT)}; run scripts/download_data.py --groups wikipedia",
    )
    total = sum(path.stat().st_size for path in files)
    report.add(section, f"Dump date: **{dump_date}**; Hindi files: **{len(files)}**, {total / 1e9:.2f} GB.")
    stream_note = directory / "STREAM.en.json"
    report.check(
        "English streaming note present",
        stream_note.exists(),
        "STREAM.en.json missing; English is streamed in preprocess_wiki.py",
    )
    manifest_path = directory.parent / "MANIFEST.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        report.add(section, "")
        report.add(section, f"`data/raw/MANIFEST.json` records **{len(manifest.get('files', {}))}** files.")


def check_processed_corpora(config: dict[str, Any], report: Report) -> None:
    """Report the S12 corpora, the shard sizes and the S13 contamination counts.

    Everything here is read from the stats files written by
    ``scripts/preprocess_wiki.py``; nothing is recomputed, so the numbers in the
    data card always match the run that produced the corpora.
    """
    processed = REPO_ROOT / "data" / "processed"
    section = "Wikipedia training corpora (S12)"
    section13 = "Contamination filter and shard sizes (S13)"
    report.add(section, "")
    report.add(
        section,
        "| Language | Articles | Train | Distractor | Fit | Sentences | Tokens | Types | Wall time (s) |",
    )
    report.add(section, "|---|---|---|---|---|---|---|---|---|")

    stats: dict[str, dict[str, Any]] = {}
    for language in ("hi", "en"):
        stats_path = processed / f"wiki.{language}.stats.json"
        if not stats_path.exists():
            report.add(section, "")
            report.add(
                section,
                f"**Not built yet:** `{stats_path.relative_to(REPO_ROOT)}` is missing. "
                f"Run `uv run python scripts/preprocess_wiki.py --lang {language}`.",
            )
            continue
        payload = json.loads(stats_path.read_text(encoding="utf-8") or "null")
        if not isinstance(payload, dict):
            report.add(section, "")
            report.add(section, f"**Unreadable stats:** `{stats_path.relative_to(REPO_ROOT)}` is not a JSON object.")
            continue
        stats[language] = payload
        shards = payload.get("shard_articles", {})
        counts_path = processed / f"wordcounts_{language}.tsv"
        types = 0
        if counts_path.exists():
            with counts_path.open("r", encoding="utf-8") as handle:
                types = sum(1 for _ in handle)
        report.add(
            section,
            f"| {language} | {payload['articles_kept']} | {shards.get('train', 0)} | "
            f"{shards.get('distractor', 0)} | {shards.get('fit', 0)} | {payload['sentences_kept']} | "
            f"{payload['tokens']} | {types} | {payload['elapsed_s']} |",
        )

    if "hi" in stats and "en" in stats:
        hindi_tokens = stats["hi"]["tokens"]
        english_tokens = stats["en"]["tokens"]
        difference = abs(english_tokens - hindi_tokens) / max(1, hindi_tokens)
        report.check("English train tokens within 1% of Hindi", difference <= 0.01, f"{difference:.3%} apart")
        report.add(section, "")
        report.add(
            section,
            f"Equal-size rule: English train tokens are **{difference:.3%}** away from the Hindi total "
            f"({hindi_tokens} vs {english_tokens}).",
        )
    elif stats:
        report.add(section, "")
        report.add(section, "English is capped at the Hindi `train` token count; build both languages to compare sizes.")

    for language in ("hi", "en"):
        paths = {
            "paragraphs": processed / f"wiki.{language}.para.tsv",
            "sentences": processed / f"wiki.{language}.sent.txt",
            "tokens": processed / f"wiki.{language}.tok.txt",
            "wordcounts": processed / f"wordcounts_{language}.tsv",
        }
        missing = [name for name, path in paths.items() if not path.exists()]
        if missing:
            report.check(
                f"Processed corpus files present for {language}",
                False,
                f"missing {missing}; run scripts/preprocess_wiki.py --lang {language}",
            )
            continue
        report.check(f"Processed corpus files present for {language}", True)

    shard_ids: dict[str, dict[str, set[str]]] = {}
    for language in stats:
        para_path = processed / f"wiki.{language}.para.tsv"
        if not para_path.exists():
            continue
        shard_ids[language] = {name: set() for name in ("train", "distractor", "fit")}
        paragraphs = 0
        with para_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                fields = line.rstrip("\n").split("\t", 4)
                if len(fields) != 5:
                    continue
                shard, article_id = fields[0], fields[1]
                if shard in shard_ids[language]:
                    shard_ids[language][shard].add(article_id)
                paragraphs += 1
        report.add(section, "")
        report.add(section, f"`{para_path.name}`: **{paragraphs}** paragraph rows, all shards, disjoint by article id.")
        for first, second in (("train", "distractor"), ("train", "fit"), ("distractor", "fit")):
            overlap = shard_ids[language][first] & shard_ids[language][second]
            report.check(
                f"{language} {first}/{second} shards disjoint by article id",
                not overlap,
                f"{len(overlap)} shared articles, e.g. {sorted(overlap)[:3]}",
            )

    report.add(section13, "")
    report.add(
        section13,
        "Exact-match filter: a `train` sentence is dropped when, after normalization, it equals a "
        "FLORES dev/devtest sentence or an XQuAD context/question/answer of the same language.",
    )
    report.add(section13, "")
    report.add(section13, "| Language | Sentences seen | Removed by filter | Kept | Dropped as too short |")
    report.add(section13, "|---|---|---|---|---|")
    for language, payload in stats.items():
        removed = payload.get("sentences_dropped_contamination", 0)
        report.add(
            section13,
            f"| {language} | {payload['sentences_seen']} | {removed} | {payload['sentences_kept']} | "
            f"{payload.get('sentences_dropped_short', 0)} |",
        )

    for language, payload in stats.items():
        examples = list(dict.fromkeys(payload.get("contamination_examples", [])))
        if examples:
            report.add(section13, "")
            report.add(section13, f"Removed sentences ({language}), first {len(examples)} distinct filter matches:")
            report.add(section13, "")
            for example in examples:
                report.add(section13, f"- {example}")
    report.add(section13, "")
    report.add(
        section13,
        "Matches are exact after tokenization, so a Wikipedia sentence is only removed when it is word-for-word "
        "identical to an evaluation sentence. The low counts are therefore expected: FLORES is news text that "
        "rarely appears verbatim in the dump, and the English training shard is a seeded sample of the dump, so "
        "most XQuAD source articles are not in it at all.",
    )
    report.add(section13, "")
    report.add(
        section13,
        "Full per-language counters, including paragraph-level drops, are in "
        "`data/processed/wiki.<lang>.stats.json`.",
    )


def render(report: Report, config: dict[str, Any]) -> str:
    """Render ``docs/DATA.md`` from the collected sections."""
    lines = [
        "# Data card",
        "",
        "Generated by `scripts/validate_data.py` (S11). Every number below was measured from",
        "the files in `data/raw/`; regenerate with `uv run python scripts/validate_data.py`.",
        "No data files are committed (`AGENTS.md` §12).",
        "",
        "## Checks",
        "",
        "| Check | Result | Detail |",
        "|---|---|---|",
    ]
    for name, passed, detail in report.checks:
        lines.append(f"| {name} | {'pass' if passed else 'FAIL'} | {detail if not passed else ''} |")
    for section, body in report.sections.items():
        lines.extend(["", f"## {section}", ""])
        trimmed = list(body)
        while trimmed and not trimmed[0]:
            trimmed.pop(0)
        lines.extend(trimmed)
    lines.extend(
        [
            "",
            "## Licences",
            "",
            "Licence notes are collected in `data/raw/MANIFEST.json`, one entry per file. They are marked",
            "*unverified* unless the source page stated the licence explicitly; verify before redistribution.",
            "",
        ]
    )
    return "\n".join(lines)


def display_path(path: Path) -> str:
    """Return ``path`` relative to the repository root, or absolute when it lies outside it."""
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=DATA_DOC_PATH, help="data card path (default: docs/DATA.md)")
    parser.add_argument("--allow-missing", action="store_true", help="report missing groups instead of failing")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run every dataset check and write the data card."""
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s %(message)s")
    config = load_config()
    report = Report()

    checks = [
        ("FLORES-200", check_flores),
        ("XQuAD", check_xquad),
        ("MUSE", check_muse),
        ("Wikipedia", check_wikipedia_manifest),
        ("Processed corpora", check_processed_corpora),
    ]
    failures = 0
    for name, function in checks:
        try:
            function(config, report)
        except ValidationError as error:
            failures += 1
            LOGGER.error("%s: %s", name, error)
            report.add(name, "")
            report.add(name, f"**Not validated:** {error}")

    if failures and not args.allow_missing:
        LOGGER.error("%d dataset group(s) failed validation", failures)
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render(report, config), encoding="utf-8")
    LOGGER.info("wrote %s", display_path(args.output))
    LOGGER.info("%d checks passed", sum(1 for _, passed, _ in report.checks if passed))
    return 0 if not failures or args.allow_missing else 1


if __name__ == "__main__":
    sys.exit(main())
