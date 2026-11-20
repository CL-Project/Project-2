"""T3: Hindi question to English paragraph retrieval on XQuAD (S27, S28).

XQuAD ships as two parallel files, one per language, with 1,190 questions over
240 paragraphs.  The paragraph tables are built from the **question id**, never
by row position: a Hindi question is linked to its English paragraph through the
id it shares with its English counterpart (PLAN S27).  Paragraph ids are assigned
to the sorted English contexts, so the numbering is deterministic and
independent of file order.

The retrieval pool is the 240 XQuAD paragraphs plus Wikipedia distractors from
the disjoint `distractor` shard (S28).  Both pools are read from
``data/processed/t3_pool_<lang>.jsonl``; a gold paragraph keeps the same ``pid``
in both languages, which is what lets a Hindi question be scored against an
English paragraph.

Scoring is plain cosine on L2-normalized vectors (the plan specifies cosine for
T3); BM25 goes through :mod:`src.rankers`.  Metrics are Recall@1, Recall@5 and
MRR over the whole pool.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from src.tasks.parallel import RetrievalResult

LOGGER = logging.getLogger("xquad")

T3_METRICS = {"Recall@1": "top1", "Recall@5": "top5", "MRR": "rr"}
GOLD_PID_PREFIX = "xq-"


@dataclass(frozen=True)
class Paragraph:
    """One XQuAD paragraph in both languages, keyed by a deterministic id."""

    paragraph_id: int
    text_en: str
    text_hi: str

    @property
    def pid(self) -> str:
        """Return the paragraph id used in the pool files (``xq-000``)."""
        return f"{GOLD_PID_PREFIX}{self.paragraph_id:03d}"


@dataclass(frozen=True)
class Question:
    """One XQuAD question in both languages and its gold paragraph."""

    qid: str
    question_hi: str
    question_en: str
    paragraph_id: int


@dataclass(frozen=True)
class XQuadData:
    """The aligned XQuAD question and paragraph tables."""

    paragraphs: list[Paragraph]
    questions: list[Question]

    @property
    def n_questions(self) -> int:
        """Number of questions."""
        return len(self.questions)

    @property
    def n_paragraphs(self) -> int:
        """Number of unique paragraphs."""
        return len(self.paragraphs)


@dataclass(frozen=True)
class PoolParagraph:
    """One candidate paragraph of a T3 pool."""

    pid: str
    text: str
    is_gold_candidate: bool


@dataclass(frozen=True)
class T3Pool:
    """Queries, candidate paragraphs and gold indices for one setting."""

    setting: str
    query_lang: str
    doc_lang: str
    query_ids: list[str]
    query_texts: list[str]
    candidate_ids: list[str]
    candidate_texts: list[str]
    gold_indices: list[list[int]]

    @property
    def n_queries(self) -> int:
        """Number of questions."""
        return len(self.query_ids)

    @property
    def n_candidates(self) -> int:
        """Size of the paragraph pool."""
        return len(self.candidate_ids)


def parse_setting(setting: str) -> tuple[str, str]:
    """Split ``"hi-en"`` into ``("hi", "en")``, validating the language codes.

    T3 evaluates the four combinations of Hindi and English on both sides, so
    same-language settings are valid here (unlike T1 and T2).
    """
    parts = setting.split("-")
    if len(parts) != 2 or any(part not in {"hi", "en"} for part in parts):
        raise ValueError(f"invalid setting {setting!r}; expected one of hi-en, en-hi, hi-hi, en-en")
    return parts[0], parts[1]


def read_jsonl(path: Path | str) -> list[dict]:
    """Read a JSONL file, dropping blank lines."""
    records: list[dict] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON ({error})") from error
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            records.append(record)
    if not records:
        raise ValueError(f"{path} contains no records")
    return records


def _index_questions(records: Sequence[dict], language: str) -> dict[str, dict]:
    """Index one language's records by question id, rejecting duplicates."""
    indexed: dict[str, dict] = {}
    for record in records:
        qid = str(record["id"])
        if qid in indexed:
            raise ValueError(f"{language}: duplicate question id {qid!r}")
        for field in ("question", "context"):
            value = record.get(field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{language}: question {qid!r} has an empty {field}")
        indexed[qid] = record
    return indexed


def load_xquad(
    raw_dir: Path | str,
    *,
    expected_questions: int | None = None,
    expected_paragraphs: int | None = None,
) -> XQuadData:
    """Load ``xquad.hi.jsonl`` and ``xquad.en.jsonl`` and align them by question id.

    Args:
        raw_dir: ``data/raw/xquad``.
        expected_questions: when given, assert the question count.
        expected_paragraphs: when given, assert the unique-paragraph count.

    The gold paragraph of a question is the paragraph its id maps to in **both**
    languages.  Two English contexts claiming the same Hindi paragraph (or the
    reverse) is a data error and raises.
    """
    root = Path(raw_dir)
    hindi = _index_questions(read_jsonl(root / "xquad.hi.jsonl"), "hi")
    english = _index_questions(read_jsonl(root / "xquad.en.jsonl"), "en")

    if set(hindi) != set(english):
        only_hi = sorted(set(hindi) - set(english))
        only_en = sorted(set(english) - set(hindi))
        raise ValueError(
            f"question ids differ across languages: {len(only_hi)} Hindi-only, "
            f"{len(only_en)} English-only, e.g. {only_hi[:3]} / {only_en[:3]}"
        )

    context_en_to_hi: dict[str, str] = {}
    for qid, record in hindi.items():
        context_en = english[qid]["context"]
        context_hi = record["context"]
        previous = context_en_to_hi.get(context_en)
        if previous is not None and previous != context_hi:
            raise ValueError(f"English paragraph {context_en[:60]!r} maps to two Hindi paragraphs")
        context_en_to_hi[context_en] = context_hi

    english_contexts = sorted(context_en_to_hi)
    paragraph_of_context = {context: index for index, context in enumerate(english_contexts)}
    paragraphs = [
        Paragraph(
            paragraph_id=paragraph_of_context[context],
            text_en=context,
            text_hi=context_en_to_hi[context],
        )
        for context in english_contexts
    ]

    questions = [
        Question(
            qid=qid,
            question_hi=hindi[qid]["question"],
            question_en=english[qid]["question"],
            paragraph_id=paragraph_of_context[english[qid]["context"]],
        )
        for qid in sorted(hindi)
    ]

    if expected_questions is not None and len(questions) != int(expected_questions):
        raise ValueError(f"{len(questions)} questions, expected {expected_questions}")
    if expected_paragraphs is not None and len(paragraphs) != int(expected_paragraphs):
        raise ValueError(f"{len(paragraphs)} unique paragraphs, expected {expected_paragraphs}")

    LOGGER.info(
        "XQuAD: %d questions, %d unique paragraphs, every id present in both languages",
        len(questions),
        len(paragraphs),
    )
    return XQuadData(paragraphs=paragraphs, questions=questions)


def read_pool(path: Path | str) -> list[PoolParagraph]:
    """Read one ``t3_pool_<lang>.jsonl`` file, validating its records."""
    paragraphs: list[PoolParagraph] = []
    seen: set[str] = set()
    for record in read_jsonl(path):
        pid = str(record.get("pid", ""))
        text = record.get("text")
        is_gold = record.get("is_gold_candidate")
        if not pid or not isinstance(text, str) or not text.strip():
            raise ValueError(f"{path}: record without a pid or text: {record!r}")
        if not isinstance(is_gold, bool):
            raise ValueError(f"{path}: is_gold_candidate must be a bool for {pid!r}")
        if pid in seen:
            raise ValueError(f"{path}: duplicate pid {pid!r}")
        seen.add(pid)
        paragraphs.append(PoolParagraph(pid=pid, text=text, is_gold_candidate=is_gold))
    if not paragraphs:
        raise ValueError(f"{path} contains no paragraphs")
    return paragraphs


def gold_pids(pool: Sequence[PoolParagraph]) -> list[str]:
    """Return the gold-candidate pids of a pool, in file order."""
    return [paragraph.pid for paragraph in pool if paragraph.is_gold_candidate]


def build_pool(
    data: XQuadData,
    *,
    setting: str,
    pools: dict[str, Sequence[PoolParagraph]],
) -> T3Pool:
    """Build the query and candidate tables for one T3 setting.

    Args:
        data: the aligned XQuAD tables.
        setting: ``query_lang-doc_lang``, one of ``hi-en``, ``en-hi``, ``hi-hi``,
            ``en-en``.
        pools: ``{"hi": pool_hi, "en": pool_en}``, read from
            ``data/processed/t3_pool_<lang>.jsonl``.

    Returns:
        A :class:`T3Pool` whose gold index of question *i* is the position of its
        paragraph in the **document-language** pool, so a Hindi question scored
        against English paragraphs keeps the same gold paragraph.
    """
    query_lang, doc_lang = parse_setting(setting)
    if query_lang not in pools or doc_lang not in pools:
        raise ValueError(f"setting {setting!r} needs pools for {sorted({query_lang, doc_lang})}")

    candidates = list(pools[doc_lang])
    position_of_pid = {paragraph.pid: index for index, paragraph in enumerate(candidates)}
    if len(position_of_pid) != len(candidates):
        raise ValueError("the document pool has duplicate pids")

    expected_gold = {paragraph.pid for paragraph in candidates if paragraph.is_gold_candidate}
    if len(expected_gold) != data.n_paragraphs:
        raise ValueError(
            f"{doc_lang} pool has {len(expected_gold)} gold paragraphs, "
            f"XQuAD has {data.n_paragraphs}; rebuild the pool"
        )

    query_ids: list[str] = []
    query_texts: list[str] = []
    gold_indices: list[list[int]] = []
    for question in data.questions:
        pid = f"{GOLD_PID_PREFIX}{question.paragraph_id:03d}"
        if pid not in position_of_pid:
            raise ValueError(f"question {question.qid!r}: gold paragraph {pid!r} is not in the pool")
        query_ids.append(question.qid)
        query_texts.append(question.question_hi if query_lang == "hi" else question.question_en)
        gold_indices.append([position_of_pid[pid]])

    LOGGER.info(
        "T3 %s: %d queries, %d candidates (%d gold, %d distractors)",
        setting,
        len(query_ids),
        len(candidates),
        len(expected_gold),
        len(candidates) - len(expected_gold),
    )
    return T3Pool(
        setting=setting,
        query_lang=query_lang,
        doc_lang=doc_lang,
        query_ids=query_ids,
        query_texts=query_texts,
        candidate_ids=[paragraph.pid for paragraph in candidates],
        candidate_texts=[paragraph.text for paragraph in candidates],
        gold_indices=gold_indices,
    )


def per_item_frame(pool: T3Pool, result: RetrievalResult) -> pd.DataFrame:
    """Return the per-item table for one T3 run (schema in ``src/results_log.py``)."""
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


def token_shingles(tokens: Sequence[str], size: int) -> set[tuple[str, ...]]:
    """Return the set of ``size``-token shingles of ``tokens``.

    A text shorter than ``size`` contributes its whole token tuple: it can only
    match a reference that contains it verbatim.
    """
    if size < 1:
        raise ValueError("size must be >= 1")
    if not tokens:
        return set()
    if len(tokens) < size:
        return {tuple(tokens)}
    return {tuple(tokens[start : start + size]) for start in range(len(tokens) - size + 1)}


def overlaps_reference(
    tokens: Sequence[str],
    reference_shingles: set[tuple[str, ...]],
    *,
    size: int,
) -> bool:
    """Return whether ``tokens`` shares a single ``size``-token shingle with the reference."""
    return bool(token_shingles(tokens, size) & reference_shingles)


def reference_shingles(texts: Iterable[str], lang: str, *, size: int) -> set[tuple[str, ...]]:
    """Return every ``size``-token shingle of ``texts`` for the overlap filter.

    Tokens come from the shared tokenizer, so the comparison is on words after
    NFC normalization and punctuation removal rather than on raw strings.
    """
    from src.textproc import tokenize

    shingles: set[tuple[str, ...]] = set()
    for text in texts:
        shingles |= token_shingles(tokenize(text, lang), size)
    return shingles


def load_pools(processed_dir: Path | str, *, prefix: str = "t3_pool") -> dict[str, list[PoolParagraph]]:
    """Load ``<prefix>_hi.jsonl`` and ``<prefix>_en.jsonl`` from ``processed_dir``."""
    root = Path(processed_dir)
    pools: dict[str, list[PoolParagraph]] = {}
    for language in ("hi", "en"):
        path = root / f"{prefix}_{language}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"{path} is missing; run scripts/build_t3_pool.py first")
        pools[language] = read_pool(path)
    return pools


def write_pool(path: Path | str, paragraphs: Iterable[PoolParagraph]) -> Path:
    """Write a pool file as JSONL with one object per paragraph."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    records = [
        {"pid": paragraph.pid, "text": paragraph.text, "is_gold_candidate": paragraph.is_gold_candidate}
        for paragraph in paragraphs
    ]
    temporary = target.with_suffix(target.suffix + ".part")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(target)
    return target
