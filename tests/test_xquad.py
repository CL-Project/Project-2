"""Tests for XQuAD loading and paragraph alignment (S27), offline and tiny.

The alignment rule is the one thing that must not slip: a Hindi question has to
reach the *same* paragraph as its English counterpart, and it must do so through
the question id, not through row order.  The fixtures here therefore shuffle the
English file deliberately.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.tasks import xquad
from src.tasks.xquad import (
    PoolParagraph,
    build_pool,
    load_xquad,
    parse_setting,
    per_item_frame,
    read_pool,
    write_pool,
)


def _record(qid: str, question: str, context: str) -> dict:
    return {"id": qid, "question": question, "context": context, "answers": {"text": ["x"], "answer_start": [0]}}


def _write_xquad(directory, *, english_order=None):
    """Write a three-question, two-paragraph XQuAD pair in ``directory``."""
    hindi = [
        _record("q1", "प्रश्न एक?", "हिंदी अनुच्छेद एक।"),
        _record("q2", "प्रश्न दो?", "हिंदी अनुच्छेद एक।"),
        _record("q3", "प्रश्न तीन?", "हिंदी अनुच्छेद दो।"),
    ]
    english = [
        _record("q1", "Question one?", "English paragraph one."),
        _record("q2", "Question two?", "English paragraph one."),
        _record("q3", "Question three?", "English paragraph two."),
    ]
    if english_order is not None:
        english = [english[index] for index in english_order]
    for name, records in (("xquad.hi.jsonl", hindi), ("xquad.en.jsonl", english)):
        path = directory / name
        path.write_text("".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records), encoding="utf-8")


def test_parse_setting_accepts_all_four_t3_settings():
    assert parse_setting("hi-en") == ("hi", "en")
    assert parse_setting("hi-hi") == ("hi", "hi")
    assert parse_setting("en-en") == ("en", "en")
    for bad in ("hi", "hi-fr", "hi-en-hi", ""):
        with pytest.raises(ValueError):
            parse_setting(bad)


def test_load_xquad_aligns_by_question_id_not_position(tmp_path):
    """The English file is in a different order; alignment must still hold."""
    _write_xquad(tmp_path, english_order=[2, 0, 1])
    data = load_xquad(tmp_path)

    assert data.n_questions == 3
    assert data.n_paragraphs == 2
    by_qid = {question.qid: question for question in data.questions}
    assert by_qid["q1"].question_hi == "प्रश्न एक?"
    assert by_qid["q1"].question_en == "Question one?"
    assert by_qid["q1"].paragraph_id == by_qid["q2"].paragraph_id
    assert by_qid["q3"].paragraph_id != by_qid["q1"].paragraph_id

    paragraphs = {paragraph.paragraph_id: paragraph for paragraph in data.paragraphs}
    gold = paragraphs[by_qid["q1"].paragraph_id]
    assert gold.text_en == "English paragraph one."
    assert gold.text_hi == "हिंदी अनुच्छेद एक।"
    assert gold.pid == "xq-000"


def test_load_xquad_paragraph_ids_follow_sorted_english_contexts(tmp_path):
    """Ids are deterministic: sorted English contexts, whatever the file order."""
    _write_xquad(tmp_path, english_order=[2, 0, 1])
    data = load_xquad(tmp_path)
    assert [paragraph.text_en for paragraph in data.paragraphs] == [
        "English paragraph one.",
        "English paragraph two.",
    ]


def test_load_xquad_rejects_mismatched_id_sets(tmp_path):
    _write_xquad(tmp_path)
    path = tmp_path / "xquad.en.jsonl"
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    records[2]["id"] = "q4"
    path.write_text("".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records), encoding="utf-8")
    with pytest.raises(ValueError, match="question ids differ"):
        load_xquad(tmp_path)


def test_load_xquad_rejects_duplicate_question_ids(tmp_path):
    _write_xquad(tmp_path)
    path = tmp_path / "xquad.hi.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    lines[2] = lines[0]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate question id"):
        load_xquad(tmp_path)


def test_load_xquad_enforces_expected_counts(tmp_path):
    _write_xquad(tmp_path)
    with pytest.raises(ValueError, match="questions, expected"):
        load_xquad(tmp_path, expected_questions=1190)
    with pytest.raises(ValueError, match="unique paragraphs, expected"):
        load_xquad(tmp_path, expected_paragraphs=240)


def test_load_xquad_rejects_a_paragraph_mapping_to_two_contexts(tmp_path):
    _write_xquad(tmp_path)
    hindi_path = tmp_path / "xquad.hi.jsonl"
    english_path = tmp_path / "xquad.en.jsonl"
    hindi = [json.loads(line) for line in hindi_path.read_text(encoding="utf-8").splitlines()]
    english = [json.loads(line) for line in english_path.read_text(encoding="utf-8").splitlines()]
    hindi[1]["context"] = "एक अलग अनुच्छेद।"
    hindi_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in hindi), encoding="utf-8")
    english_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in english), encoding="utf-8")
    with pytest.raises(ValueError, match="maps to two Hindi paragraphs"):
        load_xquad(tmp_path)


def _xy_pool(gold_count: int = 2, distractors: int = 3, lang: str = "en") -> list[PoolParagraph]:
    pool = [
        PoolParagraph(pid=f"xq-{index:03d}", text=f"gold {lang} paragraph {index}", is_gold_candidate=True)
        for index in range(gold_count)
    ]
    pool.extend(
        PoolParagraph(pid=f"dist-{index:03d}", text=f"distractor {lang} {index}", is_gold_candidate=False)
        for index in range(distractors)
    )
    return pool


def _data(tmp_path, queries: int = 3):
    """Two gold paragraphs; ``queries`` questions spread over them."""
    _write_xquad(tmp_path)
    data = load_xquad(tmp_path)
    if queries == 3:
        return data
    raise AssertionError("helper only documents the three-question fixture")


def test_build_pool_maps_gold_to_the_document_language(tmp_path):
    data = _data(tmp_path)
    pools = {"hi": _xy_pool(lang="hi"), "en": _xy_pool(lang="en")}
    pool = build_pool(data, setting="hi-en", pools=pools)

    assert pool.n_queries == 3
    assert pool.n_candidates == 5
    assert pool.query_lang == "hi" and pool.doc_lang == "en"
    assert pool.query_texts[0] == "प्रश्न एक?"
    assert pool.candidate_texts[0].startswith("gold en")
    for question, gold in zip(data.questions, pool.gold_indices):
        assert pool.candidate_ids[gold[0]] == f"xq-{question.paragraph_id:03d}"


def test_build_pool_uses_the_query_language_text(tmp_path):
    data = _data(tmp_path)
    pools = {"hi": _xy_pool(lang="hi"), "en": _xy_pool(lang="en")}
    english = build_pool(data, setting="en-hi", pools=pools)
    assert english.query_texts[0] == "Question one?"
    assert english.query_lang == "en" and english.doc_lang == "hi"
    assert english.candidate_texts[0].startswith("gold hi")


def test_build_pool_same_language_selects_one_pool(tmp_path):
    data = _data(tmp_path)
    pools = {"hi": _xy_pool(lang="hi"), "en": _xy_pool(lang="en")}
    pool = build_pool(data, setting="hi-hi", pools=pools)
    assert pool.query_lang == pool.doc_lang == "hi"
    assert all(text.startswith("gold hi") for text in pool.candidate_texts[:2])


def test_build_pool_rejects_a_pool_missing_a_gold_paragraph(tmp_path):
    data = _data(tmp_path)
    short = _xy_pool(gold_count=1)
    pools = {"hi": short, "en": short}
    with pytest.raises(ValueError, match="rebuild the pool"):
        build_pool(data, setting="hi-en", pools=pools)


def test_build_pool_rejects_duplicate_pids(tmp_path):
    data = _data(tmp_path)
    pool = _xy_pool()
    duplicated = pool + [PoolParagraph(pid=pool[0].pid, text="clone", is_gold_candidate=False)]
    with pytest.raises(ValueError, match="duplicate pids"):
        build_pool(data, setting="hi-en", pools={"hi": duplicated, "en": duplicated})


def test_pool_round_trip(tmp_path):
    pool = _xy_pool()
    path = write_pool(tmp_path / "t3_pool_en.jsonl", pool)
    assert path.exists()
    assert read_pool(path) == pool
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == len(pool)
    assert json.loads(lines[0])["is_gold_candidate"] is True


def test_read_pool_validates_records(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps({"pid": "x", "text": "", "is_gold_candidate": True}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="without a pid or text"):
        read_pool(path)

    path.write_text(json.dumps({"pid": "x", "text": "t", "is_gold_candidate": "yes"}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must be a bool"):
        read_pool(path)

    path.write_text(
        json.dumps({"pid": "x", "text": "t", "is_gold_candidate": True}) + "\n"
        + json.dumps({"pid": "x", "text": "u", "is_gold_candidate": False}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate pid"):
        read_pool(path)


def test_per_item_frame_schema_and_gold_ids(tmp_path):
    data = _data(tmp_path)
    pools = {"hi": _xy_pool(lang="hi"), "en": _xy_pool(lang="en")}
    pool = build_pool(data, setting="hi-en", pools=pools)
    from src.tasks.parallel import RetrievalResult

    result = RetrievalResult(
        ranks=np.array([1, 4, 10], dtype=np.int64),
        best_index=np.zeros(3, dtype=np.int64),
        best_score=np.ones(3, dtype=np.float32),
        gold_score=np.ones(3, dtype=np.float32),
        n_candidates=pool.n_candidates,
    )
    frame = per_item_frame(pool, result)
    assert list(frame.columns) == [
        "query_id",
        "gold_ids",
        "rank_of_best_gold",
        "top1",
        "top5",
        "rr",
        "n_candidates",
    ]
    assert isinstance(frame, pd.DataFrame)
    assert frame["top1"].tolist() == [True, False, False]
    assert frame["top5"].tolist() == [True, True, False]
    assert frame["gold_ids"].iloc[0] == [pool.candidate_ids[pool.gold_indices[0][0]]]
    np.testing.assert_allclose(frame["rr"].to_numpy(), [1.0, 0.25, 0.1])


def test_per_item_frame_rejects_wrong_length(tmp_path):
    data = _data(tmp_path)
    pools = {"hi": _xy_pool(lang="hi"), "en": _xy_pool(lang="en")}
    pool = build_pool(data, setting="hi-en", pools=pools)
    from src.tasks.parallel import RetrievalResult

    result = RetrievalResult(
        ranks=np.ones(2, dtype=np.int64),
        best_index=np.zeros(2, dtype=np.int64),
        best_score=np.ones(2, dtype=np.float32),
        gold_score=np.ones(2, dtype=np.float32),
        n_candidates=pool.n_candidates,
    )
    with pytest.raises(ValueError):
        per_item_frame(pool, result)


def test_real_xquad_alignment():
    """On the downloaded files: 1,190 questions over 240 paragraphs, ids aligned.

    Skipped on a fresh clone (``data/`` is git-ignored).
    """
    from pathlib import Path

    raw = Path(__file__).resolve().parents[1] / "data" / "raw" / "xquad"
    if not (raw / "xquad.hi.jsonl").exists():
        pytest.skip("XQuAD is not downloaded")
    data = load_xquad(raw, expected_questions=1190, expected_paragraphs=240)
    assert data.n_questions == 1190 and data.n_paragraphs == 240
    assert len({question.qid for question in data.questions}) == 1190
    assert {question.paragraph_id for question in data.questions} == set(range(240))
    assert all(question.question_hi and question.question_en for question in data.questions)
