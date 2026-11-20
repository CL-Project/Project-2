"""Tests for the T3 distractor sampler and overlap filter (S28), offline and tiny."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from src.tasks import xquad


def _load_script():
    """Import ``scripts/build_t3_pool.py`` as a module."""
    path = Path(__file__).resolve().parents[1] / "scripts" / "build_t3_pool.py"
    spec = importlib.util.spec_from_file_location("build_t3_pool", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_token_shingles_width_and_short_text():
    tokens = ["a", "b", "c", "d"]
    assert xquad.token_shingles(tokens, 2) == {("a", "b"), ("b", "c"), ("c", "d")}
    assert xquad.token_shingles(["a", "b"], 3) == {("a", "b")}
    assert xquad.token_shingles([], 3) == set()
    with pytest.raises(ValueError):
        xquad.token_shingles(tokens, 0)


def test_reference_shingles_uses_the_shared_tokenizer():
    """Punctuation and case are handled by ``tokenize``, not by raw strings."""
    shingles = xquad.reference_shingles(["The Quick brown fox."], "en", size=2)
    assert ("the", "quick") in shingles
    assert ("quick", "brown") in shingles


def test_overlap_filter_rejects_one_shared_shingle():
    reference = xquad.reference_shingles(["one two three four five six seven eight nine"], "en", size=3)
    hit = xquad.overlaps_reference("x one two three four five six seven eight nine y".split(), reference, size=3)
    miss = xquad.overlaps_reference("completely different words in a paragraph".split(), reference, size=3)
    assert hit is True
    assert miss is False


def test_overlap_filter_needs_the_full_width():
    """A run shorter than the shingle width is not an overlap."""
    reference = xquad.reference_shingles(["one two three four five six seven eight"], "en", size=8)
    partial = "one two three four five six seven".split()
    assert xquad.overlaps_reference(partial, reference, size=8) is False


def test_sample_distractors_is_deterministic_and_exact():
    module = _load_script()
    texts = [f"paragraph number {index} with some filler words here" for index in range(100)]
    first, counters = module.sample_distractors(texts, count=10, seed=42)
    second, _ = module.sample_distractors(texts, count=10, seed=42)
    assert first == second
    assert len(first) == 10
    assert len(set(first)) == 10
    assert counters["shortfall"] == 0


def test_sample_distractors_reports_a_shortfall():
    module = _load_script()
    sampled, counters = module.sample_distractors(["only one paragraph"], count=10, seed=42)
    assert sampled == ["only one paragraph"]
    assert counters["shortfall"] == 9


def test_sample_distractors_applies_the_overlap_filter():
    module = _load_script()
    reference = xquad.reference_shingles(["the shared gold sentence lives here today and tomorrow"], "en", size=4)
    texts = [
        "the shared gold sentence lives here today and tomorrow",
        "a genuinely unrelated paragraph about rivers and mountains",
    ]
    sampled, counters = module.sample_distractors(
        texts, count=5, seed=42, reference_shingles=reference, shingle_size=4
    )
    assert sampled == ["a genuinely unrelated paragraph about rivers and mountains"]
    assert counters["dropped_overlap"] == 1


def test_read_distractor_candidates_filters_shard_length_script_and_duplicates(tmp_path):
    module = _load_script()
    short = "बहुत छोटा"
    good = " ".join(f"शब्द{index}" for index in range(45))
    too_long = " ".join(f"श{index}" for index in range(300))
    rows = [
        f"train\t1\tt\t0\t{good}",
        f"distractor\t2\tt\t0\t{good}",
        f"distractor\t3\tt\t0\t{short}",
        f"distractor\t4\tt\t0\t{too_long}",
        f"distractor\t5\tt\t0\t{good}",
        f"distractor\t6\tt\t0\t{' '.join(['mostly', 'english', 'text', *[f'word{i}' for i in range(45)]])}",
        "malformed row",
    ]
    path = tmp_path / "wiki.hi.para.tsv"
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    texts, counters = module.read_distractor_candidates(
        path, lang="hi", min_tokens=40, max_tokens=250, min_devanagari_ratio=0.4
    )
    assert texts == [good]
    assert counters["distractor_rows"] == 5
    assert counters["dropped_length"] == 2
    assert counters["dropped_duplicate"] == 1
    assert counters["dropped_script"] == 1
    assert counters["malformed_rows"] == 1


def test_read_distractor_candidates_keeps_only_the_distractor_shard(tmp_path):
    module = _load_script()
    text = " ".join(["english"] * 45)
    rows = [f"train\t1\tt\t0\t{text}", f"fit\t2\tt\t0\t{text}", f"distractor\t3\tt\t0\t{text}"]
    path = tmp_path / "wiki.en.para.tsv"
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    texts, _ = module.read_distractor_candidates(
        path, lang="en", min_tokens=40, max_tokens=250, min_devanagari_ratio=0.4
    )
    assert texts == [text]


def test_length_summary_reports_quartiles():
    module = _load_script()
    summary = module.length_summary([10, 20, 30, 40])
    assert summary["n"] == 4
    assert summary["min"] == 10 and summary["max"] == 40
    assert summary["mean"] == 25.0
    assert module.length_summary([]) == {"n": 0}
