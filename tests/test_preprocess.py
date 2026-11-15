"""Tests for the deterministic parts of the Wikipedia preprocessing (S12).

Everything here is offline and tiny: no parquet, no Hub, no network.  The
network and parquet paths are exercised by the smoke runs in the step log.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from preprocess_wiki import (  # noqa: E402
    MIN_TOKENS,
    paragraphs_from_text,
    read_token_target,
    sentence_key,
    shard_for_title,
)

SHARDS = {"train": 0.90, "distractor": 0.05, "fit": 0.05}


def test_shard_assignment_is_deterministic():
    """The same title always lands in the same shard."""
    first = shard_for_title("भारत", SHARDS)
    second = shard_for_title("भारत", SHARDS)
    assert first == second
    assert first in SHARDS


def test_shard_assignment_ignores_whitespace_variants():
    """Titles that differ only in spacing normalize to the same shard."""
    assert shard_for_title("  भारत ", SHARDS) == shard_for_title("भारत", SHARDS)


def test_shard_assignment_covers_all_three_shards():
    """A few thousand titles produce all three shards in roughly the configured ratio."""
    titles = [f"article {number}" for number in range(4000)]
    counts = {name: 0 for name in SHARDS}
    for title in titles:
        counts[shard_for_title(title, SHARDS)] += 1
    assert all(count > 0 for count in counts.values())
    assert 0.85 <= counts["train"] / len(titles) <= 0.95


def test_shards_are_disjoint_by_construction():
    """One shard per title: no title is ever assigned twice."""
    titles = [f"page-{number}" for number in range(1000)]
    assigned = [shard_for_title(title, SHARDS) for title in titles]
    assert all(name in SHARDS for name in assigned)
    assert len(assigned) == len(titles)


def test_paragraphs_drop_headings_and_short_lines():
    """Headings and lines below the character threshold are not paragraphs."""
    text = "== इतिहास ==\nयह एक लंबा पैराग्राफ है जिसमें पर्याप्त जानकारी मौजूद है और इसे रखा जाना चाहिए।\n\nछोटा\n"
    paragraphs = paragraphs_from_text(text)
    assert len(paragraphs) == 1
    assert paragraphs[0].startswith("यह एक लंबा पैराग्राफ")
    assert "==" not in paragraphs[0]


def test_paragraphs_join_lines_within_a_block():
    """Continuation lines of one block become a single paragraph."""
    long_line = "यह पंक्ति काफी लंबी है और इसे पैराग्राफ का हिस्सा होना चाहिए।"
    other = "यह दूसरी पंक्ति भी उसी पैराग्राफ का हिस्सा है और पर्याप्त लंबी है।"
    paragraphs = paragraphs_from_text(f"{long_line}\n{other}\n\n")
    assert len(paragraphs) == 1
    assert long_line in paragraphs[0] and other in paragraphs[0]


def test_paragraphs_empty_text():
    """No text yields no paragraphs."""
    assert paragraphs_from_text("") == []


def test_sentence_key_ignores_punctuation_and_spacing():
    """Keys make a danda-terminated evaluation sentence match a stripped one."""
    assert sentence_key("मैं कल दिल्ली गया था।", "hi") == sentence_key("मैं   कल दिल्ली गया था", "hi")
    assert sentence_key("I went to Delhi yesterday.", "en") == sentence_key("i went to delhi yesterday", "en")


def test_sentence_key_differs_for_different_sentences():
    """Different wording gives a different key."""
    assert sentence_key("मैं कल दिल्ली गया था", "hi") != sentence_key("मैं कल मुंबई गया था", "hi")


def test_min_tokens_is_three():
    """The sentence-length floor the contamination filter also uses is 3."""
    assert MIN_TOKENS == 3


def test_read_token_target_accepts_valid_stats(tmp_path):
    """A stats file with a positive token count is read back."""
    path = tmp_path / "stats.json"
    path.write_text(json.dumps({"tokens": 1234}), encoding="utf-8")
    assert read_token_target(path) == 1234


@pytest.mark.parametrize("payload", ["", "null", "[]", '{"tokens": 0}', '{"tokens": "x"}', "{}"])
def test_read_token_target_rejects_incomplete_stats(tmp_path, payload):
    """Empty, null or non-positive stats mean the output cannot be trusted."""
    path = tmp_path / "stats.json"
    path.write_text(payload, encoding="utf-8")
    assert read_token_target(path) is None


def test_read_token_target_missing_file(tmp_path):
    """A missing stats file is not an error, it just means nothing is cached."""
    assert read_token_target(tmp_path / "absent.json") is None
