"""Contract tests for the reconciled configuration files (S00/S03).

These guard the skeleton reconciliation: the system ids, dataset splits and probe-set schema that the
rest of the pipeline assumes. They read configs/ only, so they need no network, data or GPU.
"""

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"


def load(name):
    with open(CONFIGS / name, encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_static_families_are_the_two_in_scope():
    static = load("models.yaml")["static"]
    assert set(static) == {"dim", "vocab_cap", "word2vec", "fasttext"}
    assert static["dim"] == 300
    assert static["vocab_cap"] == 200_000
    for name in ("word2vec", "fasttext"):
        assert static[name]["workers"] == 8
        assert static[name]["min_count"] == 5
    assert static["fasttext"]["min_n"] == 3 and static["fasttext"]["max_n"] == 6


def test_encoders_and_aliases():
    enc = load("models.yaml")["encoders"]
    assert set(enc) == {"labse", "me5", "muril"}
    assert enc["labse"]["max_len"] == 256
    assert enc["me5"]["query_prefix"] == "query: "
    assert enc["me5"]["passage_prefix"] == "passage: "
    assert enc["muril"]["pooling"] == "mean"


def test_muse_splits_are_disjoint_by_construction():
    muse = load("data.yaml")["muse"]
    assert muse["splits"]["train"] == "0-5000"
    assert muse["splits"]["test"] == "5000-6500"
    assert set(muse["pairs"]) == {"hi-en", "en-hi"}


def test_wikipedia_shards_and_distractors():
    data = load("data.yaml")
    shards = data["wikipedia"]["shards"]
    assert set(shards) == {"train", "distractor", "fit"}
    assert abs(sum(shards.values()) - 1.0) < 1e-9
    distractors = data["t3_distractors"]
    assert distractors["n_per_language"] == 10_000
    assert distractors["seed"] == 42
    assert distractors["source_shard"] == "distractor"


def test_probe_set_header_matches_s40_schema():
    header = (ROOT / "analysis" / "probe_set_polysemy.tsv").read_text(encoding="utf-8").splitlines()[0]
    assert header.split("\t") == [
        "id",
        "word",
        "word_romanized",
        "intended_sense",
        "hindi_sentence",
        "english_match",
        "english_contrast",
        "status",
        "annotator",
    ]


def test_task_ids_and_settings():
    tasks = load("tasks.yaml")
    assert tasks["seed"] == 42
    assert tasks["t3_xquad"]["settings"] == ["hi-en", "hi-hi", "en-hi", "en-en"]
    assert tasks["t1_bli"]["directions"] == ["hi-en", "en-hi"]
