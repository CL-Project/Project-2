"""Tests for the results schema and per-item logging (S08)."""

import json

import pandas as pd
import pytest

from src.results_log import (
    PER_ITEM_COLUMNS,
    RESULT_COLUMNS,
    append_result,
    summarize_per_item,
    timed,
    write_per_item,
)


def sample_row(**overrides):
    row = {
        "system": "labse",
        "task": "t2_flores",
        "setting": "hi-en",
        "metric": "Acc@1",
        "value": 0.97,
    }
    row.update(overrides)
    return row


def sample_per_item():
    return pd.DataFrame(
        {
            "query_id": ["q1", "q2", "q3"],
            "gold_ids": [[0, 4], [2], [1]],
            "rank_of_best_gold": [1, 3, 2],
            "top1": [True, False, False],
            "top5": [True, True, True],
            "rr": [1.0, 1 / 3, 0.5],
            "n_candidates": [100, 100, 100],
        }
    )


def test_append_result_writes_header_once_and_appends(tmp_path):
    first_id = append_result(sample_row(), results_dir=tmp_path)
    second_id = append_result(sample_row(value=0.96), results_dir=tmp_path)

    lines = (tmp_path / "results.csv").read_text(encoding="utf-8").splitlines()
    assert lines[0].split(",") == RESULT_COLUMNS
    assert len(lines) == 3
    assert sum(line.startswith("run_id,") for line in lines) == 1

    frame = pd.read_csv(tmp_path / "results.csv", keep_default_na=False)
    assert list(frame.columns) == RESULT_COLUMNS
    assert frame["value"].tolist() == [0.97, 0.96]
    assert frame["run_id"].is_unique
    assert frame["run_id"].tolist() == [first_id, second_id]
    assert (frame["git_commit"] != "").all()
    assert (frame["timestamp"] != "").all()
    assert frame["n_items"].tolist() == ["", ""]


def test_append_result_rejects_missing_and_unknown_fields(tmp_path):
    with pytest.raises(ValueError):
        append_result({"system": "labse", "task": "t2_flores"}, results_dir=tmp_path)
    with pytest.raises(ValueError):
        append_result(sample_row(typo_column=1), results_dir=tmp_path)


def test_append_result_rejects_non_numeric_value(tmp_path):
    with pytest.raises(ValueError):
        append_result(sample_row(value="not a number"), results_dir=tmp_path)


def test_write_per_item_round_trips(tmp_path):
    per_item = sample_per_item()
    path = write_per_item("t2_flores", "labse", "hi-en", per_item, results_dir=tmp_path)

    assert path == tmp_path / "per_item" / "t2_flores__labse__hi-en.csv"
    back = pd.read_csv(path)

    assert list(back.columns) == PER_ITEM_COLUMNS
    assert back["query_id"].tolist() == ["q1", "q2", "q3"]
    assert [json.loads(value) for value in back["gold_ids"]] == [[0, 4], [2], [1]]
    assert back["rank_of_best_gold"].tolist() == [1, 3, 2]
    assert back["top1"].tolist() == [True, False, False]
    assert back["top5"].tolist() == [True, True, True]
    assert back["rr"].tolist() == pytest.approx([1.0, 1 / 3, 0.5])
    assert back["n_candidates"].tolist() == [100, 100, 100]


def test_write_per_item_rejects_bad_input(tmp_path):
    with pytest.raises(ValueError):
        write_per_item("t2_flores", "labse", "hi-en", sample_per_item().drop(columns=["rr"]), results_dir=tmp_path)

    bad = sample_per_item()
    bad.loc[0, "rank_of_best_gold"] = 0
    with pytest.raises(ValueError):
        write_per_item("t2_flores", "labse", "hi-en", bad, results_dir=tmp_path)


def test_summarize_per_item_computes_metrics_with_cis():
    rows = summarize_per_item(sample_per_item(), "labse", "t2_flores", "hi-en", n_boot=200, seed=42)
    by_metric = {row["metric"]: row for row in rows}

    assert set(by_metric) == {"Acc@1", "MRR", "Recall@5"}
    assert by_metric["Acc@1"]["value"] == pytest.approx(1 / 3)
    assert by_metric["MRR"]["value"] == pytest.approx((1 + 1 / 3 + 0.5) / 3)
    assert by_metric["Recall@5"]["value"] == pytest.approx(1.0)
    for row in rows:
        assert row["n_items"] == 3
        assert row["n_excluded"] == 0
        assert row["ci_low"] <= row["value"] <= row["ci_high"]


def test_summarize_per_item_accepts_task_specific_metric_names():
    rows = summarize_per_item(
        sample_per_item(),
        "labse",
        "t1_bli",
        "hi-en",
        metrics={"P@1": "top1", "P@5": "top5"},
        n_boot=100,
    )
    assert {row["metric"] for row in rows} == {"P@1", "P@5"}


def test_summarize_rows_are_appendable(tmp_path):
    rows = summarize_per_item(sample_per_item(), "labse", "t2_flores", "hi-en", n_boot=50)
    for row in rows:
        append_result(row, results_dir=tmp_path)
    frame = pd.read_csv(tmp_path / "results.csv", keep_default_na=False)
    assert len(frame) == 3


def test_timed_captures_runtime_and_memory():
    with timed() as stats:
        sum(range(10_000))
    assert stats.runtime_s >= 0.0
    assert stats.peak_mem_mb > 0.0
