"""Tests for the markdown results table (S18).

The table script is pure formatting, so these tests use a small in-memory frame
with the real results schema.
"""

from __future__ import annotations

import pandas as pd
import pytest

from scripts.make_table import build_table, filter_results, format_cell, load_results


def _frame() -> pd.DataFrame:
    rows = [
        {
            "run_id": "r1",
            "git_commit": "abc",
            "timestamp": "2026-01-01T00:00:00+00:00",
            "system": "labse",
            "task": "t2_flores",
            "setting": "hi-en",
            "metric": "Acc@1",
            "value": "1.0",
            "ci_low": "0.998",
            "ci_high": "1.0",
            "n_items": "2009",
            "n_excluded": "0",
            "runtime_s": "11",
            "peak_mem_mb": "100",
            "notes": "max_len=256",
        },
        {
            "run_id": "r2",
            "git_commit": "abc",
            "timestamp": "2026-01-01T00:00:00+00:00",
            "system": "labse",
            "task": "t2_flores",
            "setting": "hi-en",
            "metric": "MRR",
            "value": "1.0",
            "ci_low": "0.999",
            "ci_high": "1.0",
            "n_items": "2009",
            "n_excluded": "0",
            "runtime_s": "11",
            "peak_mem_mb": "100",
            "notes": "",
        },
        {
            "run_id": "r3",
            "git_commit": "abc",
            "timestamp": "2026-01-01T00:00:00+00:00",
            "system": "bm25_mt",
            "task": "t2_flores",
            "setting": "hi-en",
            "metric": "Acc@1",
            "value": "0.958",
            "ci_low": "0.949",
            "ci_high": "0.967",
            "n_items": "2009",
            "n_excluded": "0",
            "runtime_s": "44",
            "peak_mem_mb": "900",
            "notes": "",
        },
    ]
    return pd.DataFrame(rows)


def test_filter_results_by_task_and_metric():
    frame = _frame()
    filtered = filter_results(frame, task="t2_flores", metrics=["Acc@1"])
    assert len(filtered) == 2
    assert set(filtered["system"]) == {"labse", "bm25_mt"}
    assert filter_results(frame, task="t1_bli").empty


def test_format_cell_with_and_without_ci():
    assert format_cell("0.9582", "0.949", "0.967") == "0.958 [0.949, 0.967]"
    assert format_cell("1.0") == "1.000"
    assert format_cell("") == "-"
    assert format_cell(None) == "-"


def test_build_table_places_values_and_missing_cells():
    table = build_table(_frame(), metrics=["Acc@1", "MRR"])
    lines = table.splitlines()
    assert lines[0].startswith("| system |")
    assert "hi-en Acc@1" in lines[0] and "hi-en MRR" in lines[0]
    assert "| labse | 1.000 [0.998, 1.000] | 1.000 [0.999, 1.000] |" in table
    assert "| bm25_mt | 0.958 [0.949, 0.967] | - |" in table


def test_build_table_uses_latest_run_for_a_duplicate_cell():
    frame = _frame()
    repeat = frame.iloc[[0]].copy()
    repeat["value"] = "0.5"
    repeat["run_id"] = "r4"
    combined = pd.concat([frame, repeat], ignore_index=True)
    table = build_table(combined, metrics=["Acc@1"])
    assert "0.500" in table
    assert "1.000 [0.998, 1.000]" not in table


def test_load_results_requires_existing_file(tmp_path):
    with pytest.raises(SystemExit):
        load_results(tmp_path / "missing.csv")


def test_load_results_requires_rows(tmp_path):
    path = tmp_path / "empty.csv"
    path.write_text("system,task\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        load_results(path)
