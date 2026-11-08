"""Single results schema and per-item logging used by every experiment.

``results/results.csv`` is append-only (schema in ``AGENTS.md`` §8), while
``results/per_item/<task>__<system>__<setting>.csv`` holds one row per query.
Paired bootstrap tests, the gap analyses and the failure taxonomy all read the
per-item files, so they are mandatory for every run.
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.metrics import bootstrap_ci

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = REPO_ROOT / "results"

RESULT_COLUMNS = [
    "run_id",
    "git_commit",
    "timestamp",
    "system",
    "task",
    "setting",
    "metric",
    "value",
    "ci_low",
    "ci_high",
    "n_items",
    "n_excluded",
    "runtime_s",
    "peak_mem_mb",
    "notes",
]

PER_ITEM_COLUMNS = [
    "query_id",
    "gold_ids",
    "rank_of_best_gold",
    "top1",
    "top5",
    "rr",
    "n_candidates",
]

DEFAULT_METRIC_COLUMNS = {"Acc@1": "top1", "MRR": "rr", "Recall@5": "top5"}


def _as_scalar(value: Any) -> Any:
    """Convert a numpy scalar to a plain Python scalar."""
    return value.item() if isinstance(value, np.generic) else value


def _finite_float(value: Any) -> float:
    """Return ``value`` as a finite float or raise ``ValueError``."""
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"expected a numeric value, got {value!r}") from exc
    if not np.isfinite(number):
        raise ValueError(f"expected a finite value, got {value!r}")
    return number


def _git_commit() -> str:
    """Return the short hash of the current commit, or ``"unknown"``."""
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return completed.stdout.strip() or "unknown"


def _new_run_id() -> str:
    """Return a unique run id: UTC timestamp plus a short random suffix."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    return f"{stamp}-{uuid.uuid4().hex[:8]}"


def _results_path(results_dir: Path | str) -> Path:
    """Return the path of ``results.csv`` inside ``results_dir``."""
    return Path(results_dir) / "results.csv"


def _ensure_header(path: Path) -> None:
    """Create ``path`` with the results header once, or verify the existing one."""
    if path.exists() and path.stat().st_size > 0:
        with path.open("r", encoding="utf-8", newline="") as handle:
            header = next(csv.reader(handle), [])
        if header != RESULT_COLUMNS:
            raise RuntimeError(f"{path} has an unexpected header: {header!r}")
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle).writerow(RESULT_COLUMNS)


def append_result(row: dict[str, Any], *, results_dir: Path | str = DEFAULT_RESULTS_DIR) -> str:
    """Append one metric row to ``results.csv`` and return its ``run_id``.

    ``row`` must provide ``system``, ``task``, ``setting``, ``metric`` and
    ``value``; the remaining schema columns default to empty.  ``run_id``,
    ``git_commit`` and ``timestamp`` are always generated here.  Unknown keys
    are rejected so that a typo cannot silently drop a column.
    """
    if not isinstance(row, dict):
        raise TypeError("row must be a dict")

    unknown = set(row) - set(RESULT_COLUMNS)
    if unknown:
        raise ValueError(f"unknown result columns: {sorted(unknown)}")
    missing = [name for name in ("system", "task", "setting", "metric", "value") if row.get(name) in (None, "")]
    if missing:
        raise ValueError(f"missing required result fields: {missing}")

    record: dict[str, Any] = {column: _as_scalar(row.get(column, "")) for column in RESULT_COLUMNS}
    record["run_id"] = _new_run_id()
    record["git_commit"] = _git_commit()
    record["timestamp"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    record["value"] = _finite_float(record["value"])
    for name in ("ci_low", "ci_high", "runtime_s", "peak_mem_mb"):
        if record[name] not in ("", None):
            record[name] = _finite_float(record[name])
    for name in ("n_items", "n_excluded"):
        if record[name] not in ("", None):
            record[name] = int(record[name])

    path = _results_path(results_dir)
    _ensure_header(path)
    with path.open("a", encoding="utf-8", newline="") as handle:
        csv.DictWriter(handle, fieldnames=RESULT_COLUMNS).writerow(record)
    return str(record["run_id"])


def _json_gold_ids(value: Any) -> str:
    """Normalize a gold-id collection (or JSON string) to a JSON list string."""
    if isinstance(value, str):
        parsed = json.loads(value)
        if not isinstance(parsed, list):
            raise ValueError("gold_ids must be a list")
        return json.dumps([_as_scalar(item) for item in parsed])
    if isinstance(value, (list, tuple, np.ndarray, pd.Series)):
        return json.dumps([_as_scalar(item) for item in value])
    raise TypeError("gold_ids must be a list, numpy array or JSON string")


def _bool_series(series: pd.Series, name: str) -> pd.Series:
    """Validate and cast a column to bool (accepting 0/1)."""
    if series.dtype == bool:
        return series.astype(bool)
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.isna().any() or not numeric.isin([0, 1]).all():
        raise ValueError(f"{name} must be boolean or 0/1")
    return numeric.astype(bool)


def _positive_int_series(series: pd.Series, name: str, minimum: int) -> pd.Series:
    """Validate an integer column and cast it to ``int``."""
    numeric = pd.to_numeric(series, errors="raise")
    if numeric.isna().any() or (numeric != numeric.round()).any():
        raise ValueError(f"{name} must contain integers")
    values = numeric.astype(int)
    if (values < minimum).any():
        raise ValueError(f"{name} must be >= {minimum}")
    return values


def write_per_item(
    task: str,
    system: str,
    setting: str,
    per_item: pd.DataFrame,
    *,
    results_dir: Path | str = DEFAULT_RESULTS_DIR,
) -> Path:
    """Write one per-query table and return its path.

    ``per_item`` must contain the columns in :data:`PER_ITEM_COLUMNS` (extra
    columns are dropped).  ``gold_ids`` may be a list (stored as JSON) and
    ``top1``/``top5`` may be booleans or 0/1.
    """
    if not isinstance(per_item, pd.DataFrame):
        raise TypeError("per_item must be a pandas DataFrame")

    missing = [column for column in PER_ITEM_COLUMNS if column not in per_item.columns]
    if missing:
        raise ValueError(f"per_item is missing columns: {missing}")
    if per_item.empty:
        raise ValueError("per_item must contain at least one row")

    frame = per_item.loc[:, PER_ITEM_COLUMNS].copy()
    frame["query_id"] = frame["query_id"].astype(str)
    frame["gold_ids"] = frame["gold_ids"].map(_json_gold_ids)
    frame["top1"] = _bool_series(frame["top1"], "top1")
    frame["top5"] = _bool_series(frame["top5"], "top5")
    frame["rank_of_best_gold"] = _positive_int_series(frame["rank_of_best_gold"], "rank_of_best_gold", minimum=1)
    frame["n_candidates"] = _positive_int_series(frame["n_candidates"], "n_candidates", minimum=1)
    frame["rr"] = pd.to_numeric(frame["rr"], errors="raise").astype(float)
    if ((frame["rr"] < 0.0) | (frame["rr"] > 1.0)).any():
        raise ValueError("rr must be between 0 and 1")

    path = Path(results_dir) / "per_item" / f"{task}__{system}__{setting}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def summarize_per_item(
    per_item: pd.DataFrame,
    system: str,
    task: str,
    setting: str,
    metrics: dict[str, str] | None = None,
    *,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
    n_excluded: int = 0,
) -> list[dict[str, Any]]:
    """Return result rows with bootstrap CIs for a per-item table.

    ``metrics`` maps metric names to per-item columns;  the default is
    ``Acc@1``/``top1``, ``MRR``/``rr`` and ``Recall@5``/``top5``.  Pass
    task-specific names (for example ``{"P@1": "top1", "P@5": "top5"}`` for
    T1).  Rows are ready for :func:`append_result`.
    """
    if not isinstance(per_item, pd.DataFrame):
        raise TypeError("per_item must be a pandas DataFrame")
    columns = dict(DEFAULT_METRIC_COLUMNS if metrics is None else metrics)
    if not columns:
        raise ValueError("metrics must not be empty")

    missing = [column for column in columns.values() if column not in per_item.columns]
    if missing:
        raise ValueError(f"per_item is missing metric columns: {missing}")
    if per_item.empty:
        raise ValueError("per_item must contain at least one row")

    rows: list[dict[str, Any]] = []
    for metric, column in columns.items():
        scores = pd.to_numeric(per_item[column], errors="raise").to_numpy(dtype=float)
        value, ci_low, ci_high = bootstrap_ci(scores, n=n_boot, alpha=alpha, seed=seed)
        rows.append(
            {
                "system": system,
                "task": task,
                "setting": setting,
                "metric": metric,
                "value": value,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "n_items": int(len(per_item)),
                "n_excluded": int(n_excluded),
            }
        )
    return rows


@dataclass
class TimingStats:
    """Wall time and peak memory captured by :func:`timed`."""

    runtime_s: float = 0.0
    peak_mem_mb: float = 0.0
    cuda_peak_mem_mb: float | None = None


def _peak_rss_mb() -> float:
    """Return the peak resident set size of this process in MB (Linux)."""
    import resource

    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0


@contextmanager
def timed():
    """Context manager capturing wall time and peak memory.

    ``peak_mem_mb`` is the host RSS high-water mark.  ``cuda_peak_mem_mb`` is
    recorded when ``torch`` is already imported and CUDA is available (the
    training and evaluation scripts import torch before timing, so this avoids
    a costly import inside the context manager).
    """
    stats = TimingStats()
    torch = sys.modules.get("torch")
    use_cuda = torch is not None and bool(torch.cuda.is_available())
    if use_cuda:
        torch.cuda.reset_peak_memory_stats()

    start = time.perf_counter()
    try:
        yield stats
    finally:
        stats.runtime_s = time.perf_counter() - start
        stats.peak_mem_mb = _peak_rss_mb()
        if use_cuda:
            stats.cuda_peak_mem_mb = torch.cuda.max_memory_allocated() / (1024**2)
