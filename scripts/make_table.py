"""Print a markdown results table from ``results/results.csv`` (S18, S26, S32).

Every number comes from a completed run, so the table can be regenerated at any
checkpoint and compared against the report.  Confidence intervals are printed as
``value [low, high]``; a metric that has not been run for a system shows ``-``.

Examples::

    uv run python scripts/make_table.py --task t2_flores
    uv run python scripts/make_table.py --task t2_flores --metric Acc@1 --setting hi-en
    uv run python scripts/make_table.py --task t2_flores --output docs/RESULTS_T2.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = REPO_ROOT / "results" / "results.csv"

SYSTEM_ORDER = ["bm25", "bm25_mt", "w2v", "ft", "ft_ref", "ft_cc", "labse", "me5", "muril"]


def load_results(path: Path) -> pd.DataFrame:
    """Read ``results.csv``, failing loudly when it is missing or empty."""
    if not path.exists():
        raise SystemExit(f"{path} does not exist; run an evaluation first")
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    if frame.empty:
        raise SystemExit(f"{path} has no rows")
    return frame


def filter_results(
    frame: pd.DataFrame,
    *,
    task: str | None = None,
    systems: Sequence[str] | None = None,
    settings: Sequence[str] | None = None,
    metrics: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Apply the requested filters to a results frame."""
    filtered = frame
    if task:
        filtered = filtered[filtered["task"] == task]
    if systems:
        filtered = filtered[filtered["system"].isin(list(systems))]
    if settings:
        filtered = filtered[filtered["setting"].isin(list(settings))]
    if metrics:
        filtered = filtered[filtered["metric"].isin(list(metrics))]
    return filtered


def _ordered(values: Sequence[str], preferred: Sequence[str]) -> list[str]:
    """Return ``values`` deduplicated: preferred order first, then alphabetical."""
    present = set(values)
    ordered = [value for value in preferred if value in present]
    ordered.extend(sorted(present - set(ordered)))
    return ordered


def format_cell(value: Any, ci_low: Any = "", ci_high: Any = "") -> str:
    """Format one metric cell as ``value [low, high]`` (``-`` when missing)."""
    if value is None or str(value) == "":
        return "-"
    number = float(value)
    if str(ci_low) == "" or str(ci_high) == "":
        return f"{number:.3f}"
    return f"{number:.3f} [{float(ci_low):.3f}, {float(ci_high):.3f}]"


def build_table(frame: pd.DataFrame, *, metrics: Sequence[str]) -> str:
    """Render a markdown table: one row per system, one column per setting x metric."""
    metric_list = list(metrics)
    settings = _ordered(frame["setting"].tolist(), ["hi-en", "en-hi", "hi-hi", "en-en", "hi-en_cvfreq"])
    systems = _ordered(frame["system"].tolist(), SYSTEM_ORDER)

    header = ["system"]
    for setting in settings:
        for metric in metric_list:
            header.append(f"{setting} {metric}")
    align = ["---"] + ["---:"] * (len(header) - 1)

    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(align) + " |"]
    for system in systems:
        cells = [system]
        for setting in settings:
            for metric in metric_list:
                match = frame[
                    (frame["system"] == system)
                    & (frame["setting"] == setting)
                    & (frame["metric"] == metric)
                ]
                if match.empty:
                    cells.append("-")
                else:
                    row = match.iloc[-1]  # later runs supersede earlier ones
                    cells.append(format_cell(row["value"], row["ci_low"], row["ci_high"]))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def build_notes(frame: pd.DataFrame) -> str:
    """Return a short provenance line: n_items and excluded counts per system."""
    parts = []
    for system in _ordered(frame["system"].tolist(), SYSTEM_ORDER):
        rows = frame[frame["system"] == system]
        n_items = sorted({int(float(value)) for value in rows["n_items"] if str(value) != ""})
        n_excluded = sorted({int(float(value)) for value in rows["n_excluded"] if str(value) != ""})
        item_text = "/".join(str(value) for value in n_items) if n_items else "?"
        excluded_text = "/".join(str(value) for value in n_excluded) if n_excluded else "0"
        parts.append(f"{system}: n={item_text}, excluded={excluded_text}")
    return "; ".join(parts)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS, help="path to results.csv")
    parser.add_argument("--task", default=None, help="task id filter (e.g. t2_flores)")
    parser.add_argument("--systems", nargs="+", default=None, help="system ids to include")
    parser.add_argument("--settings", nargs="+", default=None, help="settings to include (e.g. hi-en en-hi)")
    parser.add_argument("--metrics", nargs="+", default=None, help="metrics to include (e.g. Acc@1 MRR)")
    parser.add_argument("--title", default=None, help="heading printed above the table")
    parser.add_argument("--output", type=Path, default=None, help="write the table to this file as well")
    return parser.parse_args(argv)


def _default_metrics(frame: pd.DataFrame) -> list[str]:
    """Return metrics in the canonical order for the task, then any extras."""
    canonical = ["P@1", "P@5", "Acc@1", "MRR", "Recall@1", "Recall@5"]
    return _ordered(frame["metric"].tolist(), canonical)


def main(argv: list[str] | None = None) -> int:
    """Print (and optionally save) the results table."""
    args = parse_args(argv)
    frame = load_results(args.results)
    filtered = filter_results(
        frame,
        task=args.task,
        systems=args.systems,
        settings=args.settings,
        metrics=args.metrics,
    )
    if filtered.empty:
        raise SystemExit("no rows match the requested filters")

    metrics = list(args.metrics) if args.metrics else _default_metrics(filtered)
    table = build_table(filtered, metrics=metrics)
    title = args.title or f"Results: {args.task or 'all tasks'}"
    output = [f"## {title}", "", table, "", f"_{build_notes(filtered)}_", ""]
    text = "\n".join(output)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        print(f"wrote {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
