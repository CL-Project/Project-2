"""Single results schema used by every experiment.

Columns: system, task, direction, metric, value, ci_low, ci_high, n_items, runtime_s, peak_mem_mb, notes
Append rows to results/results.csv; never overwrite earlier runs (add a run_id or timestamp column if needed).
"""
