"""Evaluation metrics.

Spec:
- acc_at_k(ranks, k), precision_at_k, recall_at_k, mrr(ranks)
- bootstrap_ci(per_item_scores, n=1000, alpha=0.05, seed=42) -> (mean, low, high)
- macro_f1 for T4 is not needed in this repo (T4 is optional/dropped).
Test on toy data with known answers.
"""
