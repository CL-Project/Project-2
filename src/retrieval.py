"""Similarity search.

Spec:
- cosine_topk(queries, targets, k): chunked, returns indices and scores.
- csls_topk(queries, targets, k, csls_k=10): chunked CSLS; never materialize a full vocab x vocab matrix.
- Cap vocabulary at configs/models.yaml static.vocab_cap.
Sanity tests: identical query/target sets -> rank 1; shuffled labels -> near-chance accuracy.
"""
