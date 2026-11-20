# Contributions

Per-member record of work, updated as steps are completed (`AGENTS.md` §12, PLAN Appendix D). Contributions are graded
separately, and each member commits their own work under their own git identity. The step IDs refer to `PLAN.md`.

Member identities (fill in):

| Member | Name | Roll number | Git identity |
|---|---|---|---|
| A | **TBD** | **TBD** | **TBD** |
| B | Pranjal Baranwal | 2025114004 | `P-B <pranjalbaranwal304@gmail.com>` |

Suggested workstreams (`PLAN.md` Appendix D) — the humans decide who is A and who is B:

- **Member A (static and lexical):** S02 to S05, S07, S10 to S13, S19 to S22, S24, S25, S31, S35 to S37, S45.
- **Member B (encoders and tasks):** S06, S08, S09, S14 to S18, S23, S27 to S30, S33, S34, S38, S39, S42, S43, S46.
- **Shared:** S00 to S01, S18, S26, S32, S40, S41, S44, S47 to S49.

## Work completed

| Step | Description | Owner | Notes |
|---|---|---|---|
| S00 | Project docs and decision log | TBD | `docs/CONTRIBUTIONS.md`, `docs/DECISIONS.md` created |
| S03 | Reconcile skeleton with proposal | TBD | Dropped static family and `sib200` removed, `stanza`/`regex` added, configs extended, probe-set header updated, README RQs rewritten |
| S05 | `textproc.py` + tests | A | `src/textproc.py`, `tests/test_textproc.py` (27 tests): matras not split, NFC, nukta unification, zero-width removal, danda splitting, Devanagari digits |
| S06 | Ranking metrics and paired bootstrap | B | `src/metrics.py`, `tests/test_metrics.py` (15 tests) |
| S07 | `retrieval.py` + tests | A | `src/retrieval.py`, `tests/test_retrieval.py` (19 tests): chunked cosine/CSLS vs brute force, chunk invariance, shuffled-gold chance floor |
| S08 | Results and per-item logging utilities | B | `src/results_log.py`, `tests/test_results_log.py` (9 tests) |
| S09 | Shared synthetic test fixtures | B | `tests/conftest.py`, `tests/test_fixtures.py` (3 tests) |
| S10 | `download_data.py` | A | `scripts/download_data.py`: idempotent per-group downloads with `data/raw/MANIFEST.json` (URL, size, SHA-256, licence note). Ran for MUSE, FLORES, XQuAD, Hindi Wikipedia |
| S11 | Data validation and data card | A | `scripts/validate_data.py` → `docs/DATA.md`: FLORES 997/1,012 lines per language, XQuAD 1,190 questions / 240 paragraphs, MUSE train/test disjoint |
| S12 | `preprocess_wiki.py` | A | `scripts/preprocess_wiki.py`: 163,093 Hindi and 111,020 English articles, 1.84M / 1.76M sentences, 37.0M tokens each (0.000% apart) |
| S13 | Contamination filter report | A | Contamination and shard sections of `docs/DATA.md`; 2 Hindi and 20 English training sentences removed by the exact-match filter |
| S19 | Train Word2Vec and fastText | A | `src/embeddings/static.py`, `scripts/train_static.py`, `tests/test_static.py`; four sequential runs, vocab 140,590 (hi) / 147,988 (en), nearest-neighbour spot check |
| S20 | Shared candidate vocabularies | A | `scripts/build_candidates.py`; pools of 98,069 (hi) / 112,703 (en) after intersecting with every static vocabulary; coverage counters in `docs/DATA.md` |
| S21 | Procrustes alignment + tests | A | `src/embeddings/align.py`, `tests/test_align.py`: rotation recovery, orthogonality, OOV pair dropping, seed-accuracy direction control |
| S22 | T1 for static systems | A | `src/tasks/bli.py`, `tests/test_bli.py`; common query set, CSLS (k=10) primary and cosine secondary, full-dictionary coverage view (`<direction>_cov`) |
| S24 | Cross-validated frequent band | A | `run_eval.py --cvfreq`; 5-fold cross-validation over the MUSE train words, pooled n = 4,883 |
| S25 | SIF sentence vectors and static T2 | A | `src/embeddings/sentence_vectors.py`, `tests/test_sentence_vectors.py`; PC fitted on the held-out `fit` shard, plain-average ablation; static T2 checked against three leak controls (D29) |
| S26 | Checkpoint 2 | A | Checkpoint 2 tables and caveats in `docs/RESULTS.md`; D28–D30 in `docs/DECISIONS.md`; H1 verdict deferred to S34 |
| S27 | T3 XQuAD loading and alignment | B | `src/tasks/xquad.py` (loading, id-based paragraph alignment, pool construction, per-item frame), `tests/test_xquad.py` (17 tests) |
| S28 | T3 distractors and overlap filter | B | `scripts/build_t3_pool.py`, `tests/test_t3_pool.py` (10 tests); 10,240-paragraph pools per language, English 8-token shingle filter, pool statistics in `data/processed/t3_pool_stats.json` and `docs/DATA.md` |
| S29 | T3 BM25 baselines | B | `run_eval.py --task t3_xquad` BM25 path (`bm25` en-en/hi-hi, `bm25_mt` hi-en), `tests/test_run_eval_t3.py`; `en-en` Recall@5 = 0.924 clears the sanity bar |
| S30 | T3 encoders | B | T3 encoder path (query/passage kinds, per-pool truncation counts); LaBSE weak result investigated and recorded as D37 |
| S31 | T3 static systems | A | Paragraph-level SIF PC (20,000 `fit`-shard paragraphs, D35), query-side mapping (D36), four settings for `w2v`/`ft`/`ft_ref` plus the plain-average ablation |
| S32 | Checkpoint 3 | A | Checkpoint 3 tables in `docs/RESULTS.md`, freeze recorded as D40, experiment set frozen |
| S33 | T3b MIRACL-hi (optional) | B | `src/tasks/miracl.py`, `tests/test_miracl.py` (13 tests), MIRACL download group, T3b runner; pool of 734 relevant + 20,000 sampled passages, MRR@10/Recall@10 for six systems |

## Statement for the report

**TBD.** Write this from the table above at S48, in both members' own words.
