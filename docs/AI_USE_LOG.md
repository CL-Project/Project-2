# AI-use log

Every non-trivial piece of AI-generated code or text is recorded here (`AGENTS.md` §13): date, agent/tool, what
was produced (file or step), how it was verified, and `reviewed_by` (a human fills this in). The README and the
report include an AI-use statement built from this log.

| Date | Agent / tool | Step | What was produced | Verification | reviewed_by |
|---|---|---|---|---|---|
| 2026-11-15 | Command Code (Claude) | S05 | `src/textproc.py`, `tests/test_textproc.py` | `uv run pytest tests/test_textproc.py` — matras, NFC idempotence, nukta unification, zero-width, danda split, decimal point, Devanagari digits, Latin-in-Hindi tokens, `is_devanagari_ratio` | |
| 2026-11-15 | Command Code (Claude) | S07 | `src/retrieval.py`, `tests/test_retrieval.py` | chunked cosine and CSLS compared against brute-force reference matrices, chunk-size invariance, column-block invariance, shuffled-gold chance check, `rank_of_gold` | |
| 2026-11-15 | Command Code (Claude) | S10 | `scripts/download_data.py`, `configs/data.yaml` (`wikipedia.dump_date`) | Ran for `muse,flores,xquad,wikipedia`; `data/raw/MANIFEST.json` records URL, size, SHA-256 and licence note per file | |
| 2026-11-15 | Command Code (Claude) | S11 | `scripts/validate_data.py`; generates `docs/DATA.md` | Ran on the downloaded data: FLORES 997/1,012 lines per language, XQuAD 1,190 questions / 240 paragraphs, MUSE train-test disjointness | |
| 2026-11-15 | Command Code (Claude) | S12 | `scripts/preprocess_wiki.py` | Smoke runs on 50 Hindi and 40 English articles; full Hindi/English corpora built with equal-size rule | |
| 2026-11-15 | Command Code (Claude) | S13 | Contamination and shard sections of `docs/DATA.md` (rendered by `scripts/validate_data.py`) | Counts read from `data/processed/wiki.<lang>.stats.json`; shard disjointness asserted on article ids | |
| 2026-11-15 | Command Code (Claude) | S14 | `src/embeddings/encoders.py`, `tests/test_encoders.py`, `configs/models.yaml` (`encoder_defaults`, per-encoder `pooling`/`batch_size`, `baselines`) | 100 FLORES sentences per encoder: shape (100, 768), unit rows, identical strings cosine 1.000000, reorder max abs delta 0.0 (LaBSE, mE5) / 1.7e-05 (MuRIL fp16), 0 truncated, CUDA peak about 1.0 GB. Cache round-trip and sidecar-mismatch tests; real-model tests skipped automatically when weights are not cached | |
| 2026-11-15 | Command Code (Claude) | S15, S16 | `src/tasks/parallel.py`, `scripts/run_eval.py`, `src/seeding.py`, `tests/test_parallel.py`; T2 runs for `labse`, `me5`, `muril` | T2 hi-en Acc@1: LaBSE 1.000, mE5 0.9985, MuRIL 0.9706. LaBSE sanity: gold cosine 0.855 vs best non-gold 0.509, minimum margin 0.080, random-pairing control Acc@1 0.0005 | |
| 2026-11-15 | Command Code (Claude) | S17 | `src/mt.py`, `src/rankers.py`, `tests/test_rankers.py`, `data/processed/flores_hi2en.tsv` (cache, git-ignored); `bm25_mt` T2 run | opus-mt hi-en on 2,009 sentences (fp16, beams 4); BM25 `bm25_mt` hi-en Acc@1 0.958. Eight translations spot-checked: fluent, with named-entity and rare-word errors | |
| 2026-11-15 | Command Code (Claude) | S18 | `scripts/make_table.py`, `tests/test_make_table.py`, `docs/RESULTS.md` | Table regenerated from `results/results.csv`; 156 tests pass; per-item files verified to share identical `query_id` sets across systems | |
