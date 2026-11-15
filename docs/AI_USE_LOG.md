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
