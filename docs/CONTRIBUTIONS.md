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
| S00 | Project docs and decision log | TBD | `docs/AI_USE_LOG.md`, `docs/CONTRIBUTIONS.md`, `docs/DECISIONS.md` created |
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

## Statement for the report

**TBD.** Write this from the table above at S48, in both members' own words.
