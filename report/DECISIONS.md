# Decision log

Every pre-registered or post-hoc decision that affects results. Decisions fixed **before** evaluation are marked
*pre-registered*; anything changed later needs an entry here saying why, and a note in the report's limitations.
`AGENTS.md` sections 8 and 9 are the source of truth if this file and the rules ever disagree.

| # | Date | Decision | Status | Why / where |
|---|---|---|---|---|
| D01 | pre-registered | **One Procrustes mapping per direction.** `W_hi_en` is fitted on `hi-en.0-5000.txt`, `W_en_hi` on `en-hi.0-5000.txt`; each direction is evaluated on its own `5000-6500` test file. Hindi is mapped into English space (`X W`), English into Hindi space via `Y W^T`. | fixed | PLAN S21, AGENTS.md §10.6. A shared mapping or a shared test file would leak across directions. |
| D02 | pre-registered | **One shared candidate vocabulary per language**, top 200,000 words by our Wikipedia counts, common to every system in a task and setting. | fixed | PLAN S20, AGENTS.md §9.2. Removes pool-size as a confound between static and pretrained systems. |
| D03 | pre-registered | **SIF first principal component is fitted on held-out Wikipedia text**, never on test data: 100,000 `fit`-shard sentences for sentence-level tasks, 20,000 `fit`-shard paragraphs for paragraph-level tasks (T3), per language. | fixed | PLAN S25, AGENTS.md §10.4. Fitting the PC on the evaluation set would be leakage. |
| D04 | pre-registered | **Wikipedia article shard split**: 90% `train`, 5% `distractor`, 5% `fit`, assigned by a deterministic hash of the article title, disjoint by article id. | fixed | PLAN S12, S28. Keeps training text out of the T3 distractor pool and the PC fit. |
| D05 | pre-registered | **T3 distractor design**: 10,000 paragraphs per language from the `distractor` shard only, 40 to 250 whitespace tokens, deduplicated, sampled with seed 42. Any English distractor sharing an 8-token shingle with an XQuAD English paragraph is dropped. Pool = 240 XQuAD paragraphs + distractors. | fixed | PLAN S28, AGENTS.md §10.4. Paragraph-length distribution of gold vs distractor is reported so length cannot act as a silent shortcut. |
| D06 | pre-registered | **T1 common query set**: primary metrics use only queries that are in the shared source candidate list, in the vocabulary of *every* static system, and have at least one gold translation in the shared target candidate list. `n_items` and `n_excluded` are always reported; coverage-aware P@1 over the full test set is secondary. | fixed | PLAN S22. Makes systems comparable without hiding OOV losses. |
| D07 | pre-registered | **Best static / best pretrained selection rule**: highest MRR on **T2 hi-en**, static within `{w2v, ft}`, pretrained within `{labse, me5}`. Controls (`muril`, `ft_ref`, `ft_cc`) are never eligible. | fixed | PLAN pre-registered design table, AGENTS.md §9.3. Fixed before results so it cannot be cherry-picked. |
| D08 | pre-registered | **Global seed 42** for numpy, torch, python `random`, gensim, sampling and bootstrap (1,000 resamples, percentile CIs). | fixed | `configs/tasks.yaml`, AGENTS.md §7. |
| D09 | pre-registered | **gensim `workers=8` is not exactly reproducible**; trained vectors are saved to `data/vectors/static/` and the run-to-run variability is reported instead of claiming bit-exact reproduction. | fixed | AGENTS.md §7, §12. `workers=1` would be exactly reproducible but far too slow on this laptop. |
| D10 | pre-registered | **No tuning on evaluation data.** MUSE test pairs, FLORES, XQuAD, MIRACL and the polysemy probe are evaluation-only; any static hyperparameter change must be cross-validated on the MUSE **train** pairs (S24) and recorded here. | fixed | AGENTS.md §9.1. |

## Skeleton reconciliation (S03)

- The dropped fourth static-vector family (see `PLAN.md` §0, "Dropped on purpose") removed from `configs/models.yaml` and from the `static.py` / `train_static.py` docstrings: the proposal does not include it.
- `sib200` removed from `configs/data.yaml`: not part of the proposal.
- `stanza` and `regex` added to `pyproject.toml`. `nltk` was **not** added: nothing in the plan needs it.
- The probe-set TSV header now matches the S40 schema exactly.
- `docs/` created (S00).
- No `app/` directory and no demo code exist (confirmed).
- README research questions rewritten to the proposal's RQ1 (with H1 to H3) and RQ2.

## Open items (answers to record here)

- Report deadline and format/page limit: **TBD** (S01).
- Which member is A and which is B: **TBD** (PLAN Appendix D).
- Instructor stance on AI-assisted code: **TBD** (S01).
- GitHub repository URL: **TBD**.
