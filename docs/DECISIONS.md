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

## Data pipeline (S10 to S13)

| # | Date | Decision | Status | Why / where |
|---|---|---|---|---|
| D11 | 2026-11-15 | **Wikipedia dump pinned to `20231101`** (`configs/data.yaml` → `wikipedia.dump_date`). Hindi is downloaded fully; English is streamed and never stored. | fixed | PLAN S10/S12, `AGENTS.md` §4. Pinning the date makes the corpus reproducible; streaming English avoids a >20 GB download on a 16 GB laptop. |
| D12 | 2026-11-15 | **English `train` is capped at the Hindi `train` token count** (37,000,208 vs 37,000,208 in the current build, 0.000% apart). English articles are streamed in a seeded shuffled order (`t3_distractors.seed` = 42, shuffle buffer 10,000) and processing stops when the target is reached. | fixed | PLAN S12 step 5, `AGENTS.md` §4. Equal-sized corpora are required for the static comparison; the seeded shuffle keeps the choice random rather than "first N articles". |
| D13 | 2026-11-15 | **Contamination filter compares tokenized sentence keys**, not raw strings: a `train` sentence is dropped when its `tokenize()` output equals that of a FLORES dev/devtest sentence or an XQuAD context/question/answer sentence of the same language. Keys shorter than 3 tokens are not indexed. | fixed | `AGENTS.md` §10.4, PLAN S13. Raw-string matching fails because `sentence_split` strips the terminal danda/period; indexing short answers ("A", "W") inflates the removed count with strings that `MIN_TOKENS` would drop anyway. |
| D14 | 2026-11-15 | **`data/` contents are git-ignored** (`.gitignore`), with `.gitkeep` files retaining the directory structure; `__pycache__`, `.venv` and LaTeX build output are ignored too. | fixed | `AGENTS.md` §12. The corpora are several GB and must never be committed. |

## Encoders and T2 (S14 to S18)

| # | Date | Decision | Status | Why / where |
|---|---|---|---|---|
| D15 | 2026-11-15 | **LaBSE and mE5 are loaded through `sentence-transformers`, MuRIL through `transformers` with fp32 masked mean pooling.** | fixed | Both ST models ship a pooling module, and LaBSE also a dense projection (`2_Dense`); mean-pooling the base transformer by hand would silently produce a different representation than the model card describes. MuRIL has no ST pipeline, so it follows PLAN S14 ("`AutoModel` with mean pooling over the attention mask"). Means are accumulated in fp32 even under fp16 (`AGENTS.md` §10.7). Verified: unit rows, identical strings cosine 1.000000, reorder delta 0 (LaBSE/mE5). |
| D16 | 2026-11-15 | **MuRIL scores far above the level PLAN S16 anticipated and is left unchanged.** T2 hi-en Acc@1 = 0.971, en-hi 0.895 (LaBSE 1.000/1.000, mE5 0.999/1.000). | recorded, not tuned | PLAN S16 says explicitly: "do not 'fix' MuRIL, it is a control". It stays ineligible to be "best pretrained" under D07. The report states that the pre-run expectation was wrong and gives the measured numbers. |
| D17 | 2026-11-15 | **T2 scores with plain cosine, and gold ids are matched by normalized text, not by position.** | fixed | PLAN S15 specifies cosine for T2. Pooled FLORES dev+devtest has no duplicate sentences (2,009/2,009 unique in each language, checked), but matching by normalized text means a future duplicate can never create a spurious rank; the candidate list stays in pair order, so row *i* is pair *i* in both languages. |
| D18 | 2026-11-15 | **`bm25_mt` on T2 is `hi-en` only; other settings raise instead of running.** | fixed | PLAN S17: cross-lingual BM25 is defined only through MT of the Hindi query into English. Refusing the other directions prevents a meaningless row from being logged. |
| D19 | 2026-11-15 | **Embedding cache naming**: `<system>/<dataset>_<lang>.npy` as in `AGENTS.md` §8, plus an optional `_<kind_tag>` suffix when one dataset and language need several caches (for example a T3 query pool and a passage pool). | fixed | One cache per (system, dataset, language, kind, max_len, prefix) is required for correctness; the sidecar records every field and the text digest, and a mismatched sidecar forces re-encoding instead of reusing stale vectors. |
| D20 | 2026-11-15 | **Truncation counts are measured with the tokenizer, before truncation, including special tokens**, and stored in the cache sidecar as `n_truncated`/`truncated_fraction`. | fixed | `AGENTS.md` §10.3 requires the count to be logged and discussed; special tokens count because they consume the budget. T2 has 0 truncated inputs; T3 paragraphs are where this will matter. |

## Static pipeline and T1 (S19 to S25)

| # | Date | Decision | Status | Why / where |
|---|---|---|---|---|
| D21 | 2026-11-19 | **Static training hyperparameters made explicit**: `sg=1` (skip-gram, as PLAN S19 requires) and fastText `bucket=2,000,000` (gensim's default, written out so a gensim change cannot silently alter it); types capped at 200,000 per language with `max_final_vocab`. | fixed | PLAN S19/S20, `configs/models.yaml`. Trained vocabularies: 140,590 (hi) and 147,988 (en); the small Hindi corpus is the binding constraint. |
| D22 | 2026-11-19 | **fastText is saved as a full model and also exported to a `KeyedVectors` file holding the correct in-vocabulary vectors (word + n-gram sum).** Evaluation loads the export. | fixed | A fastText word vector is the sum of its word and n-gram vectors, so `wv.vectors_vocab` alone is wrong; `wv[word]` needs the 2.4 GB n-gram table. The export is produced once while the trained model is still in memory. |
| D23 | 2026-11-19 | **Shared T1 candidate pools are 98,069 (hi) and 112,703 (en) words**, not the 200,000 target, because the pool is the intersection with the static vocabularies (`w2v`, `ft`, first 200k lines of aligned `ft_ref`). Chance P@1 is therefore 1/112,703 for `hi-en` and 1/98,069 for `en-hi`. | recorded | PLAN S20/D02 requires one pool for all systems; the intersection is the binding constraint. Coverage counters in `docs/DATA.md`. |
| D24 | 2026-11-19 | **T1 primary retrieval is CSLS (k=10); cosine is logged as a secondary setting `<direction>_cos`.** | fixed | PLAN S22. CSLS is much stronger for the static systems (w2v `hi-en` P@1 0.376 vs 0.158 with cosine), so the primary/secondary split is not cosmetic. |
| D25 | 2026-11-19 | **Procrustes acceptance check.** On the MUSE train pairs, seed top-1 is 0.326 (`hi-en`, 7,894 usable pairs) and 0.341 (`en-hi`, 8,569) against the seed targets; applying `W.T` instead gives 0.000/0.0002. Restricting to one-to-one pairs (no target shared by several source words) raises it to 0.665/0.699. | accepted | PLAN S21 acceptance. The gap between seed and test accuracy is explained by the many-to-one structure of the seed dictionary, not by a mapping error; the transposed-mapping control rules out a direction bug. |
| D26 | 2026-11-19 | **Sentence/paragraph composition uses the in-vocabulary `KeyedVectors` export for fastText and skips OOV tokens**, counting them. The 2.4 GB n-gram table is never loaded during evaluation. | fixed | PLAN S25 says "skipping OOV words (count them)". The subword path would only matter if OOV lookups were allowed, which would also break the shared-candidate restriction of D02. |
| D27 | 2026-11-19 | **S24 folds**: the unique `hi-en` source words of `hi-en.0-5000.txt` that survive the common-set filter are assigned to 5 folds by a seeded permutation of the sorted word list; each fold's mapping is fitted on the pairs of the other four folds and the five held-out per-item sets are pooled. | fixed | PLAN S24. Reusing the S22 candidate pool keeps the cross-validated band comparable to the test band. |
| D28 | 2026-11-20 | **The full-dictionary coverage view is logged under its own setting tag `<direction>_cov`**, not merged into `<direction>`. Queries outside the common set get rank `n_candidates + 1` and are therefore counted wrong. | fixed | PLAN S22 asks for it as a *secondary* metric; keeping it as a separate tag means the primary common-set numbers cannot be accidentally read as coverage numbers, and `results.csv` stays append-only. Backfilled from the logged CSLS per-item files via `run_eval.py --coverage-only`, which re-runs no retrieval. |
| D29 | 2026-11-20 | **The static T2 SIF numbers were accepted only after three controls ruled out a leak**: shuffled gold labels (Acc@1 0.000000), no Procrustes mapping (0.0010), and random vectors (0.0000). The ablation ladder plain average 0.4515 / SIF without PC 0.7616 / SIF with PC 0.8935 is recorded in `docs/RESULTS.md`. | accepted | PLAN §15 asks for a stop-and-check when a result looks well outside the sanity ranges. It does: the plan expects static T2 to be "modest". The result survives every control, so it is reported as a finding, with the mechanism (a shared first principal component, `cos(pc_hi, pc_en) = 0.978`) named rather than hand-waved. |
| D30 | 2026-11-20 | **No H1 verdict is issued at S26.** The observed ordering is already gap(T1) = 0.134 > gap(T2) = 0.075, the reverse of H1's first inequality, but the verdict belongs to S34 with paired bootstrap CIs over the same queries. | deferred to S34 | `AGENTS.md` §9.5 fixes the decision rule; the checkpoint records the numbers without pre-empting the test. The report must not be written as though H1 were supported. |

## Open items (answers to record here)

- Report deadline and format/page limit: **TBD** (S01).
- Which member is A and which is B: **TBD** (PLAN Appendix D).
- Instructor stance on AI-assisted code: **TBD** (S01).
- GitHub repository URL: **TBD**.

## Blockers needing a human (recorded 2026-11-18)

| # | Issue | Evidence | Consequence |
|---|---|---|---|
| B1 | **CUDA is unavailable.** `torch.cuda.is_available()` is `False` and `nvidia-smi` cannot reach the driver. The GPU is an **RTX 4060 Max-Q** (`01:00.0`, AD107M), not the RTX 4050 described in `AGENTS.md` §4. The nvidia kmods are installed and signed for the running kernel `6.14.0-63.fc42.x86_64` (`/lib/modules/6.14.0-63.fc42.x86_64/extra/nvidia/nvidia.ko.xz`, signer `fedora_1775040905_c7a5f636`), but nothing is loaded (`lsmod` has no nvidia entry) and the kernel log shows `nouveau 0000:01:00.0: NVIDIA AD107` binding the device at boot. SecureBoot is enabled and `mokutil --list-enrolled` shows no MOK, so the akmod may be unsigned for this boot chain. | Needs a human: try `sudo modprobe nvidia` and read `dmesg`; if it fails on signature, enroll the akmod MOK (`sudo kmodgenca -a`, `sudo mokutil --import /etc/pki/akmods/certs/public_key.der`, reboot and enroll in the MOK manager) or blacklist `nouveau`. Also update `AGENTS.md` §4 — the hardware line names the wrong GPU. | Every GPU step (S14 to S16, S19 to S31, S39) currently runs CPU-only. PLAN Appendix B heuristics assume fp16 on the 6 GB card, and `AGENTS.md` §4 budgets assume it. No encoder result should be reported as "on GPU" until this is fixed. |
