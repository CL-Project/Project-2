# PLAN.md: ordered execution plan

Companion to `AGENTS.md` (rules) and `proposal.pdf` (the contract). Work **in order**: step IDs follow dependencies. Calendar weeks are targets taken from the proposal (wk1 baselines and encoders; wk2 static models and T1; wk3 T3 and analyses; wk4 taxonomy and report). If you are ahead, you may pull later steps forward as long as their dependencies are met.

Legend: **Owner** A / B is a *suggested* split of the two members (the humans decide who is A and B; see Appendix D). **Accept** = the check that must pass before the step is done. **H** = needs a human. Every step also obeys the Definition of Done in `AGENTS.md` section 14 (tests pass, results logged, AI-use log updated, commit made).

## Status checklist

- [x] S00 Project docs and decision log
- [x] S01 (H) Email instructors
- [x] S02 Environment check
- [x] S03 Reconcile skeleton with proposal
- [x] S04 First real commit
- [ ] S05 `textproc.py` + tests
- [x] S06 `metrics.py` + tests
- [ ] S07 `retrieval.py` + tests
- [x] S08 `results_log.py` + tests
- [x] S09 Shared test fixtures
- [ ] S10 `download_data.py`
- [ ] S11 Data validation and data card
- [ ] S12 `preprocess_wiki.py`
- [ ] S13 Contamination filter report
- [ ] S14 `encoders.py` with caching
- [ ] S15 T2 with LaBSE
- [ ] S16 T2 with mE5 and MuRIL
- [ ] S17 BM25 + MT baseline on T2
- [ ] S18 Checkpoint 1: results table v0
- [ ] S19 Train Word2Vec and fastText
- [ ] S20 Shared candidate vocabularies
- [ ] S21 `align.py` (Procrustes) + tests
- [ ] S22 T1 for static systems
- [ ] S23 T1 for encoders
- [ ] S24 T1 cross-validated very-frequent band
- [ ] S25 Sentence vectors (SIF) and T2 for static systems
- [ ] S26 Checkpoint 2: results table v1
- [ ] S27 T3 data: XQuAD loading and alignment
- [ ] S28 T3 distractors and overlap filter
- [ ] S29 T3 BM25 baselines
- [ ] S30 T3 encoders
- [ ] S31 T3 static systems
- [ ] S32 Checkpoint 3: freeze experiments
- [ ] S33 (optional) T3b MIRACL-hi
- [ ] S34 Gap analysis (H1)
- [ ] S35 A2 frequency and OOV bins
- [ ] S36 A1 morphology with Stanza
- [ ] S37 A4 vocabulary layers
- [ ] S38 A5 named-entity density
- [ ] S39 A6 word shuffling (H3)
- [ ] S40 (H) A3 polysemy probe set
- [ ] S41 H2 verdict and hypothesis table
- [ ] S42 (optional) A7 query type and length
- [ ] S43 Failure sampling export
- [ ] S44 (H) Manual taxonomy labeling
- [ ] S45 Taxonomy aggregation and figures
- [ ] S46 Report assets script
- [ ] S47 Report writing
- [ ] S48 README, AI-use statement, contributions
- [ ] S49 Final QA and submission

## 0. Pre-registered design summary

| Item | Decision |
|---|---|
| Systems | `bm25` (monolingual controls only), `bm25_mt` (hi to en via opus-mt, then English BM25), `w2v`, `ft` (trained by us), `ft_ref` (pretrained aligned fastText wiki vectors), `ft_cc` (optional: pretrained cc vectors + our Procrustes), `labse`, `me5` (multilingual-e5-base), `muril` (control, mean-pooled) |
| Tasks | T1 lexicon induction (P@1, P@5), T2 parallel sentence retrieval (Acc@1, MRR, Recall@5), T3 question to paragraph retrieval (Recall@1, Recall@5, MRR); optional T3b MIRACL-hi |
| Directions | T1, T2: `hi-en` (primary) and `en-hi`. T3: `hi-en` (primary), controls `hi-hi`, `en-hi`, `en-en` |
| Static vectors | gensim skip-gram, 300-d, window 5, min_count 5, negative 10, 5 epochs, workers 8; fastText char n-grams 3 to 6; equal-sized Hindi and English corpora |
| Alignment | Orthogonal Procrustes from MUSE train pairs (0-5000), **one mapping per direction**, L2-normalized vectors; CSLS with k=10 |
| Candidate vocabulary | top 200k words per language by our Wikipedia counts (letters only), shared by all systems |
| Sentence vectors | SIF, a = 1e-3, first principal component removed (PC fitted on held-out Wikipedia text, never on test data) |
| Encoders | LaBSE (max 256, fp16), mE5-base (max 512, fp16, `query:`/`passage:`), MuRIL-base-cased (max 512, mean pooling), all L2-normalized, embeddings cached |
| T3 pool | 240 XQuAD paragraphs + 10,000 same-language Wikipedia distractors (disjoint article shard); 1,190 questions |
| Statistics | 95% percentile bootstrap, 1,000 resamples over queries, seed 42; paired bootstrap for differences |
| Best static / best pretrained | highest MRR on T2 hi-en within `{w2v, ft}` and `{labse, me5}` |
| Hypotheses | H1 gap widens with granularity; H2 static matches or beats pretrained on frequent-word lexicon induction and fastText beats Word2Vec on inflected forms; H3 shuffling hurts only pretrained models (decision rules in `AGENTS.md` section 9) |

Dropped on purpose (do not add): GloVe, T4 classification, Gradio demo, BGE-M3.

---

## Phase 0: Admin and setup (day 1 to 2)

### S00 Project docs and decision log
**Owner:** A | **Est.:** 1 h | **Week:** 1
**Do:** create `docs/AI_USE_LOG.md` (table: date, agent/tool, step id, what was produced, verification, reviewed_by), `docs/CONTRIBUTIONS.md` (per-member list, filled in as work happens), `docs/DECISIONS.md` seeded with the decisions already fixed in this plan: per-direction Procrustes mappings; shared candidate vocabulary; SIF PC fitted on held-out Wikipedia; Wikipedia shard split (S12); distractor design (S28); common-query-set rule for T1 (S22); best-system selection rule; seed 42; gensim `workers=8` is not exactly reproducible so vectors are saved.
**Accept:** the three files exist with the above content. **Commit:** "add AI-use log, contributions and decision log".

### S01 (H) Email instructors
**Owner:** human | **Est.:** 15 min
Ask: (1) is AI-assisted code allowed and what disclosure is expected, (2) final deadline and report format/page limit, (3) confirmation that the proposal is acceptable. Record the answers in `docs/DECISIONS.md` and fill `AGENTS.md` section 16.

### S02 Environment check
**Owner:** A | **Est.:** 1 h (driver issues can take longer)
**Do:** NVIDIA driver from RPM Fusion (Secure Boot enrollment if enabled); `sudo dnf install gcc make git`; install `uv`; `uv sync`; `uv run python scripts/check_env.py`.
**Accept:** `check_env.py` prints the RTX 4050 and `CUDA available: True`; `uv run pytest` runs (even with no tests yet). If CUDA is `False`, it is almost always the driver or Secure Boot, not the lockfile.

### S03 Reconcile skeleton with proposal
**Owner:** A | **Est.:** 1 h
**Do:** apply the list in `AGENTS.md` section 17: remove `glove` from `configs/models.yaml` and the README; `uv add stanza regex` (add `nltk` only if truly needed); update the probe-set TSV header to the S40 schema; create `docs/`; extend `configs/data.yaml` with entries for MIRACL (optional), the Wikipedia shard split and distractor settings, pretrained fastText (aligned wiki and cc), and Kaikki Wiktionary; confirm no `app/` directory exists; update README research questions to match the proposal (two RQs, three hypotheses, T1 to T3).
**Accept:** `uv sync` clean; `grep -ri glove .` finds nothing outside this plan, AGENTS.md and the proposal; `uv run pytest` passes.

### S04 First real commit
**Owner:** A | **Est.:** 15 min
Commit the reconciled skeleton **with working content** (config files, `check_env.py`, docs, README). Do not make a placeholder commit. Both members make at least one commit in week 1 under their own identity.

---

## Phase 1: Shared libraries (days 2 to 4). Write tests first.

### S05 `src/textproc.py` + `tests/test_textproc.py`
**Owner:** A | **Est.:** 3 h
**Spec:**
- `normalize(text) -> str`: NFC; remove U+200B, U+200C, U+200D, U+FEFF; collapse whitespace; map Devanagari digits U+0966 to U+096F to ASCII `0` to `9`.
- `tokenize(text, lang) -> list[str]`: tokens are maximal runs matching `[\p{L}\p{M}\p{N}]+` (use the `regex` package). English: lowercase. Hindi: no case change. Punctuation (including danda) is dropped; keep an option `keep_punct=False`.
- `sentence_split(text, lang) -> list[str]`: split after danda (U+0964), `?`, `!`, and `.` followed by whitespace and not inside a number like `3.5`; strip empties.
- `is_devanagari_ratio(text) -> float`: fraction of letters in U+0900 to U+097F (used by S12).
**Tests (must exist):** words with matras (`किताबों`, `हिंदी`, `प्रधानमंत्री`) are single tokens; NFC is idempotent; a precomposed nukta letter (U+0958) and its decomposed form normalize to the same string; zero-width characters removed; danda splits sentences but `3.5` does not; Hindi digits become ASCII; Latin words inside Hindi text survive as tokens.
**Accept:** tests pass. **Commit:** "add Devanagari-aware normalization and tokenizer with tests".

### S06 `src/metrics.py` + tests
**Owner:** B | **Est.:** 2 h
**Spec:** inputs are 1-based `ranks` of the best gold item (numpy array). `acc_at_k`, `recall_at_k` (identical when one gold), `precision_at_k` for multi-gold (fraction of top-k that are gold; for T1 report "hit@k" = any gold in top-k and call it P@k as in MUSE), `mrr`. `bootstrap_ci(per_item, n=1000, alpha=0.05, seed=42) -> (mean, lo, hi)` using percentile method. `paired_bootstrap_diff(a, b, n=1000, seed=42) -> (mean_diff, lo, hi, p_two_sided_approx)` resampling item indices jointly.
**Tests:** hand-computed toy rankings; CI deterministic for a fixed seed and brackets the mean; paired diff of identical arrays is exactly 0 with CI [0, 0].
**Accept:** tests pass. **Commit:** "add ranking metrics and paired bootstrap with tests".

### S07 `src/retrieval.py` + tests
**Owner:** A | **Est.:** 4 h
**Spec:** `cosine_topk(Q, D, k, chunk=1024)` and `csls_topk(Q, D, k, S, csls_k=10, chunk=1024)` where `Q` are query vectors, `D` candidate vectors, `S` the (mapped) source-vocabulary matrix used to compute the candidate-side hubness term. CSLS(x, y) = 2 cos(x, y) - r_T(x) - r_S(y). `r_T(x)` is constant per query so it does not change the ranking for that query; `r_S(y)` is the mean cosine of candidate `y` to its `csls_k` nearest vectors in `S`. Compute `r_S` once for all candidates in chunks. Return indices and scores, plus a helper `rank_of_gold(scores_row, gold_idx_set) -> int`. Use torch on GPU when available, numpy otherwise; never allocate an N x N matrix larger than `chunk x N`.
**Tests:** chunked equals unchunked; CSLS equals a brute-force reference on a 50-vector toy set; identical query and candidate sets give rank 1 for each item; random shuffled gold labels give accuracy near chance (tolerance stated in the test).
**Accept:** tests pass. **Commit:** "add chunked cosine and CSLS retrieval with brute-force equivalence tests".

### S08 `src/results_log.py` + tests
**Owner:** B | **Est.:** 2 h
**Spec:** `append_result(row: dict)` writes to `results/results.csv` using the schema in `AGENTS.md` section 8 (creates the header once; appends; adds `run_id`, `git_commit`, `timestamp`); `write_per_item(task, system, setting, df)` writes `results/per_item/<task>__<system>__<setting>.csv` with columns `query_id, gold_ids, rank_of_best_gold, top1, top5, rr, n_candidates`; `summarize_per_item(...)` computes metric rows with bootstrap CIs through `metrics.py`. Also a context manager `timed()` capturing wall time and peak memory (host RSS and, if CUDA, `torch.cuda.max_memory_allocated`).
**Tests:** header written once; repeated calls append; per-item file round-trips.
**Accept:** tests pass. **Commit:** "add results and per-item logging utilities".

### S09 Shared test fixtures
**Owner:** B | **Est.:** 1 h
`tests/conftest.py` with tiny synthetic fixtures: 20 random unit vectors with a known orthogonal rotation between two languages (for Procrustes and BLI tests), a toy parallel corpus of 10 sentence pairs, toy rankings. No network.

---

## Phase 2: Data (days 3 to 6)

### S10 `scripts/download_data.py`
**Owner:** A | **Est.:** 3 h (plus download time) | **Week:** 1
Everything goes to `data/raw/` and is recorded in `data/raw/MANIFEST.json` (url, date, size in bytes, sha256, license note). The script is idempotent. **Verify every URL and license before relying on it; if a download fails, report the exact URL and path.**
- **MUSE dictionaries**: `https://dl.fbaipublicfiles.com/arrival/dictionaries/{src}-{tgt}.txt`, and the standard splits `{src}-{tgt}.0-5000.txt` (train) and `{src}-{tgt}.5000-6500.txt` (test), for `hi-en` and `en-hi`. Format: `source target` per line, space-separated.
- **FLORES-200**: preferred, the original tarball `https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz`; use `dev/hin_Deva.dev`, `dev/eng_Latn.dev`, `devtest/hin_Deva.devtest`, `devtest/eng_Latn.devtest` (one sentence per line). Expect about 997 + 1012 = 2,009 sentences per language (verify). Do not rely on script-based Hugging Face loaders: recent `datasets` versions dropped loading scripts.
- **XQuAD**: Hugging Face `google/xquad`, configs `xquad.hi` and `xquad.en`, split `validation` (expect 1,190 questions and 240 paragraphs per language). Fallback: the raw JSON `xquad.hi.json` and `xquad.en.json` from the deepmind/xquad GitHub repository.
- **Wikipedia**: Hugging Face `wikimedia/wikipedia` (config `<date>.hi` and `<date>.en`, e.g. `20231101.hi`; check the latest available). Hindi is downloaded fully. English is **streamed** (never download the full dump).
- **Pretrained fastText**: aligned wiki vectors (`wiki.hi.align.vec`, `wiki.en.align.vec`, see fasttext.cc "aligned vectors") for `ft_ref`; optionally Common Crawl `cc.hi.300.vec.gz` and `cc.en.300.vec.gz` for `ft_cc`. Download but never fully load (see pitfall 9).
- **Models**: pre-fetch to the Hugging Face cache: `sentence-transformers/LaBSE`, `intfloat/multilingual-e5-base`, `google/muril-base-cased`, `Helsinki-NLP/opus-mt-hi-en`. Stanza: `stanza.download("hi")` and `stanza.download("en")`.
- **Kaikki Hindi Wiktionary extract** (JSONL, from kaikki.org's Hindi dictionary page): for S37.
- **MIRACL-hi** (optional, S33): only fetch when you reach S33; check access terms.
**Accept:** manifest lists every file; sizes look plausible; no secrets or large files staged in git.

### S11 Data validation and data card
**Owner:** A | **Est.:** 2 h
`scripts/validate_data.py` prints and writes `docs/DATA.md`: counts (FLORES sentences per split and language, XQuAD questions and unique paragraphs per language, MUSE pair counts and unique source words per split), a few Hindi examples per dataset (eyeball for mojibake), checks that FLORES hi and en have equal lengths, XQuAD hi and en question ids match one-to-one, MUSE train and test source words are disjoint within each direction.
**Accept:** all assertions pass; `docs/DATA.md` committed (no data).

### S12 `scripts/preprocess_wiki.py`
**Owner:** A | **Est.:** 4 h | **Week:** 1 (launch before S19)
**Spec:**
1. For each language, iterate articles; keep the article title and id. Assign each article to a shard by a deterministic hash of its title: 90% `train`, 5% `distractor`, 5% `fit`.
2. Split article text into paragraphs (blank lines or newlines), drop headings and very short lines. For Hindi, drop paragraphs whose Devanagari-letter ratio is below 0.4 (cleans English and template residue).
3. Write `data/processed/wiki.<lang>.para.tsv` with columns `shard, article_id, title, para_idx, text` for all shards (paragraph-level, used by S28 and the PC fit in S25).
4. For the `train` shard only: sentence-split, normalize and tokenize (S05) and write `wiki.<lang>.sent.txt` (one cleaned sentence per line) and `wiki.<lang>.tok.txt` (space-joined tokens, sentences with fewer than 3 tokens dropped). Remove any sentence that exactly matches (after normalization) a FLORES or XQuAD sentence of the same language (S13 reports counts).
5. **Equal size**: process Hindi fully, record its token count `N_hi`; stream English articles in random order (seeded) until the English `train` token count reaches `N_hi`. Record both counts in `docs/DATA.md`.
6. Write `data/processed/wordcounts_<lang>.tsv` (`token<TAB>count`) from the `train` token files.
**Accept:** English and Hindi `train` token counts within 1% of each other; `wordcounts` files exist; spot-check 20 Hindi sentences for broken words. The `distractor` and `fit` shards must be disjoint from `train` (assert by article id).

### S13 Contamination filter report
**Owner:** A | **Est.:** 30 min
Report in `docs/DATA.md` how many training sentences were removed by the FLORES/XQuAD exact-match filter, and how many articles are in each shard.

---

## Phase 3: Encoders and T2 (week 1)

### S14 `src/embeddings/encoders.py` with caching
**Owner:** B | **Est.:** 4 h
**Spec:** `encode(system_id, texts, kind, batch_size=None, max_len=None) -> np.ndarray` (float32, L2-normalized, row order = input order). `kind` is one of `word`, `sentence`, `query`, `passage`. Behavior per system from `configs/models.yaml`: **LaBSE** via `sentence-transformers` (max_seq_length 256), **mE5** with prefixes (`"query: "` for `word`, `sentence` and `query`; `"passage: "` for `passage`), **MuRIL** via `transformers` `AutoModel` with mean pooling over the attention mask (accumulate in fp32). fp16 on GPU; sort inputs by length to reduce padding and restore the original order; `word` kind uses `max_len=16`, batch 512. Cache to `data/vectors/embeddings/<system>/<dataset>_<lang>.npy` with a `.json` sidecar (model id, max_len, pooling, prefix, normalized, n_items, **n_truncated**). Skip if the cache exists and the sidecar matches.
**Accept:** encoding 100 FLORES sentences with each model returns the right shape; identical strings give cosine 1.0; reordering inputs does not change per-string embeddings (up to fp16 tolerance); truncation counts logged. VRAM stays under 6 GB.

### S15 T2 with LaBSE
**Owner:** B | **Est.:** 3 h
**Spec (`src/tasks/parallel.py`, `scripts/run_eval.py --task t2_flores --system labse`):** pool FLORES dev + devtest (2,009 pairs). Direction `hi-en`: query = Hindi sentence *i*, candidates = all English sentences, gold = English sentence *i*. Cosine similarity, full 2,009 x 2,009 matrix (small). Per-item ranks, metrics (Acc@1, MRR, Recall@5) with bootstrap CIs; append to `results.csv` and write per-item files. Repeat for `en-hi`.
**Accept:** LaBSE hi-en Acc@1 is high (heuristic: at least 0.95; below 0.90 means a bug such as misaligned rows, wrong max length or missing normalization; investigate before moving on).

### S16 T2 with mE5 and MuRIL
**Owner:** B | **Est.:** 2 h
Run both. mE5 must use `"query: "` on both sides. MuRIL is a control and is expected to be markedly worse than the sentence-trained encoders (it was not trained for sentence similarity).
**Accept:** mE5 hi-en Acc@1 heuristic at least 0.85 (below 0.6: check prefixes); MuRIL well above chance (chance is about 0.0005) but likely far below LaBSE; do not "fix" MuRIL, it is a control.

### S17 BM25 + MT baseline on T2
**Owner:** B | **Est.:** 3 h
Translate the 2,009 Hindi FLORES sentences to English with `Helsinki-NLP/opus-mt-hi-en` (fp16, `num_beams=4`, `max_length=256`, batch 32); cache to `data/processed/flores_hi2en.tsv`. BM25 (`rank_bm25.BM25Okapi`, default k1=1.5 and b=0.75, tokens from S05, no stemming, no stopword list) over the English side. Direction `hi-en` only (the proposal allows cross-lingual BM25 only through MT of Hindi queries). System id `bm25_mt`.
**Accept:** results logged; translations spot-checked (10 sentences) and judged fluent enough to be a fair baseline; record translation chrF or BLEU if `sacrebleu` is added (optional).

### S18 Checkpoint 1: results table v0
**Owner:** both | **Est.:** 1 h | **End of week 1**
`scripts/make_table.py` reads `results/results.csv` and prints a markdown table (systems x T2 metrics with CIs). Commit `results/` (small files only).
**Accept:** table has `labse`, `me5`, `muril`, `bm25_mt` on T2. **Commit:** "T2 results for encoders and BM25+MT baseline".

---

## Phase 4: Static pipeline and T1 (week 2)

### S19 Train Word2Vec and fastText
**Owner:** A | **Est.:** 1 h attention, 2 to 4 h compute
`scripts/train_static.py --system {w2v,ft} --lang {hi,en}`: gensim `Word2Vec` or `FastText` on `LineSentence(data/processed/wiki.<lang>.tok.txt)` with the settings in `configs/models.yaml` (`vector_size=300, window=5, min_count=5, sg=1, negative=10, epochs=5, workers=8, seed=42`; fastText `min_n=3, max_n=6`). Save the full models (`data/vectors/static/<system>.<lang>.model`; fastText must keep its n-gram vectors) and, for Word2Vec, also `.kv`. Log vocabulary size and training time. Run in `tmux` under `systemd-inhibit`. Four runs in sequence (never in parallel). fastText with default buckets needs a few GB of RAM per language; do not run GPU jobs simultaneously.
**Accept:** nearest-neighbor sanity check on 10 frequent words per language (e.g. Hindi `भारत` should return Indian entities or Hindi geography words; English `india` similarly); report vocabulary sizes. Any hyperparameter change must be justified by MUSE-train cross-validation (S24 machinery), recorded in `docs/DECISIONS.md`, and applied identically to both languages and both systems.

### S20 Shared candidate vocabularies
**Owner:** A | **Est.:** 2 h
From `wordcounts_<lang>.tsv` keep tokens containing at least one letter (`\p{L}`) and, for Hindi, at least one Devanagari letter; take the top 200,000 by count (fewer if the vocabulary is smaller; record the size). Intersect with the vocabularies of `w2v` and `ft` (and with `ft_ref`'s first 200k lines) to get the **shared candidate list** per language; write `data/processed/candidates_<lang>.txt` with ranks. Log how many words each system lacks.
**Accept:** candidate lists exist; coverage numbers recorded in `docs/DATA.md`.

### S21 `src/embeddings/align.py` (Procrustes) + tests
**Owner:** A | **Est.:** 3 h
**Spec:** `procrustes(X, Y) -> W`: `X` and `Y` are n x d row-wise L2-normalized embeddings of translation pairs (X source = Hindi, Y target = English); compute `U, S, Vt = svd(X.T @ Y)`; `W = U @ Vt`; mapped source = `X @ W`. `fit_mapping(src_kv, tgt_kv, pairs)` keeps only pairs where both words are in the vocabularies (log the count). One mapping **per direction**: `W_hi_en` trained on `hi-en.0-5000.txt`, `W_en_hi` on `en-hi.0-5000.txt` (each direction uses its own train and test files; this avoids cross-direction leakage).
**Tests:** on the synthetic rotated fixture, Procrustes recovers the rotation (`||W - R||` small); `W.T @ W` is the identity; Procrustes on identical vectors returns the identity.
**Accept:** tests pass; on the real data, accuracy on the **seed pairs themselves** after mapping is far above test accuracy (an orientation bug shows up as near-zero accuracy on seeds). **Commit:** "add Procrustes alignment with recovery and orthogonality tests".

### S22 T1 for static systems
**Owner:** A | **Est.:** 4 h
**Spec (`src/tasks/bli.py`, `scripts/run_eval.py --task t1_bli`):**
- Queries: unique source words of `hi-en.5000-6500.txt` (direction `hi-en`) or `en-hi.5000-6500.txt` (direction `en-hi`); each source word has a set of gold translations.
- **Common query set**: keep only queries that (a) are in the shared source candidate list and in the vocabulary of **every** static system and (b) have at least one gold translation in the shared target candidate list. Report `n_items` and `n_excluded`. Primary metrics are on this common set (comparable across systems). Secondary: coverage-aware P@1 over the full test set where OOV queries count as wrong.
- Map the source vectors with the direction's `W`; retrieve among the target candidates with **CSLS (k=10)** (primary) and cosine (secondary). A query is a hit at k if any gold translation is in the top-k. Metrics: P@1, P@5 (hit@k).
- Systems: `w2v`, `ft` (our vectors), `ft_ref` (no mapping; already aligned), and optionally `ft_cc` (load the first 200k lines of the cc vectors, align with our Procrustes).
**Accept:** `ft`/`w2v` P@1 well above chance (heuristic at least 0.05; chance is about 5e-6); `ft_ref` clearly above our systems or comparable (heuristic: below 0.05 indicates a bug); report that Hindi Wikipedia is small, which explains weaker static vectors.

### S23 T1 for encoders
**Owner:** B | **Est.:** 3 h
Embed the Hindi and English candidate words and the query words with `kind="word"` for `labse`, `me5`, `muril` (about 200k words per language; minutes on the 4050; cache). Same queries, same candidate lists, same CSLS and cosine protocol as S22 (no mapping). Note: for encoders `S` in CSLS is the encoded source candidate list.
**Accept:** results and per-item files logged for all encoders; LaBSE should be clearly above chance.

### S24 T1 cross-validated very-frequent band
**Owner:** A | **Est.:** 3 h
**Why:** MUSE test words are ranks 5,000 to 6,500, so "frequent words" (H2) need a separate high-frequency band. **Do:** 5-fold cross-validation over the 5,000 MUSE **train** source words (direction `hi-en`): for each fold fit Procrustes on the other 4,000 pairs, evaluate the held-out 1,000 pairs with the S22 protocol; pool the 5,000 held-out queries. Evaluate the encoders on the same 5,000 words (no fitting). Output task id `t1_bli`, setting `hi-en_cvfreq`. Also use this machinery to justify any static hyperparameter change (never the test pairs).
**Accept:** per-item files for all systems on the cv band.

### S25 Sentence vectors (SIF) and T2 for static systems
**Owner:** A | **Est.:** 5 h
**Spec (`src/embeddings/sentence_vectors.py`):** `sif_vectors(token_lists, kv, counts, a=1e-3, pc=None)`: weight of word w is `a / (a + p(w))`, `p(w)` = count / total from `wordcounts_<lang>.tsv` (words absent from counts get `p = 1/total`); sentence vector = weighted average of word vectors, skipping OOV words (count them); then subtract the projection on the first principal component `pc`. Fit `pc` (via SVD of the 100k x 300 matrix, no centering) on **100,000 random sentences from the `fit` shard** of each language, for sentence-level tasks; for paragraph-level tasks (T3) fit a separate `pc` on 20,000 random paragraphs from the `fit` shard. Apply the mapping **first** (Hindi vectors into English space via `W_hi_en`), then SIF per language. `plain_average_vectors` is the ablation (no weights, no PC).
**Systems and runs:** `w2v`, `ft` (our), `ft_ref` (SIF using our counts for `p(w)`, with fallback `p = 1/total` for words absent from the counts), optionally `ft_cc`. Run T2 in both directions with cosine; plain-average ablation for `w2v` and `ft` on `hi-en` only.
**Tests:** SIF and averaging are invariant to word order (needed for H3); an OOV-only sentence yields a zero or flagged vector without crashing; the PC removal gives a vector orthogonal to `pc`.
**Accept:** T2 numbers logged. Static T2 may be modest; that is a finding, not a bug, provided sanity tests pass (invariance, alignment of rows, non-trivial above-chance accuracy).

### S26 Checkpoint 2: results table v1
**Owner:** both | **End of week 2**
Table of all systems on T1 (common set, CSLS primary) and T2. Commit. Review the pre-registered cut list (Appendix C) honestly against the remaining time.

---

## Phase 5: T3 paragraph retrieval (week 2 to 3; S28 may start earlier)

### S27 T3 data: XQuAD loading and alignment
**Owner:** B | **Est.:** 3 h
Load `xquad.hi` and `xquad.en` (validation). Build one **paragraph table** keyed by English context: `paragraph_id` (0 to 239), `text_en`, `text_hi`. Map each Hindi question to its English-paragraph id through the **question id** (never by position). Build the question table: `qid, question_hi, question_en, paragraph_id`.
**Accept (assert):** 1,190 questions, 240 unique paragraphs; every Hindi context maps to exactly one `paragraph_id`; every question id appears in both languages.

### S28 T3 distractors and overlap filter
**Owner:** B | **Est.:** 3 h
**Spec:** from `wiki.<lang>.para.tsv`, take paragraphs of the **`distractor` shard** only (disjoint from the static training shard). Keep paragraphs with 40 to 250 whitespace tokens, deduplicate, and sample 10,000 per language (seed 42; if the Hindi shard has fewer, use all, log the number, and say so in the report; or raise the Hindi shard share and rerun S12). **Overlap filter (English):** drop any distractor sharing at least one 8-token shingle with any XQuAD English paragraph (XQuAD is built from SQuAD v1.1 Wikipedia articles, which may also be in the dump). Save `data/processed/t3_pool_{hi,en}.jsonl` with `pid, text, is_gold_candidate`. Pools: `pool_en` = 240 XQuAD English + English distractors; `pool_hi` = 240 XQuAD Hindi + Hindi distractors.
**Accept:** pool sizes logged (expected 10,240 each unless Hindi is short); assert no distractor overlaps XQuAD text; assert paragraph length distribution of gold vs distractor paragraphs is reported (a large mismatch would let length act as a shortcut; note it in `docs/DECISIONS.md`).

### S29 T3 BM25 baselines
**Owner:** B | **Est.:** 3 h
`bm25` for `en-en` and `hi-hi` (monolingual controls, tokenizer from S05); `bm25_mt` for `hi-en` (translate the 1,190 Hindi questions to English with opus-mt, cache). Also, for completeness, the Hindi-question `en-hi`/`hi-en` BM25 without MT is not meaningful and is **not** run.
**Accept:** per-item files and metrics logged. Sanity: monolingual `en-en` BM25 Recall@5 clearly above chance (heuristic above 0.7).

### S30 T3 encoders
**Owner:** B | **Est.:** 3 h
For `labse`, `me5`, `muril`: encode questions (`kind="query"`) and both pools (`kind="passage"`; 10,240 paragraphs per language, cache) and run the four settings `hi-en`, `hi-hi`, `en-hi`, `en-en` with cosine. Log **truncation counts per pool and language** (S14 sidecar). Optional ablation if time allows: LaBSE with paragraph chunking (encode sentences, score a paragraph by the max similarity over its sentences) to separate truncation from representation quality.
**Accept:** mE5 `en-en` Recall@5 heuristic at least 0.85 (below 0.5: check `query:`/`passage:` prefixes and truncation); `hi-en` below or near `en-en`.

### S31 T3 static systems
**Owner:** A | **Est.:** 3 h
`w2v`, `ft`, `ft_ref` (optionally `ft_cc`): paragraph vectors with SIF (paragraph-level PC from S25), mapped Hindi into English space; four settings; cosine. Report OOV rates for paragraphs.
**Accept:** per-item files for all four settings. Low static T3 scores (long-paragraph averaging) are a legitimate finding; do not spend time rescuing them beyond the SIF vs plain-average check (run the plain average on `hi-en` as a one-line ablation).

### S32 Checkpoint 3: freeze experiments
**Owner:** both | **End of week 3 at the latest**
All systems x T1 to T3 in one table, every cell with a CI. **Freeze**: from here on, no new systems and no re-runs except bug fixes recorded in `docs/DECISIONS.md`. Tag the commit `experiments-frozen`. Re-apply the cut list if behind.

### S33 (optional) T3b MIRACL-hi
**Owner:** B | **Est.:** 4 to 6 h | **Only after S32**
Provide the larger Hindi-side check promised in the proposal: monolingual Hindi retrieval on the MIRACL-hi dev queries. Build a pool of **all judged-relevant passages of the dev queries plus about 20,000 random passages** from the Hindi corpus (the full corpus is about half a million passages; sample to fit the laptop and say so). Systems: `bm25`, `labse`, `me5`, `muril`, `w2v`, `ft` (monolingual, so no alignment). Metrics: MRR@10 and Recall@10 (nDCG@10 optional). Caveat: mE5 was trained on MIRACL training data (dev is held out). Check access terms first; drop this step if access or time is a problem and say so.

---

## Phase 6: Analyses (week 3)

All analyses read `results/per_item/` and cached embeddings; they never retrain. Each writes a CSV to `results/analysis/` and a figure to `results/plots/` (PDF and PNG).

### S34 Gap analysis (H1)
**Owner:** B | **Est.:** 3 h
Select the best static and best pretrained system by the pre-registered rule (S15 to S25 per-item MRR on T2 hi-en). Compute top-1 gaps (best pretrained minus best static) for T1 (common set, CSLS), T2 hi-en, T3 hi-en, with paired-bootstrap CIs, and the adjacent differences (gap T2 minus gap T1, gap T3 minus gap T2). Output `results/analysis/gaps.csv` and `results/plots/gap_by_granularity`. Also include the controls (`muril`, `ft_ref`) in a supplementary plot.
**Accept:** the H1 verdict ("supported / partially supported / not supported") is written to the hypothesis table draft per the rule in `AGENTS.md` section 9, with the numbers; humans approve wording.

### S35 A2: frequency and OOV bins
**Owner:** A | **Est.:** 3 h
Bin the common T1 test words by Hindi corpus frequency (terciles, plus the S24 very-frequent band as the top bin) and compute P@1 per system per bin with CIs; also OOV rates per system on the full test set (Word2Vec cannot embed OOV; fastText and encoders can). Output table and figure.

### S36 A1: morphology with Stanza
**Owner:** A | **Est.:** 4 h
For each T1 Hindi test word in the common set, take the first sentence in `wiki.hi.sent.txt` containing it (6 to 30 tokens; prefer one not used elsewhere), run Stanza Hindi (`processors="tokenize,pos,lemma"`, `tokenize_pretokenized=True`) on the sentence, and record the target token's `upos`, `lemma`, `feats`. **Definition:** `inflected = (surface != lemma)`; sub-buckets by UPOS (NOUN, VERB, ADJ, other) and by features present in `feats` (e.g. `Case=Obl`, `Number=Plur`, verb `VerbForm`/`Aspect`). Compute P@1 per system per bucket with CIs; H2 second clause (fastText vs Word2Vec on inflected forms) uses a paired bootstrap on the inflected subset. Manually spot-check 30 lemma assignments and report the observed Stanza error rate as a caveat. Words without a usable context sentence are excluded and counted.
**Accept:** table with n per bucket (flag buckets with n below 30).

### S37 A4: vocabulary layers
**Owner:** A | **Est.:** 4 h
Look up the Stanza lemma (fallback: the surface form) in the Kaikki Hindi JSONL. Classify with rules over `etymology_text` and category names: **Sanskritic** (Sanskrit, tatsama/tadbhava), **Perso-Arabic** (Persian or Arabic), **English**, **other loan**, **unknown**. Write `docs/vocab_layer_rules.md` with the exact rules, coverage per class, and a manual check of 30 random items. Report P@1 per class per system with CIs (flag small classes).
**Accept:** coverage reported honestly (many words will be `unknown`; do not guess).

### S38 A5: named-entity density
**Owner:** B | **Est.:** 3 h
Run Stanza English NER on the English side of the 2,009 FLORES sentences (the Hindi side is parallel, so entity density transfers). Density = tokens in entities / tokens. Bin: 0, (0, 0.1], (0.1, 0.25], above 0.25 (merge bins with fewer than 30 sentences). T2 hi-en Acc@1 and MRR per bin per system with CIs. If a Hindi NER model is available and trusted, mention it as future work only.
**Accept:** bins and counts reported.

### S39 A6: word shuffling (H3)
**Owner:** B | **Est.:** 4 h
**Control design matters:** compare three query conditions for `hi-en` on T2 (and T3): (1) original text; (2) **re-joined control**: tokenize (S05) and join with single spaces (isolates tokenization effects); (3) **shuffled**: shuffle the tokens of (2) with 5 seeds (0 to 4). Metrics: Acc@1 (T2) and Recall@1 (T3) per system; the shuffle effect is (3) minus (2), averaged over seeds, with paired bootstrap CIs. For static systems and BM25 the drop must be exactly 0 (assert; this is the H3 sanity check). Re-encode queries only (documents unchanged); cache.
**Accept:** `results/analysis/shuffle.csv`, figure of drops per pretrained system; H3 wording drafted from numbers (humans approve).

### S40 (H) A3: polysemy probe set
**Owner:** both (human-authored) | **Est.:** 3 to 4 h of human time, spread over week 1 to 3
**File:** `analysis/probe_set_polysemy.tsv`, tab-separated, header exactly: `id	word	word_romanized	intended_sense	hindi_sentence	english_match	english_contrast	status	annotator`.
**Design:** about 40 rows; at least 8 each for **कल (yesterday vs tomorrow)**, **आम (mango vs common/ordinary)**, **सोना (gold vs to sleep)**, and the remainder from other ambiguous words (e.g. बाल hair/child, कर tax/hand/doing). The Hindi sentence must contain enough context to settle the sense (for कल, the verb tense does it). `english_match` is a natural English rendering with the intended sense; `english_contrast` keeps the rest of the sentence the same but uses the *other* sense. Balance the senses across rows. `status` is `draft` until a second Hindi speaker verifies it, then `verified`; the analysis uses `verified` rows only. Illustrative drafts (to be checked by humans, not final): `कल` / "मैं कल दिल्ली गया था।" / "I went to Delhi yesterday." / "I will go to Delhi tomorrow."; `आम` / "उसने बाज़ार से मीठे आम ख़रीदे।" / "She bought sweet mangoes from the market." / a contrast sentence that reflects the "ordinary" sense.
**Agent task (A3 script `analysis/a3_polysemy.py`):** validate the TSV (all fields present; the ambiguous word occurs in `hindi_sentence`; `verified` rows only); for each system compute `cos(hi, match)` vs `cos(hi, contrast)`; a system **passes** an item if match > contrast; output a per-item pass/fail table and margins (`results/analysis/polysemy.csv`) and a LaTeX table. Static systems operate only through context words because the ambiguous word's vector is shared across senses. **Report qualitatively**; no CIs and no significance claims (n is small).

### S41 H2 verdict and hypothesis table
**Owner:** both | **Est.:** 2 h
Compute H2 from S24, S35 and S36: top-tercile and very-frequent-band P@1 of the best static vs the best pretrained (paired CIs), and `ft` vs `w2v` on inflected forms (paired CI). Write `results/analysis/hypotheses.csv` (H1, H2a, H2b, H3: verdict, numbers, CI). Humans approve the final wording.

### S42 (optional) A7: query type and length
**Owner:** B | **Est.:** 2 h
T3 hi-en Recall@1 by question length bucket and by question word (क्या, कब, कहाँ, कौन, कितना...). Skip if behind.

---

## Phase 7: Failure taxonomy (week 3 to 4). This is the core of RQ2.

### S43 Failure sampling export
**Owner:** B | **Est.:** 2 h
Using the best static and best pretrained systems (pre-registered rule), sample **50 random top-1 failures from T2 hi-en and 25 from T3 hi-en per system** (seed 42; take all failures if fewer). Write `analysis/taxonomy/failures_<system>.csv` with `item_id, task, system, query_hi, gold_en, retrieved_en, gold_rank, sim_gold, sim_retrieved, primary_label, secondary_label, notes, annotator` (the last four empty). Also write a shared overlap file of 20 items that **both** annotators label (for Cohen's kappa).

### S44 (H) Manual taxonomy labeling
**Owner:** both humans | **Est.:** 4 to 6 h total
Label each failure with a **primary** and optional **secondary** category from this codebook (extend with "other" notes rather than inventing categories mid-stream; record any codebook change in `docs/DECISIONS.md`): (1) inflection/morphology; (2) polysemy/sense ambiguity; (3) named entity or transliteration; (4) vocabulary layer mismatch (Sanskritized vs Perso-Arabic vs English loan); (5) multiword expression, light verb or idiom; (6) numbers, dates or units; (7) word order or syntactic divergence; (8) short or low-content sentence; (9) translation divergence (the reference is a free paraphrase); (10) distractor near-duplicate or topical confusion (T3); (11) truncation (T3, long Hindi paragraph cut by the encoder); (12) other (free-text note). Each human labels the overlap file independently first; compute agreement before labeling the rest. Split the remaining items.

### S45 Taxonomy aggregation and figures
**Owner:** A | **Est.:** 3 h
`analysis/a8_taxonomy.py`: Cohen's kappa on the overlap file; category counts per system and per task with the 12 categories; a stacked-bar figure; a table of 2 to 3 exemplar failures per major category (Hindi, gold, retrieved) for the report. Report agreement honestly even if low. Rows still without labels are listed, not guessed.

---

## Phase 8: Report, README, finish (week 3 to 4; start drafting during week 3)

### S46 Report assets script
**Owner:** B | **Est.:** 3 h
`scripts/make_report_assets.py` regenerates every table and figure used in the report from `results/` (LaTeX tables with CIs; figures as PDF) into `report/assets/`. The report must never contain a number that is not produced by this script or an analysis script.

### S47 Report writing (LaTeX in `report/`)
**Owner:** both | **Est.:** 2 to 3 days of human time
Use the format the instructors specify (S01). Suggested structure and what each section must contain:
1. **Introduction**: motivation, RQ1 and RQ2, contributions, and the explicit "comparison of systems, not a clean static vs pretrained effect" statement.
2. **Related work**: cite refs [1] to [11] (MUSE/CSLS, LaBSE, fastText, mE5, SIF, MuRIL, FLORES, XQuAD, Procrustes, MIRACL, Stanza); verify bibliographic details first.
3. **Data and tasks**: T1 to T3 definitions, sizes, T3 pool design and distractor choices, candidate vocabulary, common-query-set rule.
4. **Systems**: static training corpora (equal size, Hindi Wikipedia is small), alignment, SIF, encoders, baselines, controls, hyperparameters, compute and runtimes.
5. **Evaluation protocol**: metrics, bootstrap, best-system rule, hypotheses with decision rules.
6. **Results**: main table (Acc@1/MRR with CIs), controls, gap figure and H1 verdict, T3 settings and truncation, optional MIRACL.
7. **Analysis**: A1/A2 morphology and frequency, A4 vocabulary layers, A5 named entities, A6 shuffling and H3, A3 probe (qualitative), H2 verdict.
8. **Failure taxonomy** (centrepiece): codebook, agreement, distribution per system and task, exemplars, interpretation.
9. **Limitations**: in-distribution training of LaBSE and mE5, small test sets, small Hindi corpus, distractor design, Stanza and Wiktionary noise, MT quality in `bm25_mt`, contamination, non-exact reproducibility of gensim training.
10. **Conclusion**.
11. **Appendices**: reproducibility instructions, compute table, AI-use statement (from `docs/AI_USE_LOG.md`), per-member contribution statement.
Language rule: claims about causes ("because of subword n-grams") need a supporting analysis in the report; otherwise phrase as hypotheses.

### S48 README, AI-use statement, contributions
**Owner:** both | **Est.:** 3 h
README: one-paragraph summary, main results table, how to reproduce (ordered commands from `scripts/run_all.sh`), hardware used and runtimes, data sources and licenses, AI-use statement, per-member contributions, limitations pointer. Keep `scripts/run_all.sh` updated as steps complete (an ordered list of commands from download to report assets).

### S49 Final QA and submission
**Owner:** both | **Est.:** 1 day
1. Fresh clone into a temp directory; `uv sync`; `uv run pytest`; run a small end-to-end subset (e.g. T2 with LaBSE) from the README instructions.
2. Check `git log`: at least 20 meaningful commits, at least 12 distinct days, no large final-24-hour burst (course penalty above 80%), both members have commits, no data or weights tracked.
3. Verify every reference and dataset citation; check all numbers in the report against `results/`.
4. Proofread the PDF, check figures render with Devanagari text (use a font that supports it, e.g. Noto Sans Devanagari, via XeLaTeX or `fontspec`).
5. Submit the PDF and the repository link as the guidelines require.

---

## Appendix A: where results go

`results/results.csv` (aggregated, CIs), `results/per_item/*.csv` (per-query), `results/analysis/*.csv`, `results/plots/*.{pdf,png}`, `report/assets/` (generated), `docs/DATA.md`, `docs/DECISIONS.md`, `docs/AI_USE_LOG.md`, `docs/CONTRIBUTIONS.md`, `docs/vocab_layer_rules.md`.

## Appendix B: sanity ranges (heuristics, not guarantees)

If a result falls outside its range, investigate before continuing (see `AGENTS.md` section 15). These ranges come from general experience with these models, not from measurements on this setup.
- T2 chance Acc@1 is about 0.0005. LaBSE hi-en Acc@1 should be at least about 0.95. mE5 at least about 0.85. MuRIL (control) well above chance but far below LaBSE.
- T1 chance is about 5e-6 for a 200k candidate list. Our static systems should be well above chance (at least about 0.05); `ft_ref` should be higher or comparable. Accuracy on the seed pairs after fitting should be far above test accuracy.
- T3 `en-en` with mE5: Recall@5 at least about 0.85 on the 10,240-paragraph pool. `hi-en` at or below `en-en`.
- Static methods on T3 may be low; that is acceptable.
- Shuffle drop for static and BM25 is exactly 0.

## Appendix C: scope-cut order (when behind)

Never cut: T1 to T3 for `w2v`, `ft`, `labse`, `me5`, `bm25_mt`; the failure taxonomy (S43 to S45); the H1 gap analysis (S34); the H3 shuffling analysis (S39); the report. Cut in this order: (1) S33 MIRACL; (2) `ft_cc`; (3) S42 A7 and the plain-average and chunked-LaBSE ablations; (4) S24 cross-validated band (then H2's frequent-word clause uses the top tercile only, say so); (5) S38 named-entity density; (6) S37 vocabulary layers; (7) reduce S36 to NOUN and VERB buckets. Record every cut in `docs/DECISIONS.md` and in the report's limitations.

## Appendix D: suggested workstream split (two members)

- **Member A (static and lexical):** S02 to S05, S07, S10 to S13, S19 to S22, S24, S25, S31, S35 to S37, S45.
- **Member B (encoders and tasks):** S06, S08, S09, S14 to S18, S23, S27 to S30, S33, S34, S38, S39, S42, S43, S46.
- **Shared:** S00 to S01, S18, S26, S32, S40 (each writes about half of the probe set and verifies the other's), S41, S44 (each labels half plus the overlap), S47 to S49 (split report sections; each writes the sections for their own workstream).
Contributions are graded separately, so each member commits their own work under their own identity, and `docs/CONTRIBUTIONS.md` records it.

## Appendix E: suggested commit cadence

At minimum one commit after each of: S03/S04, S05, S06, S07, S08, S10, S11, S12, S14, S15, S16, S17, S18, S19 (config and script, not vectors), S21, S22, S23, S24, S25, S26, S27, S28, S29, S30, S31, S32, each analysis step, S40 (probe set), S43, S45, S46, S47 (per section), S48, S49. That gives well over 30 natural commits; spread them across days.

## Appendix F: risk register

| Risk | Mitigation |
|---|---|
| Hindi Wikipedia too small for good static vectors | Expected; report it; `ft_ref`/`ft_cc` give a reference point; do not tune on test |
| Static sentence vectors score very low on T3 | Legitimate finding; run the plain-average ablation once and move on |
| Hindi distractor shard has fewer than 10,000 paragraphs | Use all, log it, or enlarge the shard in S12; document in the report |
| LaBSE truncates Hindi paragraphs more than English | Log truncation counts; optional chunked-LaBSE ablation; discuss as a finding |
| Stanza or Wiktionary coverage is poor | Report coverage and error rates; reduce buckets (Appendix C) |
| Download blocked (FLORES, MIRACL, fastText) | Give humans the exact URL and path; use the fallbacks listed in S10 |
| CUDA not working on Fedora | Driver and Secure Boot first; encoders can run on CPU for small tasks as a last resort |
| Running out of time | Appendix C cut order; freeze at S32 |
| Over-claiming | Section 1 of `AGENTS.md`; limitations section; paired CIs |
