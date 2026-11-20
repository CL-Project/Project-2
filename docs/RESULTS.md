# Results

Generated from `results/results.csv`; every number below comes from a completed
run in this repository. Regenerate any table with:

```
uv run python scripts/make_table.py --task t2_flores
```

95% percentile-bootstrap CIs (1,000 resamples over queries, seed 42).

## Checkpoint 1 (S18): T2 FLORES-200, 2,009 pairs

Cosine similarity over L2-normalized vectors. `bm25_mt` = opus-mt hi→en
translation of the Hindi query followed by English BM25 (`k1=1.5`, `b=0.75`),
so it is defined for `hi-en` only.

| system | hi-en Acc@1 | hi-en MRR | hi-en Recall@5 | en-hi Acc@1 | en-hi MRR | en-hi Recall@5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| bm25_mt | 0.958 [0.950, 0.967] | 0.969 [0.962, 0.975] | 0.982 [0.977, 0.987] | - | - | - |
| labse | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| me5 | 0.999 [0.997, 1.000] | 0.999 [0.998, 1.000] | 1.000 [1.000, 1.000] | 1.000 [0.999, 1.000] | 1.000 [0.999, 1.000] | 1.000 [1.000, 1.000] |
| muril | 0.971 [0.963, 0.978] | 0.981 [0.975, 0.986] | 0.992 [0.988, 0.996] | 0.895 [0.882, 0.909] | 0.926 [0.916, 0.936] | 0.965 [0.957, 0.973] |

All runs: `n_items` = 2,009, `n_excluded` = 0, 0 truncated inputs (FLORES
sentences are well inside LaBSE's 256 and mE5/MuRIL's 512 token limits).

```
_bm25_mt: n=2009, excluded=0; labse: n=2009, excluded=0; me5: n=2009, excluded=0; muril: n=2009, excluded=0_
```

### What these numbers show, and what they do not

- **T2 is close to ceiling for the sentence-trained encoders**, and LaBSE is at
  exactly 1.000 in both directions (2,009/2,009). T2 therefore has very little
  headroom to show a static-vs-pretrained gap: this matters for H1, which
  predicts the gap to *widen* with granularity, and will be discussed honestly
  in the report rather than hidden.
- **The 1.000 is not a bug.** Checked directly: gold-pair cosine is 0.855 on
  average versus 0.509 for the best non-gold candidate, the smallest margin over
  all 2,009 queries is 0.080, and a random-pairing control collapses to
  Acc@1 = 0.0005 (chance). Rows are verified to be aligned in both directions.
- **MuRIL (control) is much stronger than the plan anticipated.** PLAN S16 and
  Appendix B expect MuRIL to be "markedly worse"/"far below LaBSE" because it was
  not trained for sentence similarity; it reaches 0.971 hi-en and 0.895 en-hi.
  The plan says not to "fix" a control, so it is left as is and recorded in
  `docs/DECISIONS.md` (D16). It remains a control and is never eligible to be
  "best pretrained" (D07).
- **`bm25_mt` at 0.958 is a strong lexical baseline.** Its translations were
  spot-checked on eight sentences: fluent on the majority, with clear errors on
  named entities and rare words (`Gripen` → "Grippen", `हिमस्खलन` → "Snowflone",
  `उपसंस्कृति` → "sub-oxidment"). That error rate is part of the baseline and
  will be reported as a limitation.
- **In-distribution caveat (AGENTS.md §9.6):** LaBSE was trained on translation
  pairs and mE5 on retrieval/QA data, so FLORES is in-distribution for them.
  These numbers may be optimistic and are not evidence of a general advantage.

### Reproduce

```
uv run python scripts/run_eval.py --task t2_flores --system labse
uv run python scripts/run_eval.py --task t2_flores --system me5
uv run python scripts/run_eval.py --task t2_flores --system muril
uv run python scripts/run_eval.py --task t2_flores --system bm25_mt --settings hi-en
uv run python scripts/make_table.py --task t2_flores
```

## Checkpoint 2 (S26): T1 and T2 with every system

T1 primary view = **common query set, CSLS (k=10)**. Every system is scored on the
same 98,069-word Hindi / 112,703-word English candidate pool (S20), so the numbers
are directly comparable (`AGENTS.md` §9.2). The common set keeps only MUSE test
queries that every static system can embed and that have at least one gold
translation in the shared candidate pool: 1,458 of 1,493 for `hi-en`, 1,416 of
1,500 for `en-hi`. Excluded queries are counted, never dropped silently.

### T1 bilingual lexicon induction (MUSE, P@1 / P@5)

| system | hi-en P@1 | hi-en P@5 | en-hi P@1 | en-hi P@5 | hi-en_cvfreq P@1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| w2v | 0.376 [0.351, 0.402] | 0.547 [0.521, 0.575] | 0.345 [0.320, 0.370] | 0.556 [0.531, 0.583] | 0.528 [0.514, 0.543] |
| ft | 0.336 [0.311, 0.360] | 0.499 [0.473, 0.523] | 0.310 [0.286, 0.336] | 0.523 [0.496, 0.547] | 0.490 [0.476, 0.504] |
| ft_ref (control) | 0.483 [0.459, 0.508] | 0.695 [0.672, 0.719] | 0.480 [0.455, 0.504] | 0.702 [0.677, 0.728] | 0.726 [0.714, 0.739] |
| labse | 0.510 [0.484, 0.536] | 0.735 [0.711, 0.755] | 0.456 [0.431, 0.480] | 0.720 [0.696, 0.743] | 0.566 [0.553, 0.580] |
| me5 | 0.337 [0.311, 0.361] | 0.577 [0.551, 0.601] | 0.355 [0.331, 0.382] | 0.584 [0.560, 0.611] | 0.471 [0.457, 0.486] |
| muril (control) | 0.276 [0.254, 0.299] | 0.385 [0.360, 0.409] | 0.293 [0.268, 0.316] | 0.434 [0.407, 0.460] | 0.453 [0.440, 0.468] |

`hi-en_cvfreq` is the S24 band: 5-fold cross-validation over the 5,000 MUSE
**train** words, so these are genuinely high-frequency words and no mapping ever
saw the query it scored. n = 4,883 queries, 51 excluded. The test words are ranks
5,000 to 6,500 and are therefore relatively rare, which is why the same systems
score much higher here.

### T1 secondary view: coverage over the full test dictionary

Same runs, but every MUSE test entry is scored and a query that cannot be
embedded or matched is counted **wrong** (PLAN S22). This shows what the
common-set restriction costs; it is not the primary comparison.

| system | hi-en_cov P@1 | en-hi_cov P@1 |
| --- | ---: | ---: |
| w2v | 0.367 [0.343, 0.392] | 0.325 [0.301, 0.350] |
| ft | 0.328 [0.304, 0.352] | 0.293 [0.269, 0.316] |
| ft_ref (control) | 0.472 [0.446, 0.496] | 0.453 [0.427, 0.477] |
| labse | 0.498 [0.473, 0.522] | 0.431 [0.404, 0.455] |
| me5 | 0.329 [0.305, 0.351] | 0.335 [0.311, 0.358] |
| muril (control) | 0.269 [0.246, 0.292] | 0.277 [0.253, 0.299] |

n = 1,493 (`hi-en`) and 1,500 (`en-hi`), `n_excluded` = 0 by construction. The
cost of the common-set filter is small for every system (at most 0.023 P@1,
`muril` `en-hi`), so the primary table is not flattering anyone.

### T2 FLORES-200 with the static systems added (2,009 pairs)

| system | hi-en Acc@1 | hi-en MRR | hi-en Recall@5 | en-hi Acc@1 | en-hi MRR | en-hi Recall@5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| bm25_mt | 0.958 [0.950, 0.967] | 0.969 [0.962, 0.975] | 0.982 [0.977, 0.987] | - | - | - |
| w2v | 0.893 [0.881, 0.907] | 0.925 [0.916, 0.936] | 0.964 [0.956, 0.972] | 0.823 [0.806, 0.840] | 0.876 [0.864, 0.888] | 0.941 [0.931, 0.951] |
| ft | 0.892 [0.880, 0.906] | 0.925 [0.916, 0.935] | 0.967 [0.959, 0.975] | 0.836 [0.822, 0.853] | 0.883 [0.872, 0.895] | 0.942 [0.931, 0.951] |
| ft_ref (control) | 0.941 [0.931, 0.952] | 0.962 [0.954, 0.969] | 0.985 [0.980, 0.990] | 0.893 [0.880, 0.907] | 0.933 [0.924, 0.941] | 0.980 [0.974, 0.986] |
| labse | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| me5 | 0.999 [0.997, 1.000] | 0.999 [0.998, 1.000] | 1.000 [1.000, 1.000] | 1.000 [0.999, 1.000] | 1.000 [0.999, 1.000] | 1.000 [1.000, 1.000] |
| muril (control) | 0.971 [0.963, 0.978] | 0.981 [0.975, 0.986] | 0.992 [0.988, 0.996] | 0.895 [0.882, 0.909] | 0.926 [0.916, 0.936] | 0.965 [0.957, 0.973] |

Ablation (`hi-en`, plain average instead of SIF, no PC removal): `w2v`
Acc@1 = 0.451 [0.431, 0.472], `ft` 0.417 [0.395, 0.440].

### What these numbers show, and what they do not

- **The static T2 result is real, and it was checked rather than assumed.** SIF
  composition with PC removal puts `w2v` at 0.893 Acc@1 `hi-en`; the plain
  average on the same vectors gives 0.451. Three controls were run before this
  was logged: (1) shuffled gold labels give Acc@1 = 0.000000 (chance 0.000498);
  (2) dropping the Procrustes mapping, so Hindi queries stay in Hindi space,
  gives 0.0010; (3) random vectors give 0.0000. The ablation ladder is plain
  average 0.4515 → SIF weights without PC 0.7616 → SIF with PC 0.8935. The
  principal component is nearly the same direction in both spaces
  (`cos(pc_hi, pc_en) = 0.978`), which is what makes its removal a *shared*
  transformation rather than a language-specific one.
- **Why PC removal helps so much here.** The first component of Wikipedia text
  is a "common word" direction present in both languages. Removing it before
  comparing a Hindi query with English candidates strips a large shared offset
  that would otherwise dominate cosine similarity for every pair. This is exactly
  the mechanism Arora et al. describe, and it is an observation about *this*
  setting, not a general claim.
- **This does not rescue the static systems on T1.** The same systems stay at
  0.336 to 0.376 P@1 on word-level induction, far below `ft_ref` (0.483) and
  LaBSE (0.510). Word-level and sentence-level behaviour diverge sharply, which
  is the point of running all three granularities.
- **H1's predicted ordering already looks wrong at T1 vs T2.** Using the
  pre-registered rule (best static = highest T2 `hi-en` MRR in {`w2v`, `ft`};
  best pretrained = highest in {`labse`, `me5`}), the best static is `w2v`
  (MRR 0.9253, narrowly over `ft` at 0.9248) and the best pretrained is LaBSE
  (1.000). The T1 gap is 0.510 − 0.376 = **0.134** while the T2 gap is
  1.000 − 0.925 = **0.075**. H1 predicts gap(T1) < gap(T2) < gap(T3); the first
  inequality is already reversed. **No verdict is issued here** — that is S34's
  job, with paired bootstrap CIs over the same queries — but the report must not
  be written as if H1 were on track.
- **`ft` does not beat `w2v` on T1**, contrary to H2's second clause: 0.336 vs
  0.376 (`hi-en`), 0.490 vs 0.528 (`cvfreq`). The fastText advantage has to be
  tested on the *inflected subset* before anything is claimed (S36), not on the
  aggregate.
- **The Hindi corpus is the binding constraint.** It yields 140,590 vocabulary
  types against English's 147,988, and the shared pool is 98,069 Hindi words.
  `ft_ref`, trained on a much larger corpus, beats both our systems on T1 as
  expected; that comparison is a corpus-size effect at least as much as a
  subword effect, and must be described that way.

### Scope review against the pre-registered cut list (Appendix C)

Every step in the "never cut" list is complete or on track: T1 to T3 for `w2v`, `ft`,
`labse`, `me5` and `bm25_mt` (T3 pending S27 to S31, which is next), the failure taxonomy
(S43 to S45), the H1 gap analysis (S34) and the H3 shuffling analysis (S39) all remain in
scope. Nothing has been cut yet. The optional items still open are `ft_cc`, S33 MIRACL,
S42 A7 and the chunked-LaBSE ablation; none is blocking. `S24` — the item the cut list
would drop fourth — was completed, so H2's frequent-word clause uses the real
cross-validated band rather than the top-tercile fallback.

### Reproduce

```
uv run python scripts/run_eval.py --task t1_bli --system w2v
uv run python scripts/run_eval.py --task t1_bli --system ft
uv run python scripts/run_eval.py --task t1_bli --system ft_ref
uv run python scripts/run_eval.py --task t1_bli --system labse
uv run python scripts/run_eval.py --task t1_bli --system me5
uv run python scripts/run_eval.py --task t1_bli --system muril
uv run python scripts/run_eval.py --task t1_bli --system w2v --coverage-only   # S22 secondary view
uv run python scripts/run_eval.py --task t1_bli --system w2v --cvfreq          # S24 band
uv run python scripts/run_eval.py --task t2_flores --system w2v
uv run python scripts/run_eval.py --task t2_flores --system ft
uv run python scripts/run_eval.py --task t2_flores --system ft_ref
uv run python scripts/run_eval.py --task t2_flores --system w2v --ablation     # plain average
uv run python scripts/make_table.py --task t1_bli --settings hi-en en-hi hi-en_cvfreq --metrics P@1 P@5
```

