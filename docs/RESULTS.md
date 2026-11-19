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
