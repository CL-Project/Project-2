# Static vs. Pretrained Embeddings for Hindi-English Cross-lingual Retrieval

Course project for Computational Linguistics 2.

## Research questions
**RQ1.** How do aligned static embeddings (Word2Vec, fastText) compare with pretrained encoders (LaBSE, mE5) at the word, sentence and paragraph level? The systems differ in training objective, training data and unit of representation, so we claim a *comparison of systems*, not a clean "static vs. pretrained" effect. Mean-pooled MuRIL and pretrained aligned fastText serve as controls.

Pre-stated hypotheses:
- **H1** the gap widens with granularity.
- **H2** static models match or beat pretrained ones on word-level lexicon induction for frequent words, and fastText beats Word2Vec on inflected forms.
- **H3** word shuffling hurts only the pretrained models.

**RQ2 (core).** Which properties of Hindi explain where each system fails: inflection, polysemy, Sanskritic vs. Perso-Arabic vs. English-origin vocabulary, transliterated named entities, word order or something else? The centrepiece is a manual failure taxonomy of the best static and the best pretrained system.

## Tasks
- T1 Bilingual lexicon induction (MUSE en-hi)
- T2 Parallel sentence retrieval (FLORES-200)
- T3 Hindi question to English paragraph retrieval (XQuAD)

## Setup (Fedora, uv)
1. NVIDIA driver from RPM Fusion; `sudo dnf install gcc make git`
2. Install [uv](https://docs.astral.sh/uv/) (check `dnf search uv`, or use the official installer).
3. In the repo root: `uv sync` (creates `.venv`, installs Python 3.12 if needed, and installs the exact versions in `uv.lock`).
4. Check the environment: `uv run python scripts/check_env.py`
5. Run tests: `uv run pytest`

Add a dependency with `uv add <package>`; commit `pyproject.toml` and `uv.lock` together.

## Reproduce
Python is always invoked through `uv run` from the repository root; every script is argparse-driven (`--help`).

```bash
# 1. Data (S10-S13): download, validate, build the Wikipedia corpora
uv run python scripts/download_data.py                 # data/raw + MANIFEST.json
uv run python scripts/validate_data.py                 # writes docs/DATA.md
uv run python scripts/preprocess_wiki.py               # data/processed/wiki.*

# 2. T2 parallel sentence retrieval with the pretrained encoders (S14-S17)
uv run python scripts/run_eval.py --task t2_flores --system labse         # both directions
uv run python scripts/run_eval.py --task t2_flores --system me5
uv run python scripts/run_eval.py --task t2_flores --system muril          # control
uv run python scripts/run_eval.py --task t2_flores --system bm25_mt --settings hi-en

# 3. Results table (S18)
uv run python scripts/make_table.py --task t2_flores

# 4. Static vectors and T1 (S19-S26); four training runs, sequential, a few hours
systemd-inhibit --what=idle:sleep:handle-lid-switch \
    uv run python scripts/train_static.py --all
uv run python scripts/build_candidates.py              # shared 200k candidate pools
uv run python scripts/run_eval.py --task t2_flores --system w2v
uv run python scripts/run_eval.py --task t2_flores --system ft
uv run python scripts/run_eval.py --task t2_flores --system ft_ref
uv run python scripts/run_eval.py --task t2_flores --system w2v --ablation   # plain average
for s in w2v ft ft_ref labse me5 muril; do
    uv run python scripts/run_eval.py --task t1_bli --system "$s"
done
uv run python scripts/run_eval.py --task t1_bli --system w2v --coverage-only  # S22 secondary
uv run python scripts/run_eval.py --task t1_bli --system w2v --cvfreq         # S24 band
uv run python scripts/make_table.py --task t1_bli --settings hi-en en-hi hi-en_cvfreq \
    --metrics P@1 P@5
```

Encoded vectors are cached under `data/vectors/embeddings/`; pass `--force` to re-encode.
Machine translations for `bm25_mt` are cached in `data/processed/flores_hi2en.tsv`.
Raw data, vectors and cached embeddings are git-ignored (`AGENTS.md` §12).

## Results
Current numbers, with 95% bootstrap CIs and the caveats that belong with them, are in
[`docs/RESULTS.md`](docs/RESULTS.md). **Checkpoint 2 (T1 word induction, P@1, common
candidate pool, CSLS k=10):** `w2v` 0.376, `ft` 0.336, LaBSE 0.510, mE5 0.337, with
`ft_ref` at 0.483 and MuRIL at 0.276 as controls. **Checkpoint 1 (T2, 2,009 FLORES pairs,
Acc@1):** LaBSE 1.000, mE5 0.999, MuRIL 0.971, BM25+MT 0.958, and the static SIF systems at
`w2v` 0.893 / `ft` 0.892 for `hi-en`. T2 is close to ceiling for the sentence-trained
encoders, so it leaves little room to observe a static-vs-pretrained gap; that is reported
rather than hidden.

## Repository layout
- `configs/` dataset, model and task settings
- `scripts/` runnable entry points
- `src/` library code (text processing, metrics, retrieval, embeddings, tasks)
- `tests/` unit tests
- `analysis/` linguistic analyses and the polysemy probe set
- `results/` result CSVs and plots
- `docs/` decision log, data card, contributions
- `report/` LaTeX report
- `data/` downloaded and processed data (git-ignored)

## Contributions
TODO: per-member contribution statement.

## References
- Conneau et al. (2018). Word Translation Without Parallel Data.
- Feng et al. (2022). Language-agnostic BERT Sentence Embedding.
- Bojanowski et al. (2017). Enriching Word Vectors with Subword Information.
- Wang et al. (2024). Multilingual E5 Text Embeddings: A Technical Report.
