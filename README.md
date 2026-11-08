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
TODO: `scripts/download_data.py` -> `scripts/preprocess_wiki.py` -> `scripts/train_static.py` -> `scripts/run_eval.py`

## Results
TODO: results table and the static-vs-pretrained gap figure.

## Repository layout
- `configs/` dataset, model and task settings
- `scripts/` runnable entry points
- `src/` library code (text processing, metrics, retrieval, embeddings, tasks)
- `tests/` unit tests
- `analysis/` linguistic analyses and the polysemy probe set
- `results/` result CSVs and plots
- `docs/` decision log, data card, AI-use log, contributions
- `report/` LaTeX report
- `data/` downloaded and processed data (git-ignored)

## AI-use statement
TODO: describe which parts were generated with AI assistance, how they were specified, tested and reviewed.

## Contributions
TODO: per-member contribution statement.

## References
- Conneau et al. (2018). Word Translation Without Parallel Data.
- Feng et al. (2022). Language-agnostic BERT Sentence Embedding.
- Bojanowski et al. (2017). Enriching Word Vectors with Subword Information.
- Wang et al. (2024). Multilingual E5 Text Embeddings: A Technical Report.
