# Static vs. Pretrained Embeddings for Hindi-English Cross-lingual Retrieval

Course project for Computational Linguistics 2.

## Research questions
1. How do aligned static embeddings (Word2Vec, FastText) compare with pretrained multilingual encoders (LaBSE, mE5) at word, sentence and paragraph level?
2. Which properties of Hindi (morphology, polysemy, vocabulary layers, named entities, word order) explain where each approach fails?

## Tasks
- T1 Bilingual lexicon induction (MUSE en-hi)
- T2 Parallel sentence retrieval (FLORES)
- T3 Question-to-paragraph retrieval (XQuAD)

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
