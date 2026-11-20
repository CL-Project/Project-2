"""Train and load static embeddings (S19, S20).

Two families, both 300-dimensional skip-gram models trained on the equal-sized
Wikipedia token files:

* ``w2v`` — gensim :class:`~gensim.models.Word2Vec`.  Out-of-vocabulary words
  cannot be embedded at all (``AGENTS.md`` §10.8).
* ``ft`` — gensim :class:`~gensim.models.FastText` with character n-grams 3 to 6,
  saved *as the full model* so the subword vectors survive a reload.

The trained vectors are git-ignored artefacts under ``data/vectors/static/``.
Hyperparameters come from ``configs/models.yaml``; nothing here is hard-coded.

Reproducibility caveat (recorded in ``docs/DECISIONS.md``): gensim is exactly
reproducible only with ``workers=1``.  With ``workers=8`` two runs differ
slightly, so the trained vectors are saved and every result is tied to them.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS_CONFIG_PATH = REPO_ROOT / "configs" / "models.yaml"
STATIC_ROOT = REPO_ROOT / "data" / "vectors" / "static"
PROCESSED_ROOT = REPO_ROOT / "data" / "processed"

SYSTEMS = ("w2v", "ft")
LANGUAGES = ("hi", "en")

LOGGER = logging.getLogger("static")


def load_models_config(path: Path = MODELS_CONFIG_PATH) -> dict[str, Any]:
    """Load ``configs/models.yaml``."""
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def corpus_path(lang: str, processed_root: Path | str = PROCESSED_ROOT) -> Path:
    """Return the tokenized Wikipedia corpus for ``lang``."""
    if lang not in LANGUAGES:
        raise ValueError(f"unsupported language: {lang!r}")
    return Path(processed_root) / f"wiki.{lang}.tok.txt"


def model_path(system: str, lang: str, static_root: Path | str = STATIC_ROOT) -> Path:
    """Return the saved full-model path for ``(system, lang)``."""
    _check_system(system)
    return Path(static_root) / f"{system}.{lang}.model"


def vocab_path(system: str, lang: str, static_root: Path | str = STATIC_ROOT) -> Path:
    """Return the saved ``KeyedVectors`` path for ``(system, lang)``."""
    _check_system(system)
    return Path(static_root) / f"{system}.{lang}.kv"


def _check_system(system: str) -> None:
    if system not in SYSTEMS:
        raise ValueError(f"unsupported static system {system!r}; expected one of {SYSTEMS}")


def training_params(system: str, config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return the gensim keyword arguments for ``system`` from the config.

    Unknown keys are filtered out for the model they do not belong to, so the
    config can stay flat and readable.
    """
    _check_system(system)
    models = load_models_config() if config is None else config
    static = models["static"]
    block = dict(static["word2vec" if system == "w2v" else "fasttext"])
    params: dict[str, Any] = {
        "vector_size": int(static["dim"]),
        "window": int(block["window"]),
        "min_count": int(block["min_count"]),
        "sg": int(block.get("sg", 1)),
        "negative": int(block["negative"]),
        "epochs": int(block["epochs"]),
        "workers": int(block["workers"]),
        "max_final_vocab": int(static["vocab_cap"]),
    }
    if system == "ft":
        params["min_n"] = int(block["min_n"])
        params["max_n"] = int(block["max_n"])
        if "bucket" in block:
            params["bucket"] = int(block["bucket"])
    return params


def train(
    system: str,
    lang: str,
    *,
    seed: int = 42,
    config: dict[str, Any] | None = None,
    corpus: Path | str | None = None,
    static_root: Path | str = STATIC_ROOT,
    force: bool = False,
) -> dict[str, Any]:
    """Train one ``(system, lang)`` model and save it; skip when the output exists.

    Returns:
        A summary dict with the vocabulary size, training time and paths, suitable
        for logging and for ``docs/DATA.md``.

    The full model is saved for both families.  fastText needs its n-gram
    vectors, so saving only ``.kv`` would silently disable subword information.
    """
    _check_system(system)
    if lang not in LANGUAGES:
        raise ValueError(f"unsupported language: {lang!r}")

    from gensim.models import FastText, Word2Vec
    from gensim.models.word2vec import LineSentence

    source = Path(corpus) if corpus is not None else corpus_path(lang)
    if not source.exists():
        raise FileNotFoundError(f"training corpus not found: {source}")
    destination = model_path(system, lang, static_root)
    summary_path = destination.with_suffix(".summary.json")

    if destination.exists() and not force:
        LOGGER.info("skip %s (already trained at %s)", destination.name, destination)
        return _read_summary(summary_path, destination)

    params = training_params(system, config)
    destination.parent.mkdir(parents=True, exist_ok=True)
    LOGGER.info("training %s.%s: %s, %d lines", system, lang, params, _line_count(source))

    started = time.perf_counter()
    sentences = LineSentence(str(source))
    model_class = Word2Vec if system == "w2v" else FastText
    model = model_class(sentences, seed=seed, **params)
    elapsed = time.perf_counter() - started

    model.save(str(destination))
    model.wv.save(str(vocab_path(system, lang, static_root)))
    if system == "ft":
        # Export the in-vocabulary vectors (word + n-grams) while the n-gram
        # table is still in memory; later loads can then avoid it.
        export_vocab_vectors(system, lang, static_root=static_root, model=model, force=True)
    summary = {
        "system": system,
        "lang": lang,
        "corpus": str(source.relative_to(REPO_ROOT)) if source.is_relative_to(REPO_ROOT) else str(source),
        "vector_size": int(model.wv.vector_size),
        "vocab_size": int(len(model.wv.index_to_key)),
        "train_lines": _line_count(source),
        "epochs": params["epochs"],
        "workers": params["workers"],
        "seed": seed,
        "runtime_s": round(elapsed, 2),
        "model_path": str(destination.relative_to(REPO_ROOT)) if destination.is_relative_to(REPO_ROOT) else str(destination),
    }
    summary_path.write_text(_json_dumps(summary), encoding="utf-8")
    LOGGER.info(
        "trained %s.%s: %d words, %.1f s -> %s",
        system,
        lang,
        summary["vocab_size"],
        elapsed,
        destination.name,
    )
    return summary


def _line_count(path: Path, block: int = 1 << 20) -> int:
    """Count lines without loading the file."""
    with path.open("rb") as handle:
        return sum(block.count(b"\n") for block in iter(lambda: handle.read(block), b""))


def _json_dumps(payload: dict[str, Any]) -> str:
    """Serialize a summary dict with a trailing newline."""
    import json

    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def _read_summary(path: Path, model: Path) -> dict[str, Any]:
    """Return the saved summary, or a minimal one when it is missing."""
    import json

    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"model_path": str(model), "note": "summary missing"}


def vocab_vectors_path(system: str, lang: str, static_root: Path | str = STATIC_ROOT) -> Path:
    """Return the path of the in-vocabulary ``KeyedVectors`` export."""
    _check_system(system)
    return Path(static_root) / f"{system}.{lang}.vocab.kv"


def export_vocab_vectors(
    system: str,
    lang: str,
    *,
    static_root: Path | str = STATIC_ROOT,
    model=None,
    force: bool = False,
) -> Path:
    """Export in-vocabulary fastText vectors to a small ``KeyedVectors`` file.

    A fastText vector for an in-vocabulary word is **the word vector plus the
    sum of its character n-gram vectors**, so ``wv.vectors_vocab`` is *not* that
    vector.  Taking the row alone produces meaningless neighbours; calling
    ``wv[word]`` produces the correct one but requires the full 2,000,000-row
    n-gram table (about 2.4 GB).  This export does the expensive lookup once and
    stores a 170 MB ``KeyedVectors`` that every later run can load cheaply.
    """
    _check_system(system)
    if system != "ft":
        raise ValueError("vocab export is only needed for fastText")
    path = vocab_vectors_path(system, lang, static_root)
    if path.exists() and not force:
        return path

    import numpy as np
    from gensim.models import FastText, KeyedVectors

    owns_model = model is None
    if model is None:
        model = FastText.load(str(model_path(system, lang, static_root)))
    words = list(model.wv.index_to_key)
    matrix = np.vstack([np.asarray(model.wv[word], dtype=np.float32) for word in words])
    cart = KeyedVectors(vector_size=int(model.wv.vector_size))
    cart.add_vectors(words, matrix)
    path.parent.mkdir(parents=True, exist_ok=True)
    cart.save(str(path))
    LOGGER.info("exported %s (%d x %d)", path.name, *matrix.shape)
    if owns_model:
        del model
    return path


def load(system: str, lang: str, *, static_root: Path | str = STATIC_ROOT, with_subwords: bool = False):
    """Load the trained vectors of ``(system, lang)``.

    For fastText, ``with_subwords=False`` returns the in-vocabulary export built
    by :func:`export_vocab_vectors` (word vector *plus* n-grams, but without the
    2,000,000-row n-gram table in memory).  Pass ``with_subwords=True`` only when
    genuinely out-of-vocabulary words must be embedded (``AGENTS.md`` §10.8).
    """
    _check_system(system)
    if lang not in LANGUAGES:
        raise ValueError(f"unsupported language: {lang!r}")

    from gensim.models import FastText, KeyedVectors

    if system == "ft":
        path = model_path(system, lang, static_root)
        if not path.exists():
            raise FileNotFoundError(f"no trained model at {path}; run scripts/train_static.py first")
        if with_subwords:
            return FastText.load(str(path)).wv
        export = vocab_vectors_path(system, lang, static_root)
        if not export.exists():
            export_vocab_vectors(system, lang, static_root=static_root)
        return KeyedVectors.load(str(export), mmap="r")

    path = vocab_path(system, lang, static_root)
    if path.exists():
        return KeyedVectors.load(str(path), mmap="r")
    model = model_path(system, lang, static_root)
    if not model.exists():
        raise FileNotFoundError(f"no trained model at {model}; run scripts/train_static.py first")
    from gensim.models import Word2Vec

    return Word2Vec.load(str(model)).wv


def load_pretrained(
    path: Path | str,
    *,
    limit: int = 200_000,
):
    """Load the first ``limit`` vectors of a pretrained ``.vec`` file.

    The files are frequency-sorted and enormous, so the full file is never read
    (``AGENTS.md`` §10.9).
    """
    from gensim.models import KeyedVectors

    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"pretrained vectors not found: {source}")
    LOGGER.info("loading %s (first %d vectors)", source.name, limit)
    return KeyedVectors.load_word2vec_format(str(source), limit=limit)


def nearest_neighbours(vectors, words: list[str], topn: int = 10) -> dict[str, list[tuple[str, float]]]:
    """Return the nearest neighbours of ``words`` that are in the vocabulary.

    Used for the S19 sanity check (``भारत`` should return Indian entities or
    Hindi geography words).  Words missing from the vocabulary are reported with
    an empty list rather than silently skipped.
    """
    result: dict[str, list[tuple[str, float]]] = {}
    for word in words:
        if word not in vectors:
            result[word] = []
            continue
        result[word] = [(neighbour, round(float(score), 4)) for neighbour, score in vectors.most_similar(word, topn=topn)]
    return result


def vocabulary(vectors) -> set[str]:
    """Return the vocabulary of a ``KeyedVectors``-like object as a set."""
    return set(vectors.key_to_index)


def vector_matrix(vectors, words: list[str]) -> np.ndarray:
    """Return the ``(len(words), dim)`` float32 matrix for ``words`` (all must be present)."""
    missing = [word for word in words if word not in vectors]
    if missing:
        raise KeyError(f"{len(missing)} words are not in the vocabulary, e.g. {missing[:5]}")
    return np.vstack([np.asarray(vectors[word], dtype=np.float32) for word in words])
