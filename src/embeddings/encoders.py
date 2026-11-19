"""Pretrained encoder wrappers with caching (S14): LaBSE, multilingual-E5 and MuRIL.

All three systems return row-wise L2-normalized float32 vectors in input order,
so a row index in one language matrix always refers to the same item id
(``AGENTS.md`` §10.5).  Design points:

* **LaBSE and mE5** run through ``sentence-transformers`` because their modules
  include a pooling layer and, for LaBSE, a dense projection.  Pooling the base
  transformers model by hand would silently produce a different representation.
* **MuRIL** is a control that was not trained for sentence similarity: it runs
  through ``transformers`` with mean pooling over the attention mask.  Means are
  accumulated in fp32 even when the model runs in fp16 (``AGENTS.md`` §10.7).
* **mE5 prefixes are mandatory** (``AGENTS.md`` §10.2): ``query: `` for ``word``,
  ``sentence`` and ``query``; ``passage: `` for ``passage``.
* Inputs are sorted by tokenized length so batches are not padded to the
  longest sequence in the whole call, then restored to the input order.
* Truncation is counted against each model's configured ``max_len`` and stored
  in the cache sidecar: Hindi tokenizes into more pieces than English, so the
  truncation rate is a result to report (``AGENTS.md`` §10.3).
* Embeddings are cached as float32 ``.npy`` plus a ``.json`` sidecar; a cache is
  reused only when every field of the sidecar matches the request.

The cache key is ``<system>/<dataset>_<lang>.npy`` by default, as specified in
``AGENTS.md`` §8.  When several *kinds* are encoded for the same dataset and
language (a query and a passage pool, say), pass a distinct ``kind_tag`` so the
two files do not overwrite each other; the sidecar records it.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS_CONFIG_PATH = REPO_ROOT / "configs" / "models.yaml"
CACHE_ROOT = REPO_ROOT / "data" / "vectors" / "embeddings"

KINDS = ("word", "sentence", "query", "passage")
SENTENCE_TRANSFORMERS_POOLING = "sentence_transformers"

LOGGER = logging.getLogger("encoders")


@dataclass(frozen=True)
class EncoderSettings:
    """Resolved per-system encoder settings from ``configs/models.yaml``."""

    system: str
    hf_id: str
    max_len: int
    fp16: bool
    pooling: str
    batch_size: int
    query_prefix: str = ""
    passage_prefix: str = ""

    def prefix_for(self, kind: str) -> str:
        """Return the text prefix ``kind`` requires for this system."""
        if self.system != "me5":
            return ""
        return self.passage_prefix if kind == "passage" else self.query_prefix


@dataclass
class EncodeStats:
    """Counters returned alongside an encoding run (logged and cached)."""

    n_items: int
    n_truncated: int
    max_len: int
    dim: int
    batch_size: int
    n_oov_pieces: int = 0

    @property
    def truncated_fraction(self) -> float:
        """Fraction of inputs whose token sequence exceeded ``max_len``."""
        return self.n_truncated / self.n_items if self.n_items else 0.0

    def as_dict(self) -> dict[str, Any]:
        """Return the sidecar-ready representation."""
        return {
            "n_items": self.n_items,
            "n_truncated": self.n_truncated,
            "truncated_fraction": round(self.truncated_fraction, 6),
            "max_len": self.max_len,
            "dim": self.dim,
            "batch_size": self.batch_size,
        }


def load_models_config(path: Path = MODELS_CONFIG_PATH) -> dict[str, Any]:
    """Load ``configs/models.yaml``."""
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def encoder_settings(system: str, config: dict[str, Any] | None = None) -> EncoderSettings:
    """Return the resolved settings for ``system``.

    Raises ``KeyError`` for an unknown system id so a typo fails loudly instead
    of silently falling back to a default model.
    """
    models = load_models_config() if config is None else config
    encoders = models.get("encoders", {})
    if system not in encoders:
        raise KeyError(f"unknown encoder system {system!r}; known: {sorted(encoders)}")
    entry = dict(encoders[system])
    defaults = models.get("encoder_defaults", {})
    return EncoderSettings(
        system=system,
        hf_id=entry["hf_id"],
        max_len=int(entry.get("max_len", 512)),
        fp16=bool(entry.get("fp16", False)),
        pooling=entry.get("pooling", SENTENCE_TRANSFORMERS_POOLING),
        batch_size=int(entry.get("batch_size", defaults.get("batch_size", 32))),
        query_prefix=entry.get("query_prefix", ""),
        passage_prefix=entry.get("passage_prefix", ""),
    )


def _device() -> str:
    """Return ``"cuda"`` when a GPU is available, else ``"cpu"``."""
    try:
        import torch
    except ImportError:  # pragma: no cover - torch is a hard dependency
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _display_path(path: Path) -> str:
    """Return ``path`` relative to the repository root when possible, else absolute."""
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _texts_digest(texts: Sequence[str]) -> str:
    """Return a stable digest of the input texts (cache safety check)."""
    digest = hashlib.sha256()
    for text in texts:
        digest.update(text.encode("utf-8"))
        digest.update(b"\x1e")
    return digest.hexdigest()


def _checked_texts(texts: Iterable[str], kind: str) -> list[str]:
    """Validate the input sequence and ``kind``, returning a list of strings."""
    if kind not in KINDS:
        raise ValueError(f"unsupported kind {kind!r}; expected one of {KINDS}")
    values = list(texts)
    if not values:
        raise ValueError("texts must not be empty")
    for index, value in enumerate(values):
        if not isinstance(value, str):
            raise TypeError(f"texts[{index}] must be a str, got {type(value).__name__}")
    return values


def _truncation_counts(
    tokenizer: Any, texts: Sequence[str], max_len: int, block: int = 20_000
) -> tuple[np.ndarray, int]:
    """Return per-input token lengths and the number exceeding ``max_len``.

    Lengths include the special tokens that the model adds, because that is what
    gets truncated.  Inputs are tokenized without truncation so the true length
    is visible, in blocks so a 200k-word list does not blow up memory.
    """
    if not texts:
        return np.zeros(0, dtype=np.int64), 0
    lengths = np.empty(len(texts), dtype=np.int64)
    for start in range(0, len(texts), block):
        batch = list(texts[start : start + block])
        encoded = tokenizer(
            batch, add_special_tokens=True, truncation=False, padding=False, verbose=False
        )
        lengths[start : start + len(batch)] = [len(ids) for ids in encoded["input_ids"]]
    return lengths, int(np.count_nonzero(lengths > max_len))


def _resolve_max_len(kind: str, settings: EncoderSettings, max_len: int | None, defaults: dict[str, Any]) -> int:
    """Return the effective maximum sequence length for this request."""
    if max_len is not None:
        if max_len < 1:
            raise ValueError("max_len must be >= 1")
        return int(max_len)
    if kind == "word":
        return min(settings.max_len, int(defaults.get("word_max_len", 16)))
    return settings.max_len


def _resolve_batch_size(kind: str, batch_size: int | None, settings: EncoderSettings, defaults: dict[str, Any]) -> int:
    """Return the effective batch size for this request."""
    if batch_size is not None:
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        return int(batch_size)
    if kind == "word":
        return int(defaults.get("word_batch_size", 512))
    return max(1, min(settings.batch_size, 256))


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """Return ``matrix`` as float32 with every row scaled to unit length."""
    values = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    normalized = np.zeros_like(values)
    np.divide(values, norms, out=normalized, where=norms > 0)
    return normalized


class _Encoder:
    """Lazily loaded model that encodes a list of strings into float32 rows."""

    def __init__(self, settings: EncoderSettings) -> None:
        self.settings = settings
        self.device = _device()
        self._model: Any = None
        self._tokenizer: Any = None

    def _load(self) -> None:
        """Load the tokenizer and model once, in fp16 when configured and possible."""
        if self._model is not None:
            return
        use_half = self.settings.fp16 and self.device == "cuda"
        if self.settings.pooling == SENTENCE_TRANSFORMERS_POOLING:
            import torch
            from sentence_transformers import SentenceTransformer

            kwargs = {"dtype": torch.float16} if use_half else {}
            model = SentenceTransformer(self.settings.hf_id, device=self.device, model_kwargs=kwargs)
            model.max_seq_length = self.settings.max_len
            self._model = model
            self._tokenizer = model.tokenizer
            LOGGER.info("loaded %s (%s, %s) on %s", self.settings.hf_id, self.settings.pooling, "fp16" if use_half else "fp32", self.device)
            return

        import torch
        from transformers import AutoModel, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self.settings.hf_id)
        self._model = AutoModel.from_pretrained(
            self.settings.hf_id, dtype=torch.float16 if use_half else torch.float32
        ).eval().to(self.device)
        LOGGER.info("loaded %s (mean pooling, %s) on %s", self.settings.hf_id, "fp16" if use_half else "fp32", self.device)

    def tokenizer(self) -> Any:
        """Return the model tokenizer, loading the model if necessary."""
        self._load()
        return self._tokenizer

    def _encode_sentence_transformers(self, texts: Sequence[str], batch_size: int) -> np.ndarray:
        """Encode with ``SentenceTransformer.encode`` (no normalization yet)."""
        return np.asarray(
            self._model.encode(
                list(texts),
                batch_size=batch_size,
                convert_to_numpy=True,
                normalize_embeddings=False,
                show_progress_bar=False,
            ),
            dtype=np.float32,
        )

    def _encode_mean_pooled(self, texts: Sequence[str], batch_size: int, max_len: int) -> np.ndarray:
        """Encode with ``transformers`` and masked mean pooling accumulated in fp32."""
        import torch

        outputs = []
        for start in range(0, len(texts), batch_size):
            batch = list(texts[start : start + batch_size])
            encoded = self._tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_len,
            ).to(self.device)
            with torch.inference_mode():
                hidden = self._model(**encoded).last_hidden_state
                mask = encoded["attention_mask"].unsqueeze(-1).to(torch.float32)
                pooled = (hidden.to(torch.float32) * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            outputs.append(pooled.cpu().numpy())
        return np.concatenate(outputs, axis=0).astype(np.float32, copy=False)

    def encode(self, texts: Sequence[str], kind: str, batch_size: int, max_len: int) -> np.ndarray:
        """Encode ``texts`` and return L2-normalized float32 rows in input order."""
        self._load()
        prefix = self.settings.prefix_for(kind)
        prepared = [prefix + text for text in texts]
        if self.settings.pooling == SENTENCE_TRANSFORMERS_POOLING:
            self._model.max_seq_length = max_len
            return self._encode_sentence_transformers(prepared, batch_size)
        return self._encode_mean_pooled(prepared, batch_size, max_len)

    def free(self) -> None:
        """Drop the model and free GPU memory; the next call reloads it."""
        self._model = None
        self._tokenizer = None
        try:
            import torch
        except ImportError:  # pragma: no cover
            return
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


_LOADED: dict[str, _Encoder] = {}


def _get_encoder(settings: EncoderSettings) -> _Encoder:
    """Return a process-wide encoder for ``settings``, loading it on first use."""
    encoder = _LOADED.get(settings.system)
    if encoder is None or encoder.settings != settings:
        encoder = _Encoder(settings)
        _LOADED[settings.system] = encoder
    return encoder


def release(system: str | None = None) -> None:
    """Free loaded models (all of them when ``system`` is ``None``)."""
    names = list(_LOADED) if system is None else [system]
    for name in names:
        encoder = _LOADED.pop(name, None)
        if encoder is not None:
            encoder.free()


def cache_paths(
    system: str,
    dataset: str,
    lang: str,
    kind_tag: str | None = None,
    *,
    cache_root: Path | str = CACHE_ROOT,
) -> tuple[Path, Path]:
    """Return the ``(.npy, .json)`` cache paths for this request."""
    suffix = f"_{kind_tag}" if kind_tag else ""
    stem = Path(cache_root) / system / f"{dataset}_{lang}{suffix}"
    return stem.with_suffix(".npy"), stem.with_suffix(".json")


def _sidecar_matches(meta: dict[str, Any], expected: dict[str, Any]) -> bool:
    """Return whether a cached sidecar describes exactly this request."""
    for key, value in expected.items():
        if key == "n_items":
            continue
        if meta.get(key) != value:
            return False
    return int(meta.get("n_items", -1)) == int(expected["n_items"])


def _read_sidecar(path: Path) -> dict[str, Any]:
    """Read a cache sidecar, returning an empty dict when it is unreadable."""
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_cache(array: np.ndarray, meta_path: Path, array_path: Path, meta: dict[str, Any]) -> None:
    """Write the array and its sidecar atomically enough for a re-run to trust them."""
    array_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(array_path, array.astype(np.float32, copy=False))
    temporary = meta_path.with_suffix(".json.part")
    temporary.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(meta_path)


def encode(
    system: str,
    texts: Sequence[str],
    kind: str,
    batch_size: int | None = None,
    max_len: int | None = None,
    *,
    dataset: str | None = None,
    lang: str | None = None,
    kind_tag: str | None = None,
    cache: bool = True,
    use_cache: bool = True,
    config: dict[str, Any] | None = None,
    cache_root: Path | str = CACHE_ROOT,
    return_stats: bool = False,
) -> np.ndarray | tuple[np.ndarray, EncodeStats]:
    """Encode ``texts`` with ``system`` and return float32, L2-normalized rows.

    Args:
        system: encoder id in ``configs/models.yaml`` (``labse``, ``me5``, ``muril``).
        texts: input strings; the output row order matches this order exactly.
        kind: ``word``, ``sentence``, ``query`` or ``passage``; selects the mE5
            prefix and the default ``max_len`` (``word`` uses a short one).
        batch_size: override the configured batch size.
        max_len: override the configured maximum sequence length.
        dataset: when given together with ``lang``, enables the on-disk cache.
        lang: language tag of the inputs (``hi``, ``en``), used in the cache name.
        kind_tag: extra tag when one dataset and language need several caches.
        cache: write the result to the cache.  ``use_cache`` controls reading.
        config: pre-loaded ``configs/models.yaml`` (avoids re-reading per call).
        cache_root: root directory of the embedding cache.
        return_stats: also return an :class:`EncodeStats`.

    Returns:
        ``(n_items, dim)`` float32 array, or ``(array, stats)`` when
        ``return_stats`` is true.
    """
    values = _checked_texts(texts, kind)
    models = load_models_config() if config is None else config
    settings = encoder_settings(system, models)
    defaults = models.get("encoder_defaults", {})
    effective_max_len = _resolve_max_len(kind, settings, max_len, defaults)
    effective_batch = _resolve_batch_size(kind, batch_size, settings, defaults)

    if (dataset is None) != (lang is None):
        raise ValueError("dataset and lang must be given together")
    paths: tuple[Path, Path] | None = None
    expected: dict[str, Any] = {}
    if dataset is not None and lang is not None:
        paths = cache_paths(system, dataset, lang, kind_tag, cache_root=cache_root)
        expected = {
            "system": system,
            "hf_id": settings.hf_id,
            "dataset": dataset,
            "lang": lang,
            "kind": kind,
            "kind_tag": kind_tag,
            "max_len": effective_max_len,
            "pooling": settings.pooling,
            "prefix": settings.prefix_for(kind),
            "normalized": True,
            "n_items": len(values),
            "texts_sha256": _texts_digest(values),
        }
        if use_cache and paths[0].exists() and paths[1].exists():
            meta = _read_sidecar(paths[1])
            if _sidecar_matches(meta, expected):
                cached = np.load(paths[0])
                if cached.shape[0] != len(values):
                    raise RuntimeError(
                        f"cache {paths[0]} has {cached.shape[0]} rows but {len(values)} texts were passed"
                    )
                stats = EncodeStats(
                    n_items=len(values),
                    n_truncated=int(meta.get("n_truncated", 0)),
                    max_len=effective_max_len,
                    dim=int(cached.shape[1]),
                    batch_size=effective_batch,
                )
                LOGGER.info("cache hit %s (%d x %d)", _display_path(paths[0]), *cached.shape)
                return (cached.astype(np.float32, copy=False), stats) if return_stats else cached.astype(np.float32, copy=False)

    encoder = _get_encoder(settings)
    lengths, n_truncated = _truncation_counts(encoder.tokenizer(), values, effective_max_len)
    LOGGER.info(
        "%s %s: %d texts, %d truncated at %d tokens (%.2f%%), batch %d",
        system,
        kind,
        len(values),
        n_truncated,
        effective_max_len,
        100.0 * n_truncated / len(values),
        effective_batch,
    )

    # Sort by length so padding is local; restore the input order afterwards.
    order = np.argsort(lengths, kind="stable")
    sorted_texts = [values[index] for index in order]
    raw = encoder.encode(sorted_texts, kind, effective_batch, effective_max_len)
    if raw.shape[0] != len(values):
        raise RuntimeError(f"{system} returned {raw.shape[0]} rows for {len(values)} texts")

    restored = np.empty_like(raw, dtype=np.float32)
    restored[order] = raw
    normalized = _l2_normalize(restored)
    stats = EncodeStats(
        n_items=len(values),
        n_truncated=n_truncated,
        max_len=effective_max_len,
        dim=int(normalized.shape[1]),
        batch_size=effective_batch,
    )

    if cache and paths is not None:
        meta = dict(expected)
        meta.update(
            {
                "n_truncated": n_truncated,
                "truncated_fraction": round(stats.truncated_fraction, 6),
                "dim": stats.dim,
                "batch_size": effective_batch,
            }
        )
        _write_cache(normalized, paths[1], paths[0], meta)
        LOGGER.info("wrote %s (%d x %d)", _display_path(paths[0]), *normalized.shape)

    return (normalized, stats) if return_stats else normalized
