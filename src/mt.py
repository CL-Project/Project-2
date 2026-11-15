"""Machine translation for the ``bm25_mt`` baseline (S17, S29).

Helsinki-NLP/opus-mt-hi-en translates Hindi queries to English so that English
BM25 can be used as a cross-lingual baseline.  Translations are cached to a TSV
so a re-run never re-generates them, and the cache is keyed by the exact input
text: a mismatch (different sentence list) fails instead of silently pairing the
wrong translation with a query.

Model settings (fp16, ``num_beams=4``, ``max_length=256``, batch 32) come from
``configs/models.yaml`` -> ``baselines.mt``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Sequence

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS_CONFIG_PATH = REPO_ROOT / "configs" / "models.yaml"

LOGGER = logging.getLogger("mt")

_CACHE_HEADER = ("source", "translation")


def load_config(path: Path = MODELS_CONFIG_PATH) -> dict[str, Any]:
    """Load ``configs/models.yaml``."""
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _mt_settings(config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return the resolved MT settings from config."""
    models = load_config() if config is None else config
    return dict(models["baselines"]["mt"])


def _read_cache(path: Path) -> dict[str, str]:
    """Read a two-column TSV cache, returning ``{}`` when it does not exist."""
    if not path.exists():
        return {}
    mapping: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        header = handle.readline().rstrip("\n").split("\t")
        if tuple(header) != _CACHE_HEADER:
            raise RuntimeError(f"{path} has an unexpected header: {header!r}")
        for line in handle:
            source, _, translation = line.rstrip("\n").partition("\t")
            mapping[source] = translation
    return mapping


def _write_cache(path: Path, mapping: dict[str, str]) -> None:
    """Write the translation cache atomically, preserving insertion order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tsv.part")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write("\t".join(_CACHE_HEADER) + "\n")
        for source, translation in mapping.items():
            handle.write(f"{source}\t{translation}\n")
    temporary.replace(path)


class _Translator:
    """Lazily loaded opus-mt model."""

    def __init__(self, settings: dict[str, Any]) -> None:
        self.settings = settings
        self._tokenizer: Any = None
        self._model: Any = None
        self._device = "cuda" if self._cuda() else "cpu"

    @staticmethod
    def _cuda() -> bool:
        try:
            import torch
        except ImportError:  # pragma: no cover
            return False
        return torch.cuda.is_available()

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        half = bool(self.settings.get("fp16", False)) and self._device == "cuda"
        self._tokenizer = AutoTokenizer.from_pretrained(self.settings["hf_id"])
        self._model = (
            AutoModelForSeq2SeqLM.from_pretrained(
                self.settings["hf_id"], dtype=torch.float16 if half else torch.float32
            )
            .eval()
            .to(self._device)
        )
        LOGGER.info("loaded %s (%s)", self.settings["hf_id"], "fp16" if half else "fp32")

    def translate(self, texts: Sequence[str], batch_size: int) -> list[str]:
        """Translate Hindi ``texts`` into English, preserving order."""
        import torch

        self._load()
        outputs: list[str] = []
        for start in range(0, len(texts), batch_size):
            batch = list(texts[start : start + batch_size])
            encoded = self._tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=int(self.settings.get("max_length", 256)),
            ).to(self._device)
            with torch.inference_mode():
                generated = self._model.generate(
                    **encoded,
                    num_beams=int(self.settings.get("num_beams", 4)),
                    max_length=int(self.settings.get("max_length", 256)),
                )
            outputs.extend(
                piece.strip() for piece in self._tokenizer.batch_decode(generated, skip_special_tokens=True)
            )
            LOGGER.info("translated %d/%d", min(start + batch_size, len(texts)), len(texts))
        return outputs

    def free(self) -> None:
        """Drop the model and release GPU memory."""
        self._model = None
        self._tokenizer = None
        try:
            import torch
        except ImportError:  # pragma: no cover
            return
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


_TRANSLATOR: _Translator | None = None


def translate_cached(
    texts: Sequence[str],
    cache_path: Path | str,
    *,
    force: bool = False,
    config: dict[str, Any] | None = None,
) -> list[str]:
    """Translate ``texts`` to English, reusing and extending ``cache_path``.

    Only the texts missing from the cache are translated, so adding queries to a
    later task (T3 questions after T2 sentences) costs one extra pass.  Existing
    entries are verified for consistency: if the cache already holds a
    translation for a text, it is reused verbatim.

    Args:
        texts: Hindi source sentences.
        cache_path: TSV cache to read and update.
        force: ignore the cache and re-translate every input.
        config: pre-loaded ``configs/models.yaml``.

    Returns:
        English translations in input order.
    """
    if not texts:
        raise ValueError("texts must not be empty")
    path = Path(cache_path)
    mapping = {} if force else _read_cache(path)

    missing = [text for text in dict.fromkeys(texts) if text not in mapping]
    if missing:
        global _TRANSLATOR
        settings = _mt_settings(config)
        if _TRANSLATOR is None or _TRANSLATOR.settings != settings:
            _TRANSLATOR = _Translator(settings)
        LOGGER.info("%d unique texts, %d missing from cache; translating", len(set(texts)), len(missing))
        translations = _TRANSLATOR.translate(missing, int(settings.get("batch_size", 32)))
        if len(translations) != len(missing):
            raise RuntimeError(f"{len(translations)} translations for {len(missing)} texts")
        for source, translation in zip(missing, translations):
            mapping[source] = translation
        _write_cache(path, mapping)
        LOGGER.info("wrote %s (%d entries)", path, len(mapping))

    return [mapping[text] for text in texts]


def release() -> None:
    """Free the loaded translation model."""
    global _TRANSLATOR
    if _TRANSLATOR is not None:
        _TRANSLATOR.free()
        _TRANSLATOR = None
