"""Shared synthetic fixtures for the test suite (S09).

Everything here is tiny, deterministic and offline: no downloads, no GPU and no
real datasets.  Fixtures return fresh objects per test, so tests may mutate them.
"""

from __future__ import annotations

import numpy as np
import pytest

EMBEDDING_COUNT = 20
EMBEDDING_DIM = 8
SEED = 42


def _row_normalize(matrix: np.ndarray) -> np.ndarray:
    """Return ``matrix`` with every row L2-normalized."""
    return matrix / np.linalg.norm(matrix, axis=1, keepdims=True)


@pytest.fixture
def synthetic_embeddings() -> dict[str, np.ndarray]:
    """20 unit vectors plus a known orthogonal rotation between two languages.

    ``source`` is the Hindi-side matrix, ``target = source @ rotation`` the
    English-side matrix, and ``rotation`` the true mapping recovered by
    Procrustes (``source @ rotation`` maps into the target space).
    """
    rng = np.random.default_rng(SEED)
    source = _row_normalize(rng.normal(size=(EMBEDDING_COUNT, EMBEDDING_DIM)))
    rotation, upper = np.linalg.qr(rng.normal(size=(EMBEDDING_DIM, EMBEDDING_DIM)))
    rotation = rotation * np.sign(np.diag(upper))
    target = source @ rotation
    return {"source": source, "target": target, "rotation": rotation}


@pytest.fixture
def toy_parallel_corpus() -> list[tuple[str, str]]:
    """Ten Hindi-English sentence pairs (danda, matras and nukta included)."""
    return [
        ("मैं कल दिल्ली गया था।", "I went to Delhi yesterday."),
        ("वह किताब पढ़ रही है।", "She is reading a book."),
        ("बच्चे बगीचे में खेल रहे हैं।", "The children are playing in the garden."),
        ("हमने बाज़ार से आम ख़रीदे।", "We bought mangoes from the market."),
        ("प्रधानमंत्री ने देश को संबोधित किया।", "The Prime Minister addressed the nation."),
        ("कृपया खिड़की बंद कर दीजिए।", "Please close the window."),
        ("यह फ़िल्म बहुत अच्छी है।", "This film is very good."),
        ("वह सोमवार को स्कूल नहीं गया।", "He did not go to school on Monday."),
        ("मुझे हिंदी सीखना पसंद है।", "I like learning Hindi."),
        ("रेलगाड़ी दस मिनट देर से चल रही है।", "The train is running ten minutes late."),
    ]


@pytest.fixture
def toy_rankings() -> np.ndarray:
    """1-based best-gold ranks of eight queries with hand-computable metrics.

    Acc@1 = 1/8, Acc@3 = 4/8, MRR = (1 + 1/2 + 1/2 + 1/3 + 1/5 + 1/8 + 1/10 + 1/20) / 8.
    """
    return np.array([1, 2, 2, 3, 5, 8, 10, 20], dtype=np.int64)
