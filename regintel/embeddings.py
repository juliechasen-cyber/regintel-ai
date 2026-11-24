"""Embedding generation.

The default production embedder uses Sentence Transformers locally (no data
leaves the machine). The model is configurable. When the model cannot be loaded
— e.g. in offline CI or on a machine without the dependency — the factory falls
back to a deterministic ``HashingEmbedder`` so the end-to-end pipeline still
runs and is testable. The fallback is clearly labelled and is NOT suitable for
production retrieval quality.
"""

from __future__ import annotations

import hashlib
import math
from typing import Protocol, Sequence, runtime_checkable

import numpy as np


@runtime_checkable
class Embedder(Protocol):
    """Minimal embedding interface used throughout the pipeline."""

    name: str
    dimension: int

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Return an (n, dimension) float32 array of unit-norm embeddings."""
        ...


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (matrix / norms).astype(np.float32)


class HashingEmbedder:
    """Deterministic, dependency-free embedder for tests / offline fallback.

    Hashes token n-grams into a fixed-width vector (the "hashing trick"). It has
    no semantic understanding, but it is stable, fast and good enough to exercise
    the full retrieval/generation path without downloading a model.
    """

    def __init__(self, dimension: int = 256):
        self.dimension = dimension
        self.name = f"hashing-{dimension}d (offline fallback — not for production)"

    def _tokens(self, text: str) -> list[str]:
        words = text.lower().split()
        grams = list(words)
        grams += [f"{a}_{b}" for a, b in zip(words, words[1:])]  # bigrams
        return grams or [""]

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        vectors = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for row, text in enumerate(texts):
            for tok in self._tokens(text):
                h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
                idx = h % self.dimension
                sign = 1.0 if (h // self.dimension) % 2 == 0 else -1.0
                vectors[row, idx] += sign
        return _l2_normalize(vectors)


class SentenceTransformerEmbedder:
    """Wraps a local Sentence Transformers model. Imported lazily."""

    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer  # lazy import

        self._model = SentenceTransformer(model_name)
        self.name = model_name
        self.dimension = int(self._model.get_sentence_embedding_dimension())

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        vectors = self._model.encode(
            list(texts),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vectors.astype(np.float32)


def build_embedder(model_name: str, allow_offline_fallback: bool = True) -> Embedder:
    """Return a Sentence Transformers embedder, or the hashing fallback.

    Raises if the model cannot load and ``allow_offline_fallback`` is False.
    """
    try:
        return SentenceTransformerEmbedder(model_name)
    except Exception as exc:  # pragma: no cover - exercised only without the dep
        if not allow_offline_fallback:
            raise RuntimeError(
                f"Could not load embedding model {model_name!r}: {exc}"
            ) from exc
        return HashingEmbedder()


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D vectors."""
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)
