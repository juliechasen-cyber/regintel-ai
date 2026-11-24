"""Central, environment-driven configuration.

All tunables live here so the pipeline's behaviour is explicit and reproducible.
Values can be overridden with environment variables (optionally loaded from a
local ``.env`` file via python-dotenv when installed).
"""

from __future__ import annotations

import os
from dataclasses import dataclass


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv  # type: ignore
    except Exception:
        return
    load_dotenv()


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


@dataclass
class Config:
    """Resolved configuration for one RegIntel session."""

    # --- Chunking ---
    chunk_tokens: int = 800
    chunk_overlap_tokens: int = 120
    respect_page_boundaries: bool = True

    # --- Embeddings ---
    # Default is a general-purpose retrieval model. Kept configurable per spec.
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    # When Sentence Transformers / its model cannot be loaded (e.g. offline CI),
    # the pipeline falls back to a deterministic hashing embedder so the system
    # still runs end-to-end. Set to False to require the real model.
    allow_offline_embedder: bool = True

    # --- Vector store ---
    chroma_path: str = "./data/chroma"
    collection_name: str = "regintel"

    # --- Retrieval ---
    top_k: int = 6

    # --- Generation (Anthropic) ---
    # Model ids follow the Claude 5 family. Sonnet is a sensible default for
    # grounded Q&A; override with REGINTEL_MODEL.
    anthropic_model: str = "claude-sonnet-5-5"
    max_tokens: int = 1024
    temperature: float = 0.0  # deterministic, grounded answers

    @classmethod
    def from_env(cls) -> "Config":
        _load_dotenv()
        return cls(
            chunk_tokens=_env_int("REGINTEL_CHUNK_TOKENS", 800),
            chunk_overlap_tokens=_env_int("REGINTEL_CHUNK_OVERLAP", 120),
            respect_page_boundaries=_env("REGINTEL_RESPECT_PAGES", "1") != "0",
            embedding_model=_env(
                "REGINTEL_EMBEDDING_MODEL",
                "sentence-transformers/all-MiniLM-L6-v2",
            ),
            allow_offline_embedder=_env("REGINTEL_ALLOW_OFFLINE_EMBEDDER", "1") != "0",
            chroma_path=_env("REGINTEL_CHROMA_PATH", "./data/chroma"),
            collection_name=_env("REGINTEL_COLLECTION", "regintel"),
            top_k=_env_int("REGINTEL_TOP_K", 6),
            anthropic_model=_env("REGINTEL_MODEL", "claude-sonnet-5-5"),
            max_tokens=_env_int("REGINTEL_MAX_TOKENS", 1024),
            temperature=_env_float("REGINTEL_TEMPERATURE", 0.0),
        )
