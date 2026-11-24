"""Semantic retrieval over the vector store.

Embeds the query with the same embedder used for ingestion and returns the
top-k most similar chunks, optionally filtered by regulator / document type.

Keyword/hybrid retrieval and reranking are deliberately deferred to a later
phase (see docs/rag_pipeline.md). The ``Retriever`` interface is stable so those
can be layered in without changing callers.
"""

from __future__ import annotations

from typing import Any, Optional

from .embeddings import Embedder
from .schema import RetrievedChunk
from .vectorstore import VectorStore


class Retriever:
    def __init__(self, embedder: Embedder, store: VectorStore, top_k: int = 6):
        self.embedder = embedder
        self.store = store
        self.top_k = top_k

    def retrieve(
        self,
        question: str,
        top_k: Optional[int] = None,
        regulator: Optional[str] = None,
        document_type: Optional[str] = None,
    ) -> list[RetrievedChunk]:
        where: dict[str, Any] = {}
        if regulator:
            where["regulator"] = regulator
        if document_type:
            where["document_type"] = document_type

        query_vec = self.embedder.embed([question])[0]
        return self.store.query(query_vec, top_k or self.top_k, where or None)
