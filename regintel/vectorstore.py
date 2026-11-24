"""Vector storage and metadata-filtered similarity search.

Two implementations behind one interface:

* ``ChromaVectorStore`` — ChromaDB with persistent local storage (the default
  for real use). Imported lazily.
* ``InMemoryVectorStore`` — a numpy cosine-similarity store used by the tests and
  as an offline fallback. Same semantics, no persistence.

Both prevent duplicate ingestion by keying on the content-addressed
``chunk_id`` and both support filtering by ``regulator`` and ``document_type``.
"""

from __future__ import annotations

from typing import Any, Optional, Protocol, Sequence

import numpy as np

from .schema import Chunk, RetrievedChunk


class VectorStore(Protocol):
    def add(self, chunks: Sequence[Chunk], embeddings: np.ndarray) -> int: ...

    def query(
        self,
        embedding: np.ndarray,
        top_k: int,
        where: Optional[dict[str, Any]] = None,
    ) -> list[RetrievedChunk]: ...

    def existing_ids(self, chunk_ids: Sequence[str]) -> set[str]: ...

    def count(self) -> int: ...

    def list_documents(self) -> dict[str, dict[str, Any]]: ...


def _matches(meta: dict[str, Any], where: Optional[dict[str, Any]]) -> bool:
    if not where:
        return True
    for key, value in where.items():
        if meta.get(key) != value:
            return False
    return True


class InMemoryVectorStore:
    """Numpy-backed cosine store. Deterministic and dependency-light."""

    def __init__(self) -> None:
        self._ids: list[str] = []
        self._id_set: set[str] = set()
        self._chunks: list[Chunk] = []
        self._metas: list[dict[str, Any]] = []
        self._vectors: list[np.ndarray] = []

    def existing_ids(self, chunk_ids: Sequence[str]) -> set[str]:
        return {cid for cid in chunk_ids if cid in self._id_set}

    def add(self, chunks: Sequence[Chunk], embeddings: np.ndarray) -> int:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings length mismatch")
        added = 0
        for chunk, vector in zip(chunks, embeddings):
            cid = chunk.chunk_id
            if cid in self._id_set:  # dedup
                continue
            self._ids.append(cid)
            self._id_set.add(cid)
            self._chunks.append(chunk)
            self._metas.append(chunk.store_metadata())
            self._vectors.append(np.asarray(vector, dtype=np.float32))
            added += 1
        return added

    def query(self, embedding: np.ndarray, top_k: int,
              where: Optional[dict[str, Any]] = None) -> list[RetrievedChunk]:
        if not self._vectors:
            return []
        q = np.asarray(embedding, dtype=np.float32)
        qn = np.linalg.norm(q) or 1.0
        candidates: list[tuple[float, int]] = []
        for i, vector in enumerate(self._vectors):
            if not _matches(self._metas[i], where):
                continue
            denom = (np.linalg.norm(vector) * qn) or 1.0
            score = float(np.dot(vector, q) / denom)
            candidates.append((score, i))
        candidates.sort(key=lambda t: t[0], reverse=True)
        results: list[RetrievedChunk] = []
        for rank, (score, i) in enumerate(candidates[:top_k]):
            results.append(RetrievedChunk(chunk=self._chunks[i], score=score, rank=rank))
        return results

    def count(self) -> int:
        return len(self._ids)

    def list_documents(self) -> dict[str, dict[str, Any]]:
        docs: dict[str, dict[str, Any]] = {}
        for meta in self._metas:
            doc_id = meta.get("doc_id")
            if doc_id is None:
                continue
            entry = docs.setdefault(doc_id, {"chunks": 0})
            entry["chunks"] += 1
            for key in ("title", "regulator", "document_type", "publication_date",
                        "source_url", "version_note"):
                if key in meta and key not in entry:
                    entry[key] = meta[key]
        return docs


class ChromaVectorStore:
    """ChromaDB persistent store. Imported lazily so Chroma is optional."""

    def __init__(self, path: str, collection_name: str):
        import chromadb  # lazy import

        self._client = chromadb.PersistentClient(path=path)
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def existing_ids(self, chunk_ids: Sequence[str]) -> set[str]:
        if not chunk_ids:
            return set()
        got = self._collection.get(ids=list(chunk_ids))
        return set(got.get("ids", []) or [])

    def add(self, chunks: Sequence[Chunk], embeddings: np.ndarray) -> int:
        ids = [c.chunk_id for c in chunks]
        existing = self.existing_ids(ids)
        new = [(i, c) for i, c in enumerate(chunks) if c.chunk_id not in existing]
        if not new:
            return 0
        self._collection.add(
            ids=[c.chunk_id for _, c in new],
            embeddings=[embeddings[i].tolist() for i, _ in new],
            documents=[c.text for _, c in new],
            metadatas=[c.store_metadata() for _, c in new],
        )
        return len(new)

    def query(self, embedding: np.ndarray, top_k: int,
              where: Optional[dict[str, Any]] = None) -> list[RetrievedChunk]:
        res = self._collection.query(
            query_embeddings=[np.asarray(embedding, dtype=np.float32).tolist()],
            n_results=top_k,
            where=self._to_chroma_where(where),
        )
        results: list[RetrievedChunk] = []
        ids = (res.get("ids") or [[]])[0]
        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]
        for rank, (text, meta, dist) in enumerate(zip(docs, metas, dists)):
            chunk = _chunk_from_store(text, meta)
            results.append(RetrievedChunk(chunk=chunk, score=1.0 - float(dist), rank=rank))
        return results

    @staticmethod
    def _to_chroma_where(where: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
        if not where:
            return None
        if len(where) == 1:
            return {k: {"$eq": v} for k, v in where.items()}
        return {"$and": [{k: {"$eq": v}} for k, v in where.items()]}

    def count(self) -> int:
        return int(self._collection.count())

    def list_documents(self) -> dict[str, dict[str, Any]]:
        got = self._collection.get(include=["metadatas"])
        docs: dict[str, dict[str, Any]] = {}
        for meta in got.get("metadatas", []) or []:
            doc_id = meta.get("doc_id")
            if doc_id is None:
                continue
            entry = docs.setdefault(doc_id, {"chunks": 0})
            entry["chunks"] += 1
            for key in ("title", "regulator", "document_type", "publication_date",
                        "source_url", "version_note"):
                if key in meta and key not in entry:
                    entry[key] = meta[key]
        return docs


def _chunk_from_store(text: str, meta: dict[str, Any]) -> Chunk:
    from .schema import DocumentMetadata

    return Chunk(
        doc_id=meta.get("doc_id", "unknown"),
        chunk_index=int(meta.get("chunk_index", 0)),
        page_number=int(meta.get("page_number", 0)),
        text=text,
        section=meta.get("section"),
        metadata=DocumentMetadata(
            title=meta.get("title"),
            regulator=meta.get("regulator"),
            publication_date=meta.get("publication_date"),
            document_type=meta.get("document_type"),
            source_url=meta.get("source_url"),
            version_note=meta.get("version_note"),
        ),
    )


def build_vector_store(path: str, collection_name: str,
                       allow_offline_fallback: bool = True) -> VectorStore:
    try:
        return ChromaVectorStore(path, collection_name)
    except Exception:  # pragma: no cover - exercised only without chromadb
        if not allow_offline_fallback:
            raise
        return InMemoryVectorStore()
