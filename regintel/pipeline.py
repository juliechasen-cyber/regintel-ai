"""End-to-end RAG orchestration.

``RegIntelPipeline`` owns the four moving parts (embedder, vector store,
retriever, generator) and exposes two operations:

* ``ingest_document`` / ``ingest_pdf_file`` — chunk, embed and store a document,
  skipping chunks already present (duplicate prevention);
* ``ask`` — retrieve top-k evidence and generate a grounded, cited answer.

Every component is injectable, which is what makes the system testable without
network access, an API key, or a downloaded model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from .chunking import chunk_document, estimate_tokens
from .config import Config
from .embeddings import Embedder, build_embedder
from .generation import Generator, LLMClient
from .retrieval import Retriever
from .schema import Answer, Chunk, Document, DocumentMetadata
from .vectorstore import VectorStore, build_vector_store


@dataclass
class IngestResult:
    doc_id: str
    pages: int
    chunks_created: int
    chunks_added: int
    chunks_skipped_as_duplicate: int


class RegIntelPipeline:
    def __init__(
        self,
        config: Config,
        embedder: Embedder,
        store: VectorStore,
        llm_client: Optional[LLMClient] = None,
    ):
        self.config = config
        self.embedder = embedder
        self.store = store
        self.retriever = Retriever(embedder, store, top_k=config.top_k)
        self.generator = (
            Generator(
                llm_client,
                model=config.anthropic_model,
                max_tokens=config.max_tokens,
                temperature=config.temperature,
            )
            if llm_client is not None
            else None
        )

    # ------------------------------------------------------------------ ingest

    def chunk(self, document: Document) -> list[Chunk]:
        return chunk_document(
            document,
            chunk_tokens=self.config.chunk_tokens,
            overlap_tokens=self.config.chunk_overlap_tokens,
            respect_page_boundaries=self.config.respect_page_boundaries,
            count_tokens=estimate_tokens,
        )

    def ingest_document(self, document: Document) -> IngestResult:
        chunks = self.chunk(document)
        if not chunks:
            return IngestResult(document.doc_id, len(document.pages), 0, 0, 0)

        # Duplicate prevention: only embed what the store doesn't already have.
        ids = [c.chunk_id for c in chunks]
        existing = self.store.existing_ids(ids)
        fresh = [c for c in chunks if c.chunk_id not in existing]

        added = 0
        if fresh:
            vectors = self.embedder.embed([c.text for c in fresh])
            added = self.store.add(fresh, vectors)

        return IngestResult(
            doc_id=document.doc_id,
            pages=len(document.pages),
            chunks_created=len(chunks),
            chunks_added=added,
            chunks_skipped_as_duplicate=len(chunks) - added,
        )

    def ingest_pdf_file(
        self,
        path: str,
        doc_id: Optional[str] = None,
        metadata_override: Optional[DocumentMetadata] = None,
    ) -> IngestResult:
        from .ingestion import ingest_pdf  # lazy: pulls in PyMuPDF

        document = ingest_pdf(path, doc_id=doc_id, metadata_override=metadata_override)
        return self.ingest_document(document)

    # -------------------------------------------------------------------- ask

    def retrieve(self, question: str, **kwargs: Any):
        return self.retriever.retrieve(question, **kwargs)

    def ask(
        self,
        question: str,
        top_k: Optional[int] = None,
        regulator: Optional[str] = None,
        document_type: Optional[str] = None,
    ) -> Answer:
        if self.generator is None:
            raise RuntimeError(
                "No LLM client configured. Build the pipeline with an "
                "AnthropicClient (requires ANTHROPIC_API_KEY) to generate answers."
            )
        retrieved = self.retriever.retrieve(
            question, top_k=top_k, regulator=regulator, document_type=document_type
        )
        return self.generator.generate(question, retrieved)

    # ------------------------------------------------------------------ library

    def library(self) -> dict[str, dict[str, Any]]:
        return self.store.list_documents()

    def chunk_count(self) -> int:
        return self.store.count()


def build_pipeline(
    config: Optional[Config] = None,
    with_llm: bool = True,
) -> RegIntelPipeline:
    """Construct a pipeline from configuration, with graceful degradation.

    Falls back to the offline embedder / in-memory store when the heavy
    dependencies are unavailable, so the app always starts. The UI reports which
    components are actually live.
    """
    config = config or Config.from_env()
    embedder = build_embedder(config.embedding_model, config.allow_offline_embedder)
    store = build_vector_store(config.chroma_path, config.collection_name)

    llm_client: Optional[LLMClient] = None
    if with_llm:
        try:
            from .generation import AnthropicClient

            llm_client = AnthropicClient()
        except Exception:
            llm_client = None

    return RegIntelPipeline(config, embedder, store, llm_client)
