"""Core data structures shared across the RAG pipeline.

These are plain dataclasses with no heavy dependencies so every stage of the
pipeline speaks the same vocabulary and the objects are trivial to serialize,
inspect in tests, and render in the UI.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, asdict
from typing import Any, Optional


# Controlled vocabularies kept small and explicit. They are used for metadata
# filtering (regulator / document type) and for the UI facets. Values are not
# guessed from content; they come from the uploader or from PDF metadata.
DOCUMENT_TYPES = (
    "regulation",          # codified rule (e.g. in the CFR)
    "statute",             # law passed by Congress
    "supervisory_guidance",  # examiner manuals, bulletins, FAQs
    "enforcement",         # consent orders, civil money penalties
    "other",
)


@dataclass(frozen=True)
class DocumentMetadata:
    """Descriptive metadata for a source document.

    Every field is optional. Unknown fields stay ``None`` rather than being
    guessed — downstream code and the UI render ``None`` as "unknown". This is a
    deliberate anti-hallucination choice: a citation never carries a fabricated
    regulator, date or title.
    """

    title: Optional[str] = None
    regulator: Optional[str] = None          # e.g. "FinCEN", "OCC", "FFIEC", "FRB"
    publication_date: Optional[str] = None   # ISO 8601 string when known
    document_type: Optional[str] = None       # one of DOCUMENT_TYPES
    source_url: Optional[str] = None
    # Vintage warning. Set when the indexed text is a dated snapshot that a later
    # revision may have superseded (e.g. an examination-manual section captured
    # from an agency letter). Surfaced on every citation: a compliance answer
    # must never present guidance of unknown vintage as current.
    version_note: Optional[str] = None
    extra: dict[str, Any] = field(default_factory=dict)

    _STORED_FIELDS = (
        "title", "regulator", "publication_date", "document_type",
        "source_url", "version_note",
    )

    def as_filter_dict(self) -> dict[str, Any]:
        """Metadata flattened for the vector store (only non-null scalars)."""
        out: dict[str, Any] = {}
        for key in self._STORED_FIELDS:
            value = getattr(self, key)
            if value is not None:
                out[key] = value
        return out


@dataclass(frozen=True)
class Page:
    """A single extracted PDF page with its original 1-based page number."""

    page_number: int
    text: str


@dataclass(frozen=True)
class Document:
    """A source document: its id, metadata and extracted pages."""

    doc_id: str
    metadata: DocumentMetadata
    pages: tuple[Page, ...]

    @property
    def full_text(self) -> str:
        return "\n".join(p.text for p in self.pages)


@dataclass(frozen=True)
class Chunk:
    """A retrievable unit of text with provenance back to one source page.

    ``chunk_id`` is content-addressed (doc_id + page + index + text hash) so that
    re-ingesting the same document produces identical ids — this is how duplicate
    ingestion is prevented at the storage layer.
    """

    doc_id: str
    chunk_index: int
    page_number: int
    text: str
    section: Optional[str] = None
    metadata: DocumentMetadata = field(default_factory=DocumentMetadata)

    @property
    def chunk_id(self) -> str:
        digest = hashlib.sha256(self.text.encode("utf-8")).hexdigest()[:16]
        return f"{self.doc_id}:p{self.page_number}:c{self.chunk_index}:{digest}"

    def store_metadata(self) -> dict[str, Any]:
        """Flat metadata dict for the vector store (Chroma requires scalars)."""
        meta = self.metadata.as_filter_dict()
        meta.update(
            doc_id=self.doc_id,
            chunk_index=self.chunk_index,
            page_number=self.page_number,
        )
        if self.section is not None:
            meta["section"] = self.section
        return meta


@dataclass(frozen=True)
class RetrievedChunk:
    """A chunk returned by retrieval, with its relevance score and rank.

    ``score`` is a cosine similarity in [-1, 1]; higher is more similar.
    """

    chunk: Chunk
    score: float
    rank: int


@dataclass(frozen=True)
class Citation:
    """A citation that must correspond to exactly one retrieved chunk.

    It is constructed from retrieved chunks only (see ``citations.py``); the LLM
    never invents the page number, title or regulator — it only references a
    marker like ``[1]`` that maps to one of these.
    """

    marker: str               # e.g. "[1]"
    chunk_id: str
    doc_id: str
    title: Optional[str]
    regulator: Optional[str]
    page_number: int
    document_type: Optional[str] = None
    source_url: Optional[str] = None
    publication_date: Optional[str] = None
    version_note: Optional[str] = None

    def human_label(self) -> str:
        title = self.title or self.doc_id
        reg = f"{self.regulator}, " if self.regulator else ""
        date = f", {self.publication_date}" if self.publication_date else ""
        return f"{title} ({reg}p. {self.page_number}{date})"


@dataclass(frozen=True)
class Answer:
    """The final RAG result handed to the UI."""

    question: str
    text: str
    citations: tuple[Citation, ...]
    retrieved: tuple[RetrievedChunk, ...]
    insufficient_evidence: bool
    model: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "text": self.text,
            "insufficient_evidence": self.insufficient_evidence,
            "model": self.model,
            "citations": [asdict(c) for c in self.citations],
            "retrieved": [
                {
                    "chunk_id": r.chunk.chunk_id,
                    "score": r.score,
                    "rank": r.rank,
                    "doc_id": r.chunk.doc_id,
                    "page_number": r.chunk.page_number,
                    "text": r.chunk.text,
                }
                for r in self.retrieved
            ],
        }
