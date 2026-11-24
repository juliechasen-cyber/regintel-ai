"""End-to-end pipeline tests: ingest -> chunk -> embed -> store -> retrieve -> generate.

Runs with the offline embedder, the in-memory store and a fake LLM client, so the
whole RAG workflow is exercised without a model download, a vector DB or an API
key. The real-PDF test is skipped automatically when no PDF is present.
"""

import os

from regintel.config import Config
from regintel.embeddings import HashingEmbedder
from regintel.ingestion import clean_text, default_doc_id, document_from_pages
from regintel.pipeline import RegIntelPipeline
from regintel.schema import DocumentMetadata
from regintel.vectorstore import InMemoryVectorStore

DATA_PDFS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "data", "pdfs")


class EchoCitingClient:
    """Fake LLM that cites whichever passages it was given — enough to verify
    that the plumbing carries real evidence through to real citations."""

    def __init__(self):
        self.calls = []

    def create_message(self, *, model, system, messages, max_tokens, temperature):
        self.calls.append(messages[0]["content"])
        return "Based on the retrieved evidence, the requirement applies [1]."


def _pipeline(top_k=3, chunk_tokens=120, overlap=20):
    config = Config(
        chunk_tokens=chunk_tokens,
        chunk_overlap_tokens=overlap,
        top_k=top_k,
        anthropic_model="claude-sonnet-5-5",
    )
    return RegIntelPipeline(
        config=config,
        embedder=HashingEmbedder(dimension=256),
        store=InMemoryVectorStore(),
        llm_client=EchoCitingClient(),
    )


def _sample_doc(doc_id="fincen-cdd", regulator="FinCEN"):
    pages = [
        (1, "Customer Due Diligence Requirements\n\n"
            "A covered financial institution must establish and maintain written "
            "procedures that are reasonably designed to identify and verify "
            "beneficial owners of legal entity customers."),
        (2, "Suspicious Activity Reporting\n\n"
            "A bank must file a suspicious activity report with respect to certain "
            "transactions conducted or attempted by, at, or through the bank."),
        (3, "Recordkeeping\n\n"
            "A bank shall retain records of each transaction for the period "
            "specified by the applicable recordkeeping provisions."),
    ]
    return document_from_pages(
        doc_id, pages,
        DocumentMetadata(
            title="CDD Rule Consolidated FAQs",
            regulator=regulator,
            document_type="supervisory_guidance",
            source_url="https://www.fincen.gov/example",
        ),
    )


# ------------------------------------------------------------------- ingestion

def test_ingest_reports_counts():
    p = _pipeline()
    result = p.ingest_document(_sample_doc())
    assert result.doc_id == "fincen-cdd"
    assert result.pages == 3
    assert result.chunks_created > 0
    assert result.chunks_added == result.chunks_created
    assert result.chunks_skipped_as_duplicate == 0
    assert p.chunk_count() == result.chunks_added


def test_re_ingesting_the_same_document_adds_nothing():
    p = _pipeline()
    first = p.ingest_document(_sample_doc())
    before = p.chunk_count()
    second = p.ingest_document(_sample_doc())
    assert second.chunks_added == 0
    assert second.chunks_skipped_as_duplicate == second.chunks_created
    assert p.chunk_count() == before, "duplicate ingestion must not grow the store"
    assert first.chunks_created == second.chunks_created


def test_empty_document_ingests_cleanly():
    p = _pipeline()
    result = p.ingest_document(document_from_pages("empty", [(1, "")]))
    assert result.chunks_created == 0
    assert result.chunks_added == 0
    assert p.chunk_count() == 0


def test_library_lists_ingested_documents():
    p = _pipeline()
    p.ingest_document(_sample_doc("doc-a", "FinCEN"))
    p.ingest_document(_sample_doc("doc-b", "FFIEC"))
    lib = p.library()
    assert set(lib) == {"doc-a", "doc-b"}
    assert lib["doc-b"]["regulator"] == "FFIEC"


# ------------------------------------------------------------------- retrieval

def test_retrieval_finds_the_relevant_page():
    p = _pipeline()
    p.ingest_document(_sample_doc())
    results = p.retrieve("beneficial owners of legal entity customers")
    assert results
    assert results[0].chunk.page_number == 1, "CDD text lives on page 1"
    assert "beneficial owner" in results[0].chunk.text.lower()


def test_retrieval_respects_top_k():
    p = _pipeline(top_k=2)
    p.ingest_document(_sample_doc())
    assert len(p.retrieve("bank")) <= 2
    assert len(p.retrieve("bank", top_k=1)) == 1


def test_retrieval_metadata_filter_excludes_other_regulators():
    p = _pipeline()
    p.ingest_document(_sample_doc("doc-a", "FinCEN"))
    p.ingest_document(_sample_doc("doc-b", "FFIEC"))
    only_ffiec = p.retrieve("suspicious activity report", regulator="FFIEC", top_k=10)
    assert only_ffiec
    assert {r.chunk.doc_id for r in only_ffiec} == {"doc-b"}


def test_retrieval_on_empty_library_returns_nothing():
    assert _pipeline().retrieve("anything") == []


# ------------------------------------------------------------- full RAG answer

def test_ask_produces_a_cited_grounded_answer():
    p = _pipeline()
    p.ingest_document(_sample_doc())
    answer = p.ask("What must a bank do about beneficial ownership?")

    assert answer.insufficient_evidence is False
    assert answer.citations, "a grounded answer must carry at least one citation"
    assert answer.model == "claude-sonnet-5-5"

    # The citation must point at a real page of a real ingested document.
    cite = answer.citations[0]
    assert cite.doc_id == "fincen-cdd"
    assert cite.page_number in (1, 2, 3)
    assert cite.regulator == "FinCEN"
    assert cite.title == "CDD Rule Consolidated FAQs"

    # And it must correspond to a chunk that was actually retrieved.
    retrieved_ids = {r.chunk.chunk_id for r in answer.retrieved}
    assert cite.chunk_id in retrieved_ids


def test_ask_sends_real_evidence_to_the_model():
    p = _pipeline()
    p.ingest_document(_sample_doc())
    p.ask("beneficial ownership procedures")
    sent = p.generator.client.calls[0]
    assert "EVIDENCE:" in sent
    assert "beneficial owner" in sent.lower()
    assert "page 1" in sent


def test_ask_with_empty_library_reports_insufficient_evidence():
    p = _pipeline()
    answer = p.ask("What is the SAR filing deadline?")
    assert answer.insufficient_evidence is True
    assert answer.citations == ()
    assert p.generator.client.calls == [], "no model call without evidence"


def test_ask_without_llm_client_raises_clearly():
    p = RegIntelPipeline(Config(), HashingEmbedder(32), InMemoryVectorStore(), None)
    try:
        p.ask("q")
    except RuntimeError as exc:
        assert "No LLM client" in str(exc)
        return
    raise AssertionError("expected RuntimeError without an LLM client")


# ------------------------------------------------------------ text normalization

def test_clean_text_joins_hyphenated_line_wraps():
    assert "institution" in clean_text("insti-\ntution")


def test_clean_text_collapses_whitespace():
    assert clean_text("a    b\n\n\n\nc") == "a b\n\nc"


def test_default_doc_id_slugifies_filenames():
    assert default_doc_id("/x/FinCEN CDD Rule (2026).pdf") == "fincen-cdd-rule-2026"


# ----------------------------------------------------------- real PDF (if any)

def test_real_pdf_ingests_with_correct_pages_and_metadata():
    """Exercises PyMuPDF against an actual downloaded regulator PDF."""
    if not os.path.isdir(DATA_PDFS):
        print("    (skipped: no data/pdfs directory)")
        return
    pdfs = [f for f in sorted(os.listdir(DATA_PDFS)) if f.lower().endswith(".pdf")]
    if not pdfs:
        print("    (skipped: no PDFs downloaded yet)")
        return
    try:
        import pymupdf  # noqa: F401
    except ImportError:
        try:
            import fitz  # noqa: F401
        except ImportError:
            print("    (skipped: PyMuPDF not installed)")
            return

    path = os.path.join(DATA_PDFS, pdfs[0])
    p = _pipeline(chunk_tokens=800, overlap=120)
    result = p.ingest_pdf_file(path)

    assert result.pages > 0, "a real PDF must yield at least one page"
    assert result.chunks_created > 0
    assert result.chunks_added == result.chunks_created

    # Page numbers must be real and 1-based, never zero or beyond the page count.
    lib = p.library()
    assert result.doc_id in lib
    retrieved = p.retrieve("customer due diligence", top_k=5)
    assert retrieved, "expected at least one passage from a BSA/AML document"
    for r in retrieved:
        assert 1 <= r.chunk.page_number <= result.pages

    # The sidecar metadata (if present) must flow into citations.
    if os.path.exists(path + ".meta.json"):
        assert retrieved[0].chunk.metadata.regulator in ("FinCEN", "FFIEC")
        assert retrieved[0].chunk.metadata.source_url

    # Re-ingesting the same real PDF must be a no-op.
    again = p.ingest_pdf_file(path)
    assert again.chunks_added == 0
