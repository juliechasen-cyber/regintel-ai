"""Tests for the recursive chunker."""

from regintel.chunking import (
    chunk_document,
    estimate_tokens,
    word_token_counter,
    _looks_like_heading,
)
from regintel.ingestion import document_from_pages
from regintel.schema import DocumentMetadata


def _para(word: str, n: int) -> str:
    return " ".join([word] * n)


def test_estimate_tokens_is_zero_for_blank():
    assert estimate_tokens("") == 0
    assert estimate_tokens("   \n ") == 0
    assert estimate_tokens("abcd") == 1


def test_chunks_respect_token_budget():
    text = "\n\n".join(_para(f"w{i}", 100) for i in range(10))
    doc = document_from_pages("d1", [(1, text)])
    chunks = chunk_document(
        doc, chunk_tokens=120, overlap_tokens=20, count_tokens=word_token_counter
    )
    assert len(chunks) > 1
    for c in chunks:
        # Budget is enforced at merge time; allow the final atomic piece to tip
        # a chunk slightly over, but never by more than one paragraph.
        assert word_token_counter(c.text) <= 120 + 100


def test_overlap_repeats_content_between_consecutive_chunks():
    # Distinct sentences so we can detect shared text across the boundary.
    sentences = [f"sentence number {i} about compliance." for i in range(60)]
    doc = document_from_pages("d2", [(1, " ".join(sentences))])
    chunks = chunk_document(
        doc, chunk_tokens=50, overlap_tokens=15, count_tokens=word_token_counter
    )
    assert len(chunks) >= 2
    first_words = set(chunks[0].text.split())
    second_words = set(chunks[1].text.split())
    assert first_words & second_words, "expected overlapping tokens between chunks"


def test_zero_overlap_produces_no_shared_sentences():
    sentences = [f"unique{i} alpha beta gamma delta." for i in range(40)]
    doc = document_from_pages("d3", [(1, " ".join(sentences))])
    chunks = chunk_document(
        doc, chunk_tokens=30, overlap_tokens=0, count_tokens=word_token_counter
    )
    assert len(chunks) >= 2
    # No "uniqueN" token should appear in two different chunks.
    seen: dict[str, int] = {}
    for i, c in enumerate(chunks):
        for tok in c.text.split():
            if tok.startswith("unique"):
                assert tok not in seen or seen[tok] == i, f"{tok} duplicated"
                seen[tok] = i


def test_metadata_and_page_numbers_are_preserved():
    meta = DocumentMetadata(title="BSA Manual", regulator="FFIEC", document_type="supervisory_guidance")
    pages = [(1, _para("alpha", 200)), (2, _para("beta", 200)), (7, _para("gamma", 200))]
    doc = document_from_pages("bsa", pages, meta)
    chunks = chunk_document(
        doc, chunk_tokens=80, overlap_tokens=10, count_tokens=word_token_counter
    )
    page_numbers = {c.page_number for c in chunks}
    assert page_numbers == {1, 2, 7}, "original page numbers must be preserved"
    for c in chunks:
        assert c.doc_id == "bsa"
        assert c.metadata.regulator == "FFIEC"
        assert c.metadata.title == "BSA Manual"
    # chunk_index is globally sequential across pages
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_chunks_never_span_pages_by_default():
    # Each page's text must appear only in chunks tagged with that page.
    pages = [(1, _para("pageone", 150)), (2, _para("pagetwo", 150))]
    doc = document_from_pages("d4", pages)
    chunks = chunk_document(
        doc, chunk_tokens=1000, overlap_tokens=50, count_tokens=word_token_counter
    )
    for c in chunks:
        if c.page_number == 1:
            assert "pagetwo" not in c.text
        if c.page_number == 2:
            assert "pageone" not in c.text


def test_heading_detection():
    assert _looks_like_heading("Customer Due Diligence")
    assert _looks_like_heading("31 CFR 1020.220")
    assert not _looks_like_heading(
        "A bank must establish and maintain written procedures that are reasonably designed."
    )
    assert not _looks_like_heading("")


def test_heading_detection_rejects_pdf_artifacts():
    # Page numbers and letter-spaced running headers are common in regulator
    # PDFs and must not be recorded as section titles.
    assert not _looks_like_heading("6")
    assert not _looks_like_heading("17")
    assert not _looks_like_heading("F I N C E N G U I D A N C E 6")
    assert not _looks_like_heading("- 12 -")
    assert not _looks_like_heading("Z")


def test_heading_stays_with_following_text():
    # A heading followed by body text, sized so the greedy merge would otherwise
    # end a chunk right on the heading.
    body_a = _para("alpha", 40)
    body_b = _para("beta", 40)
    text = f"{body_a}\n\nCustomer Due Diligence\n\n{body_b}"
    doc = document_from_pages("d5", [(1, text)])
    chunks = chunk_document(
        doc, chunk_tokens=45, overlap_tokens=0, count_tokens=word_token_counter
    )
    for c in chunks:
        stripped = c.text.strip()
        if stripped.endswith("Customer Due Diligence"):
            raise AssertionError("heading was stranded at the end of a chunk")
    # And the heading is recorded as the section for the text it introduces.
    beta_chunks = [c for c in chunks if "beta" in c.text]
    assert beta_chunks
    assert any(c.section == "Customer Due Diligence" for c in beta_chunks)


def test_empty_pages_produce_no_chunks():
    doc = document_from_pages("d6", [(1, "   "), (2, "")])
    assert chunk_document(doc, count_tokens=word_token_counter) == []


def test_oversized_single_word_is_split_by_character_fallback():
    doc = document_from_pages("d7", [(1, "x" * 5000)])
    chunks = chunk_document(
        doc, chunk_tokens=100, overlap_tokens=0, count_tokens=estimate_tokens
    )
    assert len(chunks) > 1
    assert all(c.text for c in chunks)
