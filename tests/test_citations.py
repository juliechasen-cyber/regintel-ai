"""Tests for citation construction and validation.

These encode the core anti-hallucination contract: a citation may only exist if
it maps to a chunk that was actually retrieved.
"""

from regintel.citations import (
    build_citations,
    format_sources_block,
    markers_in_text,
    used_citations,
)
from regintel.schema import Chunk, DocumentMetadata, RetrievedChunk


def _retrieved(n: int) -> list[RetrievedChunk]:
    out = []
    for i in range(n):
        chunk = Chunk(
            doc_id=f"doc{i}",
            chunk_index=i,
            page_number=10 + i,
            text=f"evidence text {i}",
            metadata=DocumentMetadata(
                title=f"Title {i}",
                regulator="FinCEN" if i % 2 == 0 else "OCC",
                document_type="regulation",
                source_url=f"https://example.gov/{i}" if i == 0 else None,
            ),
        )
        out.append(RetrievedChunk(chunk=chunk, score=1.0 - i * 0.1, rank=i))
    return out


def test_build_citations_numbers_from_one():
    cites = build_citations(_retrieved(3))
    assert [c.marker for c in cites] == ["[1]", "[2]", "[3]"]


def test_citation_carries_title_regulator_and_real_page():
    cites = build_citations(_retrieved(2))
    assert cites[0].title == "Title 0"
    assert cites[0].regulator == "FinCEN"
    assert cites[0].page_number == 10
    assert cites[1].page_number == 11


def test_citation_includes_source_url_when_available():
    cites = build_citations(_retrieved(2))
    assert cites[0].source_url == "https://example.gov/0"
    assert cites[1].source_url is None


def test_every_citation_maps_to_a_retrieved_chunk():
    retrieved = _retrieved(3)
    cites = build_citations(retrieved)
    retrieved_ids = {r.chunk.chunk_id for r in retrieved}
    for c in cites:
        assert c.chunk_id in retrieved_ids


def test_markers_in_text_extracts_unique_ordered():
    assert markers_in_text("claim [2] and [1] and again [2].") == [2, 1]
    assert markers_in_text("no markers here") == []


def test_used_citations_filters_to_cited_only():
    cites = build_citations(_retrieved(4))
    valid, invalid = used_citations("Banks must file [1] within the period [3].", cites)
    assert [c.marker for c in valid] == ["[1]", "[3]"]
    assert invalid == []


def test_invented_marker_is_rejected():
    cites = build_citations(_retrieved(2))
    valid, invalid = used_citations("Per [1] and also [9] and [42].", cites)
    assert [c.marker for c in valid] == ["[1]"]
    assert invalid == [9, 42]


def test_uncited_answer_yields_no_citations():
    cites = build_citations(_retrieved(3))
    valid, invalid = used_citations("A confident answer with no sources.", cites)
    assert valid == []
    assert invalid == []


def test_human_label_falls_back_to_doc_id_without_title():
    chunk = Chunk(doc_id="fincen-cdd", chunk_index=0, page_number=5, text="x",
                  metadata=DocumentMetadata())
    cites = build_citations([RetrievedChunk(chunk=chunk, score=0.5, rank=0)])
    label = cites[0].human_label()
    assert "fincen-cdd" in label
    assert "p. 5" in label


def test_format_sources_block_renders_markers_and_urls():
    cites = build_citations(_retrieved(2))
    block = format_sources_block(cites)
    assert "[1]" in block and "[2]" in block
    assert "https://example.gov/0" in block
    assert "p. 10" in block


def test_format_sources_block_handles_empty():
    assert format_sources_block([]) == "No sources cited."


# ------------------------------------------------------ document vintage (staleness)

def _dated_retrieved(version_note=None, pub_date="February 2021"):
    chunk = Chunk(
        doc_id="ffiec-cip", chunk_index=0, page_number=3, text="manual text",
        metadata=DocumentMetadata(
            title="FFIEC Manual — CIP", regulator="FFIEC",
            document_type="supervisory_guidance",
            publication_date=pub_date, version_note=version_note,
        ),
    )
    return [RetrievedChunk(chunk=chunk, score=0.9, rank=0)]


def test_citation_carries_publication_date_and_version_note():
    note = "Snapshot dated February 2021; verify against the current InfoBase."
    c = build_citations(_dated_retrieved(note))[0]
    assert c.publication_date == "February 2021"
    assert c.version_note == note


def test_human_label_includes_the_date():
    c = build_citations(_dated_retrieved())[0]
    assert "February 2021" in c.human_label()
    assert "p. 3" in c.human_label()


def test_sources_block_surfaces_the_vintage_warning():
    note = "THIS TEXT IS LIKELY SUPERSEDED - check the current InfoBase."
    block = format_sources_block(build_citations(_dated_retrieved(note)))
    assert "LIKELY SUPERSEDED" in block, "a stale-document warning must be shown"


def test_sources_block_has_no_warning_line_when_document_is_current():
    block = format_sources_block(build_citations(_dated_retrieved(None)))
    assert "!" not in block
