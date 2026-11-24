"""Tests for the embedder contract and the vector store (dedup + filtering)."""

import numpy as np

from regintel.embeddings import HashingEmbedder, cosine_similarity
from regintel.schema import Chunk, DocumentMetadata
from regintel.vectorstore import InMemoryVectorStore


def _chunk(doc_id: str, idx: int, page: int, text: str, **meta) -> Chunk:
    return Chunk(
        doc_id=doc_id,
        chunk_index=idx,
        page_number=page,
        text=text,
        metadata=DocumentMetadata(**meta),
    )


# ----------------------------------------------------------------- embeddings

def test_hashing_embedder_shape_and_norm():
    emb = HashingEmbedder(dimension=64)
    vectors = emb.embed(["suspicious activity report", "currency transaction report"])
    assert vectors.shape == (2, 64)
    assert vectors.dtype == np.float32
    for row in vectors:
        assert abs(float(np.linalg.norm(row)) - 1.0) < 1e-5


def test_hashing_embedder_is_deterministic():
    a = HashingEmbedder(dimension=64).embed(["beneficial ownership"])
    b = HashingEmbedder(dimension=64).embed(["beneficial ownership"])
    assert np.allclose(a, b)


def test_hashing_embedder_separates_unrelated_text():
    emb = HashingEmbedder(dimension=256)
    v = emb.embed([
        "customer due diligence beneficial ownership rule",
        "customer due diligence beneficial ownership rule",
        "capital adequacy leverage ratio tier one",
    ])
    same = cosine_similarity(v[0], v[1])
    different = cosine_similarity(v[0], v[2])
    assert same > 0.99
    assert different < same


def test_empty_text_embeds_without_error():
    vectors = HashingEmbedder(dimension=32).embed([""])
    assert vectors.shape == (1, 32)


def test_cosine_similarity_handles_zero_vector():
    assert cosine_similarity(np.zeros(4), np.ones(4)) == 0.0


# --------------------------------------------------------------- vector store

def test_add_and_count():
    store = InMemoryVectorStore()
    emb = HashingEmbedder(dimension=32)
    chunks = [_chunk("d1", 0, 1, "alpha text"), _chunk("d1", 1, 2, "beta text")]
    added = store.add(chunks, emb.embed([c.text for c in chunks]))
    assert added == 2
    assert store.count() == 2


def test_duplicate_ingestion_is_prevented():
    store = InMemoryVectorStore()
    emb = HashingEmbedder(dimension=32)
    chunks = [_chunk("d1", 0, 1, "alpha text")]
    vectors = emb.embed([c.text for c in chunks])
    assert store.add(chunks, vectors) == 1
    assert store.add(chunks, vectors) == 0, "re-adding the same chunk must be a no-op"
    assert store.count() == 1


def test_existing_ids_reports_known_chunks():
    store = InMemoryVectorStore()
    emb = HashingEmbedder(dimension=32)
    c1 = _chunk("d1", 0, 1, "alpha")
    c2 = _chunk("d1", 1, 1, "beta")
    store.add([c1], emb.embed([c1.text]))
    found = store.existing_ids([c1.chunk_id, c2.chunk_id])
    assert found == {c1.chunk_id}


def test_chunk_id_is_content_addressed():
    a = _chunk("d1", 0, 1, "same text")
    b = _chunk("d1", 0, 1, "same text")
    c = _chunk("d1", 0, 1, "different text")
    assert a.chunk_id == b.chunk_id
    assert a.chunk_id != c.chunk_id


def test_query_orders_by_similarity():
    store = InMemoryVectorStore()
    emb = HashingEmbedder(dimension=256)
    texts = [
        "suspicious activity report filing deadline thirty days",
        "capital adequacy and leverage ratio requirements",
        "branch opening application procedures",
    ]
    chunks = [_chunk("d1", i, i + 1, t) for i, t in enumerate(texts)]
    store.add(chunks, emb.embed(texts))

    results = store.query(emb.embed(["suspicious activity report filing deadline"])[0], top_k=3)
    assert len(results) == 3
    assert results[0].chunk.text == texts[0]
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)
    assert [r.rank for r in results] == [0, 1, 2]


def test_query_respects_top_k():
    store = InMemoryVectorStore()
    emb = HashingEmbedder(dimension=64)
    chunks = [_chunk("d1", i, i + 1, f"text {i}") for i in range(5)]
    store.add(chunks, emb.embed([c.text for c in chunks]))
    assert len(store.query(emb.embed(["text"])[0], top_k=2)) == 2


def test_metadata_filtering_by_regulator_and_type():
    store = InMemoryVectorStore()
    emb = HashingEmbedder(dimension=64)
    chunks = [
        _chunk("d1", 0, 1, "fincen rule text", regulator="FinCEN", document_type="regulation"),
        _chunk("d2", 0, 1, "occ bulletin text", regulator="OCC", document_type="supervisory_guidance"),
        _chunk("d3", 0, 1, "fincen guidance text", regulator="FinCEN", document_type="supervisory_guidance"),
    ]
    store.add(chunks, emb.embed([c.text for c in chunks]))
    q = emb.embed(["text"])[0]

    fincen = store.query(q, top_k=10, where={"regulator": "FinCEN"})
    assert {r.chunk.doc_id for r in fincen} == {"d1", "d3"}

    guidance = store.query(q, top_k=10, where={"document_type": "supervisory_guidance"})
    assert {r.chunk.doc_id for r in guidance} == {"d2", "d3"}

    both = store.query(q, top_k=10, where={"regulator": "FinCEN", "document_type": "regulation"})
    assert {r.chunk.doc_id for r in both} == {"d1"}


def test_query_on_empty_store_returns_nothing():
    store = InMemoryVectorStore()
    assert store.query(np.zeros(8), top_k=5) == []


def test_add_rejects_length_mismatch():
    store = InMemoryVectorStore()
    try:
        store.add([_chunk("d1", 0, 1, "a")], np.zeros((2, 8)))
    except ValueError:
        return
    raise AssertionError("expected ValueError on length mismatch")


def test_list_documents_summarizes_library():
    store = InMemoryVectorStore()
    emb = HashingEmbedder(dimension=32)
    chunks = [
        _chunk("d1", 0, 1, "a", title="Rule A", regulator="FinCEN"),
        _chunk("d1", 1, 2, "b", title="Rule A", regulator="FinCEN"),
        _chunk("d2", 0, 1, "c", title="Manual B", regulator="FFIEC"),
    ]
    store.add(chunks, emb.embed([c.text for c in chunks]))
    lib = store.list_documents()
    assert set(lib) == {"d1", "d2"}
    assert lib["d1"]["chunks"] == 2
    assert lib["d1"]["title"] == "Rule A"
    assert lib["d2"]["regulator"] == "FFIEC"
