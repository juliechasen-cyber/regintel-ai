---
noteId: "9c37e370e84211f192c02f52450d7c90"
tags: []

---

# Architecture

## What this system is, and is not

RegIntel AI is a **retrieval-augmented generation pipeline**. Control flow is
fixed and written in Python. The language model is used for exactly one bounded
step — turning retrieved passages into prose — and it cannot choose tools, take
actions, or decide what to do next.

This is a deliberate choice, not a limitation. In a compliance function the
value is in traceability: a reviewer must be able to see precisely which text
produced an answer. An agent that decides its own retrieval strategy makes that
audit trail non-deterministic. A fixed pipeline makes the same question return
the same evidence.

## Component diagram

```
                         INGESTION (offline, free)
  ┌──────────┐    ┌───────────┐    ┌──────────┐    ┌────────────┐
  │   PDF    │───►│ PyMuPDF   │───►│ Chunker  │───►│  Embedder  │
  │  + meta  │    │ extract   │    │ 800/120  │    │ (local ST) │
  │  sidecar │    │ + clean   │    │ per page │    │            │
  └──────────┘    └───────────┘    └──────────┘    └─────┬──────┘
                                                          │
                                                   ┌──────▼──────┐
                                                   │  ChromaDB   │
                                                   │ persistent  │
                                                   │ + dedup     │
                                                   └──────┬──────┘
                         QUERY (online)                   │
  ┌──────────┐    ┌───────────┐    ┌──────────┐           │
  │ Question │───►│ Embedder  │───►│ Retrieve │◄──────────┘
  └──────────┘    └───────────┘    │  top-k   │
                                   │ + filter │
                                   └────┬─────┘
                                        │ passages [1..k]
                                   ┌────▼─────────────┐
                                   │ Prompt builder   │
                                   │ evidence block   │
                                   └────┬─────────────┘
                                        │
                                   ┌────▼─────┐
                                   │  Claude  │  ← the only model call
                                   └────┬─────┘
                                        │ prose with [n] markers
                                   ┌────▼──────────────┐
                                   │ Citation validator│  ← drops invented [n]
                                   └────┬──────────────┘
                                        │
                                   ┌────▼─────┐
                                   │  Answer  │ text + citations + evidence
                                   └──────────┘
```

## Module responsibilities

| Module | Responsibility | Heavy dependency |
|---|---|---|
| `schema.py` | Shared dataclasses. The vocabulary every stage speaks. | none |
| `config.py` | All tunables, environment-driven. | none |
| `ingestion.py` | PDF → pages of clean text, with provenance. | PyMuPDF (lazy) |
| `chunking.py` | Pages → retrievable chunks with metadata. | none |
| `embeddings.py` | Text → unit-norm vectors. | sentence-transformers (lazy) |
| `vectorstore.py` | Store, dedup, filtered similarity search. | chromadb (lazy) |
| `retrieval.py` | Question → ranked passages. | none |
| `generation.py` | Passages + question → grounded prose. | anthropic (lazy) |
| `citations.py` | Build and validate citations. | none |
| `evaluation.py` | Retrieval metrics; answer metrics. | none |
| `pipeline.py` | Wires the above; owns ingest and ask. | none |

## The key design decision: dependency injection at every boundary

Each heavy dependency sits behind a small interface (`Embedder`, `VectorStore`,
`LLMClient`) and is imported lazily inside the concrete implementation.

```python
pipeline = RegIntelPipeline(
    config=Config(),
    embedder=HashingEmbedder(256),      # or SentenceTransformerEmbedder(...)
    store=InMemoryVectorStore(),        # or ChromaVectorStore(path, name)
    llm_client=FakeClient(),            # or AnthropicClient()
)
```

This buys three things that matter:

1. **The full RAG path is testable with no API key, no model download and no
   network.** The end-to-end test ingests, chunks, embeds, stores, retrieves,
   generates and validates citations in milliseconds. This is why the test suite
   is meaningful rather than a thin veneer over mocks.
2. **Graceful degradation is explicit.** `build_pipeline` falls back to the
   offline embedder and in-memory store when the real ones are unavailable, and
   the CLI and UI *report which components are live* rather than hiding it. A
   silent fallback that quietly produces bad retrieval is a trap; a loud one is a
   feature.
3. **Phase-2 work doesn't touch callers.** Adding BM25 or a reranker means a new
   `Retriever`, not a refactor.

## Data flow invariants

These hold throughout and are covered by tests:

- **A chunk belongs to exactly one page** (default). Therefore every citation's
  page number is real, not interpolated.
- **`chunk_id` is content-addressed** — `doc_id` + page + index + SHA-256 of the
  text. Re-ingesting an identical document is a no-op at the storage layer, which
  is how duplicate ingestion is prevented without a separate bookkeeping table.
- **Unknown metadata is `None`, never a guess.** It renders as `(unknown)`.
- **A `Citation` cannot be constructed except from a `RetrievedChunk`.**
- **The model is never called with zero evidence.**

## Storage model

ChromaDB with cosine distance, one collection, persisted to `data/chroma/`.

Each record holds the chunk text, its embedding, and flat scalar metadata:
`doc_id`, `chunk_index`, `page_number`, `section`, plus `title`, `regulator`,
`document_type`, `publication_date`, `source_url` where known. Chroma requires
scalar metadata values, which is why `DocumentMetadata.as_filter_dict()` flattens
and drops nulls.

Filtering is server-side in Chroma (`$eq`, combined with `$and` for multiple
facets) and in-process for the in-memory store, with identical semantics so tests
and production agree.

## Failure modes and handling

| Failure | Behaviour |
|---|---|
| `sentence-transformers` missing or model won't load | Falls back to hashing embedder; **warned loudly** in CLI `status` and the UI sidebar. Set `REGINTEL_ALLOW_OFFLINE_EMBEDDER=0` to fail hard instead. |
| `chromadb` missing | Falls back to in-memory store; warned that data is not persisted. |
| No `ANTHROPIC_API_KEY` | Retrieval and ingestion work fully; `ask` refuses with a clear message. |
| Zero retrieval results | Short-circuits to an "insufficient evidence" answer with no API call. |
| Model cites a passage that wasn't retrieved | Citation dropped; a note naming the bad marker is appended. |
| Source host blocks automated download | Reported per-file with the URL, so it can be fetched manually. Sidecars are still written. |
| Scanned/image-only PDF | Yields empty text and zero chunks. **Not currently handled** — OCR is out of scope; the page count/chunk count mismatch makes it visible. |

## Scaling notes

Current scale is a single-machine demo: tens of documents, thousands of chunks,
ChromaDB on local disk. What changes at larger scale:

- **Embedding** becomes the ingestion bottleneck; batch and run on GPU.
- **ChromaDB** is fine into the low millions of chunks. Beyond that, a dedicated
  vector service with replication is warranted.
- **Retrieval quality**, not throughput, is the real constraint — which is why
  the roadmap prioritises hybrid search and reranking over infrastructure.
