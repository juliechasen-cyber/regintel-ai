---
noteId: "86edc110e84211f192c02f52450d7c90"
tags: []

---

# RegIntel AI

**Retrieval-Augmented Generation for U.S. banking regulatory research.**

RegIntel AI lets a bank compliance professional ask a natural-language question
about BSA/AML obligations and get an answer that is grounded in, and cited back
to, the actual regulatory documents — down to the page.

It is a **RAG application**, not an autonomous agent: a deterministic pipeline of
ingest → chunk → embed → retrieve → generate → cite. There are no tools, no
planning loop, and no self-directed actions. Every claim traces to a passage the
system retrieved and can show you.

```
Question ──► Embed ──► Vector search ──► Top-k passages ──► Claude ──► Cited answer
                            │                                              │
                       ChromaDB                                  every [n] maps to
                    (local, persistent)                          a retrieved page
```

---

## Why this exists

A compliance officer researching a question like *"who counts as a beneficial
owner when a trust holds 25% of a legal entity customer?"* has to work across the
FFIEC BSA/AML Examination Manual, FinCEN rules and FAQs, and agency guidance.
Keyword search finds documents; it doesn't answer questions. A general chatbot
answers questions but invents citations — which in a regulated function is worse
than no answer.

RegIntel is built around that specific failure mode. See
[docs/business_case.md](docs/business_case.md).

## The anti-hallucination design

This is the core of the project, enforced in code rather than by asking the model
nicely:

| Risk | Control | Where |
|---|---|---|
| Invented page or section references | Citations are **constructed from retrieved chunks**, never parsed from model prose. The model only emits `[n]` markers. | `citations.py` |
| Citing something not retrieved | Markers that don't map to a retrieved chunk are **dropped and flagged** in the answer. | `generation.py` |
| Answering from model memory | System prompt restricts the model to the evidence block; `INSUFFICIENT_EVIDENCE` is a required response path. | `generation.py` |
| Answering with no evidence at all | Zero retrieval results **short-circuits before any API call**. | `generation.py` |
| Fabricated document metadata | Unknown regulator/title/date stay `None` and render as `(unknown)`. Metadata comes from a verified sidecar, never inferred from text. | `ingestion.py` |
| Wrong page numbers | Chunks never span pages by default, so every citation's page is exact. | `chunking.py` |
| Confusing guidance with binding law | The prompt requires the model to distinguish regulation/statute from supervisory guidance. | `generation.py` |
| Flattering evaluation scores | Unlabelled questions are reported as *skipped*, and aggregate metrics return `None` — never `0.0`, which would read as a real score. | `evaluation.py` |

## Quick start

```bash
git clone <this-repo> && cd regintel-ai
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env     # add your ANTHROPIC_API_KEY

# Fetch the public source documents listed in data/sources.json
python scripts/fetch_sources.py

# Build the index
python -m regintel.cli ingest data/pdfs

# Retrieval only — free, no API key needed
python -m regintel.cli search "who is a beneficial owner"

# Full RAG answer — one billed Claude call
python -m regintel.cli ask "Who is a beneficial owner of a legal entity customer?"

# Or use the UI
streamlit run app/streamlit_app.py
```

Check what's actually configured at any time:

```bash
python -m regintel.cli status
```

### A note on the source documents

`data/sources.json` lists 10 public FFIEC and FinCEN documents. The downloader
fetches what it can; **`bsaaml.ffiec.gov` sits behind a Cloudflare bot challenge
and refuses automated requests**, so those 9 PDFs must be downloaded in a normal
browser and saved into `data/pdfs/` using the filenames the script prints. The
script writes each document's `.meta.json` sidecar either way, so metadata and
citations are correct as soon as the PDF is in place.

PDFs and the vector index are gitignored; the repo stays code-only and the corpus
is reproducible from the manifest.

## Project layout

```
regintel/
  schema.py       Dataclasses shared by every stage (Document, Chunk, Citation, Answer)
  config.py       Environment-driven configuration, all tunables in one place
  ingestion.py    PyMuPDF extraction, text cleaning, running-header removal
  chunking.py     Configurable recursive splitter (~800 tokens / ~120 overlap)
  embeddings.py   Sentence Transformers + deterministic offline fallback
  vectorstore.py  ChromaDB (persistent) + in-memory store, dedup, metadata filters
  retrieval.py    Semantic top-k search with regulator / type filtering
  generation.py   Grounded prompt construction + Claude call + citation validation
  citations.py    Citation building and marker validation
  evaluation.py   Retrieval metrics and answer metrics, kept separate
  cli.py          status / ingest / search / ask / library
app/
  streamlit_app.py  Research, document library, upload
eval/
  questions.json       Evaluation dataset (ground truth labelled by a human)
  evaluate_retrieval.py  Free, deterministic, no LLM
  evaluate_answers.py    Citation validity + refusal accuracy (billed)
  label_helper.py        Surfaces candidate passages for a reviewer to label
scripts/
  fetch_sources.py   Download source PDFs + write metadata sidecars
tests/               109 tests
docs/                architecture, rag_pipeline, evaluation, business_case
```

## Testing

```bash
pytest tests/ -q        # or: python run_tests.py   (no pytest needed)
```

`run_tests.py` is a zero-dependency runner so the suite executes on a bare Python
install. The heavy dependencies are injected, so the full RAG path — including
generation — is testable with no API key, no model download and no network.

### Verification status

Stated precisely, because "it works" is not a claim worth making loosely:

| Component | Status |
|---|---|
| Chunking, citations, retrieval ranking, dedup, eval metrics | **Verified** — 109 passing tests |
| PDF ingestion (PyMuPDF 2.1) | **Verified against a real document** — FinCEN CDD FAQs, 24 pages → 42 chunks, page numbers exact |
| ChromaDB persistence + cross-process dedup | **Verified** — ChromaDB 1.5.9; re-ingest adds 0 chunks |
| Metadata filtering | **Verified** — `--regulator FinCEN` returns results, `--regulator OCC` correctly returns none |
| Running-header removal | **Verified** — FinCEN page banners no longer pollute chunk text |
| Claude generation | **Logic tested with a fake client.** Prompt construction, insufficient-evidence handling and citation validation are covered by tests. **No live API call has been made in this build**, so live answer quality is unmeasured. |
| Sentence Transformers embeddings | **Not yet exercised.** `sentence-transformers` was not installed during the build, so retrieval above ran on the offline hashing embedder. Install it for real semantic quality — scores shown in testing are not representative. |
| Streamlit UI | **Runs clean** — Streamlit 1.67; app serves HTTP 200 headless and the full script body executes without error. Not yet click-tested in a browser by a human. |
| Evaluation harness | **Runs** — `evaluate_retrieval.py` and `label_helper.py` execute against the real index. With no labels present the harness correctly reports `n/a` rather than a score. |

Nothing in the evaluation harness has been run against labelled ground truth,
because `eval/questions.json` ships deliberately unlabelled. Retrieval metrics
are implemented and unit-tested, but **no retrieval score has been measured yet.**

## Roadmap

Phased on purpose; the interfaces are stable so each lands without touching
callers.

- **Phase 2** — hybrid retrieval (BM25 keyword + dense, score fusion). Regulatory
  language is full of exact terms (`31 CFR 1010.230`) where lexical matching beats
  embeddings.
- **Phase 2** — cross-encoder reranking of the top-k before generation.
- **Phase 3** — labelled ground truth for the full question set, then a measured
  retrieval baseline to regress against.
- **Phase 3** — multi-document comparison ("where do the OCC and FinCEN differ
  on X?").

## Cost

| Component | Cost |
|---|---|
| Python, PyMuPDF, Sentence Transformers, ChromaDB, Streamlit, NumPy | $0 — all local |
| Embedding generation | $0 — runs on your machine |
| Ingestion, chunking, retrieval, retrieval evaluation | $0 |
| Claude API (answer generation only) | Usage-based |

Only generation costs money. `search`, `ingest` and
`eval/evaluate_retrieval.py` never call a paid API.

## Limitations

- **Not legal advice.** A research tool that surfaces sourced passages. Read the
  cited source before relying on anything.
- **Answers are bounded by the corpus.** If a document isn't ingested, the right
  answer is "insufficient evidence" — and that is the intended behaviour.
- **Currency is the user's responsibility.** Regulations change; the index is a
  snapshot. `publication_date` is shown as supplied so staleness is visible.
- **Page-bounded chunks** can split an obligation that straddles a page break.
  The trade-off buys exact citations; set `REGINTEL_RESPECT_PAGES=0` to prefer
  continuity over page precision.

## License

MIT — see [LICENSE](LICENSE).
