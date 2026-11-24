---
noteId: "c0cad580e84211f192c02f52450d7c90"
tags: []

---

# The RAG pipeline, stage by stage

Each stage below explains *what it does*, *why it's built that way*, and *what
was traded away*. The trade-offs are the interesting part.

---

## 1. Ingestion

**Input:** a PDF, plus an optional `<file>.pdf.meta.json` sidecar.
**Output:** a `Document` — ordered `Page` objects with 1-based page numbers.

### Text extraction

PyMuPDF (`page.get_text("text")`) in document order. Page numbers come from the
loop index + 1 and are carried untouched through every later stage.

### Cleaning

Three normalizations, in order:

1. **De-hyphenation.** `insti-\ntution` → `institution`. PDF line wrapping
   otherwise splits words, which breaks both embedding and keyword matching.
2. **Whitespace collapse.** Runs of spaces/tabs → one space; 3+ newlines → 2.
   Paragraph structure is preserved because the chunker uses `\n\n` as its
   highest-priority split boundary.
3. **Running header/footer removal.** See below.

### Running header removal — why it needs three guards

Regulator PDFs stamp a banner on every page. In the FinCEN CDD FAQs it extracts
as `F I N C E N  G U I D A N C E  7` (letter-spaced, with the page number). Left
in, it appears in most chunks, where it both wastes context and pulls every
chunk's embedding toward a meaningless shared centroid.

Removing repeated lines naively is dangerous: it will happily delete real
obligations. Three guards make it safe, and each one exists because the simpler
version failed:

1. **Position** — only the first/last 2 lines of a page are candidates. Mid-page
   body text is never touched.
2. **Length** — a candidate must be ≤60 chars and ≤8 words after de-letter-spacing.
   A banner is short; an obligation is a sentence.
3. **Frequency** — its page-number-insensitive signature must appear on ≥50% of
   pages, and the document must have ≥4 pages.

De-letter-spacing is applied *before* the length check. Without it,
`F I N C E N G U I D A N C E 7` counts as 15 words, fails the ≤8-word guard, and
survives — which is exactly the bug that showed up on the real document.

The signature also strips digits so `BANNER 7` and `BANNER 8` match.

> **Trade-off:** a genuine short heading that happens to appear at the top of
> most pages would be removed. In practice banners are the only text with that
> profile, and the alternative — banner text in every chunk — is worse.

### Metadata, and the refusal to guess

Fields come from the sidecar first, then the PDF's own metadata dictionary.
**Nothing is inferred from document text.** A regulator name is not extracted by
looking for "FinCEN" in the body, because a FinCEN document can discuss OCC rules.

Unknown fields stay `None` and render as `(unknown)`. This is the whole point: a
citation that says `(unknown)` is useful, and a citation that says "OCC" because a
regex got lucky is a liability.

---

## 2. Chunking

**Input:** a `Document`. **Output:** `Chunk` objects, each tied to one page.

### Parameters

| Parameter | Default | Rationale |
|---|---|---|
| `chunk_tokens` | 800 | Large enough to hold a complete regulatory provision with its conditions; small enough that 6 chunks leave room in context. |
| `chunk_overlap_tokens` | 120 | ~15%. Keeps an obligation intact when it straddles a boundary. |
| `respect_page_boundaries` | `True` | Guarantees exact citation pages. |

### The algorithm

A recursive character splitter, re-implemented rather than imported so the core
has no heavy dependency and every branch is unit-tested.

**Step 1 — descend to atomic pieces.** Split on the highest-priority separator;
for any piece still over budget, recurse with the next:

```
["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""]
```

Paragraphs before lines before sentences before clauses before words, with
character slicing as a last resort (so a 5,000-character unbroken string still
chunks). Separators are re-attached to their pieces so punctuation survives.

**Step 2 — merge greedily into chunks.** Accumulate pieces until the next one
would exceed `chunk_tokens`, then flush.

**Step 3 — carry overlap.** On flush, the tail pieces of the chunk just emitted
whose combined size is ≤ `overlap_tokens` are carried into the next chunk.

**Step 4 — don't strand a heading.** If a chunk would end on heading-like pieces,
they're held back and start the next chunk instead — a heading belongs with the
text it introduces, not orphaned at the end of the previous passage.

### Heading detection

Deliberately conservative, because a bad heading becomes bad chunk metadata.
A line qualifies only if it is 2–10 words, ≤80 chars, contains letters, doesn't
end in sentence punctuation, and isn't mostly single characters (which rejects
letter-spaced banners). Bare page numbers are rejected by the ≥2-word rule.

The running `section` is tracked across pages and attached to each chunk, so a
citation can say *which* section a passage came from.

### Token counting

Pluggable via `count_tokens`. Default is a `len(text)/4` heuristic — model-agnostic
and dependency-free. Tests inject a word counter for deterministic assertions.

> **Trade-off:** the heuristic is approximate, so a chunk may land somewhat over
> or under 800 real tokens. For chunk sizing this is immaterial — it is a
> retrieval granularity knob, not a hard context limit. Pass a real tokenizer if
> exact budgeting matters.

### Why page-bounded chunks

With `respect_page_boundaries=True`, a chunk never spans pages, so
`chunk.page_number` is always a real page.

The alternative — chunking across the whole document — produces more coherent
passages but forces a citation to either name a range or pick a page arbitrarily.
For compliance research, "page 7" must mean page 7. An obligation split across a
page break is partially recovered by overlap within each page and by retrieving
multiple chunks.

---

## 3. Embedding

**Default:** `sentence-transformers/all-MiniLM-L6-v2`, run locally. Configurable
via `REGINTEL_EMBEDDING_MODEL`.

Local embedding means **no document content leaves the machine during
ingestion** — which for a bank evaluating this is often the deciding
consideration. Only the final question and the retrieved passages reach an
external API, and only at generation time.

Vectors are L2-normalized, so cosine similarity reduces to a dot product.

### The offline fallback, and why it's labelled so loudly

If the model can't load, `build_embedder` returns a `HashingEmbedder` — the
hashing trick over word unigrams and bigrams. It is deterministic and lets the
whole pipeline run and be tested with zero downloads.

It has **no semantic understanding.** "SAR filing deadline" and "suspicious
activity report timeline" are near-unrelated to it. That's why its `name` carries
`(offline fallback — not for production)` and why the CLI and UI warn when it's
active. A fallback that silently degrades retrieval quality would be the most
dangerous thing in this codebase.

Set `REGINTEL_ALLOW_OFFLINE_EMBEDDER=0` to raise instead.

---

## 4. Vector storage

ChromaDB, persistent, cosine space, one collection.

Stored per chunk: text, embedding, and flat scalar metadata (`doc_id`,
`chunk_index`, `page_number`, `section`, `title`, `regulator`, `document_type`,
`publication_date`, `source_url`).

### Duplicate prevention

`chunk_id = f"{doc_id}:p{page}:c{index}:{sha256(text)[:16]}"`

Because the id is content-addressed, re-ingesting the same document produces the
same ids and the store rejects them. No separate ledger of ingested files, and it
survives process restarts — verified: re-running `ingest` on an already-indexed
document adds 0 chunks.

A *changed* document yields different text hashes and so ingests as new chunks,
which is the desired behaviour for a revised rule.

### Metadata filtering

`regulator` and `document_type` filter server-side (`$eq`, `$and` for both). This
is what makes "only tell me what FinCEN says" possible, and it's the hook the
UI's facet selectors use.

---

## 5. Retrieval

Embed the question with the *same* embedder used at ingestion, cosine search for
top-k (default 6), optionally filtered.

Returns `RetrievedChunk` — the chunk, its similarity score, and its rank. Scores
are surfaced in the CLI and UI because a top hit at 0.58 and one at 0.95 warrant
different confidence from the reader.

### Deferred to Phase 2, on purpose

**Hybrid (BM25 + dense) retrieval.** Regulatory text is full of exact tokens —
`31 CFR 1010.230`, `FinCEN Form 112` — where lexical match beats embeddings
outright. Dense-only retrieval is the single biggest known quality gap.

**Cross-encoder reranking.** Retrieve ~30, rerank to 6. Reliable precision gain
at a latency cost.

Both land behind the existing `Retriever` interface without touching callers.

---

## 6. Generation

One Claude call. The model receives a system prompt of rules and a user message
containing a numbered evidence block plus the question.

```
EVIDENCE:
[1] (FinCEN CDD Rule Consolidated FAQs | FinCEN | type: supervisory_guidance | page 7)
B.16. What process should a covered financial institution use to identify...

[2] (...)

QUESTION:
Who is a beneficial owner of a legal entity customer?

Answer using only the evidence above, with [n] citations. If the evidence is
insufficient, reply with INSUFFICIENT_EVIDENCE.
```

The five prompt rules: answer only from the evidence; cite every factual claim
with `[n]`; distinguish binding authority from guidance; emit
`INSUFFICIENT_EVIDENCE` rather than guessing; be precise and don't give legal
advice.

`temperature=0.0` — this is extraction and synthesis, not creative writing, and
reproducibility matters for audit.

### Why the model only emits numbers

The model never writes a page number, title or section reference. It writes
`[2]`. Everything a citation displays is assembled in code from the retrieved
chunk's real metadata.

This is the structural reason the system can't fabricate a citation. The model's
only citation-shaped output is an integer, and an integer either maps to a
retrieved passage or it doesn't.

### Post-generation validation

1. Parse the `[n]` markers actually used.
2. Keep those mapping to a retrieved chunk; collect the rest as invalid.
3. If any were invalid, append a note naming them and stating they were removed.
4. Detect `INSUFFICIENT_EVIDENCE`, strip the token, set the flag.

A reviewer sees that the model overreached, rather than the overreach being
silently swallowed or silently rendered.

---

## 7. Citations

Built from retrieved chunks, numbered `[1..k]` in rank order. Each carries
`chunk_id`, `doc_id`, title, regulator, **real page number**, document type and
source URL where known.

Rendered as e.g.
`[1] FinCEN CDD Rule Consolidated FAQs (FinCEN, p. 7) — https://...`

Only citations the answer *used* are returned; retrieved-but-uncited passages
remain visible under "Retrieved evidence" but aren't presented as sources.

---

## End-to-end, with real numbers

Measured on the one document downloaded during this build (FinCEN CDD
Consolidated FAQs):

```
$ python -m regintel.cli ingest data/pdfs
  fincen-cdd-rule-consolidated-faqs: 24 pages -> 42 chunks, 42 added

$ python -m regintel.cli ingest data/pdfs          # again
  fincen-cdd-rule-consolidated-faqs: 24 pages -> 42 chunks, 0 added (42 already present)

$ python -m regintel.cli search "who is a beneficial owner of a legal entity customer" -k 3
  [1] score=0.582  FinCEN CDD Rule Consolidated FAQs  page 7
       section: B. Beneficial Ownership Information Collection under the CDD Rule
       B.16. What process should a covered financial institution use to identify and verify...
```

Those similarity scores were produced by the **hashing fallback embedder**, not a
semantic model, and are not representative of real retrieval quality. Install
`sentence-transformers` before drawing any conclusion from them.
