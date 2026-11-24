# Business case

> **A note on figures.** This document contains no market-size, cost-saving or
> penalty statistics. Numbers like those circulate widely in vendor material and
> are usually unsourced or stale. Where a quantity would strengthen the argument,
> this document names the number a buyer should measure **in their own
> institution** instead of asserting an industry figure. Any benchmark added
> later should carry a citation and a date.

---

## The problem

A BSA/AML analyst's research question — *"when a trust owns 25% or more of a
legal entity customer, who is treated as the beneficial owner?"* — has an answer
that lives across several documents: the FinCEN CDD rule and its consolidated
FAQs, the FFIEC BSA/AML Examination Manual sections on CDD and CIP, and
supplementary agency guidance.

Finding it today means knowing which document to open, then searching inside it.
That requires the thing the institution is short of: a person who already knows
where the answer lives.

### Why the two obvious tools both fail

**Keyword search** returns documents, not answers. It can't bridge vocabulary —
a search for "beneficial owner" won't surface a passage phrased around
"natural person who owns or controls". The analyst still reads the manual.

**A general-purpose chatbot** answers fluently and cites confidently — and
sometimes the citation does not exist. A plausible-looking reference to a
section number that was never written is worse than no answer, because it
*passes review*. In a function whose output is examined by regulators, an
unverifiable answer has negative value.

That gap — better than keyword search, verifiable unlike a chatbot — is what
RegIntel occupies.

---

## What RegIntel does differently

Three properties, each a direct response to one of those failures:

**1. Every claim carries a verifiable citation.**
Not "according to FinCEN guidance" but *FinCEN CDD Rule Consolidated FAQs,
page 7*, with the passage shown. The reviewer can check it in seconds. The
system is structurally incapable of citing a page it didn't retrieve — the model
emits only an index number, and everything displayed is assembled from the
retrieved chunk's real metadata.

**2. It declines rather than guesses.**
Asked something the corpus doesn't cover, it says the evidence is insufficient.
This is the feature most likely to be undervalued in a demo and most likely to
decide an actual deployment: a tool that is wrong 5% of the time without saying
which 5% cannot be used for regulatory work at all.

**3. Documents never leave the institution during ingestion.**
Extraction, chunking, embedding and vector storage are entirely local. Only the
question and the retrieved passages reach an external API, and only at the
generation step. For an institution with data-residency constraints, this is
frequently the precondition for any evaluation.

---

## Who it's for

| User | Use |
|---|---|
| **BSA/AML analyst** | Day-to-day research with a citation they can paste into a file memo. |
| **BSA officer / compliance manager** | Consistency — the same question returns the same sourced answer regardless of who asks. |
| **Internal audit / independent testing** | Locating the guidance applicable to a finding, with provenance. |
| **New hire** | The steepest part of the learning curve is knowing where things are written. This substitutes for that knowledge while it's being built. |
| **Examination prep** | Assembling the authority behind a stated position, quickly. |

---

## Where the value actually comes from

Three mechanisms, in order of how defensible they are:

**1. Research time.** The clearest effect. An analyst who would have spent 20–40
minutes locating and reading the right manual section can locate the passage in
under a minute and spend the remaining time *reading* it. The tool doesn't
replace the reading — it replaces the searching.

*Measure in your institution:* time-to-answer for a sample of 20 real research
questions, before and after.

**2. Consistency.** Two analysts asking the same question get the same sourced
passages. Inconsistent internal positions are an examination finding in their own
right, and they arise naturally when the answer depends on who you asked.

*Measure:* variance in documented positions on recurring questions.

**3. Onboarding.** New staff become productive on research tasks sooner, because
corpus familiarity is supplied rather than acquired.

*Measure:* time until a new analyst completes research tasks without escalation.

### What it explicitly does not do

Stated plainly, because overselling this category is how these tools lose trust:

- It does not make compliance decisions.
- It does not replace legal review.
- It does not monitor transactions or file reports.
- It does not track regulatory change — the index is a snapshot, and
  `publication_date` is displayed as supplied so staleness is visible.
- It does not remove the obligation to read the source. It makes reading the
  source fast.

---

## Cost

| Component | Cost |
|---|---|
| Python, PyMuPDF, Sentence Transformers, ChromaDB, Streamlit, NumPy | $0, open source, local |
| Document ingestion, chunking, embedding | $0 — compute only |
| Retrieval, and all retrieval evaluation | $0 |
| Claude API — answer generation only | Usage-based |

Only generation costs anything. Ingesting a thousand pages is free; so is every
`search` and the entire retrieval regression suite. Per-question cost is a
function of retrieved context size and model choice, both configurable
(`REGINTEL_TOP_K`, `REGINTEL_MODEL` — Haiku for cheap, Sonnet as default, Opus for
hardest questions).

For an institution-specific figure: multiply expected monthly questions by the
token cost at your chosen model and `top_k`. The pipeline reports both, so this
is measurable rather than estimated.

---

## Risks, honestly

| Risk | Mitigation | Residual |
|---|---|---|
| Analyst over-trusts an answer | Citations and full retrieved evidence always shown; "not legal advice" stated in the UI. | Real. Mitigated by training, not by software. |
| Corpus is stale | `publication_date` displayed; ingestion is cheap and repeatable. | Requires an owner for corpus currency. |
| Relevant document never retrieved | Hit rate@k is the headline retrieval metric; refusal path covers the known-unknown case. | A silent miss is the main failure mode. Hybrid retrieval (Phase 2) targets it. |
| Model misreads a retrieved passage | Passage shown alongside the claim; temperature 0. | Real, and the reason the reader verifies. |
| Dependence on an external API | Only generation is external; retrieval degrades to passage search, which remains useful on its own. | Acceptable — `search` works with no API at all. |

The honest summary of residual risk: **this tool makes a knowledgeable person
faster. It does not make an unknowledgeable person safe.** Positioning it as the
latter is how it would cause harm.

---

## Why this is a credible engineering portfolio piece

What it demonstrates beyond "I called an LLM API":

- **A grounding problem solved structurally.** Citations can't be fabricated
  because the model's only citation output is an integer validated against
  retrieved chunks. That's an architectural guarantee, not a prompt instruction.
- **Honest evaluation.** Retrieval and answer quality measured separately;
  unlabelled ground truth reported as unmeasured rather than scored zero — a
  property enforced by its own tests.
- **Testability as a design constraint.** Every heavy dependency is injected, so
  the full RAG path runs in milliseconds with no API key, no model download and no
  network. 109 tests.
- **Real-document engineering.** Running-header removal, letter-spaced banner
  normalization, de-hyphenation and page-exact citations are all things that only
  surface once a real regulator PDF goes through the pipeline. Each of those came
  from a bug observed on an actual document, not from anticipation.
- **Stated limitations.** The verification table in the README distinguishes what
  is tested, what is verified against real data, and what has not been exercised
  at all.
