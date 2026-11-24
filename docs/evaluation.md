---
noteId: "da67b030e84211f192c02f52450d7c90"
tags: []

---

# Evaluation

## The core principle: two separate evaluations

Retrieval and generation are measured **separately**, and mixing them is the most
common way RAG evaluation goes wrong.

Consider a good-looking answer. It could mean:

- retrieval found the right passages and the model used them well; or
- retrieval found the wrong passages and the model papered over it with fluent
  prose; or
- retrieval found nothing useful and the model answered from memory.

A single end-to-end "is the answer good?" score cannot distinguish these. They
require opposite fixes — better embeddings vs. a better prompt — so a blended
metric tells you something has gone wrong without telling you where.

Hence:

| | Retrieval evaluation | Answer evaluation |
|---|---|---|
| Question | Did the right passages come back? | Given those passages, did the model behave? |
| LLM involved | No | Yes |
| Cost | $0 | Billed per question |
| Deterministic | Yes | No |
| Script | `eval/evaluate_retrieval.py` | `eval/evaluate_answers.py` |

Retrieval evaluation being free and deterministic means it can run on every
change. That's where most quality work actually happens.

---

## Retrieval metrics

Ground truth is a set of `(doc_id, page)` pairs a human judged responsive. A
retrieved chunk is relevant if its `(doc_id, page_number)` is in that set.

Matching on `(doc, page)` rather than `chunk_id` matters: chunk boundaries shift
when `chunk_tokens` changes, which would invalidate every label. Page labels
survive re-chunking, so the dataset keeps its value across configuration changes.

| Metric | Definition | Reads as |
|---|---|---|
| **Precision@k** | share of the top-k that are relevant | how much of what we showed the model was useful |
| **Recall@k** | share of relevant pages that appear in top-k | did we find what exists |
| **Hit rate@k** | 1 if any relevant page is in top-k | could this question be answered at all |
| **MRR** | 1/(rank of first relevant result) | how near the top the good passage sat |

Distinct `(doc, page)` pairs are counted, so three chunks from one relevant page
don't inflate precision as three separate wins.

**Hit rate is the metric to watch first.** If a relevant page never enters the
top-k, no amount of prompt work can produce a correct answer — that's a hard
ceiling. Precision and MRR then tune how efficiently context is spent.

---

## Answer metrics

Scored structurally wherever possible, because structural checks are objective,
free of judge-model bias, and catch exactly the failure this project is about.

| Metric | What it catches |
|---|---|
| **Citation validity rate** | share of answers where every `[n]` maps to a retrieved chunk and every page is real |
| **Invented citation count** | markers the model emitted that didn't map to anything — should be 0 |
| **Answers with ≥1 citation** | uncited assertions among non-refused answers |
| **Refusal accuracy** | did it decline exactly when the corpus couldn't support an answer |

### Refusal testing

`eval/questions.json` contains a `refusal` section of questions the corpus
provably cannot answer — a corporate tax rate, a Basel III capital ratio, a
specific bank's SAR volume, a HIPAA question, a request for legal advice about a
particular customer.

These are the highest-value tests in the suite. A system that answers them is
drawing on model memory, which is the exact behaviour that makes an AI tool
unusable in a regulated function. They also need **no page labels**, so they can
be run before any ground-truth labelling work.

`evaluate_answers.py --refusal-only` is the cheapest meaningful quality signal
available.

### What is not measured automatically

**Factual correctness of answer prose.** Deliberately. Deciding whether a summary
of a regulatory obligation is correct requires a compliance SME. An LLM-as-judge
score here would be a number that *looks* like an answer-quality measure while
actually measuring agreement between two models with correlated blind spots.

Groundedness is approached from the other side instead: constrain the model to
the evidence, verify every citation structurally, and always show the reader the
passages. The reader does the final verification — which, for compliance
research, is the correct division of labour.

---

## Ground truth: why it ships empty

`eval/questions.json` contains 12 answerable questions with `"relevant": []` and
`"ground_truth_status": "unlabelled"`.

This is deliberate, and it is the honest choice. Labelling which pages answer a
BSA/AML question is a compliance judgement. Auto-labelling it from the
retriever's own output would measure the retriever against its own opinion — a
guaranteed-high score that means nothing.

So the harness is built to make unlabelled data impossible to misreport:

- unlabelled questions are reported as `skipped_no_ground_truth`, not scored 0;
- the retriever **isn't even called** for them;
- aggregate metrics return **`None`**, rendered as `n/a (no labelled ground
  truth)` — never `0.0`, which a reader could mistake for a measured score.

Both behaviours are enforced by tests (`test_summary_is_none_not_zero_when_nothing_labelled`,
`test_unlabelled_case_does_not_call_the_retriever`).

### Current measured state

**No retrieval metric has been measured for this project.** The metrics are
implemented and unit-tested against synthetic fixtures; they have not been run
against labelled regulatory ground truth, because no labels exist yet.

Stating that plainly is better than publishing a precision figure that rests on
labels nobody verified.

### Labelling workflow

```bash
python eval/label_helper.py                       # all unlabelled questions
python eval/label_helper.py --id cdd-trust-beneficial-owner --k 8 --chars 600
```

For each question it prints the top retrieved passages with `doc_id`, page,
section and text, plus a pre-formatted candidate label:

```json
"relevant": [{"doc_id": "fincen-cdd-rule-consolidated-faqs", "pages": [6, 7, 8]}]
```

A reviewer reads each passage, **deletes the pages that aren't genuinely
responsive**, and pastes the result in. The script never writes labels itself.

Two cautions worth stating, since this workflow is the dataset's weak point:

1. It only surfaces what the current retriever finds, so it inherits that
   retriever's blind spots. A page the retriever never surfaces won't get
   labelled, which inflates recall. Periodically label from the document directly.
2. Several questions carry `requires_document`, naming a PDF not yet downloaded.
   Those can't be labelled until the FFIEC documents are in place.

---

## Running the evaluations

```bash
# Free, deterministic, no API key. Run this constantly.
python eval/evaluate_retrieval.py
python eval/evaluate_retrieval.py --k 10 --json eval/results/retrieval-k10.json

# Billed — one API call per question.
python eval/evaluate_answers.py --refusal-only     # cheapest useful run
python eval/evaluate_answers.py --limit 5
python eval/evaluate_answers.py --all --yes
```

`evaluate_answers.py` prints the model and the number of billed calls and asks
for confirmation before spending anything (`--yes` to skip).

---

## What a good result looks like

Targets to hold the system to once labelling is done. These are **goals, not
measurements** — nothing here has been observed yet.

| Metric | Target | Reasoning |
|---|---|---|
| Hit rate@6 | > 0.90 | A relevant page missing from top-6 is an unanswerable question. |
| Recall@6 | > 0.80 | Multi-page obligations need most of their pages present. |
| Precision@6 | > 0.50 | Some irrelevant context is tolerable; the model is instructed to ignore it. |
| MRR | > 0.70 | The best passage should usually be first or second. |
| Invented citations | **0** | Non-negotiable. Any nonzero value is a bug, not a tuning problem. |
| Refusal accuracy | > 0.95 | Declining out-of-corpus questions is the product. |

Invented citations are the one metric with no acceptable nonzero value. If the
validator ever reports one, the fix belongs in `generation.py` or the prompt, not
in the threshold.

## Regression workflow

1. Change one thing (chunk size, embedding model, k, separators).
2. Run `evaluate_retrieval.py` — free, so it can run on every commit.
3. Compare the JSON against the previous run.
4. Run `evaluate_answers.py --refusal-only` before anything ships.
5. Record what moved and why.

Changing chunking invalidates nothing in the labels, because ground truth is
page-based. That's the property that makes this loop cheap enough to actually
use.
