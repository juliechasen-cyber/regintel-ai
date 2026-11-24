"""Evaluation metrics.

Two deliberately separate layers (per the project spec):

**Retrieval evaluation** — did the right passages come back? Measured with
precision@k, recall@k, hit rate and MRR against human-labelled ground truth
(which document/pages actually answer the question). No LLM involved, so it is
cheap, deterministic and repeatable.

**Answer evaluation** — given the retrieved evidence, was the generated answer
well-behaved? Measured structurally wherever possible:
* citation validity — every ``[n]`` maps to a retrieved passage;
* citation grounding — cited passages exist and carry real page numbers;
* refusal correctness — questions with no supporting evidence must be refused.

Keeping them apart matters: a bad answer from good retrieval is a prompt problem,
while a good-looking answer from bad retrieval is luck. Mixing the two hides both.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

from .citations import markers_in_text
from .schema import Answer, RetrievedChunk


# --------------------------------------------------------------------- retrieval

def _relevant_keys(doc_id: str, pages: Sequence[int]) -> set[tuple[str, int]]:
    return {(doc_id, int(p)) for p in pages}


def _retrieved_keys(retrieved: Sequence[RetrievedChunk]) -> list[tuple[str, int]]:
    return [(r.chunk.doc_id, r.chunk.page_number) for r in retrieved]


def precision_at_k(retrieved: Sequence[RetrievedChunk],
                   relevant: set[tuple[str, int]], k: int) -> float:
    """Fraction of the top-k retrieved passages that are relevant.

    Counts distinct (doc, page) pairs so several chunks from the same relevant
    page are not double-counted as separate wins.
    """
    if k <= 0:
        return 0.0
    top = _retrieved_keys(retrieved)[:k]
    if not top:
        return 0.0
    hits = sum(1 for key in top if key in relevant)
    return hits / len(top)


def recall_at_k(retrieved: Sequence[RetrievedChunk],
                relevant: set[tuple[str, int]], k: int) -> float:
    """Fraction of the relevant pages that appear in the top-k."""
    if not relevant:
        return 0.0
    top = set(_retrieved_keys(retrieved)[:k])
    return len(top & relevant) / len(relevant)


def hit_at_k(retrieved: Sequence[RetrievedChunk],
             relevant: set[tuple[str, int]], k: int) -> float:
    """1.0 if any relevant page is in the top-k, else 0.0."""
    top = set(_retrieved_keys(retrieved)[:k])
    return 1.0 if top & relevant else 0.0


def reciprocal_rank(retrieved: Sequence[RetrievedChunk],
                    relevant: set[tuple[str, int]]) -> float:
    """1/(rank of the first relevant passage), 1-based. 0.0 if none found."""
    for i, key in enumerate(_retrieved_keys(retrieved), start=1):
        if key in relevant:
            return 1.0 / i
    return 0.0


@dataclass
class RetrievalScore:
    question_id: str
    precision: float
    recall: float
    hit: float
    mrr: float
    k: int
    evaluated: bool = True   # False when the question has no ground truth yet


@dataclass
class RetrievalReport:
    k: int
    scores: list[RetrievalScore] = field(default_factory=list)
    skipped_no_ground_truth: list[str] = field(default_factory=list)

    @property
    def evaluated_scores(self) -> list[RetrievalScore]:
        return [s for s in self.scores if s.evaluated]

    def _mean(self, attr: str) -> Optional[float]:
        rows = self.evaluated_scores
        if not rows:
            return None
        return sum(getattr(s, attr) for s in rows) / len(rows)

    def summary(self) -> dict:
        """Aggregate metrics. ``None`` when nothing could be evaluated — never 0.0,
        which would read as a real measured score."""
        return {
            "k": self.k,
            "questions_evaluated": len(self.evaluated_scores),
            "questions_skipped_no_ground_truth": len(self.skipped_no_ground_truth),
            f"precision_at_{self.k}": self._mean("precision"),
            f"recall_at_{self.k}": self._mean("recall"),
            f"hit_rate_at_{self.k}": self._mean("hit"),
            "mrr": self._mean("mrr"),
        }


def evaluate_retrieval(
    cases: Iterable[dict],
    retrieve_fn,
    k: int = 6,
) -> RetrievalReport:
    """Score retrieval over labelled cases.

    Each case needs ``id``, ``question`` and, to be scored, a ``relevant``
    list of ``{"doc_id": ..., "pages": [...]}``. Cases without ground truth are
    reported as skipped rather than silently counted as failures.
    """
    report = RetrievalReport(k=k)
    for case in cases:
        qid = case.get("id", case.get("question", "?"))
        relevant: set[tuple[str, int]] = set()
        for entry in case.get("relevant") or []:
            doc_id = entry.get("doc_id")
            pages = entry.get("pages") or []
            if doc_id and pages:
                relevant |= _relevant_keys(doc_id, pages)

        if not relevant:
            report.skipped_no_ground_truth.append(qid)
            report.scores.append(
                RetrievalScore(qid, 0.0, 0.0, 0.0, 0.0, k, evaluated=False)
            )
            continue

        retrieved = retrieve_fn(case["question"], k)
        report.scores.append(
            RetrievalScore(
                question_id=qid,
                precision=precision_at_k(retrieved, relevant, k),
                recall=recall_at_k(retrieved, relevant, k),
                hit=hit_at_k(retrieved, relevant, k),
                mrr=reciprocal_rank(retrieved, relevant),
                k=k,
            )
        )
    return report


# ----------------------------------------------------------------------- answers

@dataclass
class AnswerScore:
    question_id: str
    cited_markers: int
    valid_citations: int
    invalid_citations: int
    citations_well_formed: bool      # no invented markers, all pages real
    refused: bool
    expected_refusal: Optional[bool]
    refusal_correct: Optional[bool]


@dataclass
class AnswerReport:
    scores: list[AnswerScore] = field(default_factory=list)

    def summary(self) -> dict:
        n = len(self.scores)
        if n == 0:
            return {"questions": 0}
        judged = [s for s in self.scores if s.refusal_correct is not None]
        answered = [s for s in self.scores if not s.refused]
        return {
            "questions": n,
            "citation_validity_rate": (
                sum(1 for s in self.scores if s.citations_well_formed) / n
            ),
            "answers_with_at_least_one_citation": (
                sum(1 for s in answered if s.valid_citations > 0) / len(answered)
                if answered else None
            ),
            "invented_citation_count": sum(s.invalid_citations for s in self.scores),
            "refusal_accuracy": (
                sum(1 for s in judged if s.refusal_correct) / len(judged)
                if judged else None
            ),
        }


def score_answer(question_id: str, answer: Answer,
                 expected_refusal: Optional[bool] = None) -> AnswerScore:
    """Structurally score one generated answer.

    This checks what can be checked without a human or a judge model: that the
    answer's citations point at passages that were really retrieved, and that it
    refused exactly when it should have.
    """
    retrieved_ids = {r.chunk.chunk_id for r in answer.retrieved}
    valid = [c for c in answer.citations if c.chunk_id in retrieved_ids]
    markers_used = markers_in_text(answer.text)
    # Markers the model emitted that survived validation vs. those dropped.
    invalid_count = max(0, len(markers_used) - len(answer.citations))

    pages_real = all(c.page_number >= 1 for c in answer.citations)
    well_formed = (len(valid) == len(answer.citations)) and pages_real and invalid_count == 0

    refused = answer.insufficient_evidence
    refusal_correct = None if expected_refusal is None else (refused == expected_refusal)

    return AnswerScore(
        question_id=question_id,
        cited_markers=len(markers_used),
        valid_citations=len(valid),
        invalid_citations=invalid_count,
        citations_well_formed=well_formed,
        refused=refused,
        expected_refusal=expected_refusal,
        refusal_correct=refusal_correct,
    )


def evaluate_answers(cases: Iterable[dict], ask_fn) -> AnswerReport:
    """Generate and score an answer for each case.

    ``cases`` may set ``"expected_refusal": true`` for questions the corpus
    cannot support — those test that the system declines instead of inventing.
    """
    report = AnswerReport()
    for case in cases:
        qid = case.get("id", case.get("question", "?"))
        answer = ask_fn(case["question"])
        report.scores.append(
            score_answer(qid, answer, case.get("expected_refusal"))
        )
    return report
