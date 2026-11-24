"""Tests for the evaluation metrics and the dataset contract."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.dataset import DatasetError, load_dataset, labelled_cases, unlabelled_cases, validate_dataset
from regintel.evaluation import (
    AnswerReport,
    evaluate_answers,
    evaluate_retrieval,
    hit_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    score_answer,
)
from regintel.schema import Answer, Chunk, Citation, DocumentMetadata, RetrievedChunk


def _ret(pairs):
    """pairs: list of (doc_id, page) -> RetrievedChunk list in rank order."""
    out = []
    for i, (doc_id, page) in enumerate(pairs):
        chunk = Chunk(doc_id=doc_id, chunk_index=i, page_number=page,
                      text=f"text {doc_id} {page}", metadata=DocumentMetadata())
        out.append(RetrievedChunk(chunk=chunk, score=1.0 - i * 0.1, rank=i))
    return out


# --------------------------------------------------------------- retrieval math

def test_precision_at_k():
    retrieved = _ret([("d", 1), ("d", 2), ("d", 3), ("d", 4)])
    relevant = {("d", 1), ("d", 3)}
    assert precision_at_k(retrieved, relevant, 4) == 0.5
    assert precision_at_k(retrieved, relevant, 2) == 0.5
    assert precision_at_k(retrieved, relevant, 1) == 1.0


def test_recall_at_k():
    retrieved = _ret([("d", 1), ("d", 2)])
    relevant = {("d", 1), ("d", 5)}
    assert recall_at_k(retrieved, relevant, 2) == 0.5
    assert recall_at_k(retrieved, {("d", 9)}, 2) == 0.0


def test_recall_is_zero_with_no_ground_truth():
    assert recall_at_k(_ret([("d", 1)]), set(), 5) == 0.0


def test_hit_at_k():
    retrieved = _ret([("d", 4), ("d", 1)])
    assert hit_at_k(retrieved, {("d", 1)}, 2) == 1.0
    assert hit_at_k(retrieved, {("d", 1)}, 1) == 0.0
    assert hit_at_k(retrieved, {("d", 7)}, 2) == 0.0


def test_reciprocal_rank():
    retrieved = _ret([("d", 9), ("d", 1), ("d", 2)])
    assert reciprocal_rank(retrieved, {("d", 1)}) == 0.5
    assert reciprocal_rank(retrieved, {("d", 9)}) == 1.0
    assert reciprocal_rank(retrieved, {("d", 99)}) == 0.0


def test_document_identity_matters_not_just_page():
    retrieved = _ret([("other-doc", 1)])
    assert precision_at_k(retrieved, {("the-doc", 1)}, 1) == 0.0
    assert hit_at_k(retrieved, {("the-doc", 1)}, 1) == 0.0


def test_precision_counts_top_k_only():
    retrieved = _ret([("d", 5), ("d", 1)])
    assert precision_at_k(retrieved, {("d", 1)}, 1) == 0.0


def test_empty_retrieval_scores_zero():
    assert precision_at_k([], {("d", 1)}, 5) == 0.0
    assert hit_at_k([], {("d", 1)}, 5) == 0.0
    assert reciprocal_rank([], {("d", 1)}) == 0.0


# ------------------------------------------------------------ retrieval report

def test_unlabelled_cases_are_skipped_not_scored_zero():
    cases = [
        {"id": "labelled", "question": "q1",
         "relevant": [{"doc_id": "d", "pages": [1]}]},
        {"id": "unlabelled", "question": "q2", "relevant": []},
    ]
    report = evaluate_retrieval(
        cases, retrieve_fn=lambda q, k: _ret([("d", 1)]), k=3
    )
    assert report.skipped_no_ground_truth == ["unlabelled"]
    assert len(report.evaluated_scores) == 1
    summary = report.summary()
    assert summary["questions_evaluated"] == 1
    assert summary["questions_skipped_no_ground_truth"] == 1
    assert summary["precision_at_3"] is not None


def test_summary_is_none_not_zero_when_nothing_labelled():
    cases = [{"id": "a", "question": "q", "relevant": []}]
    report = evaluate_retrieval(cases, retrieve_fn=lambda q, k: _ret([("d", 1)]), k=5)
    summary = report.summary()
    # None signals "not measured". 0.0 would read as a real, bad score.
    assert summary["precision_at_5"] is None
    assert summary["recall_at_5"] is None
    assert summary["mrr"] is None
    assert summary["questions_evaluated"] == 0


def test_unlabelled_case_does_not_call_the_retriever():
    calls = []

    def retrieve(q, k):
        calls.append(q)
        return _ret([("d", 1)])

    evaluate_retrieval([{"id": "a", "question": "q", "relevant": []}], retrieve, 3)
    assert calls == []


# ----------------------------------------------------------------- answer score

def _answer(text, citations, retrieved, insufficient=False):
    return Answer(question="q", text=text, citations=tuple(citations),
                  retrieved=tuple(retrieved), insufficient_evidence=insufficient,
                  model="m")


def _citation(marker, chunk):
    return Citation(marker=marker, chunk_id=chunk.chunk_id, doc_id=chunk.doc_id,
                    title=None, regulator=None, page_number=chunk.page_number)


def test_well_formed_answer_scores_clean():
    retrieved = _ret([("d", 1), ("d", 2)])
    cites = [_citation("[1]", retrieved[0].chunk)]
    score = score_answer("q1", _answer("Claim [1].", cites, retrieved))
    assert score.citations_well_formed is True
    assert score.valid_citations == 1
    assert score.invalid_citations == 0


def test_invented_marker_is_counted_against_the_answer():
    retrieved = _ret([("d", 1)])
    cites = [_citation("[1]", retrieved[0].chunk)]
    # The text cites [1] and [4]; only [1] survived validation.
    score = score_answer("q1", _answer("Claim [1]. Bad [4].", cites, retrieved))
    assert score.invalid_citations == 1
    assert score.citations_well_formed is False


def test_citation_not_in_retrieved_set_is_not_valid():
    retrieved = _ret([("d", 1)])
    orphan = Chunk(doc_id="d", chunk_index=99, page_number=50, text="elsewhere",
                   metadata=DocumentMetadata())
    cites = [_citation("[1]", orphan)]
    score = score_answer("q1", _answer("Claim [1].", cites, retrieved))
    assert score.valid_citations == 0
    assert score.citations_well_formed is False


def test_refusal_accuracy_correct_and_incorrect():
    retrieved = _ret([("d", 1)])
    good = score_answer("r1", _answer("no evidence", [], retrieved, insufficient=True),
                        expected_refusal=True)
    assert good.refused is True
    assert good.refusal_correct is True

    bad = score_answer("r2", _answer("Confident claim.", [], retrieved,
                                     insufficient=False), expected_refusal=True)
    assert bad.refusal_correct is False


def test_expected_refusal_none_leaves_judgement_unset():
    score = score_answer("q", _answer("x", [], _ret([("d", 1)])))
    assert score.refusal_correct is None


def test_answer_report_summary():
    retrieved = _ret([("d", 1)])
    cites = [_citation("[1]", retrieved[0].chunk)]
    report = AnswerReport(scores=[
        score_answer("a", _answer("Claim [1].", cites, retrieved), expected_refusal=False),
        score_answer("b", _answer("nope", [], retrieved, insufficient=True),
                     expected_refusal=True),
    ])
    summary = report.summary()
    assert summary["questions"] == 2
    assert summary["refusal_accuracy"] == 1.0
    assert summary["invented_citation_count"] == 0
    assert summary["citation_validity_rate"] == 1.0


def test_evaluate_answers_passes_questions_through():
    asked = []
    retrieved = _ret([("d", 1)])

    def ask(q):
        asked.append(q)
        return _answer("nope", [], retrieved, insufficient=True)

    cases = [{"id": "r1", "question": "unanswerable?", "expected_refusal": True}]
    report = evaluate_answers(cases, ask)
    assert asked == ["unanswerable?"]
    assert report.summary()["refusal_accuracy"] == 1.0


def test_empty_answer_report_summary():
    assert AnswerReport().summary() == {"questions": 0}


# --------------------------------------------------------------------- dataset

def test_shipped_dataset_is_valid():
    data = load_dataset()
    assert data["answerable"] and data["refusal"]
    # Refusal cases must all be marked, and answerable ids must be unique.
    assert all(c["expected_refusal"] is True for c in data["refusal"])


def test_shipped_dataset_has_no_fabricated_ground_truth():
    # Ground truth must be labelled by a human. Shipping it empty is deliberate.
    data = load_dataset()
    for case in data["answerable"]:
        if not case.get("relevant"):
            assert case.get("ground_truth_status") == "unlabelled"
    assert len(unlabelled_cases(data)) + len(labelled_cases(data)) == len(data["answerable"])


def test_validate_rejects_duplicate_ids():
    bad = {
        "answerable": [{"id": "x", "question": "a"}, {"id": "x", "question": "b"}],
        "refusal": [],
    }
    try:
        validate_dataset(bad)
    except DatasetError as exc:
        assert "duplicate" in str(exc)
        return
    raise AssertionError("expected DatasetError")


def test_validate_rejects_zero_based_pages():
    bad = {
        "answerable": [{"id": "x", "question": "a",
                        "relevant": [{"doc_id": "d", "pages": [0]}]}],
        "refusal": [],
    }
    try:
        validate_dataset(bad)
    except DatasetError as exc:
        assert "1-based" in str(exc)
        return
    raise AssertionError("expected DatasetError")


def test_validate_rejects_unmarked_refusal_case():
    bad = {"answerable": [], "refusal": [{"id": "r", "question": "q"}]}
    try:
        validate_dataset(bad)
    except DatasetError as exc:
        assert "expected_refusal" in str(exc)
        return
    raise AssertionError("expected DatasetError")


def test_validate_rejects_missing_section():
    try:
        validate_dataset({"answerable": []})
    except DatasetError as exc:
        assert "refusal" in str(exc)
        return
    raise AssertionError("expected DatasetError")
