#!/usr/bin/env python3
"""Help a reviewer label retrieval ground truth. No LLM calls, no cost.

For each unlabelled question it prints the top retrieved passages with their
doc_id and page number, so a compliance reviewer can read them and record which
pages genuinely answer the question.

The reviewer's judgement is the ground truth — this script only surfaces
candidates. It never writes labels by itself, because an auto-labelled dataset
would just be measuring the retriever against its own opinion.

Usage:
    python eval/label_helper.py
    python eval/label_helper.py --id cdd-trust-beneficial-owner --k 8 --chars 600
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.dataset import load_dataset, unlabelled_cases  # noqa: E402
from regintel.config import Config  # noqa: E402
from regintel.pipeline import build_pipeline  # noqa: E402


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=6)
    parser.add_argument("--chars", type=int, default=400)
    parser.add_argument("--id", dest="case_id", default=None)
    args = parser.parse_args(argv)

    config = Config.from_env()
    pipeline = build_pipeline(config, with_llm=False)
    if pipeline.chunk_count() == 0:
        print("The vector store is empty — ingest documents first.")
        return 1

    data = load_dataset()
    cases = unlabelled_cases(data)
    if args.case_id:
        cases = [c for c in data["answerable"] if c["id"] == args.case_id]
        if not cases:
            print(f"No answerable case with id {args.case_id!r}")
            return 1

    if not cases:
        print("Every answerable case already has ground truth. Nothing to label.")
        return 0

    print(f"Embedder: {pipeline.embedder.name}")
    print(f"{len(cases)} unlabelled case(s). Read each passage and decide which "
          "pages truly answer the question.\n")

    for case in cases:
        print("=" * 78)
        print(f"id:       {case['id']}")
        print(f"question: {case['question']}")
        if case.get("requires_document"):
            print(f"needs:    {case['requires_document']} (not ingested? label will be empty)")
        print()
        results = pipeline.retrieve(case["question"], top_k=args.k)
        if not results:
            print("  (nothing retrieved)")
        by_doc: dict[str, list[int]] = {}
        for r in results:
            c = r.chunk
            by_doc.setdefault(c.doc_id, []).append(c.page_number)
            print(f"  [{r.rank + 1}] score={r.score:.3f}  doc={c.doc_id}  "
                  f"page={c.page_number}")
            if c.section:
                print(f"       section: {c.section}")
            snippet = c.text[: args.chars].replace("\n", " ")
            print(f"       {snippet}...")
            print()
        print("  Candidate label (EDIT: keep only genuinely responsive pages):")
        entries = ", ".join(
            '{"doc_id": "%s", "pages": %s}' % (doc, sorted(set(pages)))
            for doc, pages in by_doc.items()
        )
        print(f'  "relevant": [{entries}]')
        print()

    print("=" * 78)
    print("Paste confirmed labels into eval/questions.json and set "
          '"ground_truth_status": "labelled".')
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
