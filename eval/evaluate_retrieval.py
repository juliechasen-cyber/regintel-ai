#!/usr/bin/env python3
"""Retrieval-only evaluation. No LLM calls, no API key, no cost.

Usage:
    python eval/evaluate_retrieval.py            # k from config
    python eval/evaluate_retrieval.py --k 10
    python eval/evaluate_retrieval.py --json out.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.dataset import load_dataset, labelled_cases, unlabelled_cases  # noqa: E402
from regintel.config import Config  # noqa: E402
from regintel.evaluation import evaluate_retrieval  # noqa: E402
from regintel.pipeline import build_pipeline  # noqa: E402


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--json", dest="json_out", default=None)
    args = parser.parse_args(argv)

    config = Config.from_env()
    k = args.k or config.top_k

    pipeline = build_pipeline(config, with_llm=False)  # retrieval only
    if pipeline.chunk_count() == 0:
        print("The vector store is empty — ingest documents first "
              "(python -m regintel.cli ingest data/pdfs).")
        return 1

    data = load_dataset()
    cases = data["answerable"]

    print(f"Embedder:      {pipeline.embedder.name}")
    print(f"Store chunks:  {pipeline.chunk_count()}")
    print(f"k:             {k}")
    print(f"Labelled:      {len(labelled_cases(data))} of {len(cases)} answerable cases")
    print()

    report = evaluate_retrieval(
        cases,
        retrieve_fn=lambda q, kk: pipeline.retrieve(q, top_k=kk),
        k=k,
    )

    for score in report.scores:
        if not score.evaluated:
            print(f"  skip  {score.question_id}  (no ground truth)")
        else:
            print(f"  eval  {score.question_id}  "
                  f"P@{k}={score.precision:.2f} R@{k}={score.recall:.2f} "
                  f"hit={score.hit:.0f} RR={score.mrr:.2f}")

    summary = report.summary()
    print("\n--- summary ---")
    for key, value in summary.items():
        shown = "n/a (no labelled ground truth)" if value is None else value
        print(f"{key}: {shown}")

    if report.skipped_no_ground_truth:
        print(f"\n{len(report.skipped_no_ground_truth)} question(s) could not be "
              "scored because no ground truth is labelled. Run "
              "`python eval/label_helper.py` to label them.")

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)
        print(f"\nWrote {args.json_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
