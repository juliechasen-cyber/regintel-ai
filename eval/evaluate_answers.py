#!/usr/bin/env python3
"""Answer-quality evaluation. Calls the Claude API — this one costs money.

Scores citation validity (no invented markers, real pages) and refusal accuracy
(does the system decline when the corpus cannot support an answer?).

Usage:
    python eval/evaluate_answers.py --refusal-only   # cheapest meaningful run
    python eval/evaluate_answers.py --limit 5
    python eval/evaluate_answers.py --all
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.dataset import load_dataset  # noqa: E402
from regintel.config import Config  # noqa: E402
from regintel.evaluation import evaluate_answers  # noqa: E402
from regintel.pipeline import build_pipeline  # noqa: E402


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refusal-only", action="store_true",
                        help="only run the cases that must be refused")
    parser.add_argument("--all", action="store_true", help="run every case")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--json", dest="json_out", default=None)
    parser.add_argument("--yes", action="store_true",
                        help="skip the cost confirmation prompt")
    args = parser.parse_args(argv)

    config = Config.from_env()
    pipeline = build_pipeline(config, with_llm=True)

    if pipeline.generator is None:
        print("No Anthropic client available. Set ANTHROPIC_API_KEY to run "
              "answer evaluation (retrieval evaluation needs no key).")
        return 1
    if pipeline.chunk_count() == 0:
        print("The vector store is empty — ingest documents first.")
        return 1

    data = load_dataset()
    if args.refusal_only:
        cases = list(data["refusal"])
    elif args.all:
        cases = list(data["answerable"]) + list(data["refusal"])
    else:
        cases = list(data["refusal"]) + list(data["answerable"])
    if args.limit:
        cases = cases[: args.limit]

    print(f"Model:  {config.anthropic_model}")
    print(f"Cases:  {len(cases)}  (each case is one billed API call)")
    if not args.yes:
        reply = input("Proceed? [y/N] ").strip().lower()
        if reply not in ("y", "yes"):
            print("Aborted — no API calls made.")
            return 0
    print()

    report = evaluate_answers(
        cases,
        ask_fn=lambda q: pipeline.ask(q),
    )

    for score in report.scores:
        flags = []
        if score.invalid_citations:
            flags.append(f"{score.invalid_citations} INVENTED")
        if score.refusal_correct is False:
            flags.append("WRONG REFUSAL BEHAVIOUR")
        status = "  ".join(flags) or "ok"
        print(f"  {score.question_id}: cites={score.valid_citations} "
              f"refused={score.refused}  {status}")

    summary = report.summary()
    print("\n--- summary ---")
    for key, value in summary.items():
        shown = "n/a" if value is None else value
        print(f"{key}: {shown}")

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)
        print(f"\nWrote {args.json_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
