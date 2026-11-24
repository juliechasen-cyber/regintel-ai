#!/usr/bin/env python3
"""Command-line interface.

    python -m regintel.cli status
    python -m regintel.cli ingest data/pdfs
    python -m regintel.cli ingest data/pdfs/one-file.pdf
    python -m regintel.cli search "beneficial ownership threshold"
    python -m regintel.cli ask "Who is a beneficial owner?"   # calls Claude (costs)
    python -m regintel.cli library

``search`` is retrieval only and free. ``ask`` adds the generation step and makes
one billed Anthropic API call.
"""

from __future__ import annotations

import argparse
import os
import sys

from .citations import format_sources_block
from .config import Config
from .pipeline import build_pipeline


def _print_status(pipeline, config: Config) -> None:
    store_kind = type(pipeline.store).__name__
    print(f"embedder:     {pipeline.embedder.name}")
    print(f"dimension:    {pipeline.embedder.dimension}")
    print(f"vector store: {store_kind}")
    if store_kind == "InMemoryVectorStore":
        print("              (ChromaDB unavailable — data is NOT persisted)")
    else:
        print(f"              path={config.chroma_path} collection={config.collection_name}")
    print(f"chunks:       {pipeline.chunk_count()}")
    print(f"model:        {config.anthropic_model}")
    # The Anthropic SDK constructs a client even with no key -- it only fails at
    # request time -- so client construction alone does not prove a key exists.
    # Check the environment first, or `status` cheerfully reports "ready" and the
    # user discovers the truth on their first billed call.
    if not os.environ.get("ANTHROPIC_API_KEY"):
        llm_state = "no ANTHROPIC_API_KEY (retrieval still works)"
    elif pipeline.generator is not None:
        llm_state = "ready"
    else:
        llm_state = "ANTHROPIC_API_KEY is set, but the client could not be built"
    print(f"llm:          {llm_state}")
    print(f"chunking:     {config.chunk_tokens} tokens / {config.chunk_overlap_tokens} overlap")
    print(f"top_k:        {config.top_k}")


def _iter_pdfs(target: str) -> list[str]:
    if os.path.isfile(target):
        return [target]
    if os.path.isdir(target):
        return [
            os.path.join(target, f)
            for f in sorted(os.listdir(target))
            if f.lower().endswith(".pdf")
        ]
    return []


def cmd_ingest(args, pipeline, config) -> int:
    paths = _iter_pdfs(args.target)
    if not paths:
        print(f"No PDFs found at {args.target}")
        return 1
    total_added = 0
    for path in paths:
        try:
            result = pipeline.ingest_pdf_file(path)
        except ImportError:
            print("PyMuPDF is not installed. pip install pymupdf")
            return 1
        except Exception as exc:
            print(f"  FAILED {os.path.basename(path)}: {exc}")
            continue
        total_added += result.chunks_added
        note = ""
        if result.chunks_skipped_as_duplicate:
            note = f" ({result.chunks_skipped_as_duplicate} already present)"
        print(f"  {result.doc_id}: {result.pages} pages -> "
              f"{result.chunks_created} chunks, {result.chunks_added} added{note}")
    print(f"\n{total_added} new chunks. Store now holds {pipeline.chunk_count()}.")
    return 0


def cmd_search(args, pipeline, config) -> int:
    results = pipeline.retrieve(
        args.question, top_k=args.k, regulator=args.regulator,
        document_type=args.document_type,
    )
    if not results:
        print("Nothing retrieved. Is the store empty? Try: python -m regintel.cli status")
        return 1
    for r in results:
        c = r.chunk
        title = c.metadata.title or c.doc_id
        print(f"\n[{r.rank + 1}] score={r.score:.3f}  {title}  page {c.page_number}")
        if c.metadata.regulator:
            print(f"     regulator: {c.metadata.regulator}")
        if c.section:
            print(f"     section:   {c.section}")
        print(f"     {c.text[:args.chars].replace(chr(10), ' ')}...")
    return 0


def cmd_ask(args, pipeline, config) -> int:
    if pipeline.generator is None:
        print("No Anthropic client. Set ANTHROPIC_API_KEY in your environment or .env")
        return 1
    answer = pipeline.ask(
        args.question, top_k=args.k, regulator=args.regulator,
        document_type=args.document_type,
    )
    print(answer.text)
    print()
    if answer.insufficient_evidence:
        print("[evidence judged insufficient — no answer asserted]")
    print("Sources:")
    print(format_sources_block(list(answer.citations)))
    return 0


def cmd_library(args, pipeline, config) -> int:
    lib = pipeline.library()
    if not lib:
        print("No documents ingested yet.")
        return 0
    for doc_id, info in sorted(lib.items()):
        print(f"\n{doc_id}")
        print(f"  title:      {info.get('title', '(unknown)')}")
        print(f"  regulator:  {info.get('regulator', '(unknown)')}")
        print(f"  type:       {info.get('document_type', '(unknown)')}")
        print(f"  date:       {info.get('publication_date', '(unknown)')}")
        print(f"  chunks:     {info.get('chunks', 0)}")
        if info.get("source_url"):
            print(f"  source:     {info['source_url']}")
    return 0


def cmd_status(args, pipeline, config) -> int:
    _print_status(pipeline, config)
    return 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="regintel", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("status", help="show the active configuration")
    p.set_defaults(func=cmd_status, needs_llm=True)

    p = sub.add_parser("ingest", help="ingest a PDF or a directory of PDFs")
    p.add_argument("target")
    p.set_defaults(func=cmd_ingest, needs_llm=False)

    p = sub.add_parser("search", help="retrieval only (free)")
    p.add_argument("question")
    p.add_argument("-k", type=int, default=None)
    p.add_argument("--regulator", default=None)
    p.add_argument("--document-type", default=None)
    p.add_argument("--chars", type=int, default=300)
    p.set_defaults(func=cmd_search, needs_llm=False)

    p = sub.add_parser("ask", help="retrieval + Claude generation (billed)")
    p.add_argument("question")
    p.add_argument("-k", type=int, default=None)
    p.add_argument("--regulator", default=None)
    p.add_argument("--document-type", default=None)
    p.set_defaults(func=cmd_ask, needs_llm=True)

    p = sub.add_parser("library", help="list ingested documents")
    p.set_defaults(func=cmd_library, needs_llm=False)

    args = parser.parse_args(argv)
    config = Config.from_env()
    pipeline = build_pipeline(config, with_llm=args.needs_llm)
    return args.func(args, pipeline, config)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
