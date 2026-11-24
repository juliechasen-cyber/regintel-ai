"""First-run corpus bootstrap, for hosted deployments.

The repository is code-only: source PDFs and the vector index are gitignored and
rebuilt from ``data/sources.json``. That is fine locally, where you run two
commands, but a hosted demo has no shell — so when ``REGINTEL_BOOTSTRAP=1`` the
app builds its own corpus on first start.

Every document in the manifest is fetched from a host that permits automated
access (govinfo, FDIC, FinCEN), so this needs no credentials and no manual step.
"""

from __future__ import annotations

import os
import subprocess
import sys
from typing import Callable, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PDF_DIR = os.path.join(ROOT, "data", "pdfs")
FETCH = os.path.join(ROOT, "scripts", "fetch_sources.py")

Progress = Optional[Callable[[str], None]]


def bootstrap_enabled() -> bool:
    return os.environ.get("REGINTEL_BOOTSTRAP", "0") == "1"


def local_pdf_count() -> int:
    if not os.path.isdir(PDF_DIR):
        return 0
    return len([f for f in os.listdir(PDF_DIR) if f.lower().endswith(".pdf")])


def fetch_corpus(progress: Progress = None) -> tuple[int, str]:
    """Download the manifest's PDFs. Returns (pdf_count, log tail)."""
    if progress:
        progress("Downloading source documents...")
    proc = subprocess.run(
        [sys.executable, FETCH],
        cwd=ROOT, capture_output=True, text=True, timeout=900,
    )
    tail = "\n".join((proc.stdout or "").strip().splitlines()[-3:])
    return local_pdf_count(), tail


def ensure_corpus(pipeline, progress: Progress = None) -> dict:
    """Make sure the pipeline has an index, building one if it is empty.

    Safe to call on every start: it is a no-op once the store has content, and
    ingestion is idempotent because chunk ids are content-addressed.
    """
    report = {"bootstrapped": False, "pdfs": local_pdf_count(),
              "chunks_before": pipeline.chunk_count(), "chunks_after": None,
              "log": ""}

    if report["chunks_before"] > 0:
        report["chunks_after"] = report["chunks_before"]
        return report

    if report["pdfs"] == 0:
        count, log = fetch_corpus(progress)
        report["pdfs"] = count
        report["log"] = log
        if count == 0:
            report["chunks_after"] = 0
            return report

    if progress:
        progress(f"Indexing {report['pdfs']} documents (embedding runs locally)...")

    for name in sorted(os.listdir(PDF_DIR)):
        if not name.lower().endswith(".pdf"):
            continue
        try:
            pipeline.ingest_pdf_file(os.path.join(PDF_DIR, name))
        except Exception as exc:  # one bad PDF must not break the whole boot
            report["log"] += f"\nfailed {name}: {exc}"

    report["bootstrapped"] = True
    report["chunks_after"] = pipeline.chunk_count()
    return report
