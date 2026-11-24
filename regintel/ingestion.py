"""PDF ingestion using PyMuPDF (fitz).

Responsibilities:
* extract text per page, preserving the original 1-based page numbers;
* lightly clean/normalize whitespace and de-hyphenate line-wrapped words;
* read document metadata (title, regulator, date, type) from the PDF's own
  metadata and an optional sidecar JSON — NOT by guessing from the text.

Anti-hallucination rule: any field we cannot read is left ``None``. The caller
(or the UI) may supply verified values explicitly; nothing here invents a
regulator, title or date.
"""

from __future__ import annotations

import json
import os
import re
from typing import Optional

from .schema import Document, DocumentMetadata, Page

_WS_RE = re.compile(r"[ \t]+")
_MULTI_NL_RE = re.compile(r"\n{3,}")
_HYPHEN_WRAP_RE = re.compile(r"(\w)-\n(\w)")


def clean_text(text: str) -> str:
    """Normalize whitespace and join words split across line breaks."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _HYPHEN_WRAP_RE.sub(r"\1\2", text)      # de-hyphenate wrapped words
    text = _WS_RE.sub(" ", text)                    # collapse runs of spaces/tabs
    text = _MULTI_NL_RE.sub("\n\n", text)           # cap blank-line runs
    lines = [ln.strip() for ln in text.split("\n")]
    return "\n".join(lines).strip()


_DIGITS_RE = re.compile(r"\d+")


def despace_letters(text: str, min_run: int = 3) -> str:
    """Collapse runs of letter-spaced single characters into words.

    Regulator PDFs often render a banner with tracking applied, so the extracted
    text is "F I N C E N  G U I D A N C E" rather than "FINCEN GUIDANCE". Without
    this, such a line looks like 14 separate words and escapes the
    running-header heuristics.

    Only runs of ``min_run`` or more single-character tokens are collapsed, so
    ordinary text containing "a" or "I" is untouched.
    """
    tokens = text.split()
    out: list[str] = []
    run: list[str] = []

    def flush() -> None:
        if len(run) >= min_run:
            out.append("".join(run))
        else:
            out.extend(run)
        run.clear()

    for tok in tokens:
        if len(tok) == 1 and tok.isalpha():
            run.append(tok)
        else:
            flush()
            out.append(tok)
    flush()
    return " ".join(out)


def _header_signature(line: str) -> str:
    """Normalize a line so the same running header matches across pages.

    Letter-spacing is collapsed and digits are dropped (page numbers vary), so
    "F I N C E N G U I D A N C E 7" and "... 8" share one signature.
    """
    return _DIGITS_RE.sub("", despace_letters(line)).lower().strip(" -—–\t")


#: How many lines at the top and bottom of a page can be a running header/footer.
_EDGE_LINES = 2
#: A running banner is short. Longer repeated lines are treated as body text.
_MAX_HEADER_CHARS = 60
_MAX_HEADER_WORDS = 8


def _edge_indices(n_lines: int) -> set[int]:
    """Line positions eligible to be a header/footer: the page's top and bottom."""
    if n_lines <= 2 * _EDGE_LINES:
        return set(range(n_lines))
    return set(range(_EDGE_LINES)) | set(range(n_lines - _EDGE_LINES, n_lines))


def _header_candidate(line: str) -> bool:
    """Could this line be a running banner? Judged after de-letter-spacing."""
    stripped = despace_letters(line.strip())
    return (
        bool(stripped)
        and len(stripped) <= _MAX_HEADER_CHARS
        and len(stripped.split()) <= _MAX_HEADER_WORDS
    )


def strip_repeated_lines(
    pages: list[Page],
    min_fraction: float = 0.5,
    min_pages: int = 4,
) -> list[Page]:
    """Remove running headers/footers that repeat across most pages.

    Regulator PDFs put a banner on every page ("FINCEN GUIDANCE 7", "BSA/AML
    Examination Manual"). Left in, that text pollutes every chunk and distorts
    embeddings.

    Three guards keep real content safe:

    1. **Position** — only the first/last ``_EDGE_LINES`` lines of a page can
       qualify. Body text in the middle of a page is never removed.
    2. **Length** — a candidate must be short (a banner, not a sentence).
    3. **Frequency** — its page-number-insensitive signature must appear on at
       least ``min_fraction`` of pages.

    Short documents are skipped entirely, since repetition across a handful of
    pages is not evidence of a running header.
    """
    if len(pages) < min_pages:
        return pages

    counts: dict[str, int] = {}
    for page in pages:
        lines = page.text.splitlines()
        edges = _edge_indices(len(lines))
        # Count each signature once per page, not once per occurrence.
        sigs = {
            _header_signature(ln)
            for i, ln in enumerate(lines)
            if i in edges and _header_candidate(ln)
        }
        for sig in sigs:
            if sig:
                counts[sig] = counts.get(sig, 0) + 1

    threshold = max(2, int(len(pages) * min_fraction))
    boilerplate = {sig for sig, n in counts.items() if n >= threshold}
    if not boilerplate:
        return pages

    cleaned: list[Page] = []
    for page in pages:
        lines = page.text.splitlines()
        edges = _edge_indices(len(lines))
        kept = [
            ln for i, ln in enumerate(lines)
            if not (
                i in edges
                and _header_candidate(ln)
                and _header_signature(ln) in boilerplate
            )
        ]
        cleaned.append(
            Page(page_number=page.page_number,
                 text=_MULTI_NL_RE.sub("\n\n", "\n".join(kept)).strip())
        )
    return cleaned


def default_doc_id(path: str) -> str:
    base = os.path.basename(path)
    stem, _ = os.path.splitext(base)
    return re.sub(r"[^A-Za-z0-9_-]+", "-", stem).strip("-").lower() or "document"


def _load_sidecar(pdf_path: str) -> dict:
    """Load ``<pdf>.meta.json`` next to the PDF, if present. Verified metadata
    the uploader provides lives here — it is trusted, unlike the PDF text."""
    sidecar = pdf_path + ".meta.json"
    if os.path.exists(sidecar):
        try:
            with open(sidecar, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except (OSError, json.JSONDecodeError):
            return {}
    return {}


def _metadata_from_pdf(pdf_meta: dict, sidecar: dict) -> DocumentMetadata:
    def pick(*keys: str) -> Optional[str]:
        for source in (sidecar, pdf_meta):
            for key in keys:
                val = source.get(key)
                if isinstance(val, str) and val.strip():
                    return val.strip()
        return None

    doc_type = pick("document_type")
    return DocumentMetadata(
        title=pick("title", "Title"),
        regulator=pick("regulator"),              # only from sidecar; never guessed
        publication_date=pick("publication_date", "creationDate", "CreationDate"),
        document_type=doc_type,
        source_url=pick("source_url", "url"),
    )


def _import_pymupdf():
    """Import PyMuPDF under whichever name is available.

    PyMuPDF >= 1.24 exposes ``pymupdf``; the legacy ``fitz`` alias still works but
    emits a FutureWarning. Prefer the modern name and fall back for old versions.
    """
    try:
        import pymupdf  # type: ignore

        return pymupdf
    except ImportError:
        import fitz  # type: ignore

        return fitz


def ingest_pdf(
    pdf_path: str,
    *,
    doc_id: Optional[str] = None,
    metadata_override: Optional[DocumentMetadata] = None,
    remove_running_headers: bool = True,
) -> Document:
    """Extract a :class:`Document` from a PDF file.

    ``metadata_override`` wins over anything read from the file — this is how the
    UI passes human-verified regulator/title/type values.
    """
    fitz = _import_pymupdf()

    doc_id = doc_id or default_doc_id(pdf_path)
    sidecar = _load_sidecar(pdf_path)

    pages: list[Page] = []
    with fitz.open(pdf_path) as pdf:
        pdf_meta = dict(pdf.metadata or {})
        for i, page in enumerate(pdf):  # 0-based loop -> 1-based page_number
            raw = page.get_text("text")
            cleaned = clean_text(raw)
            pages.append(Page(page_number=i + 1, text=cleaned))

    if remove_running_headers:
        pages = strip_repeated_lines(pages)

    metadata = metadata_override or _metadata_from_pdf(pdf_meta, sidecar)
    return Document(doc_id=doc_id, metadata=metadata, pages=tuple(pages))


def document_from_pages(
    doc_id: str,
    pages: list[tuple[int, str]],
    metadata: Optional[DocumentMetadata] = None,
) -> Document:
    """Build a Document from already-extracted (page_number, text) pairs.

    Useful for tests and for non-PDF sources, without importing PyMuPDF.
    """
    return Document(
        doc_id=doc_id,
        metadata=metadata or DocumentMetadata(),
        pages=tuple(Page(page_number=n, text=clean_text(t)) for n, t in pages),
    )
