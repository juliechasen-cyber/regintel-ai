"""Citation construction and validation.

The contract (from the spec):
* every citation must correspond to a retrieved chunk;
* a citation carries the document title, regulator and the real page number, plus
  a source URL when available;
* the model must NOT invent section or page references.

We enforce this structurally. Retrieved chunks are numbered ``[1]..[k]`` and the
model is told to cite only those markers. After generation we parse the markers
the model actually used and keep only those that map to a real retrieved chunk —
any invented marker (e.g. ``[9]`` when 6 were retrieved) is dropped and surfaced
as a warning, never rendered as a fact.
"""

from __future__ import annotations

import re
from typing import Iterable

from .schema import Citation, RetrievedChunk

_MARKER_RE = re.compile(r"\[(\d+)\]")


def build_citations(retrieved: Iterable[RetrievedChunk]) -> list[Citation]:
    """Create one candidate citation per retrieved chunk, numbered from [1]."""
    citations: list[Citation] = []
    for i, r in enumerate(retrieved, start=1):
        c = r.chunk
        citations.append(
            Citation(
                marker=f"[{i}]",
                chunk_id=c.chunk_id,
                doc_id=c.doc_id,
                title=c.metadata.title,
                regulator=c.metadata.regulator,
                page_number=c.page_number,
                document_type=c.metadata.document_type,
                source_url=c.metadata.source_url,
            )
        )
    return citations


def markers_in_text(text: str) -> list[int]:
    """Return the 1-based citation numbers referenced in ``text``, in order."""
    seen: list[int] = []
    for m in _MARKER_RE.finditer(text):
        n = int(m.group(1))
        if n not in seen:
            seen.append(n)
    return seen


def used_citations(text: str, candidates: list[Citation]) -> tuple[list[Citation], list[int]]:
    """Filter ``candidates`` to those the answer actually cites.

    Returns ``(valid_citations, invalid_marker_numbers)``. An invalid marker is
    one the model used that does not map to any retrieved chunk — evidence the
    answer referenced something that was not provided, which the caller can flag.
    """
    by_number = {int(c.marker.strip("[]")): c for c in candidates}
    valid: list[Citation] = []
    invalid: list[int] = []
    for n in markers_in_text(text):
        if n in by_number:
            valid.append(by_number[n])
        else:
            invalid.append(n)
    return valid, invalid


def format_sources_block(citations: list[Citation]) -> str:
    """Render a human-readable sources list for display under an answer."""
    if not citations:
        return "No sources cited."
    lines = []
    for c in citations:
        url = f" — {c.source_url}" if c.source_url else ""
        lines.append(f"{c.marker} {c.human_label()}{url}")
    return "\n".join(lines)
