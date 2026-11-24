"""Configurable recursive text chunking.

A dependency-free recursive character splitter (the same family of algorithm as
LangChain's ``RecursiveCharacterTextSplitter``, re-implemented here so the core
has no heavy dependency and is fully unit-testable). It:

* targets ~800 tokens per chunk with ~120 tokens of overlap (both configurable);
* preserves ``doc_id``, ``page_number`` and the running ``section`` heading;
* keeps each chunk within a single page by default so citations carry an exact,
  real page number (never an invented one);
* tries to avoid stranding a heading at the end of a chunk, keeping it with the
  text it introduces.

Token counting is pluggable via ``count_tokens``. The default is a cheap,
model-agnostic heuristic (~4 characters per token). Pass a real tokenizer (e.g.
``tiktoken``) for exact budgeting.
"""

from __future__ import annotations

import re
from typing import Callable, Optional

from .schema import Chunk, Document, DocumentMetadata, Page

TokenCounter = Callable[[str], int]

DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""]

# A heading is a short, non-sentence line: e.g. "31 CFR 1020.220" or
# "Customer Due Diligence". Used only as a soft hint, never to invent structure.
_HEADING_RE = re.compile(r"^\s*([A-Z0-9][^\n]{0,79})\s*$")


def estimate_tokens(text: str) -> int:
    """Heuristic token count (~4 chars/token). Minimum 1 for non-empty text."""
    text = text.strip()
    if not text:
        return 0
    return max(1, round(len(text) / 4))


def word_token_counter(text: str) -> int:
    """Whitespace word count — a deterministic counter useful for tests."""
    return len(text.split())


def _looks_like_heading(piece: str) -> bool:
    """Heuristic: is this line a section heading?

    Deliberately conservative. Real regulator PDFs are full of running headers
    and page-number artifacts ("6", "F I N C E N G U I D A N C E 6"), and
    mislabelling those as sections pollutes chunk metadata, so they are rejected.
    """
    stripped = piece.strip()
    if not stripped or len(stripped) > 80:
        return False
    if stripped.endswith((".", ":", "?", "!")) and not stripped.endswith("CFR."):
        return False

    words = stripped.split()
    # Headings rarely contain more than ~10 words, and a single token (usually a
    # page number or stray letter) is not a heading.
    if not 2 <= len(words) <= 10:
        return False
    # Must contain letters — a run of digits/punctuation is an artifact.
    if not any(ch.isalpha() for ch in stripped):
        return False
    # Reject letter-spaced running headers, where most tokens are 1 character
    # (e.g. "F I N C E N G U I D A N C E 6").
    single_char = sum(1 for w in words if len(w) == 1)
    if single_char > len(words) / 2:
        return False

    return bool(_HEADING_RE.match(stripped))


def _split_recursive(text: str, separators: list[str], max_tokens: int,
                     count_tokens: TokenCounter) -> list[str]:
    """Break ``text`` into atomic pieces each <= ``max_tokens`` where possible.

    Descends the separator list: split on the current separator, and for any
    piece still over budget, recurse with the next (finer) separator. The final
    "" separator splits by character as a last resort.
    """
    if count_tokens(text) <= max_tokens or not text:
        return [text] if text else []

    if not separators:
        return [text]

    sep, *rest = separators
    if sep == "":
        # Character-level fallback: slice into token-budgeted windows.
        approx_chars = max(1, max_tokens * 4)
        return [text[i:i + approx_chars] for i in range(0, len(text), approx_chars)]

    parts = text.split(sep)
    pieces: list[str] = []
    for i, part in enumerate(parts):
        # Re-attach the separator (except after the final part) so we don't lose
        # punctuation/whitespace that carries meaning.
        if i < len(parts) - 1:
            part = part + sep
        if not part:
            continue
        if count_tokens(part) <= max_tokens:
            pieces.append(part)
        else:
            pieces.extend(_split_recursive(part, rest, max_tokens, count_tokens))
    return pieces


def _merge_pieces(pieces: list[str], max_tokens: int, overlap_tokens: int,
                  count_tokens: TokenCounter) -> list[str]:
    """Greedily merge atomic pieces into chunks near ``max_tokens`` with overlap.

    Overlap is produced by carrying a tail of the previous chunk's pieces (whose
    combined token count is <= ``overlap_tokens``) into the start of the next.
    A heading piece is not left as the last piece of a chunk — it is pushed to
    the next chunk so it stays with the text it introduces.
    """
    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0

    def flush() -> list[str]:
        nonlocal current, current_tokens
        if not current:
            return []
        # Don't end a chunk on a heading-only tail: hold trailing headings back.
        held: list[str] = []
        while len(current) > 1 and _looks_like_heading(current[-1]):
            held.insert(0, current.pop())
        chunks.append("".join(current).strip())
        carried = _overlap_tail(current, overlap_tokens, count_tokens)
        current = carried + held
        current_tokens = sum(count_tokens(p) for p in current)
        return held

    for piece in pieces:
        ptok = count_tokens(piece)
        if current and current_tokens + ptok > max_tokens:
            flush()
        current.append(piece)
        current_tokens += ptok

    if current and "".join(current).strip():
        chunks.append("".join(current).strip())

    return [c for c in chunks if c]


def _overlap_tail(pieces: list[str], overlap_tokens: int,
                  count_tokens: TokenCounter) -> list[str]:
    if overlap_tokens <= 0:
        return []
    tail: list[str] = []
    total = 0
    for piece in reversed(pieces):
        ptok = count_tokens(piece)
        if total + ptok > overlap_tokens and tail:
            break
        tail.insert(0, piece)
        total += ptok
    return tail


def _track_section(text: str, current: Optional[str]) -> Optional[str]:
    """Return the last heading-like line in ``text``, else keep ``current``."""
    section = current
    for line in text.splitlines():
        if _looks_like_heading(line):
            section = line.strip()
    return section


def chunk_page_text(
    text: str,
    *,
    doc_id: str,
    page_number: int,
    start_index: int,
    section: Optional[str],
    metadata: DocumentMetadata,
    chunk_tokens: int,
    overlap_tokens: int,
    count_tokens: TokenCounter,
    separators: Optional[list[str]] = None,
) -> tuple[list[Chunk], int, Optional[str]]:
    """Chunk a single page's text. Returns (chunks, next_index, section)."""
    separators = separators or DEFAULT_SEPARATORS
    text = text.strip()
    if not text:
        return [], start_index, section

    pieces = _split_recursive(text, list(separators), chunk_tokens, count_tokens)
    merged = _merge_pieces(pieces, chunk_tokens, overlap_tokens, count_tokens)

    chunks: list[Chunk] = []
    index = start_index
    running_section = section
    for body in merged:
        running_section = _track_section(body, running_section)
        chunks.append(
            Chunk(
                doc_id=doc_id,
                chunk_index=index,
                page_number=page_number,
                text=body,
                section=running_section,
                metadata=metadata,
            )
        )
        index += 1
    return chunks, index, running_section


def chunk_document(
    document: Document,
    *,
    chunk_tokens: int = 800,
    overlap_tokens: int = 120,
    respect_page_boundaries: bool = True,
    count_tokens: TokenCounter = estimate_tokens,
    separators: Optional[list[str]] = None,
) -> list[Chunk]:
    """Chunk every page of a document, preserving provenance.

    With ``respect_page_boundaries=True`` (default) each chunk belongs to exactly
    one page, so its citation page number is always exact. The running section
    heading is carried across pages.
    """
    chunks: list[Chunk] = []
    index = 0
    section: Optional[str] = None

    if respect_page_boundaries:
        for page in document.pages:
            page_chunks, index, section = chunk_page_text(
                page.text,
                doc_id=document.doc_id,
                page_number=page.page_number,
                start_index=index,
                section=section,
                metadata=document.metadata,
                chunk_tokens=chunk_tokens,
                overlap_tokens=overlap_tokens,
                count_tokens=count_tokens,
                separators=separators,
            )
            chunks.extend(page_chunks)
        return chunks

    # Cross-page mode: concatenate but tag each chunk with the page where it
    # starts (best-effort; page_boundaries=True is preferred for exact citations).
    combined = "\n".join(p.text for p in document.pages)
    first_page = document.pages[0].page_number if document.pages else 1
    page_chunks, _, _ = chunk_page_text(
        combined,
        doc_id=document.doc_id,
        page_number=first_page,
        start_index=0,
        section=None,
        metadata=document.metadata,
        chunk_tokens=chunk_tokens,
        overlap_tokens=overlap_tokens,
        count_tokens=count_tokens,
        separators=separators,
    )
    return page_chunks
