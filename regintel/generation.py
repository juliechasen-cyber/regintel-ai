"""Grounded answer generation with Claude (official Anthropic Python SDK).

The generator:
* builds a numbered evidence block from the retrieved chunks;
* instructs Claude to answer ONLY from that evidence, to cite ``[n]`` markers,
  to distinguish regulation/statute from supervisory guidance, and to say so
  explicitly when the evidence is insufficient rather than inventing an answer;
* validates the citations the model used against the retrieved chunks.

The Anthropic client is injected, so tests run with a fake client and no API
key. ``build_messages`` and ``build_system_prompt`` are pure functions and are
unit-tested directly.
"""

from __future__ import annotations

import inspect
from typing import Any, Optional, Protocol

from .citations import build_citations, used_citations
from .schema import Answer, Citation, RetrievedChunk

INSUFFICIENT_MARKER = "INSUFFICIENT_EVIDENCE"

SYSTEM_PROMPT = f"""You are RegIntel AI, a research assistant for U.S. banking \
compliance professionals (BSA/AML, supervisory guidance, and related regulation).

Follow these rules without exception:

1. Answer ONLY using the numbered EVIDENCE passages provided in the user message. \
Do not use prior knowledge to add facts, figures, citations, dates, or section \
numbers that are not present in the evidence.
2. Cite every factual claim with the bracketed passage number(s) it comes from, \
e.g. "...within 30 days [2]." Use only the numbers shown in the evidence. Never \
invent a passage number, a page number, or a section/CFR reference.
3. Distinguish binding authority from guidance. When the evidence is a regulation \
or statute, say so; when it is supervisory guidance (examiner manuals, bulletins, \
FAQs), label it as guidance that interprets but does not itself create law.
4. If the evidence does not contain enough to answer, respond with the single \
token {INSUFFICIENT_MARKER} on its own line, followed by a one-sentence \
explanation of what is missing. Do not guess.
5. If a passage carries a VINTAGE WARNING, say so in your answer and tell the \
reader to verify against the current version. Never present a dated snapshot as \
the current state of the guidance.
6. Be precise and concise. Quote short phrases from the evidence when wording \
matters. Do not provide legal advice; provide sourced research.
"""


class LLMClient(Protocol):
    """Subset of the Anthropic client surface the generator depends on."""

    def create_message(self, *, model: str, system: str, messages: list[dict],
                       max_tokens: int, temperature: float) -> str:
        ...


class AnthropicClient:
    """Adapter over the official ``anthropic`` SDK. Imported lazily."""

    def __init__(self, api_key: Optional[str] = None):
        import anthropic  # lazy import

        self._client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()

    @staticmethod
    def _supports_temperature(create) -> bool:
        """Does this SDK's messages.create accept a ``temperature`` argument?

        The Anthropic SDK dropped ``temperature`` (it is absent in 1.11.0, where
        ``output_config`` carries ``effort``/``format`` instead). Passing it to a
        newer SDK raises TypeError, and omitting it on an older one silently
        loses determinism — so detect rather than assume.
        """
        try:
            return "temperature" in inspect.signature(create).parameters
        except (TypeError, ValueError):  # pragma: no cover - exotic SDK stubs
            return False

    def create_message(self, *, model: str, system: str, messages: list[dict],
                       max_tokens: int, temperature: float) -> str:
        create = self._client.messages.create
        kwargs: dict[str, Any] = dict(
            model=model,
            system=system,
            messages=messages,
            max_tokens=max_tokens,
        )
        if self._supports_temperature(create):
            kwargs["temperature"] = temperature

        resp = create(**kwargs)
        parts = [b.text for b in resp.content if getattr(b, "type", None) == "text"]
        return "".join(parts).strip()


def build_system_prompt() -> str:
    return SYSTEM_PROMPT


def format_evidence(retrieved: list[RetrievedChunk]) -> str:
    """Render retrieved chunks as a numbered evidence block for the prompt."""
    if not retrieved:
        return "(no evidence passages were retrieved)"
    blocks = []
    for i, r in enumerate(retrieved, start=1):
        c = r.chunk
        header_bits = []
        if c.metadata.title:
            header_bits.append(c.metadata.title)
        if c.metadata.regulator:
            header_bits.append(c.metadata.regulator)
        if c.metadata.document_type:
            header_bits.append(f"type: {c.metadata.document_type}")
        if c.metadata.publication_date:
            header_bits.append(f"dated: {c.metadata.publication_date}")
        header_bits.append(f"page {c.page_number}")
        header = " | ".join(header_bits)
        note = f"\n(VINTAGE WARNING: {c.metadata.version_note})" if c.metadata.version_note else ""
        blocks.append(f"[{i}] ({header}){note}\n{c.text}")
    return "\n\n".join(blocks)


def build_messages(question: str, retrieved: list[RetrievedChunk]) -> list[dict]:
    evidence = format_evidence(retrieved)
    content = (
        f"EVIDENCE:\n{evidence}\n\n"
        f"QUESTION:\n{question}\n\n"
        f"Answer using only the evidence above, with [n] citations. "
        f"If the evidence is insufficient, reply with {INSUFFICIENT_MARKER}."
    )
    return [{"role": "user", "content": content}]


def _parse_insufficient(text: str) -> tuple[bool, str]:
    if INSUFFICIENT_MARKER in text:
        cleaned = text.replace(INSUFFICIENT_MARKER, "").strip()
        return True, cleaned or "The retrieved evidence does not cover this question."
    return False, text


class Generator:
    def __init__(self, client: LLMClient, model: str, max_tokens: int = 1024,
                 temperature: float = 0.0):
        self.client = client
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

    def generate(self, question: str, retrieved: list[RetrievedChunk]) -> Answer:
        # Guard: with no evidence we never call the model with an empty context
        # in a way that could invite ungrounded output.
        candidates = build_citations(retrieved)

        if not retrieved:
            return Answer(
                question=question,
                text="The document library returned no relevant passages for this "
                     "question, so there is no evidence to answer from.",
                citations=(),
                retrieved=(),
                insufficient_evidence=True,
                model=self.model,
            )

        raw = self.client.create_message(
            model=self.model,
            system=build_system_prompt(),
            messages=build_messages(question, retrieved),
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        insufficient, text = _parse_insufficient(raw)

        valid, invalid = used_citations(text, candidates)
        if invalid:
            note = (
                "\n\n> Note: the model referenced passage number(s) "
                f"{', '.join(f'[{n}]' for n in invalid)} that were not in the "
                "retrieved evidence; those references have been removed."
            )
            text = text + note

        return Answer(
            question=question,
            text=text,
            citations=tuple(valid),
            retrieved=tuple(retrieved),
            insufficient_evidence=insufficient,
            model=self.model,
        )
