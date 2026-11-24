"""Tests for grounded generation.

A fake LLM client stands in for the Anthropic SDK, so these run with no API key
and no network. They verify the prompt contract and the post-generation
validation, which is where groundedness is actually enforced.
"""

from regintel.generation import (
    INSUFFICIENT_MARKER,
    Generator,
    build_messages,
    build_system_prompt,
    format_evidence,
)
from regintel.schema import Chunk, DocumentMetadata, RetrievedChunk


class FakeClient:
    """Records the call it received and returns a scripted reply."""

    def __init__(self, reply: str):
        self.reply = reply
        self.calls: list[dict] = []

    def create_message(self, *, model, system, messages, max_tokens, temperature):
        self.calls.append(
            dict(model=model, system=system, messages=messages,
                 max_tokens=max_tokens, temperature=temperature)
        )
        return self.reply


def _retrieved(n: int = 2) -> list[RetrievedChunk]:
    out = []
    for i in range(n):
        chunk = Chunk(
            doc_id=f"doc{i}",
            chunk_index=i,
            page_number=100 + i,
            text=f"A bank shall do thing {i}.",
            metadata=DocumentMetadata(
                title=f"Doc {i}", regulator="FinCEN", document_type="regulation"
            ),
        )
        out.append(RetrievedChunk(chunk=chunk, score=0.9 - i * 0.1, rank=i))
    return out


# ------------------------------------------------------------------- prompting

def test_system_prompt_states_the_grounding_rules():
    sp = build_system_prompt()
    assert INSUFFICIENT_MARKER in sp
    for expected in ["ONLY", "Cite every factual claim", "guidance", "Never invent"]:
        assert expected in sp, f"missing rule text: {expected}"


def test_format_evidence_numbers_passages_and_shows_real_pages():
    text = format_evidence(_retrieved(2))
    assert "[1]" in text and "[2]" in text
    assert "page 100" in text and "page 101" in text
    assert "FinCEN" in text
    assert "A bank shall do thing 0." in text


def test_format_evidence_handles_no_passages():
    assert "no evidence" in format_evidence([]).lower()


def test_build_messages_includes_question_and_evidence():
    msgs = build_messages("What is the SAR deadline?", _retrieved(1))
    assert len(msgs) == 1 and msgs[0]["role"] == "user"
    content = msgs[0]["content"]
    assert "EVIDENCE:" in content
    assert "QUESTION:" in content
    assert "What is the SAR deadline?" in content
    assert INSUFFICIENT_MARKER in content


# ------------------------------------------------------------------ generation

def test_generate_passes_config_to_the_client():
    client = FakeClient("Answer grounded in [1].")
    gen = Generator(client, model="claude-sonnet-5-5", max_tokens=512, temperature=0.0)
    gen.generate("q", _retrieved(1))
    call = client.calls[0]
    assert call["model"] == "claude-sonnet-5-5"
    assert call["max_tokens"] == 512
    assert call["temperature"] == 0.0


def test_generate_returns_only_cited_citations():
    client = FakeClient("Banks must do thing 0 [1]. Nothing about the second.")
    gen = Generator(client, model="m")
    answer = gen.generate("q", _retrieved(2))
    assert [c.marker for c in answer.citations] == ["[1]"]
    assert answer.citations[0].page_number == 100
    assert answer.insufficient_evidence is False


def test_generate_strips_and_flags_invented_markers():
    client = FakeClient("Grounded claim [1]. Invented claim [7].")
    gen = Generator(client, model="m")
    answer = gen.generate("q", _retrieved(2))
    assert [c.marker for c in answer.citations] == ["[1]"]
    assert "[7]" in answer.text  # the note names the bad marker
    assert "have been removed" in answer.text


def test_insufficient_evidence_is_detected_and_flagged():
    client = FakeClient(f"{INSUFFICIENT_MARKER}\nThe passages do not cover filing deadlines.")
    gen = Generator(client, model="m")
    answer = gen.generate("q", _retrieved(2))
    assert answer.insufficient_evidence is True
    assert INSUFFICIENT_MARKER not in answer.text
    assert "do not cover" in answer.text
    assert answer.citations == ()


def test_no_retrieval_short_circuits_without_calling_the_model():
    client = FakeClient("should never be used")
    gen = Generator(client, model="m")
    answer = gen.generate("q", [])
    assert client.calls == [], "the model must not be called with zero evidence"
    assert answer.insufficient_evidence is True
    assert answer.citations == ()
    assert answer.retrieved == ()


def test_answer_retains_retrieved_passages_for_display():
    client = FakeClient("Answer [1][2].")
    gen = Generator(client, model="m")
    answer = gen.generate("q", _retrieved(2))
    assert len(answer.retrieved) == 2
    assert answer.retrieved[0].chunk.page_number == 100


def test_answer_to_dict_is_serializable():
    import json

    client = FakeClient("Answer [1].")
    gen = Generator(client, model="m")
    answer = gen.generate("q", _retrieved(2))
    payload = json.dumps(answer.to_dict())
    assert "citations" in payload and "retrieved" in payload


# ------------------------------------------------------------- document vintage

def _retrieved_with_note(note="Snapshot dated July 2023; may be superseded."):
    chunk = Chunk(
        doc_id="ffiec-x", chunk_index=0, page_number=4, text="A bank shall do X.",
        metadata=DocumentMetadata(
            title="FFIEC Manual — X", regulator="FFIEC",
            publication_date="July 2023", version_note=note,
        ),
    )
    return [RetrievedChunk(chunk=chunk, score=0.9, rank=0)]


def test_prompt_requires_disclosing_a_vintage_warning():
    sp = build_system_prompt()
    assert "VINTAGE WARNING" in sp
    assert "current version" in sp


def test_evidence_block_shows_date_and_vintage_warning_to_the_model():
    text = format_evidence(_retrieved_with_note())
    assert "dated: July 2023" in text
    assert "VINTAGE WARNING" in text
    assert "may be superseded" in text


def test_evidence_block_omits_warning_for_current_documents():
    text = format_evidence(_retrieved_with_note(None))
    assert "VINTAGE WARNING" not in text
