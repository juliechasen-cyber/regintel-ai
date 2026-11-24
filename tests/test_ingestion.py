"""Tests for PDF text cleaning and running-header removal."""

from regintel.ingestion import (
    _header_candidate,
    _header_signature,
    despace_letters,
    clean_text,
    default_doc_id,
    document_from_pages,
    strip_repeated_lines,
)
from regintel.schema import Page


# ------------------------------------------------------------------- cleaning

def test_clean_text_dehyphenates_across_lines():
    assert clean_text("finan-\ncial insti-\ntution") == "financial institution"


def test_clean_text_normalizes_newlines_and_spaces():
    assert clean_text("a\r\nb\r\n\r\n\r\n\r\nc  \t d") == "a\nb\n\nc d"


def test_clean_text_strips_leading_and_trailing_space():
    assert clean_text("   hello   ") == "hello"


def test_clean_text_handles_empty():
    assert clean_text("") == ""
    assert clean_text("\n\n\n") == ""


def test_default_doc_id():
    assert default_doc_id("/a/b/FFIEC BSA_AML Manual.pdf") == "ffiec-bsa_aml-manual"
    assert default_doc_id("x.pdf") == "x"


# ------------------------------------------------------- running header removal

def test_header_signature_ignores_page_numbers():
    assert _header_signature("FINCEN GUIDANCE 7") == _header_signature("FINCEN GUIDANCE 8")
    assert _header_signature("- 12 -") == _header_signature("- 13 -")


def _page_with_banner(i: int, body: str) -> Page:
    """A realistic page: banner, prose body, page-number footer."""
    return Page(i, f"FINCEN GUIDANCE {i}\n{body}\n- {i} -")


def test_repeated_header_and_footer_are_removed():
    pages = [
        _page_with_banner(
            i,
            f"A covered financial institution must perform obligation number {i} "
            f"as described in the applicable provisions of this guidance document.",
        )
        for i in range(1, 9)
    ]
    cleaned = strip_repeated_lines(pages)
    for page in cleaned:
        assert "GUIDANCE" not in page.text, "running header should be gone"
        assert "- " not in page.text.split("\n")[-1] or page.text.strip() != ""
        assert f"obligation number {page.page_number}" in page.text, "body must survive"


def test_body_prose_is_never_removed_even_when_identical_across_pages():
    # The same long sentence on every page is body text, not a banner: it exceeds
    # the length guard, so frequency alone must not strip it.
    sentence = ("A bank shall establish and maintain written procedures that are "
                "reasonably designed to identify and verify beneficial owners.")
    pages = [_page_with_banner(i, sentence) for i in range(1, 9)]
    cleaned = strip_repeated_lines(pages)
    for page in cleaned:
        assert sentence in page.text


def test_mid_page_lines_are_protected_by_position():
    # A short line repeated on every page but in the MIDDLE of the page must
    # survive, because only page edges can be headers.
    pages = [
        Page(i, "\n".join([
            f"BANNER {i}",
            "first body line with enough words to be prose content here",
            "Shared Short Line",
            "second body line with enough words to be prose content here",
            f"- {i} -",
        ]))
        for i in range(1, 9)
    ]
    cleaned = strip_repeated_lines(pages)
    for page in cleaned:
        assert "Shared Short Line" in page.text
        assert "BANNER" not in page.text


def test_infrequent_short_line_survives():
    # A short edge line on only 2 of 8 pages is below the frequency threshold.
    pages = []
    for i in range(1, 9):
        first = "Rare Note Here" if i in (1, 2) else f"BANNER {i}"
        pages.append(Page(i, f"{first}\nbody prose line with plenty of words {i}"))
    cleaned = strip_repeated_lines(pages)
    kept = [p for p in cleaned if "Rare Note Here" in p.text]
    assert len(kept) == 2


def test_short_documents_are_left_alone():
    # With only 3 pages, repetition is not evidence of a running header.
    pages = [Page(i, f"SAME LINE\nbody {i}") for i in range(1, 4)]
    cleaned = strip_repeated_lines(pages)
    for page in cleaned:
        assert "SAME LINE" in page.text


def test_page_numbers_survive_header_stripping():
    pages = [Page(i, f"BANNER {i}\nbody {i}") for i in range(1, 9)]
    cleaned = strip_repeated_lines(pages)
    assert [p.page_number for p in cleaned] == list(range(1, 9))


def test_stripping_does_not_crash_on_empty_pages():
    pages = [Page(i, "") for i in range(1, 9)]
    cleaned = strip_repeated_lines(pages)
    assert all(p.text == "" for p in cleaned)


def test_document_from_pages_cleans_and_preserves_numbers():
    doc = document_from_pages("d", [(3, "insti-\ntution  text"), (9, "more")])
    assert doc.pages[0].page_number == 3
    assert doc.pages[0].text == "institution text"
    assert doc.pages[1].page_number == 9


def test_full_text_joins_pages():
    doc = document_from_pages("d", [(1, "alpha"), (2, "beta")])
    assert "alpha" in doc.full_text and "beta" in doc.full_text


# --------------------------------------------------------------- letter spacing

def test_despace_letters_collapses_tracked_banners():
    assert despace_letters("F I N C E N G U I D A N C E 7") == "FINCENGUIDANCE 7"
    # Non-alphabetic tokens break a run, so each side collapses separately.
    assert despace_letters("B S A / A M L") == "BSA / AML"


def test_despace_letters_leaves_normal_prose_alone():
    text = "A bank must verify I am a customer of record"
    assert despace_letters(text) == text


def test_despace_letters_ignores_short_runs():
    # Two single letters in a row is below min_run and must not merge.
    assert despace_letters("a b cat") == "a b cat"


def test_letter_spaced_banner_is_treated_as_a_header_candidate():
    assert _header_candidate("F I N C E N G U I D A N C E 7")


def test_letter_spaced_running_header_is_removed():
    pages = [
        Page(i, "\n".join([
            f"F I N C E N G U I D A N C E {i}",
            f"Body prose for page {i} with enough words to count as content here.",
        ]))
        for i in range(1, 9)
    ]
    cleaned = strip_repeated_lines(pages)
    for page in cleaned:
        assert "F I N C E N" not in page.text
        assert f"Body prose for page {page.page_number}" in page.text


# ------------------------------------------------- edge window sizing (regression)

def test_edge_window_always_reserves_a_middle_band():
    # Regression: the footer window was widened for dense printed pages (GPO CFR
    # footers run ~9 lines). On short pages that swallowed everything, so
    # mid-page text lost its position protection.
    from regintel.ingestion import _edge_indices
    for n in range(3, 40):
        edges = _edge_indices(n)
        assert len(edges) < n, f"page of {n} lines left no protected middle"
    # Dense page: the footer window must still be wide enough for GPO junk.
    dense = _edge_indices(48)
    assert 47 in dense and 40 in dense, "footer window too narrow for printer blocks"
    assert 20 not in dense, "mid-page line must not be an edge candidate"


def test_gpo_style_footer_block_is_removed():
    # Nine-line typesetting footer, as GPO's CFR PDFs emit.
    body = [f"regulatory body text line {i} with sufficient words to be prose"
            for i in range(14)]
    pages = []
    for p in range(1, 9):
        pages.append(Page(p, "\n".join(
            [f"{1190 + p}", "Financial Crimes Enforc. Network, Treas."]
            + body
            + ["VerDate Sep<11>2014", "Jkt 265133", "PO 00000",
               f"Frm 0120{p}", "Fmt 8010", "Sfmt 8010",
               "Y:\\SGML\\265133.XXX", "265133",
               "skersey on DSK4WB1RN3PROD with CFR"]
        )))
    cleaned = strip_repeated_lines(pages)
    for page in cleaned:
        for junk in ("Sfmt", "Fmt 8010", "PO 00000", "SGML", "DSK4WB1RN3PROD", "Jkt"):
            assert junk not in page.text, f"{junk!r} survived"
        assert "regulatory body text line 7" in page.text, "body text was destroyed"
