"""RegIntel AI — Streamlit regulatory research assistant.

Run with:
    streamlit run app/streamlit_app.py

Three tabs:
* **Research** — ask a question, read the grounded answer, inspect every
  retrieved passage and its citation.
* **Document library** — what has been ingested, by regulator and type.
* **Upload** — add a PDF, with verified metadata supplied by the uploader.

Design note: the UI never shows a claim without showing the evidence behind it.
Retrieved passages are always available, and the system state (which embedder,
which store, whether an API key is present) is shown in the sidebar rather than
hidden, so a reviewer can tell exactly what produced an answer.
"""

from __future__ import annotations

import os
import sys
import tempfile

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from regintel.citations import format_sources_block  # noqa: E402
from regintel.config import Config  # noqa: E402
from regintel.pipeline import build_pipeline  # noqa: E402
from regintel.schema import DOCUMENT_TYPES, DocumentMetadata  # noqa: E402

st.set_page_config(page_title="RegIntel AI", page_icon="§", layout="wide")


@st.cache_resource(show_spinner="Loading embedding model and vector store...")
def get_pipeline():
    config = Config.from_env()
    return build_pipeline(config, with_llm=True), config


pipeline, config = get_pipeline()


# ----------------------------------------------------------------------- sidebar

with st.sidebar:
    st.title("RegIntel AI")
    st.caption("Evidence-backed research over U.S. banking regulatory documents.")

    st.subheader("System")
    store_kind = type(pipeline.store).__name__
    st.text(f"chunks indexed: {pipeline.chunk_count()}")
    st.text(f"embedder: {pipeline.embedder.name[:38]}")
    st.text(f"store: {store_kind}")
    st.text(f"model: {config.anthropic_model}")

    if store_kind == "InMemoryVectorStore":
        st.warning("ChromaDB unavailable — index is in memory and will be lost "
                   "when this process stops. `pip install chromadb`", icon=":material/warning:")
    if "hashing" in pipeline.embedder.name:
        st.warning("Using the offline hashing embedder, not a semantic model. "
                   "Retrieval quality will be poor. "
                   "`pip install sentence-transformers`", icon=":material/warning:")
    if pipeline.generator is None:
        st.error("No ANTHROPIC_API_KEY — retrieval works, but answers cannot be "
                 "generated.", icon=":material/warning:")

    st.subheader("Retrieval")
    top_k = st.slider("Passages to retrieve (k)", 1, 20, config.top_k)

    library = pipeline.library()
    regulators = sorted({v.get("regulator") for v in library.values() if v.get("regulator")})
    doc_types = sorted({v.get("document_type") for v in library.values() if v.get("document_type")})

    regulator = st.selectbox("Regulator", ["(any)"] + regulators)
    doc_type = st.selectbox("Document type", ["(any)"] + doc_types)
    regulator = None if regulator == "(any)" else regulator
    doc_type = None if doc_type == "(any)" else doc_type

    st.divider()
    st.caption("Research tool, not legal advice. Always read the cited source "
               "before relying on an answer.")


tab_research, tab_library, tab_upload = st.tabs(
    ["Research", "Document library", "Upload"]
)


# ---------------------------------------------------------------------- research

with tab_research:
    st.header("Ask a regulatory question")

    question = st.text_area(
        "Question",
        placeholder="e.g. Who is treated as a beneficial owner when a trust owns "
                    "25% or more of a legal entity customer?",
        height=90,
    )
    col_ask, col_search = st.columns([1, 1])
    do_ask = col_ask.button("Answer with citations", type="primary",
                            width="stretch",
                            disabled=pipeline.generator is None)
    do_search = col_search.button("Retrieve passages only (free)",
                                  width="stretch")

    # Bound before the branch chain: the render guard below reads it on any path
    # where do_ask is True, including the early-exit branches.
    answer = None

    if (do_ask or do_search) and not question.strip():
        st.warning("Enter a question first.")

    elif pipeline.chunk_count() == 0:
        if do_ask or do_search:
            st.info("No documents indexed yet. Add PDFs on the Upload tab, or run "
                    "`python -m regintel.cli ingest data/pdfs`.")

    elif do_search:
        results = pipeline.retrieve(question, top_k=top_k, regulator=regulator,
                                    document_type=doc_type)
        st.subheader(f"{len(results)} passage(s)")
        if not results:
            st.info("Nothing matched those filters.")
        for r in results:
            c = r.chunk
            title = c.metadata.title or c.doc_id
            with st.expander(
                f"[{r.rank + 1}] {title} — page {c.page_number} "
                f"(similarity {r.score:.3f})",
                expanded=r.rank == 0,
            ):
                meta_line = " · ".join(
                    x for x in [c.metadata.regulator, c.metadata.document_type,
                                c.section] if x
                )
                if meta_line:
                    st.caption(meta_line)
                if c.metadata.version_note:
                    st.warning(c.metadata.version_note, icon=":material/history:")
                st.write(c.text)
                if c.metadata.source_url:
                    st.markdown(f"[Open source document]({c.metadata.source_url})")

    elif do_ask:
        try:
            with st.spinner("Retrieving evidence and generating a grounded answer..."):
                answer = pipeline.ask(question, top_k=top_k, regulator=regulator,
                                      document_type=doc_type)
        except Exception as exc:
            # An API key that is missing, invalid or out of quota must read as a
            # clear message, not a Streamlit traceback. Retrieval is unaffected.
            st.error(
                f"The answer could not be generated: {type(exc).__name__}: {exc}\n\n"
                "Check that `ANTHROPIC_API_KEY` in your `.env` is a valid key. "
                "Retrieval does not need one — use **Retrieve passages only**.",
                icon=":material/error:",
            )

    if do_ask and answer is not None:

        if answer.insufficient_evidence:
            st.warning("The retrieved evidence does not support an answer. "
                       "RegIntel declines rather than guessing.", icon=":material/warning:")

        st.subheader("Answer")
        st.markdown(answer.text)

        st.subheader("Sources")
        if answer.citations:
            for c in answer.citations:
                line = f"**{c.marker}** {c.human_label()}"
                if c.document_type:
                    line += f" — _{c.document_type}_"
                st.markdown(line)
                if c.version_note:
                    st.warning(c.version_note, icon=":material/history:")
                if c.source_url:
                    st.caption(c.source_url)
        else:
            st.caption("No sources cited.")

        st.subheader(f"Retrieved evidence ({len(answer.retrieved)} passages)")
        st.caption("Everything the model was given. A claim not supported here "
                   "is not supported at all.")
        for r in answer.retrieved:
            c = r.chunk
            title = c.metadata.title or c.doc_id
            with st.expander(
                f"[{r.rank + 1}] {title} — page {c.page_number} "
                f"(similarity {r.score:.3f})"
            ):
                meta_line = " · ".join(
                    x for x in [c.metadata.regulator, c.metadata.document_type,
                                c.section] if x
                )
                if meta_line:
                    st.caption(meta_line)
                st.write(c.text)

        with st.expander("Raw result (JSON)"):
            st.json(answer.to_dict())


# ----------------------------------------------------------------------- library

with tab_library:
    st.header("Document library")
    library = pipeline.library()
    if not library:
        st.info("Nothing ingested yet.")
    else:
        st.caption(f"{len(library)} document(s), {pipeline.chunk_count()} chunks.")
        rows = []
        for doc_id, info in sorted(library.items()):
            rows.append({
                "doc_id": doc_id,
                "title": info.get("title") or "(unknown)",
                "regulator": info.get("regulator") or "(unknown)",
                "type": info.get("document_type") or "(unknown)",
                "date": info.get("publication_date") or "(unknown)",
                "chunks": info.get("chunks", 0),
                "vintage warning": "yes" if info.get("version_note") else "",
                "source": info.get("source_url") or "",
            })
        st.dataframe(rows, width="stretch", hide_index=True, height=560)
        st.caption('"(unknown)" means the field was not supplied — RegIntel '
                   "leaves it blank rather than inferring it from the text.")


# ------------------------------------------------------------------------ upload

with tab_upload:
    st.header("Add a PDF")
    st.caption("Metadata you enter here is treated as verified and is what "
               "citations will display. Leave a field blank if you are not sure — "
               "blank is better than wrong.")

    uploaded = st.file_uploader("PDF", type=["pdf"])

    col1, col2 = st.columns(2)
    title = col1.text_input("Title")
    regulator_in = col2.text_input("Regulator", placeholder="FinCEN, OCC, FFIEC, FRB, FDIC")
    col3, col4 = st.columns(2)
    doc_type_in = col3.selectbox("Document type", ["(leave blank)"] + list(DOCUMENT_TYPES))
    pub_date = col4.text_input("Publication date", placeholder="2026-05-06")
    source_url = st.text_input("Source URL", placeholder="https://www.fincen.gov/...")

    if st.button("Ingest document", type="primary", disabled=uploaded is None):
        metadata = DocumentMetadata(
            title=title.strip() or None,
            regulator=regulator_in.strip() or None,
            document_type=None if doc_type_in == "(leave blank)" else doc_type_in,
            publication_date=pub_date.strip() or None,
            source_url=source_url.strip() or None,
        )
        # Write to a temp file because PyMuPDF reads from a path.
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(uploaded.getbuffer())
            tmp_path = tmp.name
        try:
            with st.spinner("Extracting, chunking and embedding..."):
                from regintel.ingestion import default_doc_id

                result = pipeline.ingest_pdf_file(
                    tmp_path,
                    doc_id=default_doc_id(uploaded.name),
                    metadata_override=metadata,
                )
            if result.chunks_added:
                st.success(
                    f"Ingested **{result.doc_id}** — {result.pages} pages, "
                    f"{result.chunks_created} chunks, {result.chunks_added} new."
                )
            else:
                st.info(
                    f"**{result.doc_id}** is already indexed "
                    f"({result.chunks_skipped_as_duplicate} duplicate chunks "
                    "skipped). Nothing was added."
                )
            st.cache_resource.clear()
        except ImportError:
            st.error("PyMuPDF is not installed. `pip install pymupdf`")
        except Exception as exc:
            st.error(f"Ingestion failed: {exc}")
        finally:
            os.unlink(tmp_path)
