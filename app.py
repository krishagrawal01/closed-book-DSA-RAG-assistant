import io
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st
from pypdf import PdfReader

from utils.pdf_loader import split_pdf_text
from utils.pdf_registry import (
    add_pdf_entry,
    delete_pdf_file,
    ensure_data_dirs,
    get_pdf_entry,
    load_registry,
    make_collection_name,
    remove_pdf_entry,
    save_pdf_file,
    update_pdf_entry,
)
from utils.rag_chain import (
    DEFAULT_PERSIST_DIRECTORY,
    LLM_MODEL,
    answer_question_with_sources,
    delete_chroma_collection,
    store_chunks_in_chroma,
    validate_ollama_model,
)

st.set_page_config(
    page_title="Closed-Book DSA RAG Assistant",
    page_icon="📚",
    layout="wide",
)

ensure_data_dirs()

model_ready, model_error = validate_ollama_model(LLM_MODEL)
if not model_ready:
    st.error(model_error)
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "selected_pdf" not in st.session_state:
    st.session_state.selected_pdf = None

if "active_collection_name" not in st.session_state:
    st.session_state.active_collection_name = None

if "active_chunk_count" not in st.session_state:
    st.session_state.active_chunk_count = 0


def extract_pdf_text(source: io.BytesIO | Path | str) -> str:
    if isinstance(source, (Path, str)):
        reader = PdfReader(str(source))
    else:
        reader = PdfReader(source)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages).strip()


def activate_pdf(entry: dict) -> None:
    st.session_state.selected_pdf = entry["filename"]
    st.session_state.active_collection_name = entry["collection_name"]
    st.session_state.active_chunk_count = entry["chunk_count"]


def index_pdf_file(filename: str, pdf_path: Path, *, reindex: bool = False) -> tuple[bool, str]:
    text = extract_pdf_text(pdf_path)
    if not text.strip():
        return False, "The uploaded PDF did not contain extractable text."

    chunks = split_pdf_text(text)
    if not chunks:
        return False, "The uploaded PDF did not produce any text chunks."

    existing_entry = get_pdf_entry(filename)
    collection_name = (
        existing_entry["collection_name"]
        if existing_entry
        else make_collection_name(filename)
    )

    delete_chroma_collection(collection_name, persist_directory=DEFAULT_PERSIST_DIRECTORY)
    store_chunks_in_chroma(
        chunks,
        collection_name=collection_name,
        persist_directory=DEFAULT_PERSIST_DIRECTORY,
        source=filename,
    )

    file_path = str(pdf_path)
    if existing_entry:
        update_pdf_entry(
            filename,
            chunk_count=len(chunks),
            collection_name=collection_name,
            file_path=file_path,
            upload_date=datetime.now(timezone.utc).isoformat(),
        )
        entry = get_pdf_entry(filename)
    else:
        entry = add_pdf_entry(
            filename,
            collection_name=collection_name,
            chunk_count=len(chunks),
            file_path=file_path,
        )

    activate_pdf(entry)
    if reindex:
        return True, f"Reindexed **{filename}** ({len(chunks)} chunks)."
    return True, f"Indexed **{filename}** ({len(chunks)} chunks)."


def handle_upload(uploaded_file) -> None:
    filename = uploaded_file.name
    existing_entry = get_pdf_entry(filename)

    if existing_entry and Path(existing_entry["file_path"]).exists():
        activate_pdf(existing_entry)
        st.session_state.messages = []
        st.info("PDF already indexed.")
        return

    pdf_path = save_pdf_file(filename, uploaded_file.getbuffer())
    success, message = index_pdf_file(filename, pdf_path)
    st.session_state.messages = []

    if success:
        st.success(message)
    else:
        delete_pdf_file(str(pdf_path))
        st.session_state.selected_pdf = None
        st.session_state.active_collection_name = None
        st.session_state.active_chunk_count = 0
        st.warning(message)


def delete_selected_pdf(filename: str) -> None:
    entry = get_pdf_entry(filename)
    if not entry:
        return

    delete_chroma_collection(
        entry["collection_name"],
        persist_directory=DEFAULT_PERSIST_DIRECTORY,
    )
    delete_pdf_file(entry["file_path"])
    remove_pdf_entry(filename)

    st.session_state.messages = []
    st.session_state.selected_pdf = None
    st.session_state.active_collection_name = None
    st.session_state.active_chunk_count = 0


def generate_response(question: str) -> tuple[str, list[str], bool]:
    if not st.session_state.active_collection_name:
        return "Please upload or select an indexed PDF first.", [], False

    allow_outside_knowledge = st.session_state.get("allow_outside_knowledge", False)

    try:
        answer, sources = answer_question_with_sources(
            question,
            collection_name=st.session_state.active_collection_name,
            persist_directory=DEFAULT_PERSIST_DIRECTORY,
            allow_outside_knowledge=allow_outside_knowledge,
        )
        return answer, sources, allow_outside_knowledge
    except ValueError as error:
        return str(error), [], allow_outside_knowledge
    except Exception as error:
        return (
            f"Something went wrong while generating an answer: {error}",
            [],
            allow_outside_knowledge,
        )


def render_mode_badge(allow_outside_knowledge: bool) -> None:
    badge = "🧠 Assisted Mode" if allow_outside_knowledge else "🔒 Closed-Book Mode"
    st.markdown(f"**{badge}**")


def render_assistant_message(
    content: str,
    sources: list[str] | None = None,
    allow_outside_knowledge: bool = False,
) -> None:
    render_mode_badge(allow_outside_knowledge)
    st.markdown(content)
    if sources:
        with st.expander(f"Source chunks ({len(sources)})"):
            for index, chunk in enumerate(sources, start=1):
                st.markdown(f"**Chunk {index}**")
                st.text(chunk)


st.title("Closed-Book DSA RAG Assistant")
st.caption("Upload a PDF and ask questions grounded in your document.")

registry_entries = load_registry()
indexed_filenames = [entry["filename"] for entry in registry_entries]

with st.sidebar:
    st.header("Document")

    st.toggle(
        "Allow Limited Outside Knowledge",
        value=False,
        key="allow_outside_knowledge",
        help="When enabled, the assistant may supplement missing details from model knowledge.",
    )

    if indexed_filenames:
        default_index = 0
        if st.session_state.selected_pdf in indexed_filenames:
            default_index = indexed_filenames.index(st.session_state.selected_pdf)

        selected_filename = st.selectbox(
            "Indexed PDFs",
            indexed_filenames,
            index=default_index,
        )
        selected_entry = get_pdf_entry(selected_filename)
        if selected_entry:
            activate_pdf(selected_entry)
            st.caption(f"{selected_entry['chunk_count']} chunks indexed")
            st.caption(f"Uploaded: {selected_entry['upload_date'][:10]}")

        col_delete, col_reindex = st.columns(2)
        with col_delete:
            if st.button("Delete PDF", use_container_width=True):
                delete_selected_pdf(selected_filename)
                st.rerun()
        with col_reindex:
            if st.button("Reindex PDF", use_container_width=True):
                with st.spinner("Reindexing PDF..."):
                    pdf_path = Path(selected_entry["file_path"])
                    success, message = index_pdf_file(
                        selected_filename,
                        pdf_path,
                        reindex=True,
                    )
                st.session_state.messages = []
                if success:
                    st.success(message)
                else:
                    st.warning(message)
                st.rerun()
    else:
        st.caption("No PDFs indexed yet.")

    uploaded_file = st.file_uploader(
        "Upload PDF",
        type=["pdf"],
        help="Upload a closed-book reference (e.g. DSA notes or textbook chapter).",
    )

    if uploaded_file is not None:
        if st.session_state.get("last_upload_name") != uploaded_file.name:
            with st.spinner("Processing PDF..."):
                handle_upload(uploaded_file)
            st.session_state.last_upload_name = uploaded_file.name
            st.rerun()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            render_assistant_message(
                message["content"],
                message.get("sources"),
                message.get("allow_outside_knowledge", False),
            )
        else:
            st.markdown(message["content"])

if prompt := st.chat_input("Ask a question about your PDF..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching document and generating answer..."):
            answer, sources, allow_outside_knowledge = generate_response(prompt)
        render_assistant_message(answer, sources, allow_outside_knowledge)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": sources,
            "allow_outside_knowledge": allow_outside_knowledge,
        }
    )
