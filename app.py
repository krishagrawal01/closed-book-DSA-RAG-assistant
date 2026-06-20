import io
import shutil

import streamlit as st
from pypdf import PdfReader

from utils.pdf_loader import split_pdf_text
from utils.rag_chain import (
    DEFAULT_PERSIST_DIRECTORY,
    answer_question_with_sources,
    store_chunks_in_chroma,
)

st.set_page_config(
    page_title="Closed-Book DSA RAG Assistant",
    page_icon="📚",
    layout="wide",
)

if "messages" not in st.session_state:
    st.session_state.messages = []

if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None

if "pdf_indexed" not in st.session_state:
    st.session_state.pdf_indexed = False

if "chunk_count" not in st.session_state:
    st.session_state.chunk_count = 0


def extract_pdf_text(uploaded_file: io.BytesIO) -> str:
    reader = PdfReader(uploaded_file)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages).strip()


def index_pdf(uploaded_file) -> tuple[bool, str]:
    text = extract_pdf_text(uploaded_file)
    if not text.strip():
        return False, "The uploaded PDF did not contain extractable text."

    chunks = split_pdf_text(text)
    if not chunks:
        return False, "The uploaded PDF did not produce any text chunks."

    shutil.rmtree(DEFAULT_PERSIST_DIRECTORY, ignore_errors=True)
    store_chunks_in_chroma(chunks, source=uploaded_file.name)

    st.session_state.pdf_indexed = True
    st.session_state.chunk_count = len(chunks)
    return True, text


def generate_response(question: str) -> tuple[str, list[str]]:
    if not st.session_state.pdf_indexed:
        return "Please upload a PDF first so I can answer questions from your document.", []

    try:
        return answer_question_with_sources(question)
    except ValueError as error:
        return str(error), []
    except Exception as error:
        return f"Something went wrong while generating an answer: {error}", []


def render_assistant_message(content: str, sources: list[str] | None = None) -> None:
    st.markdown(content)
    if sources:
        with st.expander(f"Source chunks ({len(sources)})"):
            for index, chunk in enumerate(sources, start=1):
                st.markdown(f"**Chunk {index}**")
                st.text(chunk)


st.title("Closed-Book DSA RAG Assistant")
st.caption("Upload a PDF and ask questions grounded in your document.")

with st.sidebar:
    st.header("Document")
    uploaded_file = st.file_uploader(
        "Upload PDF",
        type=["pdf"],
        help="Upload a closed-book reference (e.g. DSA notes or textbook chapter).",
    )

    if uploaded_file is not None:
        if st.session_state.pdf_name != uploaded_file.name:
            with st.spinner("Indexing PDF..."):
                indexed, result = index_pdf(uploaded_file)
                st.session_state.pdf_name = uploaded_file.name
                st.session_state.messages = []

                if not indexed:
                    st.session_state.pdf_indexed = False
                    st.session_state.chunk_count = 0
                    st.warning(result)

        if st.session_state.pdf_indexed:
            st.success(f"Loaded **{uploaded_file.name}**")
            st.caption(f"{st.session_state.chunk_count} chunks indexed")

    if st.session_state.pdf_name:
        if st.button("Clear document"):
            shutil.rmtree(DEFAULT_PERSIST_DIRECTORY, ignore_errors=True)
            st.session_state.pdf_name = None
            st.session_state.pdf_indexed = False
            st.session_state.chunk_count = 0
            st.session_state.messages = []
            st.rerun()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            render_assistant_message(message["content"], message.get("sources"))
        else:
            st.markdown(message["content"])

if prompt := st.chat_input("Ask a question about your PDF..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching document and generating answer..."):
            answer, sources = generate_response(prompt)
        render_assistant_message(answer, sources)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )
