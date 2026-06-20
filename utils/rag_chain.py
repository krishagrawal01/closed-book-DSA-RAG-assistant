from pathlib import Path

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama

DEFAULT_COLLECTION_NAME = "pdf_chunks"
DEFAULT_PERSIST_DIRECTORY = "./chroma_db"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
LLM_MODEL = "gemma3:4b"
TOP_K = 3

RAG_PROMPT = ChatPromptTemplate.from_template(
    """You are a closed-book study assistant for data structures and algorithms.

Answer the question using ONLY the context below. Do not use outside knowledge.
If the context does not contain enough information, reply exactly with:
"I cannot find that information in the uploaded document."

Context:
{context}

Question: {question}

Answer:"""
)


def _get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def _get_llm() -> ChatOllama:
    """Return the local Ollama LLM (Gemma3:4b) for answer generation."""
    return ChatOllama(
        model=LLM_MODEL,
        temperature=0,
    )


def _format_docs(docs) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


def store_chunks_in_chroma(
    chunks: list[str],
    *,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    persist_directory: str = DEFAULT_PERSIST_DIRECTORY,
    source: str | None = None,
) -> Chroma:
    """Embed text chunks locally and store them in ChromaDB."""
    if not chunks:
        raise ValueError("No chunks provided to store.")

    Path(persist_directory).mkdir(parents=True, exist_ok=True)

    embeddings = _get_embeddings()
    metadatas = [
        {"chunk_index": index, **({"source": source} if source else {})}
        for index in range(len(chunks))
    ]

    return Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        metadatas=metadatas,
        collection_name=collection_name,
        persist_directory=persist_directory,
    )


def _load_chroma(
    *,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    persist_directory: str = DEFAULT_PERSIST_DIRECTORY,
) -> Chroma:
    return Chroma(
        collection_name=collection_name,
        embedding_function=_get_embeddings(),
        persist_directory=persist_directory,
    )


def get_retriever(
    k: int = TOP_K,
    *,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    persist_directory: str = DEFAULT_PERSIST_DIRECTORY,
):
    """Return a retriever that fetches the top-k most relevant chunks."""
    vectorstore = _load_chroma(
        collection_name=collection_name,
        persist_directory=persist_directory,
    )
    return vectorstore.as_retriever(search_kwargs={"k": k})


def retrieve_relevant_chunks(
    question: str,
    k: int = TOP_K,
    *,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    persist_directory: str = DEFAULT_PERSIST_DIRECTORY,
) -> list[str]:
    """Return the top-k chunk texts most relevant to a user question."""
    retriever = get_retriever(
        k=k,
        collection_name=collection_name,
        persist_directory=persist_directory,
    )
    documents = retriever.invoke(question)
    return [doc.page_content for doc in documents]


def create_rag_chain(
    k: int = TOP_K,
    *,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    persist_directory: str = DEFAULT_PERSIST_DIRECTORY,
):
    """Build a local RAG chain using Ollama (Gemma3:4b) over retrieved chunks."""
    retriever = get_retriever(
        k=k,
        collection_name=collection_name,
        persist_directory=persist_directory,
    )
    llm = _get_llm()

    return (
        {"context": retriever | _format_docs, "question": RunnablePassthrough()}
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )


def answer_question(
    question: str,
    k: int = TOP_K,
    *,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    persist_directory: str = DEFAULT_PERSIST_DIRECTORY,
) -> str:
    """Answer a user question using only the top-k retrieved document chunks."""
    answer, _ = answer_question_with_sources(
        question,
        k=k,
        collection_name=collection_name,
        persist_directory=persist_directory,
    )
    return answer


def answer_question_with_sources(
    question: str,
    k: int = TOP_K,
    *,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    persist_directory: str = DEFAULT_PERSIST_DIRECTORY,
) -> tuple[str, list[str]]:
    """Answer a question with the local Ollama LLM and return source chunks."""
    retriever = get_retriever(
        k=k,
        collection_name=collection_name,
        persist_directory=persist_directory,
    )
    documents = retriever.invoke(question)
    source_chunks = [doc.page_content for doc in documents]

    if not source_chunks:
        return "I cannot find that information in the uploaded document.", []

    llm = _get_llm()
    chain = RAG_PROMPT | llm | StrOutputParser()
    answer = chain.invoke({"context": _format_docs(documents), "question": question})
    return answer, source_chunks
