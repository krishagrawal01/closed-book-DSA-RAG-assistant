from pathlib import Path
import logging

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama

DEFAULT_COLLECTION_NAME = "pdf_chunks"
DEFAULT_PERSIST_DIRECTORY = "./data/chroma_db"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
LLM_MODEL = "qwen3:4b"
TOP_K = 3

logger = logging.getLogger(__name__)

RAG_PROMPT = ChatPromptTemplate.from_template(
    """You are a closed-book study assistant for data structures and algorithms.

Answer the question using ONLY the context below. Do not use outside knowledge.
Keep answers concise and student-friendly.

If the context does not contain enough information, reply exactly with:
"I cannot find that information in the uploaded document."

Otherwise, format every answer exactly like this:

### Definition
Short definition (1-2 sentences)

### Explanation
Simple explanation in easy language (2-4 sentences)

### Example
Small example if available (write "Not available in the document." if the context has no example)

### Key Points
- 3-5 bullet points summarizing the most important ideas

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


def validate_ollama_model(model: str = LLM_MODEL) -> tuple[bool, str]:
    """Check whether the configured Ollama model is installed and reachable."""
    try:
        import ollama
    except ImportError:
        return False, (
            f"The ollama package is not installed. "
            f"Install dependencies, then run: ollama pull {model}"
        )

    try:
        response = ollama.list()
        installed = {entry.model for entry in response.models}
        if model in installed or f"{model}:latest" in installed:
            return True, ""

        model_base = model.split(":")[0]
        for name in installed:
            if name == model or name.startswith(f"{model}:") or name.split(":")[0] == model_base:
                return True, ""

        return False, (
            f"Ollama model '{model}' is not installed. "
            f"Run: ollama pull {model}"
        )
    except Exception as error:
        return False, (
            f"Could not connect to Ollama ({error}). "
            f"Make sure Ollama is running, then run: ollama pull {model}"
        )


def _get_llm() -> ChatOllama:
    """Return the local Ollama LLM (Qwen3:4b) for answer generation."""
    return ChatOllama(
        model="qwen3:4b",
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


def delete_chroma_collection(
    collection_name: str,
    *,
    persist_directory: str = DEFAULT_PERSIST_DIRECTORY,
) -> bool:
    """Delete a Chroma collection from the persistent store."""
    import chromadb
    from chromadb.errors import NotFoundError

    Path(persist_directory).mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=persist_directory)
    try:
        client.delete_collection(name=collection_name)
        logger.info("Deleted Chroma collection: %s", collection_name)
        return True
    except NotFoundError:
        logger.warning(
            "Chroma collection not found, skipping deletion: %s",
            collection_name,
        )
        return False
    except ValueError:
        logger.warning(
            "Chroma collection not found, skipping deletion: %s",
            collection_name,
        )
        return False


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
    """Build a local RAG chain using Ollama (Qwen3:4b) over retrieved chunks."""
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
    """Answer a question with the local Ollama LLM (Qwen3:4b) and return source chunks."""
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
