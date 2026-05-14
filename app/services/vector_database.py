import html
import logging
import os
import re
from pathlib import Path
from threading import Lock

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredImageLoader,
    UnstructuredWordDocumentLoader,
)
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import (
    HTMLHeaderTextSplitter,
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from app.database import engine
from app.settings import settings
from external_models import get_embeddings


def html_to_plain_text(raw_html: str) -> str:
    raw_html = re.sub(
        r"(?is)<(script|style|noscript|svg).*?>.*?</\1>",
        " ",
        raw_html,
    )
    raw_html = re.sub(r"(?is)<!--.*?-->", " ", raw_html)

    raw_html = re.sub(
        r"(?i)</(p|div|section|article|h1|h2|h3|h4|li|pre|tr|td|th)>",
        "\n",
        raw_html,
    )

    text = re.sub(r"(?is)<[^>]+>", " ", raw_html)
    text = html.unescape(text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def load_html_documents(file_path: str) -> list[Document]:
    headers_to_split_on = [("h1", "Header 1"), ("h2", "Header 2"), ("h3", "Header 3")]

    try:
        splitter = HTMLHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
        docs = splitter.split_text_from_file(file_path)
        docs = [doc for doc in docs if doc.page_content.strip()]
        if docs:
            return docs
    except Exception:
        logging.exception(
            "HTMLHeaderTextSplitter failed, fallback to plain text: %s", file_path
        )

    raw_html = Path(file_path).read_text(encoding="utf-8", errors="ignore")
    text = html_to_plain_text(raw_html)

    if not text:
        raise ValueError(f"HTML file produced empty text after cleaning: {file_path}")

    return [Document(page_content=text, metadata={"source": file_path})]


def vectors_embedding(file_path: str, document_id: str, user_id: str):
    file_type = os.path.splitext(file_path)[-1].lower()

    # 1. Route to the correct Loader AND Splitter
    match file_type:
        case ".md":
            loader = TextLoader(file_path)
            headers_to_split_on = [
                ("#", "Header 1"),
                ("##", "Header 2"),
                ("###", "Header 3"),
            ]
            raw_text = loader.load()[0].page_content
            docs = MarkdownHeaderTextSplitter(
                headers_to_split_on=headers_to_split_on
            ).split_text(raw_text)

        case ".html":
            docs = load_html_documents(file_path)

        case ".pdf" | ".txt" | ".rst" | ".doc" | ".docx" | ".png" | ".jpg":
            if file_type == ".pdf":
                loader = PyPDFLoader(file_path)
            elif file_type in (".txt", ".rst"):
                loader = TextLoader(file_path)
            elif file_type in [".doc", ".docx"]:
                loader = UnstructuredWordDocumentLoader(file_path)
            else:
                loader = UnstructuredImageLoader(file_path)

            docs = loader.load()

        case _:
            raise ValueError(f"Unsupported file type: {file_type}")

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(docs)

    # 2. Inject Metadata
    for chunk in chunks:
        # Markdown splitters create 'metadata', standard splitters retain existing metadata
        if "source" not in chunk.metadata:
            chunk.metadata["source"] = file_path
        chunk.metadata["document_id"] = str(document_id)

    # 3. Embed and Store
    embedding_service = settings.services.embedding_model

    embeddings_model = get_embeddings(**embedding_service.model_dump())

    Chroma.from_documents(
        client=engine.chroma_client,
        documents=chunks,
        embedding=embeddings_model,
        collection_name=f"user-{user_id}",
    )

    invalidate_bm25_retriever(user_id)


def delete_vectors(user_id: str, document_id: str):
    """
    Deletes vectors from a user's Chroma collection based on the document_id metadata.
    """
    # 1. Initialize the same embedding model used for insertion
    embeddings_model = get_embeddings(**settings.services.embedding_model.model_dump())

    # 2. Connect to the existing Chroma collection for this user
    vectorstore = Chroma(
        client=engine.chroma_client,
        collection_name=f"user-{user_id}",
        embedding_function=embeddings_model,
    )

    # 3. Delete documents using a metadata filter ('where' clause)
    try:
        vectorstore.delete(where={"document_id": str(document_id)})
        invalidate_bm25_retriever(user_id)
    except Exception as e:
        logging.error(e)


bm25_cache_lock = Lock()
bm25_retriever_cache: dict[str, BM25Retriever] = {}


def _bm25_cache_key(user_id: str) -> str:
    return f"user-{user_id}"


def invalidate_bm25_retriever(user_id: str) -> None:
    """
    Clear the cached BM25 retriever for one user's knowledge base.

    This must be called after documents are embedded or deleted, otherwise
    BM25 may keep using an outdated in-memory index.
    """
    cache_key = _bm25_cache_key(str(user_id))
    with bm25_cache_lock:
        bm25_retriever_cache.pop(cache_key, None)


def get_bm25_retriever(vectorstore, user_id: str, k: int = 4) -> BM25Retriever | None:
    """
    Build or reuse a BM25 retriever scoped to one user's Chroma collection.
    """
    cache_key = _bm25_cache_key(str(user_id))

    with bm25_cache_lock:
        cached_retriever = bm25_retriever_cache.get(cache_key)
        if cached_retriever is not None:
            cached_retriever.k = k
            return cached_retriever

    all_db_docs = vectorstore.get()
    documents = all_db_docs.get("documents") if all_db_docs else []
    metadatas = all_db_docs.get("metadatas") if all_db_docs else []

    if not documents:
        logging.warning(
            "No documents found when building BM25 retriever for %s", cache_key
        )
        return None

    valid_pairs = []
    for index, text in enumerate(documents):
        if not text or not str(text).strip():
            continue
        metadata = (
            metadatas[index] if index < len(metadatas) and metadatas[index] else {}
        )
        valid_pairs.append((str(text), metadata))

    if not valid_pairs:
        logging.warning(
            "No valid text chunks found when building BM25 retriever for %s", cache_key
        )
        return None

    texts, metadata_list = zip(*valid_pairs)

    retriever = BM25Retriever.from_texts(
        texts=list(texts),
        metadatas=list(metadata_list),
    )
    retriever.k = k

    with bm25_cache_lock:
        bm25_retriever_cache[cache_key] = retriever

    return retriever
