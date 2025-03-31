"""Persistance des chunks : fichier JSONL (pour BM25) et base vectorielle ChromaDB."""

import json
import logging
from pathlib import Path

import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from rag.config import Settings

logger = logging.getLogger(__name__)


def save_chunks(chunks: list[Document], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            record = {"page_content": chunk.page_content, "metadata": chunk.metadata}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_chunks(path: Path) -> list[Document]:
    with path.open(encoding="utf-8") as f:
        return [Document(**json.loads(line)) for line in f if line.strip()]


def get_vectorstore(settings: Settings, embeddings: Embeddings, collection: str) -> Chroma:
    if settings.chroma_mode == "http":
        client = chromadb.HttpClient(host=settings.chroma_host, port=settings.chroma_port)
    else:
        client = chromadb.PersistentClient(path=str(settings.chroma_persist_dir))
    return Chroma(
        client=client,
        collection_name=collection,
        embedding_function=embeddings,
        collection_metadata={"hnsw:space": "cosine"},
    )


def index_chunks(
    vectorstore: Chroma, chunks: list[Document], batch_size: int = 256, reset: bool = True
) -> None:
    if reset:
        vectorstore.reset_collection()
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        vectorstore.add_documents(batch, ids=[c.metadata["chunk_id"] for c in batch])
    logger.info("%d chunks indexés dans ChromaDB", len(chunks))
