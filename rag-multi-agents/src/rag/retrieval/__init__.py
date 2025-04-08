"""Retrieval : dense (ChromaDB), sparse (BM25) et hybride (fusion RRF)."""

from langchain_core.documents import Document

# (document, score) — plus le score est élevé, plus le document est pertinent
ScoredDocument = tuple[Document, float]
