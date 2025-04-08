"""Retriever dense : similarité cosinus entre embeddings dans ChromaDB."""

from langchain_chroma import Chroma

from rag.retrieval import ScoredDocument


class DenseRetriever:
    def __init__(self, vectorstore: Chroma):
        self.vectorstore = vectorstore

    def search(self, query: str, k: int = 10) -> list[ScoredDocument]:
        results = self.vectorstore.similarity_search_with_score(query, k=k)
        # Chroma renvoie une distance cosinus : on la convertit en similarité
        return [(doc, 1.0 - distance) for doc, distance in results]
