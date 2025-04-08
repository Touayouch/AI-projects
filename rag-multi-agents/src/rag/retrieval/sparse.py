"""Retriever lexical BM25, efficace sur les termes exacts (sigles, articles de loi...)."""

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from rag.retrieval import ScoredDocument
from rag.retrieval.text import tokenize


class BM25Retriever:
    def __init__(self, chunks: list[Document]):
        self.chunks = chunks
        self.index = BM25Okapi([tokenize(c.page_content) for c in chunks])

    def search(self, query: str, k: int = 10) -> list[ScoredDocument]:
        tokens = tokenize(query)
        if not tokens:
            return []
        scores = self.index.get_scores(tokens)
        top = scores.argsort()[::-1][:k]
        return [(self.chunks[i], float(scores[i])) for i in top if scores[i] > 0]
