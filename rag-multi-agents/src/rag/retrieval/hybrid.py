"""Retriever hybride : fusion des résultats dense et BM25 par Reciprocal Rank Fusion.

RRF (Cormack et al., 2009) ne combine que les rangs, pas les scores : pas besoin de
normaliser des scores BM25 (non bornés) et cosinus (entre -1 et 1) sur une même échelle.

    score(d) = somme sur chaque retriever r de  poids_r / (rrf_k + rang_r(d))
"""

from typing import Protocol

from rag.retrieval import ScoredDocument


class Retriever(Protocol):
    def search(self, query: str, k: int = 10) -> list[ScoredDocument]: ...


def reciprocal_rank_fusion(
    rankings: list[list[ScoredDocument]], weights: list[float] | None = None, rrf_k: int = 60
) -> list[ScoredDocument]:
    weights = weights or [1.0] * len(rankings)
    scores: dict[str, float] = {}
    documents = {}
    for ranking, weight in zip(rankings, weights, strict=True):
        for rank, (doc, _) in enumerate(ranking, start=1):
            key = doc.metadata["chunk_id"]
            documents[key] = doc
            scores[key] = scores.get(key, 0.0) + weight / (rrf_k + rank)
    ordered = sorted(scores, key=scores.get, reverse=True)
    return [(documents[key], scores[key]) for key in ordered]


class HybridRetriever:
    def __init__(
        self,
        dense: Retriever,
        sparse: Retriever,
        fetch_k: int = 30,
        rrf_k: int = 60,
        weights: tuple[float, float] = (1.0, 1.0),
    ):
        self.dense = dense
        self.sparse = sparse
        self.fetch_k = fetch_k
        self.rrf_k = rrf_k
        self.weights = weights

    def search(self, query: str, k: int = 10) -> list[ScoredDocument]:
        rankings = [
            self.dense.search(query, k=self.fetch_k),
            self.sparse.search(query, k=self.fetch_k),
        ]
        return reciprocal_rank_fusion(rankings, list(self.weights), self.rrf_k)[:k]
