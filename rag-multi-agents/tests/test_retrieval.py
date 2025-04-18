from langchain_core.documents import Document

from rag.evaluation.retrieval import Question, evaluate, first_relevant_rank
from rag.retrieval.hybrid import HybridRetriever, reciprocal_rank_fusion
from rag.retrieval.sparse import BM25Retriever
from rag.retrieval.text import tokenize


def chunk(chunk_id: str, text: str = "") -> Document:
    return Document(
        page_content=text, metadata={"chunk_id": chunk_id, "doc_id": chunk_id.split("-")[0]}
    )


class StaticRetriever:
    def __init__(self, docs):
        self.docs = docs

    def search(self, query, k=10):
        return [(d, 1.0) for d in self.docs[:k]]


def test_tokenize_french():
    tokens = tokenize("L'employeur verse l'indemnité de congés payés au salarié en CDD")
    assert "employeur" in tokens
    assert "l" not in tokens and "de" not in tokens and "au" not in tokens
    assert tokenize("congés") == tokenize("conges") == tokenize("Congé")
    assert "cdd" in tokens


def test_tokenize_keeps_legal_references():
    assert "l1234-9" in tokenize("article L1234-9 du code du travail")


def test_bm25_ranks_exact_term_first():
    chunks = [
        chunk("F1-0", "Le congé de paternité dure 25 jours."),
        chunk("F2-0", "La prime de partage de la valeur (PPV) est exonérée."),
        chunk("F3-0", "Le salarié a droit à des congés payés."),
    ]
    results = BM25Retriever(chunks).search("Comment fonctionne la PPV ?", k=3)
    assert results[0][0].metadata["chunk_id"] == "F2-0"
    assert len(results) == 1  # les chunks sans terme commun sont exclus


def test_rrf_favors_documents_ranked_by_both():
    a, b, c = chunk("A-0"), chunk("B-0"), chunk("C-0")
    fused = reciprocal_rank_fusion([[(a, 0.9), (b, 0.8)], [(c, 12.0), (b, 10.0)]])
    assert [d.metadata["chunk_id"] for d, _ in fused] == ["B-0", "A-0", "C-0"]


def test_rrf_weights():
    a, c = chunk("A-0"), chunk("C-0")
    fused = reciprocal_rank_fusion([[(a, 1.0)], [(c, 1.0)]], weights=[1.0, 2.0])
    assert fused[0][0].metadata["chunk_id"] == "C-0"


def test_hybrid_retriever_truncates_to_k():
    dense = StaticRetriever([chunk(f"D{i}-0") for i in range(10)])
    sparse = StaticRetriever([chunk(f"S{i}-0") for i in range(10)])
    assert len(HybridRetriever(dense, sparse).search("q", k=4)) == 4


def test_first_relevant_rank():
    assert first_relevant_rank(["F2"], ["F1", "F2", "F2"]) == 2
    assert first_relevant_rank(["F9"], ["F1", "F2"]) is None


def test_evaluate_metrics():
    retriever = StaticRetriever([chunk("F1-0"), chunk("F2-0"), chunk("F3-0")])
    questions = [
        Question("q1", ["F1"]),
        Question("q2", ["F3"]),
        Question("q3", ["F9"]),
    ]
    scores = evaluate(retriever, questions, k=3)
    assert scores.ranks == [1, 3, None]
    assert scores.hit_at(1) == 1 / 3
    assert scores.hit_at(3) == 2 / 3
    assert abs(scores.mrr() - (1 + 1 / 3) / 3) < 1e-9
