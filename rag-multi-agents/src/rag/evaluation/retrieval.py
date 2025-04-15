"""Évaluation du retrieval sur un jeu de questions annotées.

Une question est annotée avec la ou les fiches qui contiennent la réponse. La pertinence
est jugée au niveau de la fiche : un chunk est pertinent s'il provient d'une de ces fiches.
Cela permet de comparer équitablement les stratégies de chunking entre elles.

Métriques (k = 10 par défaut) :
- hit@k : proportion de questions avec au moins un chunk pertinent dans le top k
- MRR@k : moyenne de 1 / rang du premier chunk pertinent (0 s'il n'est pas dans le top k)
"""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from rag.retrieval.hybrid import Retriever

CUTOFFS = (1, 3, 5, 10)


@dataclass
class Question:
    question: str
    doc_ids: list[str]
    type: str = ""


@dataclass
class RetrievalScores:
    ranks: list[int | None] = field(default_factory=list)
    latencies: list[float] = field(default_factory=list)

    def hit_at(self, k: int) -> float:
        return sum(r is not None and r <= k for r in self.ranks) / len(self.ranks)

    def mrr(self, k: int = 10) -> float:
        return sum(1 / r for r in self.ranks if r is not None and r <= k) / len(self.ranks)

    @property
    def latency_ms(self) -> float:
        return 1000 * sum(self.latencies) / len(self.latencies)


def load_questions(path: Path) -> list[Question]:
    with path.open(encoding="utf-8") as f:
        return [Question(**json.loads(line)) for line in f if line.strip()]


def first_relevant_rank(doc_ids: list[str], retrieved_doc_ids: list[str]) -> int | None:
    for rank, doc_id in enumerate(retrieved_doc_ids, start=1):
        if doc_id in doc_ids:
            return rank
    return None


def evaluate(retriever: Retriever, questions: list[Question], k: int = 10) -> RetrievalScores:
    scores = RetrievalScores()
    for q in questions:
        start = time.perf_counter()
        results = retriever.search(q.question, k=k)
        scores.latencies.append(time.perf_counter() - start)
        retrieved = [doc.metadata["doc_id"] for doc, _ in results]
        scores.ranks.append(first_relevant_rank(q.doc_ids, retrieved))
    return scores


def report_table(results: dict[str, RetrievalScores]) -> str:
    header = "| Méthode | " + " | ".join(f"hit@{k}" for k in CUTOFFS) + " | MRR@10 | Latence |"
    lines = [header, "|" + " --- |" * (len(CUTOFFS) + 3)]
    for name, s in results.items():
        hits = " | ".join(f"{s.hit_at(k):.2f}" for k in CUTOFFS)
        lines.append(f"| {name} | {hits} | {s.mrr():.3f} | {s.latency_ms:.0f} ms |")
    return "\n".join(lines)
