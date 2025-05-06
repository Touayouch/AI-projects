"""Évaluation de bout en bout du système multi-agents avec RAGAS.

Trois étapes, chacune reprenable (les résultats sont écrits ligne par ligne en JSONL, une
évaluation interrompue repart de la dernière question traitée) :

1. ``run_pipeline`` : pose chaque question au graphe et enregistre statut, réponse,
   brouillons successifs, contextes et latence.
2. ``score_records`` : calcule avec RAGAS la fidélité (*faithfulness*) et la pertinence
   (*answer relevancy*) de chaque brouillon. Scorer le premier brouillon ET la réponse
   finale mesure l'apport de la boucle validation / régénération.
3. ``build_report`` : agrège les scores, le routage (hors périmètre, sans réponse) et la
   détection des réponses hors sujet dans un rapport Markdown.
"""

import asyncio
import json
import logging
import statistics
import time
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Protocol

from rag.evaluation.retrieval import Question

logger = logging.getLogger(__name__)

ANSWERABLE = ("lexical", "paraphrase")
ABSTENTIONS = ("out_of_scope", "no_answer", "unverified")
# En dessous de ces seuils, une réponse est considérée hors sujet / non fidèle
RELEVANCY_THRESHOLD = 0.5
FAITHFULNESS_THRESHOLD = 0.5


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# --- 1. Exécution du pipeline ----------------------------------------------------------


def run_pipeline(graph, questions: list[Question], path: Path) -> list[dict]:
    done = {r["question"] for r in read_jsonl(path)}
    todo = [q for q in questions if q.question not in done]
    logger.info("Pipeline : %d questions déjà traitées, %d à traiter", len(done), len(todo))

    for i, q in enumerate(todo, start=1):
        start = time.perf_counter()
        state = graph.invoke({"question": q.question})
        record = {
            "question": q.question,
            "type": q.type,
            "doc_ids": q.doc_ids,
            "status": state["status"],
            "answer": state["answer"],
            "drafts": state.get("drafts", []),
            "contexts": [d.page_content for d in state.get("relevant_documents", [])],
            "retrieved_doc_ids": [d.metadata["doc_id"] for d in state.get("documents", [])],
            "steps": state["steps"],
            "latency_s": round(time.perf_counter() - start, 1),
        }
        append_jsonl(path, record)
        logger.info(
            "[%d/%d] %s (%.0f s) %s",
            i,
            len(todo),
            record["status"],
            record["latency_s"],
            q.question,
        )
    return read_jsonl(path)


# --- 2. Scores RAGAS -------------------------------------------------------------------


class Scorer(Protocol):
    async def score(self, question: str, answer: str, contexts: list[str]) -> dict: ...


class RagasScorer:
    """Faithfulness et AnswerRelevancy de RAGAS, avec un LLM servi par Ollama.

    Ollama expose une API compatible OpenAI, utilisée par les métriques RAGAS récentes.
    """

    def __init__(self, llm_model: str, embedding_model: str, base_url: str):
        from openai import AsyncOpenAI
        from ragas.embeddings import embedding_factory
        from ragas.llms import llm_factory
        from ragas.metrics.collections import AnswerRelevancy, Faithfulness

        client = AsyncOpenAI(base_url=f"{base_url}/v1", api_key="ollama", timeout=900)
        llm = llm_factory(
            llm_model, provider="openai", client=client, temperature=0.0, max_tokens=2048
        )
        embeddings = embedding_factory("openai", model=embedding_model, client=client)
        self.faithfulness = Faithfulness(llm=llm)
        self.relevancy = AnswerRelevancy(llm=llm, embeddings=embeddings)

    async def score(self, question: str, answer: str, contexts: list[str]) -> dict:
        faithfulness = await self.faithfulness.ascore(
            user_input=question, response=answer, retrieved_contexts=contexts
        )
        relevancy = await self.relevancy.ascore(user_input=question, response=answer)
        return {
            "faithfulness": round(float(faithfulness.value), 4),
            "answer_relevancy": round(float(relevancy.value), 4),
        }


async def _score_with_retry(
    scorer: Scorer, question: str, answer: str, contexts: list[str], attempts: int = 2
) -> dict | None:
    for attempt in range(1, attempts + 1):
        try:
            return await scorer.score(question, answer, contexts)
        except Exception as exc:  # JSON invalide du juge, serveur indisponible...
            logger.warning("Échec du scoring (essai %d/%d) : %s", attempt, attempts, exc)
            await asyncio.sleep(5)
    return None


async def score_records(scorer: Scorer, records: list[dict], path: Path) -> dict[str, dict]:
    """Score chaque brouillon des questions ayant donné lieu à une génération."""
    done = {r["question"]: r for r in read_jsonl(path)}
    todo = [r for r in records if r["drafts"] and r["question"] not in done]
    logger.info("Scoring : %d questions déjà scorées, %d à scorer", len(done), len(todo))

    for i, record in enumerate(todo, start=1):
        start = time.perf_counter()
        draft_scores = [
            await _score_with_retry(scorer, record["question"], draft, record["contexts"])
            for draft in record["drafts"]
        ]
        scored = {"question": record["question"], "draft_scores": draft_scores}
        append_jsonl(path, scored)
        done[record["question"]] = scored
        logger.info(
            "[%d/%d] scores %s (%.0f s)", i, len(todo), draft_scores, time.perf_counter() - start
        )
    return done


# --- 3. Rapport ------------------------------------------------------------------------


def mean(values: Iterable[float | None]) -> float | None:
    values = [v for v in values if v is not None]
    return statistics.mean(values) if values else None


def fmt(value: float | None, digits: int = 2) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def is_off_topic(scores: dict | None) -> bool:
    return scores is not None and (
        scores["answer_relevancy"] < RELEVANCY_THRESHOLD
        or scores["faithfulness"] < FAITHFULNESS_THRESHOLD
    )


def build_report(records: list[dict], scores: dict[str, dict], judge: str) -> str:
    by_type = Counter(r["type"] for r in records)
    answerable = [r for r in records if r["type"] in ANSWERABLE]
    answered = [r for r in records if r["status"] == "answered" and r["question"] in scores]
    final = [scores[r["question"]]["draft_scores"][-1] for r in answered]
    final = [s for s in final if s is not None]

    # Référence « sans validateur » : le premier brouillon aurait été renvoyé tel quel
    generated = [r for r in records if r["question"] in scores]
    first = [scores[r["question"]]["draft_scores"][0] for r in generated]
    first = [s for s in first if s is not None]
    # Réponses effectivement montrées à l'utilisateur par le système
    shown = [
        scores[r["question"]]["draft_scores"][-1]
        for r in generated
        if r["status"] == "answered" and scores[r["question"]]["draft_scores"][-1] is not None
    ]
    remediated = [r for r in generated if len(r["drafts"]) > 1]
    rejected = [r for r in generated if r["status"] == "unverified"]

    def hit(r: dict, k: int = 5) -> bool:
        return any(d in r["doc_ids"] for d in r["retrieved_doc_ids"][:k])

    status_rows = []
    for qtype in ("lexical", "paraphrase", "out_of_scope", "unanswerable"):
        group = [r for r in records if r["type"] == qtype]
        counts = Counter(r["status"] for r in group)
        status_rows.append(
            f"| {qtype} | {len(group)} | "
            + " | ".join(
                str(counts.get(s, 0))
                for s in ("answered", "no_answer", "unverified", "out_of_scope")
            )
            + " |"
        )

    out_scope = [r for r in records if r["type"] == "out_of_scope"]
    unanswerable = [r for r in records if r["type"] == "unanswerable"]
    false_refusals = [r for r in answerable if r["status"] == "out_of_scope"]
    latencies = [r["latency_s"] for r in answerable]

    lines = [
        "# Évaluation RAGAS du système multi-agents",
        "",
        f"{len(records)} questions ({', '.join(f'{n} {t}' for t, n in by_type.items())}). "
        f"Juge RAGAS : `{judge}` (local, via Ollama).",
        "",
        "## Qualité des réponses (RAGAS)",
        "",
        f"Sur les {len(final)} réponses fournies par le système (statut `answered`) :",
        "",
        "| Métrique | Score moyen |",
        "| --- | --- |",
        f"| Fidélité (*faithfulness*) | **{fmt(mean(s['faithfulness'] for s in final))}** |",
        f"| Pertinence (*answer relevancy*) | "
        f"**{fmt(mean(s['answer_relevancy'] for s in final))}** |",
        "",
        "## Détection et remédiation des réponses hors sujet",
        "",
        f"Une réponse est comptée *hors sujet / non fidèle* si sa pertinence < "
        f"{RELEVANCY_THRESHOLD} ou sa fidélité < {FAITHFULNESS_THRESHOLD}.",
        "",
        "| | Sans validateur (1er brouillon) | Avec validateur + remédiation |",
        "| --- | --- | --- |",
        f"| Réponses renvoyées | {len(first)} | {len(shown)} |",
        f"| Fidélité moyenne | {fmt(mean(s['faithfulness'] for s in first))} | "
        f"{fmt(mean(s['faithfulness'] for s in shown))} |",
        f"| Pertinence moyenne | {fmt(mean(s['answer_relevancy'] for s in first))} | "
        f"{fmt(mean(s['answer_relevancy'] for s in shown))} |",
        f"| Réponses hors sujet / non fidèles renvoyées | "
        f"{sum(is_off_topic(s) for s in first)} | {sum(is_off_topic(s) for s in shown)} |",
        "",
        f"- Réponses rejetées par le validateur puis régénérées : {len(remediated)}",
        f"- Réponses rejetées deux fois, remplacées par les sources (`unverified`) : "
        f"{len(rejected)}",
        "",
        "## Routage",
        "",
        "| Type | Questions | answered | no_answer | unverified | out_of_scope |",
        "| --- | --- | --- | --- | --- | --- |",
        *status_rows,
        "",
        f"- Questions hors périmètre refusées : "
        f"{sum(r['status'] == 'out_of_scope' for r in out_scope)}/{len(out_scope)}",
        f"- Questions sans réponse dans le corpus où le système s'abstient : "
        f"{sum(r['status'] in ABSTENTIONS for r in unanswerable)}/{len(unanswerable)}",
        f"- Questions légitimes refusées à tort comme hors périmètre : "
        f"{len(false_refusals)}/{len(answerable)}",
        f"- Retrieval (hit@5, questions avec réponse) : "
        f"{fmt(mean(float(hit(r)) for r in answerable))}",
        "",
        "## Latence (CPU, sans GPU)",
        "",
        f"- Médiane : {fmt(statistics.median(latencies), 0)} s par question avec réponse"
        if latencies
        else "- —",
        f"- Moyenne : {fmt(mean(latencies), 0)} s",
    ]
    return "\n".join(lines) + "\n"
