"""Orchestration multi-agents avec LangGraph.

analyze ──(hors périmètre)──► out_of_scope
   │
   ▼
retrieve ──► grade ──(aucun document utile)──► rewrite ──► retrieve   (1 fois max)
               │                     └──(tentatives épuisées)──► no_answer
               ▼
            generate ──► validate ──(rejetée)──► generate avec retour du validateur
                            │                    (1 fois max, puis réponse prudente)
                            ▼
                         finalize
"""

import operator
from dataclasses import dataclass
from typing import Annotated, Literal, TypedDict

import httpx
from langchain_core.documents import Document
from langchain_core.exceptions import OutputParserException
from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, START, StateGraph
from langgraph.types import RetryPolicy

from rag.agents.generator import Generator
from rag.agents.grader import DocumentGrader
from rag.agents.query import QueryAgent
from rag.agents.validator import AnswerValidator
from rag.retrieval.hybrid import Retriever

OUT_OF_SCOPE_ANSWER = (
    "Je ne peux répondre qu'aux questions sur le droit du travail, l'emploi et la protection "
    "sociale en France (fiches service-public.gouv.fr)."
)
NO_ANSWER = (
    "Je n'ai pas trouvé d'information fiable sur ce sujet dans les fiches service-public.gouv.fr."
)
UNVERIFIED_ANSWER = (
    "Je ne peux pas donner de réponse suffisamment fiable à cette question. "
    "Les fiches suivantes pourraient vous aider :"
)

# Erreurs passagères : serveur Ollama redémarré, ou JSON mal formé par le petit modèle
RETRY_POLICY = RetryPolicy(
    initial_interval=5.0,
    max_attempts=3,
    retry_on=(httpx.TransportError, ConnectionError, OutputParserException),
)

Status = Literal["answered", "unverified", "no_answer", "out_of_scope"]


class RAGState(TypedDict, total=False):
    question: str
    search_query: str
    in_scope: bool
    documents: list[Document]
    relevant_documents: list[Document]
    answer: str
    feedback: str
    validation: dict
    retrieval_attempts: int
    generation_attempts: int
    status: Status
    sources: list[dict]
    steps: Annotated[list[str], operator.add]


@dataclass
class GraphConfig:
    k: int = 5
    max_retrieval_attempts: int = 2
    max_generation_attempts: int = 2
    grader_chunk_chars: int = 800


def sources_from(documents: list[Document]) -> list[dict]:
    return [
        {
            "n": i,
            "title": doc.metadata.get("title", ""),
            "section": doc.metadata.get("section", ""),
            "url": doc.metadata.get("url", ""),
            "chunk_id": doc.metadata.get("chunk_id", ""),
        }
        for i, doc in enumerate(documents, start=1)
    ]


def build_graph(llm: BaseChatModel, retriever: Retriever, config: GraphConfig | None = None):
    config = config or GraphConfig()
    query_agent = QueryAgent(llm)
    grader = DocumentGrader(llm, max_chars=config.grader_chunk_chars)
    generator = Generator(llm)
    validator = AnswerValidator(llm)

    # --- Nœuds ---------------------------------------------------------------------------

    def analyze(state: RAGState) -> RAGState:
        analysis = query_agent.analyze(state["question"])
        return {
            "in_scope": analysis.in_scope,
            "search_query": state["question"],
            "retrieval_attempts": 0,
            "generation_attempts": 0,
            "steps": [f"analyze: in_scope={analysis.in_scope}"],
        }

    def retrieve(state: RAGState) -> RAGState:
        results = retriever.search(state["search_query"], k=config.k)
        documents = [doc for doc, _ in results]
        return {
            "documents": documents,
            "retrieval_attempts": state["retrieval_attempts"] + 1,
            "steps": [f"retrieve: {len(documents)} chunks pour {state['search_query']!r}"],
        }

    def grade(state: RAGState) -> RAGState:
        relevant = grader.grade(state["question"], state["documents"])
        return {
            "relevant_documents": relevant,
            "steps": [f"grade: {len(relevant)}/{len(state['documents'])} chunks pertinents"],
        }

    def rewrite(state: RAGState) -> RAGState:
        query = query_agent.rewrite(state["question"], state["search_query"])
        return {"search_query": query, "steps": [f"rewrite: {query!r}"]}

    def generate(state: RAGState) -> RAGState:
        answer = generator.generate(
            state["question"], state["relevant_documents"], state.get("feedback")
        )
        return {
            "answer": answer,
            "generation_attempts": state["generation_attempts"] + 1,
            "steps": [f"generate: tentative {state['generation_attempts'] + 1}"],
        }

    def validate(state: RAGState) -> RAGState:
        validation = validator.validate(
            state["question"], state["relevant_documents"], state["answer"]
        )
        return {
            "validation": validation.model_dump(),
            "feedback": validation.feedback,
            "steps": [
                f"validate: grounded={validation.grounded} "
                f"answers_question={validation.answers_question}"
            ],
        }

    def finalize(state: RAGState) -> RAGState:
        documents = state["relevant_documents"]
        if state["validation"]["grounded"] and state["validation"]["answers_question"]:
            return {"status": "answered", "sources": sources_from(documents)}
        # Remédiation épuisée : on ne renvoie pas une réponse jugée non fiable
        return {
            "status": "unverified",
            "answer": UNVERIFIED_ANSWER,
            "sources": sources_from(documents),
            "steps": ["finalize: réponse rejetée, repli sur les sources"],
        }

    def out_of_scope(state: RAGState) -> RAGState:
        return {"status": "out_of_scope", "answer": OUT_OF_SCOPE_ANSWER, "sources": []}

    def no_answer(state: RAGState) -> RAGState:
        return {"status": "no_answer", "answer": NO_ANSWER, "sources": []}

    # --- Routage -------------------------------------------------------------------------

    def route_after_analyze(state: RAGState) -> str:
        return "retrieve" if state["in_scope"] else "out_of_scope"

    def route_after_grade(state: RAGState) -> str:
        if state["relevant_documents"]:
            return "generate"
        if state["retrieval_attempts"] < config.max_retrieval_attempts:
            return "rewrite"
        return "no_answer"

    def route_after_validate(state: RAGState) -> str:
        v = state["validation"]
        if v["grounded"] and v["answers_question"]:
            return "finalize"
        if state["generation_attempts"] < config.max_generation_attempts:
            return "generate"
        return "finalize"

    # --- Graphe --------------------------------------------------------------------------

    graph = StateGraph(RAGState)
    for name, node in [
        ("analyze", analyze),
        ("retrieve", retrieve),
        ("grade", grade),
        ("rewrite", rewrite),
        ("generate", generate),
        ("validate", validate),
        ("finalize", finalize),
        ("out_of_scope", out_of_scope),
        ("no_answer", no_answer),
    ]:
        graph.add_node(name, node, retry_policy=RETRY_POLICY)

    graph.add_edge(START, "analyze")
    graph.add_conditional_edges("analyze", route_after_analyze, ["retrieve", "out_of_scope"])
    graph.add_edge("retrieve", "grade")
    graph.add_conditional_edges("grade", route_after_grade, ["generate", "rewrite", "no_answer"])
    graph.add_edge("rewrite", "retrieve")
    graph.add_edge("generate", "validate")
    graph.add_conditional_edges("validate", route_after_validate, ["generate", "finalize"])
    for terminal in ("finalize", "out_of_scope", "no_answer"):
        graph.add_edge(terminal, END)

    return graph.compile()
