"""API FastAPI du système RAG multi-agents.

    GET  /health        état du service et de la connexion à Ollama
    POST /search        recherche seule (dense, BM25 ou hybride), rapide
    POST /query         réponse complète via le graphe multi-agents
    POST /query/stream  idem, en Server-Sent Events : une étape du graphe par événement

Les retrievers et le graphe sont construits une seule fois au démarrage (chargement de
l'index BM25 et de ChromaDB).
"""

import json
import logging
import time
from collections.abc import Callable, Iterator
from contextlib import asynccontextmanager
from dataclasses import dataclass

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse

from rag import __version__
from rag.api.schemas import (
    HealthResponse,
    QueryRequest,
    QueryResponse,
    SearchHit,
    SearchRequest,
    SearchResponse,
)
from rag.config import Settings, get_settings
from rag.graph import GraphConfig, build_graph
from rag.models import get_llm
from rag.retrieval.factory import build_retrievers
from rag.retrieval.hybrid import Retriever

logger = logging.getLogger("rag.api")

# Clés d'état accumulées au fil des nœuds (les autres sont remplacées)
ACCUMULATED = ("steps", "drafts")


@dataclass
class Services:
    graph: object
    retrievers: dict[str, Retriever]


def load_services(settings: Settings) -> Services:
    retrievers = build_retrievers(settings, settings.chunk_strategy, ["dense", "sparse", "hybrid"])
    graph = build_graph(
        get_llm(settings), retrievers["hybrid"], GraphConfig.from_settings(settings)
    )
    return Services(graph=graph, retrievers=retrievers)


def ollama_available(base_url: str) -> bool:
    try:
        return httpx.get(f"{base_url}/api/version", timeout=2).status_code == 200
    except httpx.HTTPError:
        return False


def to_response(question: str, state: dict, latency: float, verbose: bool) -> QueryResponse:
    return QueryResponse(
        question=question,
        answer=state["answer"],
        status=state["status"],
        sources=state.get("sources", []),
        steps=state.get("steps") if verbose else None,
        latency_s=round(latency, 1),
    )


def sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def create_app(services_factory: Callable[[Settings], Services] = load_services) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        settings = get_settings()
        logging.basicConfig(level=settings.log_level)
        app.state.services = services_factory(settings)
        logger.info("Services chargés")
        yield

    app = FastAPI(
        title="RAG multi-agents",
        description="Questions-réponses sur les fiches service-public.gouv.fr "
        "(droit du travail, protection sociale).",
        version=__version__,
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        logger.info(
            "%s %s -> %d (%.2f s)",
            request.method,
            request.url.path,
            response.status_code,
            time.perf_counter() - start,
        )
        return response

    def services(request: Request) -> Services:
        return request.app.state.services

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        settings = get_settings()
        ollama = ollama_available(settings.ollama_base_url)
        return HealthResponse(
            status="ok" if ollama else "degraded",
            version=__version__,
            llm_model=settings.llm_model,
            embedding_model=settings.embedding_model,
            ollama=ollama,
        )

    @app.post("/search", response_model=SearchResponse)
    def search(body: SearchRequest, request: Request) -> SearchResponse:
        retriever = services(request).retrievers[body.method]
        try:
            results = retriever.search(body.query, k=body.k)
        except httpx.HTTPError as exc:
            raise HTTPException(503, f"Modèle d'embeddings indisponible : {exc}") from exc
        hits = [
            SearchHit(
                rank=rank,
                score=round(score, 4),
                chunk_id=doc.metadata["chunk_id"],
                doc_id=doc.metadata["doc_id"],
                title=doc.metadata.get("title", ""),
                section=doc.metadata.get("section", ""),
                url=doc.metadata.get("url", ""),
                content=doc.page_content,
            )
            for rank, (doc, score) in enumerate(results, start=1)
        ]
        return SearchResponse(query=body.query, method=body.method, hits=hits)

    # Endpoints synchrones : FastAPI les exécute dans un thread, sans bloquer la boucle
    @app.post("/query", response_model=QueryResponse)
    def query(body: QueryRequest, request: Request) -> QueryResponse:
        start = time.perf_counter()
        try:
            state = services(request).graph.invoke({"question": body.question})
        except httpx.HTTPError as exc:
            raise HTTPException(503, f"LLM indisponible : {exc}") from exc
        return to_response(body.question, state, time.perf_counter() - start, body.verbose)

    @app.post("/query/stream")
    def query_stream(body: QueryRequest, request: Request) -> StreamingResponse:
        graph = services(request).graph

        def events() -> Iterator[str]:
            start = time.perf_counter()
            state: dict = {}
            try:
                for update in graph.stream({"question": body.question}, stream_mode="updates"):
                    for node, values in update.items():
                        for key, value in (values or {}).items():
                            if key in ACCUMULATED:
                                state[key] = state.get(key, []) + value
                            else:
                                state[key] = value
                        yield sse("step", {"node": node, "steps": (values or {}).get("steps", [])})
            except httpx.HTTPError as exc:
                yield sse("error", {"detail": f"LLM indisponible : {exc}"})
                return
            response = to_response(body.question, state, time.perf_counter() - start, True)
            yield sse("result", response.model_dump())

        return StreamingResponse(events(), media_type="text/event-stream")

    return app


app = create_app()
