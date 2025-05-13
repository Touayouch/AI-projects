"""Schémas d'entrée / sortie de l'API."""

from typing import Literal

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(
        min_length=3, max_length=1000, examples=["Combien de jours de congé pour un Pacs ?"]
    )
    verbose: bool = Field(default=False, description="Inclure les étapes du graphe")


class Source(BaseModel):
    n: int
    title: str
    section: str
    url: str
    chunk_id: str


class QueryResponse(BaseModel):
    question: str
    answer: str
    status: Literal["answered", "unverified", "no_answer", "out_of_scope"]
    sources: list[Source]
    steps: list[str] | None = None
    latency_s: float


class SearchRequest(BaseModel):
    query: str = Field(min_length=2, max_length=1000)
    method: Literal["dense", "sparse", "hybrid"] = "hybrid"
    k: int = Field(default=5, ge=1, le=20)


class SearchHit(BaseModel):
    rank: int
    score: float
    chunk_id: str
    doc_id: str
    title: str
    section: str
    url: str
    content: str


class SearchResponse(BaseModel):
    query: str
    method: str
    hits: list[SearchHit]


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    version: str
    llm_model: str
    embedding_model: str
    ollama: bool
