"""Construction des retrievers à partir des chunks indexés en phase d'ingestion."""

from typing import Literal

from rag.config import Settings
from rag.indexing import get_vectorstore, load_chunks
from rag.models import get_embeddings
from rag.retrieval.dense import DenseRetriever
from rag.retrieval.hybrid import HybridRetriever, Retriever
from rag.retrieval.sparse import BM25Retriever

Method = Literal["dense", "sparse", "hybrid"]


def build_retrievers(
    settings: Settings, strategy: str, methods: list[Method]
) -> dict[Method, Retriever]:
    chunks_path = settings.processed_dir / f"chunks_{strategy}.jsonl"
    if not chunks_path.exists():
        raise SystemExit(f"{chunks_path} introuvable, lancer `rag ingest --strategy {strategy}`")

    retrievers: dict[Method, Retriever] = {}
    if {"sparse", "hybrid"} & set(methods):
        retrievers["sparse"] = BM25Retriever(load_chunks(chunks_path))
    if {"dense", "hybrid"} & set(methods):
        collection = f"{settings.chroma_collection}_{strategy}"
        vectorstore = get_vectorstore(settings, get_embeddings(settings), collection)
        retrievers["dense"] = DenseRetriever(vectorstore)
    if "hybrid" in methods:
        retrievers["hybrid"] = HybridRetriever(
            retrievers["dense"],
            retrievers["sparse"],
            fetch_k=settings.hybrid_fetch_k,
            rrf_k=settings.rrf_k,
            weights=(settings.dense_weight, settings.sparse_weight),
        )
    return {m: retrievers[m] for m in methods}
