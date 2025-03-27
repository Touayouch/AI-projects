"""Accès aux modèles (embeddings, et LLM à partir de la phase 3)."""

import logging

from langchain_core.embeddings import Embeddings
from langchain_ollama import OllamaEmbeddings

from rag.config import Settings

logger = logging.getLogger(__name__)


class BatchedEmbeddings(Embeddings):
    """Envoie les textes par lots au modèle, pour ne pas saturer le serveur."""

    def __init__(self, embeddings: Embeddings, batch_size: int = 64):
        self.embeddings = embeddings
        self.batch_size = batch_size

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            vectors.extend(self.embeddings.embed_documents(texts[start : start + self.batch_size]))
            done = min(start + self.batch_size, len(texts))
            if done % (self.batch_size * 20) == 0 or done == len(texts):
                logger.info("Embeddings : %d/%d", done, len(texts))
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self.embeddings.embed_query(text)


def get_embeddings(settings: Settings) -> Embeddings:
    return BatchedEmbeddings(
        OllamaEmbeddings(model=settings.embedding_model, base_url=settings.ollama_base_url),
        batch_size=settings.embedding_batch_size,
    )
