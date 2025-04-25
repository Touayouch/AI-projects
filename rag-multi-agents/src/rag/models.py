"""Accès aux modèles : embeddings et LLM."""

import hashlib
import json
import logging
import sqlite3
import time
from pathlib import Path

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_ollama import ChatOllama, OllamaEmbeddings

from rag.config import Settings

logger = logging.getLogger(__name__)


class CachedEmbeddings(Embeddings):
    """Envoie les textes par lots au modèle et met les vecteurs en cache (SQLite).

    Sur CPU, l'embedding du corpus prend plusieurs heures : le cache évite de recalculer
    un texte déjà vu (relance de l'ingestion, réglage du chunking...).
    """

    def __init__(
        self,
        embeddings: Embeddings,
        model_name: str,
        cache_path: Path | None = None,
        batch_size: int = 64,
        max_retries: int = 3,
    ):
        self.embeddings = embeddings
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.db = None
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.db = sqlite3.connect(cache_path)
            self.db.execute("CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, vector TEXT)")

    def _key(self, text: str) -> str:
        return hashlib.sha256(f"{self.model_name}\n{text}".encode()).hexdigest()

    def _lookup(self, keys: list[str]) -> dict[str, list[float]]:
        if self.db is None:
            return {}
        found = {}
        for start in range(0, len(keys), 500):
            batch = keys[start : start + 500]
            placeholders = ",".join("?" * len(batch))
            rows = self.db.execute(
                f"SELECT key, vector FROM cache WHERE key IN ({placeholders})", batch
            )
            found.update({key: json.loads(vector) for key, vector in rows})
        return found

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        keys = [self._key(t) for t in texts]
        cached = self._lookup(list(set(keys)))
        missing = list({k: t for k, t in zip(keys, texts, strict=True) if k not in cached}.items())
        logger.info("Embeddings : %d textes, %d en cache", len(texts), len(texts) - len(missing))

        for start in range(0, len(missing), self.batch_size):
            batch = missing[start : start + self.batch_size]
            vectors = self._embed_with_retry([t for _, t in batch])
            cached.update({k: v for (k, _), v in zip(batch, vectors, strict=True)})
            if self.db is not None:
                self.db.executemany(
                    "INSERT OR REPLACE INTO cache VALUES (?, ?)",
                    [(k, json.dumps(v)) for (k, _), v in zip(batch, vectors, strict=True)],
                )
                self.db.commit()
            done = min(start + self.batch_size, len(missing))
            if done % (self.batch_size * 10) == 0 or done == len(missing):
                logger.info("Embeddings calculés : %d/%d", done, len(missing))

        return [cached[k] for k in keys]

    def _embed_with_retry(self, texts: list[str]) -> list[list[float]]:
        """Réessaie en cas d'erreur passagère (serveur Ollama redémarré, surchargé...)."""
        for attempt in range(1, self.max_retries + 1):
            try:
                return self.embeddings.embed_documents(texts)
            except Exception as exc:
                if attempt == self.max_retries:
                    raise
                delay = 10 * attempt
                logger.warning("Échec de l'embedding (%s), nouvel essai dans %d s", exc, delay)
                time.sleep(delay)
        raise AssertionError("unreachable")

    def embed_query(self, text: str) -> list[float]:
        return self.embeddings.embed_query(text)


def get_embeddings(settings: Settings) -> Embeddings:
    return CachedEmbeddings(
        OllamaEmbeddings(model=settings.embedding_model, base_url=settings.ollama_base_url),
        model_name=settings.embedding_model,
        cache_path=settings.processed_dir / "embeddings_cache.sqlite",
        batch_size=settings.embedding_batch_size,
    )


def get_llm(settings: Settings) -> BaseChatModel:
    if settings.llm_provider == "ollama":
        return ChatOllama(
            model=settings.llm_model,
            base_url=settings.ollama_base_url,
            temperature=settings.llm_temperature,
            num_ctx=settings.llm_num_ctx,
            # Sans cela, Ollama considère Qwen3 comme un modèle à raisonnement et n'applique
            # pas la contrainte JSON des sorties structurées (et la latence explose sur CPU).
            reasoning=False,
        )
    # Fournisseurs API : nécessitent l'extra correspondant (uv sync --extra anthropic)
    from langchain.chat_models import init_chat_model

    return init_chat_model(
        settings.llm_model,
        model_provider=settings.llm_provider,
        temperature=settings.llm_temperature,
    )
