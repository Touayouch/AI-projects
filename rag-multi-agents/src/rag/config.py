"""Configuration centralisée, chargée depuis les variables d'environnement / .env."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # LLM
    llm_provider: Literal["ollama", "anthropic", "openai"] = "ollama"
    llm_model: str = "qwen3-instruct:4b"
    llm_temperature: float = 0.0
    llm_num_ctx: int = 8192
    ollama_base_url: str = "http://localhost:11434"

    # Embeddings (servis par Ollama)
    embedding_model: str = "bge-m3"
    embedding_batch_size: int = 64

    # Corpus : fiches pratiques service-public.gouv.fr (open data DILA)
    corpus_url: str = (
        "https://lecomarquage.service-public.gouv.fr/vdd/3.5/part/zip/vosdroits-latest.zip"
    )
    corpus_themes: list[str] = ["Travail - Formation", "Social - Santé"]

    # Données
    data_dir: Path = Path("data/raw")
    processed_dir: Path = Path("data/processed")

    # Chunking
    chunk_strategy: Literal["semantic", "recursive"] = "semantic"
    chunk_max_chars: int = 1500
    chunk_min_chars: int = 200
    chunk_breakpoint_percentile: float = 90.0

    # Retrieval hybride
    retrieval_k: int = 5
    hybrid_fetch_k: int = 30
    rrf_k: int = 60
    dense_weight: float = 1.0
    sparse_weight: float = 1.0

    # Graphe multi-agents
    max_retrieval_attempts: int = 2  # recherche initiale + 1 reformulation
    max_generation_attempts: int = 2  # génération initiale + 1 correction
    grader_chunk_chars: int = 800  # extrait de chunk montré au grader

    # ChromaDB : "local" (fichiers dans chroma_persist_dir) ou "http" (docker compose)
    chroma_mode: Literal["local", "http"] = "local"
    chroma_persist_dir: Path = Path("data/chroma")
    chroma_host: str = "localhost"
    chroma_port: int = 8001
    chroma_collection: str = "documents"

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8080
    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    return Settings()
