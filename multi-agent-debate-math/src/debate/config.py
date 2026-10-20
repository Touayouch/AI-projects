"""Configuration centralisée, chargée depuis les variables d'environnement / .env."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Serveur compatible OpenAI : Ollama en local (développement), vLLM sur GPU (expériences)
    llm_base_url: str = "http://localhost:11434/v1"
    llm_api_key: str = "EMPTY"
    default_model: str = "qwen2.5-instruct:1.5b"
    # Un serveur par modèle (débat hétérogène sur vLLM), en JSON :
    # LLM_ENDPOINTS='{"Qwen/Qwen2.5-1.5B-Instruct": "http://localhost:8001/v1", ...}'
    llm_endpoints: dict[str, str] = {}

    # Échantillonnage : une température > 0 est nécessaire pour que le vote et le débat
    # disposent de réponses diverses
    temperature: float = 0.7
    max_tokens: int = 2048  # 1024 tronquait certaines solutions de MATH-500
    seed: int = 0

    request_timeout: float = 600.0
    max_concurrency: int = 8  # requêtes simultanées (vLLM les regroupe en batch)

    data_dir: Path = Path("data")
    results_dir: Path = Path("results")
    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    return Settings()
