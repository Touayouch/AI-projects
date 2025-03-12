"""Point d'entrée FastAPI."""

from fastapi import FastAPI

from rag import __version__
from rag.config import get_settings

app = FastAPI(title="RAG multi-agents", version=__version__)


@app.get("/health")
def health() -> dict:
    settings = get_settings()
    return {
        "status": "ok",
        "version": __version__,
        "llm_provider": settings.llm_provider,
        "llm_model": settings.llm_model,
    }
