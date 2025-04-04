from rag.config import Settings


def test_defaults():
    settings = Settings(_env_file=None)
    assert settings.llm_provider == "ollama"
    assert settings.chroma_port == 8001


def test_env_override(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("CHROMA_PORT", "9000")
    settings = Settings(_env_file=None)
    assert settings.llm_provider == "ollama"
    assert settings.chroma_port == 9000
