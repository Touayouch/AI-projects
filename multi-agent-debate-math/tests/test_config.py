from debate.config import Settings


def test_defaults_target_local_ollama():
    settings = Settings(_env_file=None)
    assert settings.llm_base_url == "http://localhost:11434/v1"
    assert settings.temperature > 0  # nécessaire pour la diversité des réponses


def test_env_override(monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", "http://gpu:8000/v1")
    monkeypatch.setenv("MAX_CONCURRENCY", "64")
    settings = Settings(_env_file=None)
    assert settings.llm_base_url == "http://gpu:8000/v1"
    assert settings.max_concurrency == 64
