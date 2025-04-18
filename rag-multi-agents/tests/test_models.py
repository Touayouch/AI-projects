import pytest
from langchain_core.embeddings import Embeddings

import rag.models
from rag.models import CachedEmbeddings


class CountingEmbeddings(Embeddings):
    def __init__(self):
        self.calls: list[str] = []

    def embed_documents(self, texts):
        self.calls.extend(texts)
        return [[float(len(t)), 1.0] for t in texts]

    def embed_query(self, text):
        return self.embed_documents([text])[0]


def test_cache_avoids_recomputing(tmp_path):
    inner = CountingEmbeddings()
    cache = tmp_path / "cache.sqlite"
    first = CachedEmbeddings(inner, "m", cache, batch_size=2).embed_documents(["a", "bb", "a"])
    second = CachedEmbeddings(inner, "m", cache).embed_documents(["bb", "ccc"])

    assert first == [[1.0, 1.0], [2.0, 1.0], [1.0, 1.0]]
    assert second == [[2.0, 1.0], [3.0, 1.0]]
    assert inner.calls == ["a", "bb", "ccc"]


def test_cache_is_per_model(tmp_path):
    inner = CountingEmbeddings()
    cache = tmp_path / "cache.sqlite"
    CachedEmbeddings(inner, "m1", cache).embed_documents(["a"])
    CachedEmbeddings(inner, "m2", cache).embed_documents(["a"])
    assert inner.calls == ["a", "a"]


class FlakyEmbeddings(CountingEmbeddings):
    def __init__(self, failures: int):
        super().__init__()
        self.failures = failures

    def embed_documents(self, texts):
        if self.failures:
            self.failures -= 1
            raise ConnectionError("Server disconnected")
        return super().embed_documents(texts)


def test_retries_transient_errors(monkeypatch):
    monkeypatch.setattr(rag.models.time, "sleep", lambda s: None)
    embeddings = CachedEmbeddings(FlakyEmbeddings(failures=2), "m", max_retries=3)
    assert embeddings.embed_documents(["a"]) == [[1.0, 1.0]]


def test_gives_up_after_max_retries(monkeypatch):
    monkeypatch.setattr(rag.models.time, "sleep", lambda s: None)
    embeddings = CachedEmbeddings(FlakyEmbeddings(failures=3), "m", max_retries=3)
    with pytest.raises(ConnectionError):
        embeddings.embed_documents(["a"])
