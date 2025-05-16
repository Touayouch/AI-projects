import json

import pytest
from fastapi.testclient import TestClient

from rag.agents.grader import GradeResult
from rag.agents.query import QueryAnalysis
from rag.agents.validator import Validation
from rag.api import main
from rag.api.main import Services, create_app
from rag.graph import build_graph
from tests.fakes import FakeLLM, FakeRetriever


def answering_llm():
    return FakeLLM(
        [
            QueryAnalysis(in_scope=True),
            GradeResult(relevant=[1]),
            Validation(grounded=True, answers_question=True),
        ],
        answers=["4 jours [1]."],
    )


@pytest.fixture
def client_factory(monkeypatch):
    monkeypatch.setattr(main, "ollama_available", lambda url: True)

    def make(llm):
        retriever = FakeRetriever()
        services = Services(
            graph=build_graph(llm, retriever),
            retrievers={"hybrid": retriever, "dense": retriever, "sparse": retriever},
        )
        return TestClient(create_app(lambda settings: services))

    return make


def test_health(client_factory):
    with client_factory(answering_llm()) as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["ollama"] is True


def test_query(client_factory):
    with client_factory(answering_llm()) as client:
        response = client.post("/query", json={"question": "Congé pour un Pacs ?"})
    body = response.json()
    assert response.status_code == 200
    assert body["status"] == "answered"
    assert body["answer"] == "4 jours [1]."
    assert body["sources"][0]["chunk_id"] == "F1-0"
    assert body["steps"] is None


def test_query_verbose(client_factory):
    with client_factory(answering_llm()) as client:
        body = client.post("/query", json={"question": "Congé Pacs ?", "verbose": True}).json()
    assert [s.split(":")[0] for s in body["steps"]] == [
        "analyze",
        "retrieve",
        "grade",
        "generate",
        "validate",
    ]


def test_query_validates_input(client_factory):
    with client_factory(answering_llm()) as client:
        assert client.post("/query", json={"question": "a"}).status_code == 422
        assert client.post("/query", json={}).status_code == 422


def test_query_stream(client_factory):
    with client_factory(answering_llm()) as client:
        response = client.post("/query/stream", json={"question": "Congé Pacs ?"})
    events = [
        (block.split("\n")[0].removeprefix("event: "), json.loads(block.split("\n")[1][6:]))
        for block in response.text.strip().split("\n\n")
    ]
    assert [name for name, _ in events] == ["step"] * 6 + ["result"]
    assert [data["node"] for _, data in events[:-1]] == [
        "analyze",
        "retrieve",
        "grade",
        "generate",
        "validate",
        "finalize",
    ]
    result = events[-1][1]
    assert result["status"] == "answered"
    assert result["answer"] == "4 jours [1]."
    assert len(result["steps"]) == 5


def test_search(client_factory):
    with client_factory(answering_llm()) as client:
        body = client.post("/search", json={"query": "congés", "k": 2}).json()
    assert [h["chunk_id"] for h in body["hits"]] == ["F1-0", "F1-1"]
    assert body["hits"][0]["rank"] == 1


def test_search_rejects_unknown_method(client_factory):
    with client_factory(answering_llm()) as client:
        assert client.post("/search", json={"query": "x y", "method": "foo"}).status_code == 422
