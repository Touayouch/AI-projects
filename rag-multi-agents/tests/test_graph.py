from rag.agents.grader import GradeResult
from rag.agents.query import QueryAnalysis, RewrittenQuery
from rag.agents.validator import Validation
from rag.graph import NO_ANSWER, OUT_OF_SCOPE_ANSWER, UNVERIFIED_ANSWER, build_graph
from tests.fakes import FakeLLM, FakeRetriever


def run(llm, retriever=None):
    retriever = retriever or FakeRetriever()
    return build_graph(llm, retriever).invoke({"question": "Combien de congés ?"})


def test_happy_path():
    llm = FakeLLM(
        [
            QueryAnalysis(in_scope=True),
            GradeResult(relevant=[1, 3]),
            Validation(grounded=True, answers_question=True),
        ],
        answers=["2,5 jours par mois [1]."],
    )
    state = run(llm)

    assert state["status"] == "answered"
    assert state["answer"] == "2,5 jours par mois [1]."
    assert [s["chunk_id"] for s in state["sources"]] == ["F1-0", "F1-2"]
    assert [s.split(":")[0] for s in state["steps"]] == [
        "analyze",
        "retrieve",
        "grade",
        "generate",
        "validate",
    ]


def test_out_of_scope_skips_retrieval():
    retriever = FakeRetriever()
    state = run(FakeLLM([QueryAnalysis(in_scope=False)]), retriever)

    assert state["status"] == "out_of_scope"
    assert state["answer"] == OUT_OF_SCOPE_ANSWER
    assert retriever.queries == []


def test_rewrites_query_then_gives_up():
    retriever = FakeRetriever()
    llm = FakeLLM(
        [
            QueryAnalysis(in_scope=True),
            GradeResult(relevant=[]),
            RewrittenQuery(search_query="q2"),
            GradeResult(relevant=[]),
        ]
    )
    state = run(llm, retriever)

    assert retriever.queries == ["Combien de congés ?", "q2"]
    assert state["status"] == "no_answer"
    assert state["answer"] == NO_ANSWER


def test_rewrite_then_answer():
    llm = FakeLLM(
        [
            QueryAnalysis(in_scope=True),
            GradeResult(relevant=[]),
            RewrittenQuery(search_query="q2"),
            GradeResult(relevant=[2]),
            Validation(grounded=True, answers_question=True),
        ],
        answers=["Réponse [1]."],
    )
    assert run(llm)["status"] == "answered"


def test_regenerates_with_validator_feedback():
    llm = FakeLLM(
        [
            QueryAnalysis(in_scope=True),
            GradeResult(relevant=[1]),
            Validation(grounded=False, answers_question=True, feedback="chiffre inventé"),
            Validation(grounded=True, answers_question=True),
        ],
        answers=["Mauvaise réponse.", "Bonne réponse [1]."],
    )
    state = run(llm)

    assert state["status"] == "answered"
    assert state["answer"] == "Bonne réponse [1]."
    assert state["drafts"] == ["Mauvaise réponse.", "Bonne réponse [1]."]
    assert "chiffre inventé" in llm.prompts[-2]  # le retour est transmis au générateur


def test_falls_back_when_validation_keeps_failing():
    llm = FakeLLM(
        [
            QueryAnalysis(in_scope=True),
            GradeResult(relevant=[1]),
            Validation(grounded=True, answers_question=False, feedback="hors sujet"),
            Validation(grounded=True, answers_question=False, feedback="hors sujet"),
        ],
        answers=["Réponse 1", "Réponse 2"],
    )
    state = run(llm)

    assert state["status"] == "unverified"
    assert state["answer"] == UNVERIFIED_ANSWER
    assert len(state["sources"]) == 1


def test_grader_ignores_invalid_indices():
    llm = FakeLLM(
        [
            QueryAnalysis(in_scope=True),
            GradeResult(relevant=[0, 2, 2, 99]),
            Validation(grounded=True, answers_question=True),
        ],
        answers=["ok"],
    )
    assert [s["chunk_id"] for s in run(llm)["sources"]] == ["F1-1"]


def test_llm_errors_are_retried(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    llm = FakeLLM(
        [
            QueryAnalysis(in_scope=True),
            GradeResult(relevant=[1]),
            Validation(grounded=True, answers_question=True),
        ],
        answers=["ok"],
    )
    original = llm.invoke
    failures = iter([True])

    def flaky_invoke(prompt):
        if next(failures, False):
            raise ConnectionError("Server disconnected")
        return original(prompt)

    llm.invoke = flaky_invoke
    assert run(llm)["status"] == "answered"


def test_honest_non_answer_gets_no_answer_status():
    llm = FakeLLM(
        [
            QueryAnalysis(in_scope=True),
            GradeResult(relevant=[1]),
            Validation(grounded=True, answers_question=True, provides_information=False),
        ],
        answers=["Les extraits ne donnent pas cette information."],
    )
    state = run(llm)
    assert state["status"] == "no_answer"
    assert state["answer"] == "Les extraits ne donnent pas cette information."
