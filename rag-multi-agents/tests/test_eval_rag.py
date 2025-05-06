import asyncio

from rag.evaluation.rag import build_report, is_off_topic, read_jsonl, run_pipeline, score_records
from rag.evaluation.retrieval import Question


class FakeGraph:
    def __init__(self, states):
        self.states = states
        self.calls = []

    def invoke(self, inputs):
        self.calls.append(inputs["question"])
        return self.states[inputs["question"]]


class FakeScorer:
    def __init__(self):
        self.calls = []

    async def score(self, question, answer, contexts):
        self.calls.append(answer)
        good = answer.startswith("bonne")
        return {"faithfulness": 1.0 if good else 0.2, "answer_relevancy": 0.9 if good else 0.3}


def state(status, drafts, answer=None):
    return {
        "status": status,
        "answer": answer or (drafts[-1] if drafts else "refus"),
        "drafts": drafts,
        "steps": [],
    }


QUESTIONS = [
    Question("q1", ["F1"], "lexical"),
    Question("q2", ["F2"], "paraphrase"),
    Question("q3", [], "out_of_scope"),
    Question("q4", ["F4"], "lexical"),
]
STATES = {
    "q1": state("answered", ["bonne réponse"]),
    "q2": state("answered", ["mauvaise réponse", "bonne réponse corrigée"]),
    "q3": state("out_of_scope", []),
    "q4": state("unverified", ["mauvaise 1", "mauvaise 2"], answer="sources"),
}


def test_run_pipeline_resumes(tmp_path):
    path = tmp_path / "runs.jsonl"
    graph = FakeGraph(STATES)
    run_pipeline(graph, QUESTIONS[:2], path)
    records = run_pipeline(graph, QUESTIONS, path)

    assert graph.calls == ["q1", "q2", "q3", "q4"]  # q1 et q2 ne sont pas rejouées
    assert [r["status"] for r in records] == ["answered", "answered", "out_of_scope", "unverified"]


def test_score_records_scores_every_draft_once(tmp_path):
    records = run_pipeline(FakeGraph(STATES), QUESTIONS, tmp_path / "runs.jsonl")
    scorer = FakeScorer()
    asyncio.run(score_records(scorer, records, tmp_path / "scores.jsonl"))
    scores = asyncio.run(score_records(scorer, records, tmp_path / "scores.jsonl"))

    assert len(scorer.calls) == 5  # 1 + 2 + 0 + 2 brouillons, pas de re-scoring
    assert set(scores) == {"q1", "q2", "q4"}
    assert len(read_jsonl(tmp_path / "scores.jsonl")) == 3


def test_report_measures_remediation(tmp_path):
    records = run_pipeline(FakeGraph(STATES), QUESTIONS, tmp_path / "runs.jsonl")
    scores = asyncio.run(score_records(FakeScorer(), records, tmp_path / "scores.jsonl"))
    report = build_report(records, scores, "juge")

    # Sans validateur : 3 premiers brouillons dont 2 mauvais ; avec : 2 bonnes réponses
    assert "| Réponses hors sujet / non fidèles renvoyées | 2 | 0 |" in report
    assert "| Fidélité (*faithfulness*) | **1.00** |" in report
    assert "Questions hors périmètre refusées : 1/1" in report
    assert "Questions légitimes refusées à tort comme hors périmètre : 0/3" in report


def test_is_off_topic():
    assert is_off_topic({"faithfulness": 0.9, "answer_relevancy": 0.3})
    assert is_off_topic({"faithfulness": 0.4, "answer_relevancy": 0.9})
    assert not is_off_topic({"faithfulness": 0.9, "answer_relevancy": 0.8})
    assert not is_off_topic(None)
