import pytest

from debate.metrics import accuracy_at_budget, compare_at_budget, debate_summary


def call(answer, correct, round_=1, agent=0, prompt=10, completion=90):
    return {
        "answer": answer,
        "correct": correct,
        "round": round_,
        "agent": agent,
        "prompt_tokens": prompt,
        "completion_tokens": completion,
    }


def debate_record(pid, round1, round2, final):
    calls = [call(a, a == "18", 1, i) for i, a in enumerate(round1)]
    calls += [call(a, a == "18", 2, i, prompt=300) for i, a in enumerate(round2)]
    return {
        "problem_id": pid,
        "dataset": "gsm8k",
        "gold": "18",
        "method": "debate",
        "final_answer": final,
        "calls": calls,
    }


def sampling_record(pid, answers):
    return {
        "problem_id": pid,
        "dataset": "gsm8k",
        "gold": "18",
        "method": "sampling",
        "calls": [call(a, a == "18") for a in answers],
    }


def test_accuracy_at_budget_interpolates():
    curve = [(100, 0.5), (200, 0.6), (300, 0.8)]
    assert accuracy_at_budget(curve, 150) == pytest.approx(0.55)
    assert accuracy_at_budget(curve, 300) == 0.8
    assert accuracy_at_budget(curve, 50) == 0.5
    assert accuracy_at_budget(curve, 301) is None  # pas d'extrapolation


def test_debate_summary_transitions_and_rounds():
    records = [
        debate_record("a", ["20", "18", "18"], ["18", "18", "18"], "18"),
        debate_record("b", ["18", "20", "20"], ["20", "20", "20"], "20"),
    ]
    s = debate_summary(records)

    assert (s["agents"], s["rounds"]) == (3, 2)
    assert s["final_accuracy"] == 0.5
    assert s["per_round"][0]["vote_accuracy"] == 0.5
    assert s["per_round"][0]["agent_accuracy"] == pytest.approx(3 / 6)
    assert s["per_round"][1]["unanimity"] == 1.0
    assert s["transitions"][0] == {
        "correct→correct": 2,
        "correct→faux": 1,
        "faux→correct": 1,
        "faux→faux": 2,
    }
    assert s["tokens_per_problem"] == 3 * 100 + 3 * 390


def test_compare_at_budget():
    debate = [debate_record("a", ["18"] * 3, ["18"] * 3, "18")]
    sampling = [sampling_record("a", ["18", "20", "18", "18", "20", "18", "18", "18", "18", "18"])]
    comparison = compare_at_budget(debate, sampling)

    # débat : 1470 tokens = 14.7 appels d'échantillonnage (100 tokens) -> au-delà de k = 10
    assert comparison["equivalent_k_total"] == pytest.approx(14.7)
    assert comparison["vote_at_total_budget"] is None
    # tokens générés : 540 = 6 appels de 90 tokens -> k = 6
    assert comparison["equivalent_k_completion"] == pytest.approx(6)
    assert comparison["vote_at_completion_budget"] is not None


def test_compare_requires_same_problems():
    debate = [debate_record("a", ["18"] * 3, ["18"] * 3, "18")]
    with pytest.raises(ValueError, match="mêmes problèmes"):
        compare_at_budget(debate, [sampling_record("b", ["18"])])
