import asyncio

from debate.data import Problem
from debate.experiment import ExperimentConfig, prepare_output, run_experiment
from debate.graph import DebateSettings, build_debate_graph, seed_for
from debate.llm import Completion
from debate.records import read_records

PROBLEM = Problem(id="p0", dataset="gsm8k", question="Q?", answer="18")


class ScriptedClient:
    """Tour 1 : l'agent 0 se trompe (20), les autres trouvent 18.
    Tour 2 : chaque agent adopte la réponse majoritaire qu'il lit chez les autres."""

    def __init__(self, delay: float = 0.0):
        self.requests: list[dict] = []
        self.delay = delay

    async def complete(self, messages, model, temperature=0.7, max_tokens=1024, seed=None):
        self.requests.append({"messages": messages, "model": model, "seed": seed})
        await asyncio.sleep(self.delay)
        if len(messages) == 1:  # tour 1
            answer = "20" if seed % 1000 == 0 else "18"
        else:
            others = messages[-1]["content"]
            answer = "18" if others.count("boxed{18}") >= others.count("boxed{20}") else "20"
        return Completion(
            text=f"model={model} \\boxed{{{answer}}}",
            model=model,
            prompt_tokens=10 * len(messages),
            completion_tokens=20,
            latency_s=self.delay,
            finish_reason="stop",
        )


def settings(**overrides) -> DebateSettings:
    return DebateSettings(**({"models": ["m"] * 3, "n_rounds": 2} | overrides))


async def run(client, **overrides):
    graph = build_debate_graph(client, settings(**overrides))
    return await graph.ainvoke({"problem": PROBLEM})


async def test_debate_rounds_and_final_vote():
    client = ScriptedClient()
    state = await run(client)

    assert len(state["rounds"]) == 2
    assert [c["answer"] for c in state["rounds"][0]] == ["20", "18", "18"]
    assert [c["answer"] for c in state["rounds"][1]] == ["18", "18", "18"]  # l'agent 0 corrigé
    assert [(c["round"], c["agent"]) for c in state["rounds"][1]] == [(2, 0), (2, 1), (2, 2)]
    assert state["final_answer"] == "18"
    assert len(client.requests) == 6


async def test_round_two_sees_own_and_others_previous_answers():
    client = ScriptedClient()
    await run(client)
    agent0_round2 = next(r for r in client.requests if r["seed"] == seed_for(settings(), 2, 0))
    messages = agent0_round2["messages"]

    assert [m["role"] for m in messages] == ["user", "assistant", "user"]
    assert "boxed{20}" in messages[1]["content"]  # sa propre réponse
    assert messages[2]["content"].count("One agent solution") == 2  # les 2 autres agents
    assert "boxed{20}" not in messages[2]["content"]


def test_seeds_match_sampling_in_round_one():
    s = settings(seed=5)
    assert [seed_for(s, 1, i) for i in range(3)] == [5, 6, 7]  # = graines de l'échantillonnage
    assert [seed_for(s, 2, i) for i in range(3)] == [1005, 1006, 1007]


async def test_heterogeneous_models():
    client = ScriptedClient()
    await run(client, models=["a", "b", "c"], n_rounds=1)
    assert sorted(r["model"] for r in client.requests) == ["a", "b", "c"]


async def test_agents_of_a_round_run_in_parallel():
    client = ScriptedClient(delay=0.2)
    start = asyncio.get_running_loop().time()
    await run(client, n_rounds=1, models=["m"] * 5)
    assert asyncio.get_running_loop().time() - start < 0.6  # et non 5 × 0.2 s


async def test_debate_experiment_is_logged(tmp_path):
    config = ExperimentConfig(
        name="deb",
        dataset="gsm8k",
        subset=tmp_path / "s.txt",
        method="debate",
        model="m",
        n_agents=3,
        n_rounds=2,
    )
    out = prepare_output(config, tmp_path)
    await run_experiment(ScriptedClient(), config, [PROBLEM], out)
    record = read_records(out)[0]

    assert record["method"] == "debate"
    assert record["final_answer"] == "18"
    assert len(record["calls"]) == 6
    assert {(c["round"], c["agent"]) for c in record["calls"]} == {
        (r, a) for r in (1, 2) for a in range(3)
    }


def test_agent_models_cycle():
    config = ExperimentConfig(
        name="x", dataset="gsm8k", subset="s", model="m", n_agents=4, models=["a", "b"]
    )
    assert config.agent_models() == ["a", "b", "a", "b"]


async def test_round_one_can_be_reused_from_sampling():
    client = ScriptedClient()
    graph = build_debate_graph(client, settings())
    initial = [
        {
            "text": f"\\boxed{{{a}}}",
            "answer": a,
            "correct": a == "18",
            "model": "m",
            "seed": i,
            "prompt_tokens": 5,
            "completion_tokens": 7,
            "latency_s": 0,
            "finish_reason": "stop",
        }
        for i, a in enumerate(["18", "20", "18"])
    ]
    state = await graph.ainvoke({"problem": PROBLEM, "initial_calls": initial})

    assert len(client.requests) == 3  # seul le tour 2 est généré
    assert [c["answer"] for c in state["rounds"][0]] == ["18", "20", "18"]
    assert all(c["reused"] for c in state["rounds"][0])
    assert state["rounds"][0][0]["prompt_tokens"] == 5  # coût du tour 1 conservé


async def test_debate_experiment_with_initial_answers(tmp_path):
    sampling = ExperimentConfig(
        name="s", dataset="gsm8k", subset=tmp_path / "x.txt", model="m", n_samples=4
    )
    sampling_out = prepare_output(sampling, tmp_path)
    await run_experiment(ScriptedClient(), sampling, [PROBLEM], sampling_out)

    debate = ExperimentConfig(
        name="d",
        dataset="gsm8k",
        subset=tmp_path / "x.txt",
        method="debate",
        model="m",
        n_agents=3,
        n_rounds=2,
        initial_answers=sampling_out,
    )
    client = ScriptedClient()
    await run_experiment(client, debate, [PROBLEM], prepare_output(debate, tmp_path))
    record = read_records(tmp_path / "d.jsonl")[0]

    assert len(client.requests) == 3
    first_round = [c for c in record["calls"] if c["round"] == 1]
    sampled = read_records(sampling_out)[0]["calls"][:3]
    assert [c["text"] for c in first_round] == [c["text"] for c in sampled]


def test_initial_answers_require_same_model(tmp_path):
    import pytest

    from debate.experiment import load_initial_answers

    path = tmp_path / "s.jsonl"
    path.write_text(
        '{"problem_id": "p0", "method": "sampling", "dataset": "gsm8k", '
        '"calls": [{"model": "autre"}, {"model": "autre"}, {"model": "autre"}]}\n'
    )
    config = ExperimentConfig(
        name="d", dataset="gsm8k", subset="x", method="debate", model="m", initial_answers=path
    )
    with pytest.raises(ValueError, match="autre modèle"):
        load_initial_answers(config)
