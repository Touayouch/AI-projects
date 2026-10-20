import pytest

from debate.data import Problem
from debate.experiment import ExperimentConfig, prepare_output, run_experiment
from debate.llm import Completion
from debate.metrics import summarize, vote_accuracy
from debate.records import read_records


class FakeClient:
    """Renvoie une réponse déterminée par la graine : graines paires -> bonne réponse."""

    def __init__(self):
        self.calls = []

    async def complete(self, messages, model, temperature=0.7, max_tokens=1024, seed=None):
        self.calls.append(seed)
        answer = "18" if seed % 2 == 0 else "20"
        return Completion(
            text=f"Reasoning... \\boxed{{{answer}}}",
            model=model,
            prompt_tokens=50,
            completion_tokens=100,
            latency_s=0.1,
            finish_reason="stop",
        )


PROBLEMS = [Problem(id=f"p{i}", dataset="gsm8k", question=f"Q{i}", answer="18") for i in range(3)]


def config(tmp_path, **overrides) -> ExperimentConfig:
    values = dict(
        name="exp", dataset="gsm8k", subset=tmp_path / "s.txt", model="m", n_samples=3, seed=0
    )
    return ExperimentConfig(**(values | overrides))


async def test_run_sampling_logs_every_call(tmp_path):
    cfg = config(tmp_path)
    out = prepare_output(cfg, tmp_path)
    client = FakeClient()
    await run_experiment(client, cfg, PROBLEMS, out)

    records = read_records(out)
    assert sorted(r["problem_id"] for r in records) == ["p0", "p1", "p2"]
    calls = records[0]["calls"]
    assert [c["seed"] for c in calls] == [0, 1, 2]
    assert [c["answer"] for c in calls] == ["18", "20", "18"]
    assert [c["correct"] for c in calls] == [True, False, True]
    assert calls[0]["prompt_tokens"] == 50


async def test_run_sampling_resumes(tmp_path):
    cfg = config(tmp_path)
    out = prepare_output(cfg, tmp_path)
    await run_experiment(FakeClient(), cfg, PROBLEMS[:2], out)
    client = FakeClient()
    assert await run_experiment(client, cfg, PROBLEMS, out) == 1
    assert len(client.calls) == 3  # seul p2 est traité
    assert len(read_records(out)) == 3


def test_changed_config_is_refused(tmp_path):
    prepare_output(config(tmp_path), tmp_path)
    prepare_output(config(tmp_path), tmp_path)  # identique : accepté
    with pytest.raises(ValueError, match="différente"):
        prepare_output(config(tmp_path, n_samples=5), tmp_path)


async def test_metrics(tmp_path):
    cfg = config(tmp_path)
    out = prepare_output(cfg, tmp_path)
    await run_experiment(FakeClient(), cfg, PROBLEMS, out)
    records = read_records(out)

    assert vote_accuracy(records[0], 3) == 1.0  # 2 voix sur 3 pour la bonne réponse
    summary = summarize(records, ks=[1, 3])
    assert summary["single_accuracy"] == pytest.approx(2 / 3)
    assert summary["vote"][1]["accuracy"] == pytest.approx(2 / 3, abs=0.2)
    assert summary["vote"][3]["accuracy"] == 1.0
    assert summary["vote"][3]["tokens_per_problem"] == 3 * 150
    assert summary["no_answer_rate"] == 0
