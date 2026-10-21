import asyncio
import json

import httpx

from debate import cli
from debate.config import Settings
from debate.data import Problem
from debate.experiment import ExperimentConfig
from debate.llm import LLMClient
from debate.records import read_records

PROBLEMS = [Problem(id=f"p{i}", dataset="gsm8k", question=f"Q{i}", answer="18") for i in range(4)]


async def handler(request: httpx.Request) -> httpx.Response:
    await asyncio.sleep(0.005)  # crée de l'attente sur le sémaphore
    return httpx.Response(
        200,
        json={
            "id": "x",
            "object": "chat.completion",
            "created": 0,
            "model": json.loads(request.content)["model"],
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "\\boxed{18}"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},
        },
    )


def test_stage_runs_several_experiments_in_a_row(tmp_path, monkeypatch):
    """Régression : un client réutilisé entre deux asyncio.run faisait échouer le 2e
    ("Semaphore is bound to a different event loop")."""
    settings = Settings(_env_file=None, results_dir=tmp_path / "results", data_dir=tmp_path)
    monkeypatch.setattr(cli, "get_settings", lambda: settings)
    monkeypatch.setattr(cli, "load_subset", lambda *args: PROBLEMS)
    monkeypatch.setattr(
        cli,
        "make_client",
        lambda endpoints=None: LLMClient(
            "http://test/v1",
            max_concurrency=2,
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        ),
    )
    sampling = ExperimentConfig(
        name="s", dataset="gsm8k", subset=tmp_path / "x.txt", model="m", n_samples=4
    )
    debate = ExperimentConfig(
        name="d",
        dataset="gsm8k",
        subset=tmp_path / "x.txt",
        method="debate",
        model="m",
        n_agents=3,
        n_rounds=2,
        initial_answers=tmp_path / "results" / "s.jsonl",
    )
    paths = []
    for config in (sampling, debate):
        path = tmp_path / f"{config.name}.yaml"
        path.write_text(config.to_yaml())
        paths.append(path)

    cli.run_configs(paths)

    assert len(read_records(tmp_path / "results" / "s.jsonl")) == 4
    assert len(read_records(tmp_path / "results" / "d.jsonl")) == 4


def test_report_on_empty_results(tmp_path, capsys):
    import argparse

    empty = tmp_path / "vide.jsonl"
    empty.write_text("")
    cli.cmd_report(argparse.Namespace(results=empty, ks=None))
    assert "aucun résultat" in capsys.readouterr().out
