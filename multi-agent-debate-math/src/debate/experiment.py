"""Exécution des expériences et journalisation de chaque appel au LLM.

Une expérience est décrite par un fichier YAML (``configs/experiments/``). Ses résultats
sont écrits dans ``results/<nom>.jsonl``, un enregistrement par problème, contenant
TOUS les appels (texte, réponse extraite, tokens, latence) : toutes les métriques
(vote@k, coût en tokens, comparaison à budget égal) se recalculent sans relancer le modèle.

L'écriture est reprenable : les problèmes déjà présents dans le fichier sont sautés, et la
configuration est enregistrée à côté (``<nom>.config.yaml``) pour refuser une reprise avec
des paramètres différents.
"""

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

from debate.data import DatasetName, Problem
from debate.grading import equivalent, extract_answer
from debate.llm import Completion, LLMClient
from debate.prompts import solve_prompt

logger = logging.getLogger(__name__)


class ExperimentConfig(BaseModel):
    name: str
    dataset: DatasetName
    subset: Path
    method: Literal["sampling"] = "sampling"
    model: str
    n_samples: int = Field(default=8, ge=1)
    temperature: float = 0.7
    max_tokens: int = 2048
    seed: int = 0

    @classmethod
    def from_yaml(cls, path: Path) -> "ExperimentConfig":
        return cls(**yaml.safe_load(path.read_text()))

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.model_dump(mode="json"), sort_keys=False, allow_unicode=True)


def call_record(completion: Completion, seed: int, gold: str, dataset: str) -> dict:
    answer = extract_answer(completion.text, dataset)
    return {
        "model": completion.model,
        "seed": seed,
        "text": completion.text,
        "answer": answer,
        "correct": equivalent(gold, answer, dataset),
        "prompt_tokens": completion.prompt_tokens,
        "completion_tokens": completion.completion_tokens,
        "latency_s": completion.latency_s,
        "finish_reason": completion.finish_reason,
    }


def read_records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def prepare_output(config: ExperimentConfig, results_dir: Path) -> Path:
    """Crée ou vérifie le dossier de résultats ; renvoie le chemin du JSONL."""
    results_dir.mkdir(parents=True, exist_ok=True)
    config_path = results_dir / f"{config.name}.config.yaml"
    if config_path.exists():
        saved = ExperimentConfig(**yaml.safe_load(config_path.read_text()))
        if saved != config:
            raise ValueError(
                f"{config_path} contient une configuration différente : changer le nom de "
                "l'expérience ou supprimer ses résultats"
            )
    else:
        config_path.write_text(config.to_yaml())
    return results_dir / f"{config.name}.jsonl"


async def run_sampling(
    client: LLMClient, config: ExperimentConfig, problems: list[Problem], out_path: Path
) -> int:
    """N réponses indépendantes par problème (graines seed, seed+1, ...)."""
    done = {r["problem_id"] for r in read_records(out_path)}
    todo = [p for p in problems if p.id not in done]
    logger.info("%s : %d problèmes déjà traités, %d à traiter", config.name, len(done), len(todo))

    async def solve(problem: Problem) -> dict:
        seeds = [config.seed + i for i in range(config.n_samples)]
        completions = await asyncio.gather(
            *(
                client.complete(
                    solve_prompt(problem.question),
                    config.model,
                    config.temperature,
                    config.max_tokens,
                    seed=seed,
                )
                for seed in seeds
            )
        )
        return {
            "experiment": config.name,
            "method": config.method,
            "problem_id": problem.id,
            "dataset": problem.dataset,
            "gold": problem.answer,
            "metadata": problem.metadata,
            "calls": [
                call_record(c, s, problem.answer, problem.dataset)
                for c, s in zip(completions, seeds, strict=True)
            ],
        }

    start = time.perf_counter()
    with out_path.open("a", encoding="utf-8") as f:
        for i, task in enumerate(asyncio.as_completed([solve(p) for p in todo]), start=1):
            record = await task
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            correct = sum(c["correct"] for c in record["calls"])
            logger.info(
                "[%d/%d] %s : %d/%d échantillons corrects (%.0f s écoulées)",
                i,
                len(todo),
                record["problem_id"],
                correct,
                len(record["calls"]),
                time.perf_counter() - start,
            )
    return len(todo)
