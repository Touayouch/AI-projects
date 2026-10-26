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
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

from debate.data import DatasetName, Problem
from debate.llm import LLMClient
from debate.prompts import solve_prompt
from debate.records import call_record, read_records, repair_jsonl

logger = logging.getLogger(__name__)


class ExperimentIncomplete(RuntimeError):
    """Des problèmes ont échoué ; une reprise traitera uniquement ceux-là."""


Solver = Callable[[LLMClient, "ExperimentConfig", Problem], Awaitable[dict]]


class ExperimentConfig(BaseModel):
    name: str
    dataset: DatasetName
    subset: Path
    method: Literal["sampling", "debate"] = "sampling"
    model: str
    # Échantillonnage : N réponses indépendantes
    n_samples: int = Field(default=8, ge=1)
    # Débat : N agents (modèles identiques, ou liste `models` pour un débat hétérogène), R tours
    n_agents: int = Field(default=3, ge=1)
    n_rounds: int = Field(default=2, ge=1)
    models: list[str] | None = None
    # Débat : reprendre le tour 1 des N premiers échantillons d'une expérience de vote
    initial_answers: Path | None = None
    temperature: float = 0.7
    max_tokens: int = 2048
    seed: int = 0

    @classmethod
    def from_yaml(cls, path: Path) -> "ExperimentConfig":
        return cls(**yaml.safe_load(path.read_text()))

    def agent_models(self) -> list[str]:
        """Modèle de chaque agent : `models` répété cycliquement, sinon `model`."""
        pool = self.models or [self.model]
        return [pool[i % len(pool)] for i in range(self.n_agents)]

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.model_dump(mode="json"), sort_keys=False, allow_unicode=True)


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
    out_path = results_dir / f"{config.name}.jsonl"
    if removed := repair_jsonl(out_path):
        logger.warning("%s : dernière ligne incomplète retirée (%d octets)", out_path, removed)
    return out_path


async def sample_problem(client: LLMClient, config: ExperimentConfig, problem: Problem) -> dict:
    """N réponses indépendantes (graines seed, seed+1, ...)."""
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
        "calls": [
            call_record(c, s, problem.answer, problem.dataset)
            for c, s in zip(completions, seeds, strict=True)
        ]
    }


def load_initial_answers(config: ExperimentConfig) -> dict[str, list[dict]]:
    """Réponses du tour 1 par problème : les n_agents premiers échantillons d'un journal
    d'échantillonnage produit avec le même modèle, sur le même benchmark."""
    if len(set(config.agent_models())) > 1:
        raise ValueError("initial_answers n'est possible que pour un débat homogène")
    initial = {}
    for record in read_records(config.initial_answers):
        if record["method"] != "sampling" or record["dataset"] != config.dataset:
            raise ValueError(
                f"{config.initial_answers} n'est pas un échantillonnage {config.dataset}"
            )
        calls = record["calls"][: config.n_agents]
        if len(calls) < config.n_agents or any(c["model"] != config.model for c in calls):
            raise ValueError(
                f"{config.initial_answers} : échantillons insuffisants ou autre modèle"
            )
        initial[record["problem_id"]] = calls
    if not initial:
        raise ValueError(f"{config.initial_answers} est vide")
    return initial


def make_debate_solver(client: LLMClient, config: ExperimentConfig) -> Solver:
    from debate.graph import DebateSettings, build_debate_graph

    graph = build_debate_graph(
        client,
        DebateSettings(
            models=config.agent_models(),
            n_rounds=config.n_rounds,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            seed=config.seed,
        ),
    )

    initial = load_initial_answers(config) if config.initial_answers else {}

    async def solve(client: LLMClient, config: ExperimentConfig, problem: Problem) -> dict:
        state = await graph.ainvoke({"problem": problem, "initial_calls": initial.get(problem.id)})
        return {
            "calls": [call for round_ in state["rounds"] for call in round_],
            "final_answer": state["final_answer"],
        }

    return solve


async def run_experiment(
    client: LLMClient, config: ExperimentConfig, problems: list[Problem], out_path: Path
) -> int:
    """Traite les problèmes pas encore présents dans le journal ; renvoie leur nombre."""
    solve = sample_problem if config.method == "sampling" else make_debate_solver(client, config)
    done = {r["problem_id"] for r in read_records(out_path)}
    todo = [p for p in problems if p.id not in done]
    logger.info("%s : %d problèmes déjà traités, %d à traiter", config.name, len(done), len(todo))

    async def run_one(problem: Problem) -> dict:
        result = await solve(client, config, problem)
        return {
            "experiment": config.name,
            "method": config.method,
            "problem_id": problem.id,
            "dataset": problem.dataset,
            "gold": problem.answer,
            "metadata": problem.metadata,
            **result,
        }

    start = time.perf_counter()
    failures = 0
    with out_path.open("a", encoding="utf-8") as f:
        for i, task in enumerate(asyncio.as_completed([run_one(p) for p in todo]), start=1):
            # Une requête en échec (délai dépassé...) ne doit pas arrêter toute l'expérience :
            # le problème est sauté et sera traité à la reprise.
            try:
                record = await task
            except Exception as exc:
                failures += 1
                logger.warning(
                    "[%d/%d] problème sauté (%s : %s)", i, len(todo), type(exc).__name__, exc
                )
                continue
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            correct = sum(c["correct"] for c in record["calls"])
            logger.info(
                "[%d/%d] %s : %d/%d appels corrects (%.0f s écoulées)",
                i,
                len(todo),
                record["problem_id"],
                correct,
                len(record["calls"]),
                time.perf_counter() - start,
            )
    if failures:
        raise ExperimentIncomplete(f"{config.name} : {failures} problème(s) à reprendre")
    return len(todo)
