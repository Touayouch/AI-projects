"""Métriques calculées à partir des journaux d'expérience.

- Exactitude d'une réponse unique : moyenne sur tous les échantillons (estimateur moins
  bruité que « le premier échantillon »).
- vote@k : exactitude du vote majoritaire sur k échantillons, moyennée sur plusieurs
  tirages aléatoires de k échantillons parmi les N générés (estimateur standard, qui évite
  de dépendre de l'ordre des échantillons).
- Coût : tokens d'entrée + de sortie, par problème, pour chaque k.
"""

import random
import statistics

from debate.aggregation import majority_vote
from debate.grading import equivalent


def vote_accuracy(record: dict, k: int, draws: int = 20, seed: int = 0) -> float:
    """Exactitude espérée du vote sur k échantillons pour un problème (entre 0 et 1)."""
    calls = record["calls"]
    if k > len(calls):
        raise ValueError(f"k={k} > {len(calls)} échantillons disponibles")
    rng = random.Random(seed)
    subsets = [calls] if k == len(calls) else [rng.sample(calls, k) for _ in range(draws)]
    wins = 0
    for subset in subsets:
        voted, _ = majority_vote([c["answer"] for c in subset], record["dataset"])
        wins += equivalent(record["gold"], voted, record["dataset"])
    return wins / len(subsets)


def tokens_per_call(records: list[dict]) -> float:
    return statistics.mean(
        c["prompt_tokens"] + c["completion_tokens"] for r in records for c in r["calls"]
    )


def summarize(records: list[dict], ks: list[int]) -> dict:
    calls = [c for r in records for c in r["calls"]]
    per_call = tokens_per_call(records)
    return {
        "problems": len(records),
        "samples_per_problem": min(len(r["calls"]) for r in records),
        "single_accuracy": statistics.mean(c["correct"] for c in calls),
        "no_answer_rate": statistics.mean(c["answer"] is None for c in calls),
        "truncated_rate": statistics.mean(c["finish_reason"] == "length" for c in calls),
        "tokens_per_call": per_call,
        "vote": {
            k: {
                "accuracy": statistics.mean(vote_accuracy(r, k) for r in records),
                "tokens_per_problem": k * per_call,
            }
            for k in ks
        },
    }


def format_summary(name: str, summary: dict) -> str:
    lines = [
        f"## {name}",
        "",
        f"{summary['problems']} problèmes, {summary['samples_per_problem']} échantillons par "
        f"problème. Réponse unique : **{summary['single_accuracy']:.1%}** — sans réponse "
        f"extraite : {summary['no_answer_rate']:.1%}, tronquées : "
        f"{summary['truncated_rate']:.1%}, {summary['tokens_per_call']:.0f} tokens par appel.",
        "",
        "| k | vote@k | tokens / problème |",
        "| --- | --- | --- |",
    ]
    for k, v in summary["vote"].items():
        lines.append(f"| {k} | {v['accuracy']:.1%} | {v['tokens_per_problem']:.0f} |")
    return "\n".join(lines) + "\n"
