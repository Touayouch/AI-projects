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


# --- Débat ------------------------------------------------------------------------------

TRANSITIONS = ("correct→correct", "correct→faux", "faux→correct", "faux→faux")


def _rounds(record: dict) -> list[list[dict]]:
    n_rounds = max(c["round"] for c in record["calls"])
    return [
        sorted((c for c in record["calls"] if c["round"] == r), key=lambda c: c["agent"])
        for r in range(1, n_rounds + 1)
    ]


def _all_agree(calls: list[dict], dataset: str) -> bool:
    _, votes = majority_vote([c["answer"] for c in calls], dataset)
    return votes == len(calls)


def debate_summary(records: list[dict]) -> dict:
    """Exactitude finale, par tour, transitions entre tours et coût d'un débat."""
    rounds_per_record = [_rounds(r) for r in records]
    n_rounds = len(rounds_per_record[0])
    per_round = []
    for r in range(n_rounds):
        vote_correct, agent_correct, unanimous = [], [], []
        for record, rounds in zip(records, rounds_per_record, strict=True):
            calls = rounds[r]
            voted, _ = majority_vote([c["answer"] for c in calls], record["dataset"])
            vote_correct.append(equivalent(record["gold"], voted, record["dataset"]))
            agent_correct.extend(c["correct"] for c in calls)
            unanimous.append(_all_agree(calls, record["dataset"]))
        per_round.append(
            {
                "vote_accuracy": statistics.mean(vote_correct),
                "agent_accuracy": statistics.mean(agent_correct),
                "unanimity": statistics.mean(unanimous),
            }
        )

    transitions = []
    for r in range(1, n_rounds):
        counts = dict.fromkeys(TRANSITIONS, 0)
        for rounds in rounds_per_record:
            for before, after in zip(rounds[r - 1], rounds[r], strict=True):
                key = (
                    f"{'correct' if before['correct'] else 'faux'}→"
                    f"{'correct' if after['correct'] else 'faux'}"
                )
                counts[key] += 1
        transitions.append(counts)

    tokens = [sum(c["prompt_tokens"] + c["completion_tokens"] for c in r["calls"]) for r in records]
    completion = [sum(c["completion_tokens"] for c in r["calls"]) for r in records]
    return {
        "problems": len(records),
        "agents": len(rounds_per_record[0][0]),
        "rounds": n_rounds,
        "final_accuracy": statistics.mean(
            equivalent(r["gold"], r["final_answer"], r["dataset"]) for r in records
        ),
        "per_round": per_round,
        "transitions": transitions,
        "tokens_per_problem": statistics.mean(tokens),
        "completion_tokens_per_problem": statistics.mean(completion),
    }


# --- Comparaison à budget égal ---------------------------------------------------------


def vote_curve(records: list[dict], ks: list[int], completion_only: bool = False) -> list[tuple]:
    """Points (tokens par problème, exactitude du vote) pour chaque k."""
    calls = [c for r in records for c in r["calls"]]
    per_call = statistics.mean(
        c["completion_tokens"] + (0 if completion_only else c["prompt_tokens"]) for c in calls
    )
    return [(k * per_call, statistics.mean(vote_accuracy(r, k) for r in records)) for k in ks]


def accuracy_at_budget(curve: list[tuple], budget: float) -> float | None:
    """Interpolation linéaire de l'exactitude du vote à un budget donné ; None si le budget
    dépasse le plus grand k disponible (pas d'extrapolation)."""
    if budget < curve[0][0]:
        return curve[0][1]
    for (t0, a0), (t1, a1) in zip(curve, curve[1:], strict=False):
        if t0 <= budget <= t1:
            return a0 + (a1 - a0) * (budget - t0) / (t1 - t0)
    return curve[-1][1] if budget == curve[-1][0] else None


def compare_at_budget(debate_records: list[dict], sampling_records: list[dict]) -> dict:
    debate = debate_summary(debate_records)
    debate_ids = {r["problem_id"] for r in debate_records}
    sampling = [r for r in sampling_records if r["problem_id"] in debate_ids]
    if len(sampling) != len(debate_records):
        raise ValueError("Les deux expériences ne portent pas sur les mêmes problèmes")
    ks = list(range(1, min(len(r["calls"]) for r in sampling) + 1))
    total_curve = vote_curve(sampling, ks)
    completion_curve = vote_curve(sampling, ks, completion_only=True)
    return {
        "debate": debate,
        "vote_at_total_budget": accuracy_at_budget(total_curve, debate["tokens_per_problem"]),
        "vote_at_completion_budget": accuracy_at_budget(
            completion_curve, debate["completion_tokens_per_problem"]
        ),
        "equivalent_k_total": debate["tokens_per_problem"] / total_curve[0][0],
        "equivalent_k_completion": debate["completion_tokens_per_problem"] / completion_curve[0][0],
        "max_k": ks[-1],
    }


def _pct(value: float | None) -> str:
    return "au-delà des k disponibles" if value is None else f"{value:.1%}"


def format_debate(name: str, summary: dict) -> str:
    lines = [
        f"## {name}",
        "",
        f"{summary['problems']} problèmes, {summary['agents']} agents, {summary['rounds']} "
        f"tours. Exactitude finale (vote au dernier tour) : **{summary['final_accuracy']:.1%}** "
        f"— {summary['tokens_per_problem']:.0f} tokens par problème "
        f"({summary['completion_tokens_per_problem']:.0f} générés).",
        "",
        "| Tour | vote des agents | agent moyen | unanimité |",
        "| --- | --- | --- | --- |",
    ]
    for r, s in enumerate(summary["per_round"], start=1):
        lines.append(
            f"| {r} | {s['vote_accuracy']:.1%} | {s['agent_accuracy']:.1%} | {s['unanimity']:.1%} |"
        )
    for r, counts in enumerate(summary["transitions"], start=1):
        lines.append("")
        lines.append(
            f"Transitions tour {r} → {r + 1} : " + ", ".join(f"{k} {v}" for k, v in counts.items())
        )
    return "\n".join(lines) + "\n"


def format_comparison(comparison: dict) -> str:
    d = comparison["debate"]
    return (
        "## Débat vs vote majoritaire à budget égal\n\n"
        "| | Exactitude | Budget |\n| --- | --- | --- |\n"
        f"| Débat ({d['agents']} agents × {d['rounds']} tours) | **{d['final_accuracy']:.1%}** "
        f"| {d['tokens_per_problem']:.0f} tokens |\n"
        f"| Vote, même budget total (k ≈ {comparison['equivalent_k_total']:.1f}) | "
        f"{_pct(comparison['vote_at_total_budget'])} | {d['tokens_per_problem']:.0f} tokens |\n"
        f"| Vote, même budget généré (k ≈ {comparison['equivalent_k_completion']:.1f}) | "
        f"{_pct(comparison['vote_at_completion_budget'])} | "
        f"{d['completion_tokens_per_problem']:.0f} tokens générés |\n\n"
        f"Courbe du vote disponible jusqu'à k = {comparison['max_k']}.\n"
    )
