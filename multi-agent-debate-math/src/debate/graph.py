"""Débat multi-agents orchestré avec LangGraph.

    start ──► dispatch ──Send(agent 0..N-1)──► agent ──► collect ──(tour < R)──► dispatch
                                                            └──(tour = R)──► aggregate

- ``dispatch`` envoie un message ``Send`` par agent : les N agents d'un tour s'exécutent
  en parallèle (fan-out), et leurs réponses sont fusionnées par un reducer (fan-in).
- Tour 1 : chaque agent résout seul le problème (même consigne que l'échantillonnage).
- Tour r > 1 : chaque agent relit sa réponse et celles des autres au tour r-1, puis révise.
- ``aggregate`` : vote majoritaire sur les réponses du dernier tour.

Tour 1 réutilisable : si l'état d'entrée contient ``initial_calls``, les réponses du tour 1
sont reprises d'une expérience d'échantillonnage au lieu d'être régénérées. Débat et vote
partent alors exactement des mêmes réponses initiales (les serveurs regroupant les requêtes
en batch, une même graine ne garantit pas une sortie identique).

Graine de l'agent i : seed + i au tour 1, seed + 1000·(r-1) + i ensuite.
"""

import operator
from dataclasses import dataclass
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from debate.aggregation import majority_vote
from debate.data import Problem
from debate.llm import LLMClient
from debate.prompts import debate_prompt, solve_prompt
from debate.records import call_record


@dataclass(frozen=True)
class DebateSettings:
    models: list[str]  # un modèle par agent (identiques : débat homogène)
    n_rounds: int
    temperature: float = 0.7
    max_tokens: int = 2048
    seed: int = 0

    @property
    def n_agents(self) -> int:
        return len(self.models)


class DebateState(TypedDict, total=False):
    problem: Problem
    initial_calls: list[dict] | None  # réponses du tour 1 reprises de l'échantillonnage
    round: int  # tour en cours (1 à R)
    rounds: list[list[dict]]  # rounds[r-1][i] : appel de l'agent i au tour r
    current: Annotated[list[dict], operator.add]  # réponses du tour en cours (reducer)
    final_answer: str | None


class AgentInput(TypedDict):
    problem: Problem
    round: int
    agent: int
    previous: list[dict] | None  # réponses de tous les agents au tour précédent
    initial: dict | None  # réponse du tour 1 déjà connue pour cet agent


def seed_for(settings: DebateSettings, round_: int, agent: int) -> int:
    return settings.seed + agent if round_ == 1 else settings.seed + 1000 * (round_ - 1) + agent


def build_debate_graph(client: LLMClient, settings: DebateSettings):
    async def agent(inputs: AgentInput) -> DebateState:
        problem, round_, i = inputs["problem"], inputs["round"], inputs["agent"]
        if round_ == 1 and inputs.get("initial") is not None:
            return {"current": [inputs["initial"] | {"agent": i, "round": 1, "reused": True}]}
        if round_ == 1:
            messages = solve_prompt(problem.question)
        else:
            previous = inputs["previous"]
            others = [c["text"] for j, c in enumerate(previous) if j != i]
            messages = debate_prompt(problem.question, previous[i]["text"], others)
        seed = seed_for(settings, round_, i)
        completion = await client.complete(
            messages, settings.models[i], settings.temperature, settings.max_tokens, seed=seed
        )
        record = call_record(completion, seed, problem.answer, problem.dataset)
        return {"current": [record | {"agent": i, "round": round_}]}

    def dispatch(state: DebateState) -> list[Send]:
        previous = state["rounds"][-1] if state.get("rounds") else None
        initial = state.get("initial_calls")
        return [
            Send(
                "agent",
                {
                    "problem": state["problem"],
                    "round": state.get("round", 1),
                    "agent": i,
                    "previous": previous,
                    "initial": initial[i] if initial and not previous else None,
                },
            )
            for i in range(settings.n_agents)
        ]

    def collect(state: DebateState) -> DebateState:
        # L'ordre d'arrivée des réponses parallèles n'est pas garanti : on trie par agent
        this_round = sorted(
            (c for c in state["current"] if c["round"] == state.get("round", 1)),
            key=lambda c: c["agent"],
        )
        return {
            "rounds": [*state.get("rounds", []), this_round],
            "round": len(state.get("rounds", [])) + 2,
        }

    def route(state: DebateState) -> str | list[Send]:
        if len(state["rounds"]) < settings.n_rounds:
            return dispatch(state)
        return "aggregate"

    # async : LangGraph exécute les nœuds synchrones dans un pool de threads, or math-verify
    # (utilisé par le vote sur MATH-500) ne fonctionne que dans le thread principal
    async def aggregate(state: DebateState) -> DebateState:
        problem = state["problem"]
        answer, _ = majority_vote([c["answer"] for c in state["rounds"][-1]], problem.dataset)
        return {"final_answer": answer}

    graph = StateGraph(DebateState)
    graph.add_node("agent", agent)
    graph.add_node("collect", collect)
    graph.add_node("aggregate", aggregate)
    graph.add_conditional_edges(START, dispatch, ["agent"])
    graph.add_edge("agent", "collect")
    graph.add_conditional_edges("collect", route, ["agent", "aggregate"])
    graph.add_edge("aggregate", END)
    return graph.compile()
