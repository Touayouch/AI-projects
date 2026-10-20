"""Consignes données aux modèles.

Consigne standard des modèles Qwen pour les mathématiques : raisonnement pas à pas et
réponse finale dans \\boxed{}, ce qui permet une extraction fiable (voir grading.py).
"""

SOLVE = "{question}\n\nPlease reason step by step, and put your final answer within \\boxed{{}}."


def solve_prompt(question: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": SOLVE.format(question=question)}]


# Protocole de Du et al. (2023), adapté : à partir du tour 2, chaque agent voit sa propre
# réponse du tour précédent et celles des autres agents, puis révise sa réponse.
DEBATE_UPDATE = (
    "These are the solutions to the problem from other agents:\n\n{others}\n\n"
    "Using the reasoning from other agents as additional information, examine your solution "
    "and theirs step by step, and give an updated answer. Please reason step by step, and "
    "put your final answer within \\boxed{{}}."
)
OTHER_AGENT = "One agent solution:\n```\n{text}\n```"


def debate_prompt(question: str, own_previous: str, others_previous: list[str]) -> list[dict]:
    """Conversation d'un agent au tour r > 1 : seul le tour précédent est conservé, pour que
    la longueur du contexte ne croisse pas avec le nombre de tours."""
    others = "\n\n".join(OTHER_AGENT.format(text=t) for t in others_previous)
    return [
        *solve_prompt(question),
        {"role": "assistant", "content": own_previous},
        {"role": "user", "content": DEBATE_UPDATE.format(others=others)},
    ]
