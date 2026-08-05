"""Consignes données aux modèles.

Consigne standard des modèles Qwen pour les mathématiques : raisonnement pas à pas et
réponse finale dans \\boxed{}, ce qui permet une extraction fiable (voir grading.py).
"""

SOLVE = "{question}\n\nPlease reason step by step, and put your final answer within \\boxed{{}}."


def solve_prompt(question: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": SOLVE.format(question=question)}]
