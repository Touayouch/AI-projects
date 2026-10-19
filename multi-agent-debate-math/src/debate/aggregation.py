"""Vote majoritaire sur des réponses extraites.

Les réponses équivalentes sont regroupées avant le vote (``0.5`` et ``\\frac{1}{2}``
comptent comme une seule réponse) avec le même vérificateur que pour la correction.
En cas d'égalité, la réponse apparue en premier l'emporte : le résultat est déterministe.
"""

from debate.grading import equivalent


def majority_vote(answers: list[str | None], dataset: str) -> tuple[str | None, int]:
    """Renvoie (réponse majoritaire, nombre de voix) ; (None, 0) si aucune réponse."""
    clusters: list[list[str]] = []
    for answer in answers:
        if answer is None:
            continue
        for cluster in clusters:
            if equivalent(cluster[0], answer, dataset):
                cluster.append(answer)
                break
        else:
            clusters.append([answer])
    if not clusters:
        return None, 0
    best = max(clusters, key=len)  # max garde le premier groupe en cas d'égalité
    return best[0], len(best)
