# Évaluation RAGAS du système multi-agents

100 questions (43 paraphrase, 42 lexical, 8 out_of_scope, 7 unanswerable). Juge RAGAS : `qwen3-instruct:4b` (local, via Ollama).

## Qualité des réponses (RAGAS)

Sur les 85 réponses fournies par le système (statut `answered`) :

| Métrique | Score moyen |
| --- | --- |
| Fidélité (*faithfulness*) | **0.91** |
| Pertinence (*answer relevancy*) | **0.77** |

## Détection et remédiation des réponses hors sujet

Une réponse est comptée *hors sujet / non fidèle* si sa pertinence < 0.5 ou sa fidélité < 0.5.

| | Sans validateur (1er brouillon) | Avec validateur + remédiation |
| --- | --- | --- |
| Réponses renvoyées | 89 | 85 |
| Fidélité moyenne | 0.89 | 0.91 |
| Pertinence moyenne | 0.73 | 0.77 |
| Réponses hors sujet / non fidèles renvoyées | 9 | 5 |

- Réponses rejetées par le validateur puis régénérées : 1
- Réponses rejetées deux fois, remplacées par les sources (`unverified`) : 0

## Routage

| Type | Questions | answered | no_answer | unverified | out_of_scope |
| --- | --- | --- | --- | --- | --- |
| lexical | 42 | 41 | 1 | 0 | 0 |
| paraphrase | 43 | 43 | 0 | 0 | 0 |
| out_of_scope | 8 | 0 | 0 | 0 | 8 |
| unanswerable | 7 | 1 | 4 | 0 | 2 |

- Questions hors périmètre refusées : 8/8
- Questions sans réponse dans le corpus où le système s'abstient : 6/7
- Questions légitimes refusées à tort comme hors périmètre : 0/85
- Retrieval (hit@5, questions avec réponse) : 0.88

## Latence (CPU, sans GPU)

- Médiane : 87 s par question avec réponse
- Moyenne : 93 s
