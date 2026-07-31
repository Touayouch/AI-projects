# Débat multi-agents pour le raisonnement mathématique

Le débat entre plusieurs agents LLM améliore-t-il vraiment le raisonnement mathématique,
**à budget de calcul égal**, par rapport à un simple vote majoritaire ?

**Stack :** Python · vLLM · LangGraph · modèles open-weight (Qwen, Llama, Gemma)

> 🚧 Projet en cours de développement.

## Question de recherche

- **Vote majoritaire** (*self-consistency*, Wang et al., 2022) : k réponses indépendantes,
  on garde la plus fréquente.
- **Débat multi-agents** (Du et al., 2023) : N agents répondent, lisent les réponses des
  autres et révisent la leur pendant R tours, puis vote final.

Le débat coûte beaucoup plus de tokens (chaque agent relit les autres à chaque tour). Ce
projet compare les deux méthodes **à nombre de tokens équivalent**, sur GSM8K et MATH-500,
en faisant varier le nombre d'agents, le nombre de tours et la diversité des modèles, et
analyse l'évolution des réponses au fil du débat.

## Roadmap

- [x] Phase 0 : setup (client LLM compatible OpenAI, configuration, CI)
- [ ] Phase 1 : données (GSM8K, MATH-500) et vérification des réponses
- [ ] Phase 2 : réponse unique, vote majoritaire, comptage des tokens
- [ ] Phase 3 : débat avec LangGraph, comparaison à budget égal
- [ ] Phase 4 : expériences sur GPU (vLLM, Kaggle)
- [ ] Phase 5 : analyse (courbes exactitude / tokens, dynamique du débat, tests statistiques)
- [ ] Phase 6 : rapport final

## Calcul

| | Où | Serveur | Modèles |
| --- | --- | --- | --- |
| Développement | CPU local | Ollama | Qwen2.5-1.5B-Instruct |
| Expériences | Kaggle (GPU T4, gratuit) | vLLM | modèles open-weight 1,5 à 3B |

Ollama et vLLM exposent tous deux une API compatible OpenAI : le même code tourne dans les
deux environnements, seule `LLM_BASE_URL` change.

## Installation

Prérequis : Python 3.11+, [uv](https://docs.astral.sh/uv/), [Ollama](https://ollama.com)
pour le développement local.

```bash
cp .env.example .env
make install
make models                         # télécharge Qwen2.5-1.5B-Instruct pour Ollama
make ping                           # vérifie la connexion au serveur LLM
make test
```

## Licence

MIT
