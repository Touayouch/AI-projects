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
- [x] Phase 1 : données (GSM8K, MATH-500) et vérification des réponses
- [ ] Phase 2 : réponse unique, vote majoritaire, comptage des tokens
- [ ] Phase 3 : débat avec LangGraph, comparaison à budget égal
- [ ] Phase 4 : expériences sur GPU (vLLM, Kaggle)
- [ ] Phase 5 : analyse (courbes exactitude / tokens, dynamique du débat, tests statistiques)
- [ ] Phase 6 : rapport final

## Données et vérification des réponses

| Benchmark | Source | Problèmes évalués |
| --- | --- | --- |
| **GSM8K** | [openai/grade-school-math](https://github.com/openai/grade-school-math) (test) | 500 tirés au hasard parmi 1 319 (graine 0) |
| **MATH-500** | [HuggingFaceH4/MATH-500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500) | les 500 |

Les sous-ensembles sont **figés** dans [configs/subsets/](configs/subsets/) (listes
d'identifiants versionnées) : toutes les méthodes sont évaluées sur exactement les mêmes
problèmes. Des sous-ensembles `*_dev_30` servent uniquement à mettre au point le code.

**Vérification des réponses** ([src/debate/grading.py](src/debate/grading.py)), commune à
toutes les méthodes :
- le modèle reçoit la consigne standard *« reason step by step, and put your final answer
  within \boxed{} »* ; on extrait le dernier `\boxed{...}` (accolades imbriquées gérées) ;
- **GSM8K** : comparaison numérique (`\$1,234.00` = `1234`), et à défaut de `\boxed` le
  dernier nombre du texte ;
- **MATH-500** : équivalence mathématique avec
  [math-verify](https://github.com/huggingface/Math-Verify) : `\frac{1}{2}` = `0.5`,
  `2\sqrt{3}` = `\sqrt{12}`, `x=5` = `5`, `(x+1)^2` = `x^2+2x+1`, mais `[-2,7]` ≠ `(-2,7)`.

**Validation du vérificateur** : les 1 319 solutions de référence de GSM8K et les 500 de
MATH-500 sont toutes jugées correctes (`make check-grader`), et 27 cas d'équivalence et de
pièges sont couverts par les tests.

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
make check-grader                   # télécharge les données et valide le vérificateur
make test
```

## Licence

MIT
