# Système RAG multi-agents sur documents métier

Pipeline RAG multi-étapes orchestré par des agents spécialisés (ingestion, chunking sémantique,
retrieval hybride dense/sparse, validation qualité, génération), évalué automatiquement avec RAGAS.

**Stack :** LangChain · LangGraph · ChromaDB · FastAPI

> 🚧 Projet en cours de développement.

## Roadmap

- [x] Phase 0 : setup du projet
- [ ] Phase 1 : ingestion et chunking sémantique
- [ ] Phase 2 : retrieval hybride (dense + BM25 + fusion RRF)
- [ ] Phase 3 : orchestration multi-agents (LangGraph)
- [ ] Phase 4 : évaluation RAGAS et remédiation hors-sujet
- [ ] Phase 5 : API FastAPI et Docker
- [ ] Phase 6 : documentation, démo

## Structure

```
src/rag/
├── agents/      # agents spécialisés
├── retrieval/   # dense, sparse, fusion
├── api/         # FastAPI
└── config.py    # configuration (.env)
eval/            # jeu de test et évaluation RAGAS
tests/
```

## Installation

Prérequis : Python 3.11+, [uv](https://docs.astral.sh/uv/), Docker.

```bash
cp .env.example .env      # puis renseigner la clé API
make install
make chroma-up            # lance ChromaDB sur le port 8001
make test
make api                  # http://localhost:8080/health
```

## Licence

MIT
