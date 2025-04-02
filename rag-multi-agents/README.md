# Système RAG multi-agents sur documents métier

Pipeline RAG multi-étapes orchestré par des agents spécialisés (ingestion, chunking sémantique,
retrieval hybride dense/sparse, validation qualité, génération), évalué automatiquement avec RAGAS.

**Stack :** LangChain · LangGraph · ChromaDB · FastAPI · Ollama

> 🚧 Projet en cours de développement.

## Corpus

**875 fiches pratiques** de [service-public.gouv.fr](https://www.service-public.gouv.fr)
(thèmes *Travail - Formation* et *Social - Santé*) : congés, contrat de travail, licenciement,
chômage, retraite, santé... Données ouvertes publiées par la DILA sous
[Licence Ouverte](https://www.etalab.gouv.fr/licence-ouverte-open-licence/).

Un corpus juridique riche en termes exacts (CDD, CSE, RSA, article L1234-9...), idéal pour
mettre en évidence l'intérêt d'un retrieval hybride dense + lexical.

## Roadmap

- [x] Phase 0 : setup du projet
- [x] Phase 1 : ingestion et chunking sémantique
- [ ] Phase 2 : retrieval hybride (dense + BM25 + fusion RRF)
- [ ] Phase 3 : orchestration multi-agents (LangGraph)
- [ ] Phase 4 : évaluation RAGAS et remédiation hors-sujet
- [ ] Phase 5 : API FastAPI et Docker
- [ ] Phase 6 : documentation, démo

## Pipeline d'ingestion

```
fiches XML ──► IngestionAgent ──► ChunkingAgent ──► ChromaDB (dense)
               parsing XML→Markdown  sections (titres)     + chunks.jsonl (BM25, phase 2)
               nettoyage, dédup       + regroupement sémantique
```

- **IngestionAgent** : convertit la structure XML (chapitres, cas, encadrés, tableaux) en
  Markdown hiérarchique, extrait les métadonnées (thème, URL, date) et déduplique par hash.
- **ChunkingAgent** : découpe d'abord selon la structure (titres), puis regroupe les blocs
  consécutifs d'une section tant qu'ils restent sémantiquement proches (rupture au-delà du
  90e percentile des distances cosinus entre blocs voisins). Chaque chunk est préfixé par son
  fil d'Ariane (`Fiche > Chapitre > Cas`) pour rester compréhensible isolément.
  Une stratégie `recursive` (baseline) permet de comparer les deux approches.

## Structure

```
src/rag/
├── agents/        # agents spécialisés (ingestion, chunking...)
├── ingestion/     # téléchargement du corpus, parsing XML
├── retrieval/     # dense, sparse, fusion (phase 2)
├── api/           # FastAPI
├── indexing.py    # ChromaDB + export JSONL
├── models.py      # embeddings / LLM (Ollama)
├── cli.py         # commande `rag`
└── config.py      # configuration (.env)
eval/              # jeu de test et évaluation RAGAS
tests/
```

## Installation

Prérequis : Python 3.11+, [uv](https://docs.astral.sh/uv/), [Ollama](https://ollama.com).

```bash
cp .env.example .env
make install
make models             # télécharge bge-m3 (embeddings) et qwen3:4b (LLM)
make corpus             # télécharge et filtre les fiches dans data/raw/
make ingest             # ingestion + chunking sémantique + indexation ChromaDB
make ingest-recursive   # baseline pour comparaison
make test
```

Les statistiques de chunking sont écrites dans `data/processed/stats_<stratégie>.json`.

## Licence

Code sous licence MIT. Corpus : Licence Ouverte / Open Licence (DILA).
