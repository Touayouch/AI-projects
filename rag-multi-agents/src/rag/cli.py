"""Ligne de commande du projet.

rag download                    # télécharge et filtre le corpus
rag ingest --strategy semantic  # ingestion -> chunking -> indexation
"""

import argparse
import json
import logging
import statistics
import time
from pathlib import Path

from rag.agents.chunking import ChunkingAgent
from rag.agents.ingestion import IngestionAgent
from rag.config import get_settings
from rag.evaluation.retrieval import evaluate, load_questions, report_table
from rag.graph import GraphConfig, build_graph
from rag.indexing import get_vectorstore, index_chunks, save_chunks
from rag.ingestion.corpus import download_corpus
from rag.models import get_embeddings, get_llm
from rag.retrieval.factory import build_retrievers

logger = logging.getLogger("rag")


def cmd_download(args: argparse.Namespace) -> None:
    settings = get_settings()
    download_corpus(settings.corpus_url, settings.data_dir, settings.corpus_themes)


def cmd_ingest(args: argparse.Namespace) -> None:
    settings = get_settings()
    strategy = args.strategy or settings.chunk_strategy

    documents, report = IngestionAgent(settings.data_dir).run(limit=args.limit)
    if not documents:
        raise SystemExit(f"Aucun document dans {settings.data_dir}, lancer `rag download`")

    embeddings = get_embeddings(settings)
    chunker = ChunkingAgent(
        strategy=strategy,
        embeddings=embeddings,
        max_chars=settings.chunk_max_chars,
        min_chars=settings.chunk_min_chars,
        breakpoint_percentile=settings.chunk_breakpoint_percentile,
    )
    chunks = chunker.run(documents)

    settings.processed_dir.mkdir(parents=True, exist_ok=True)
    save_chunks(chunks, settings.processed_dir / f"chunks_{strategy}.jsonl")

    sizes = [len(c.page_content) for c in chunks]
    stats = {
        "strategy": strategy,
        "files": report.files,
        "documents": report.documents,
        "empty": report.empty,
        "duplicates": report.duplicates,
        "errors": len(report.errors),
        "chunks": len(chunks),
        "chunks_per_doc": round(len(chunks) / len(documents), 1),
        "chars_mean": round(statistics.mean(sizes)),
        "chars_median": round(statistics.median(sizes)),
        "chars_min": min(sizes),
        "chars_max": max(sizes),
    }
    (settings.processed_dir / f"stats_{strategy}.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2, ensure_ascii=False))

    if not args.no_index:
        collection = f"{settings.chroma_collection}_{strategy}"
        index_chunks(get_vectorstore(settings, embeddings, collection), chunks)


def cmd_search(args: argparse.Namespace) -> None:
    settings = get_settings()
    strategy = args.strategy or settings.chunk_strategy
    retriever = build_retrievers(settings, strategy, [args.method])[args.method]
    for rank, (doc, score) in enumerate(retriever.search(args.query, k=args.k), start=1):
        print(f"\n[{rank}] {doc.metadata['chunk_id']}  score={score:.4f}  {doc.metadata['url']}")
        print(doc.page_content[:400])


def cmd_eval_retrieval(args: argparse.Namespace) -> None:
    settings = get_settings()
    strategy = args.strategy or settings.chunk_strategy
    questions = load_questions(args.questions)
    retrievers = build_retrievers(settings, strategy, args.methods)

    results = {name: evaluate(r, questions) for name, r in retrievers.items()}
    by_type = {
        f"{name} ({qtype})": evaluate(r, [q for q in questions if q.type == qtype])
        for qtype in sorted({q.type for q in questions})
        for name, r in retrievers.items()
    }

    report = (
        f"# Évaluation du retrieval — chunking `{strategy}`\n\n"
        f"{len(questions)} questions annotées (`{args.questions}`), pertinence au niveau fiche.\n\n"
        f"## Global\n\n{report_table(results)}\n\n"
        f"## Par type de question\n\n{report_table(by_type)}\n"
    )
    print(report)
    output = Path("eval/results") / f"retrieval_{strategy}.md"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report, encoding="utf-8")
    logger.info("Rapport écrit dans %s", output)


def make_graph(strategy: str | None = None, with_retriever: bool = True):
    settings = get_settings()
    strategy = strategy or settings.chunk_strategy
    retriever = (
        build_retrievers(settings, strategy, ["hybrid"])["hybrid"] if with_retriever else None
    )
    config = GraphConfig(
        k=settings.retrieval_k,
        max_retrieval_attempts=settings.max_retrieval_attempts,
        max_generation_attempts=settings.max_generation_attempts,
        grader_chunk_chars=settings.grader_chunk_chars,
    )
    return build_graph(get_llm(settings), retriever, config)


def cmd_ask(args: argparse.Namespace) -> None:
    graph = make_graph(args.strategy)
    start = time.perf_counter()
    state = graph.invoke({"question": args.question})
    elapsed = time.perf_counter() - start

    if args.verbose:
        print("Étapes :")
        for step in state["steps"]:
            print(f"  - {step}")
        print()
    print(state["answer"])
    if state["sources"]:
        print("\nSources :")
        for source in state["sources"]:
            print(
                f"  [{source['n']}] {source['title']} — {source['section']}\n      {source['url']}"
            )
    print(f"\n(statut : {state['status']}, {elapsed:.0f} s)")


def cmd_graph(args: argparse.Namespace) -> None:
    print(make_graph(with_retriever=False).get_graph().draw_mermaid())


def main() -> None:
    parser = argparse.ArgumentParser(prog="rag")
    sub = parser.add_subparsers(required=True)

    sub.add_parser("download", help="Télécharge le corpus").set_defaults(func=cmd_download)

    ingest = sub.add_parser("ingest", help="Ingestion, chunking et indexation")
    ingest.add_argument("--strategy", choices=["semantic", "recursive"])
    ingest.add_argument("--limit", type=int, help="Nombre maximum de fiches (debug)")
    ingest.add_argument("--no-index", action="store_true", help="Ne pas indexer dans ChromaDB")
    ingest.set_defaults(func=cmd_ingest)

    search = sub.add_parser("search", help="Recherche dans le corpus indexé")
    search.add_argument("query")
    search.add_argument("--method", choices=["dense", "sparse", "hybrid"], default="hybrid")
    search.add_argument("--strategy", choices=["semantic", "recursive"])
    search.add_argument("-k", type=int, default=5)
    search.set_defaults(func=cmd_search)

    evaluation = sub.add_parser("eval-retrieval", help="Compare les méthodes de retrieval")
    evaluation.add_argument("--strategy", choices=["semantic", "recursive"])
    evaluation.add_argument(
        "--methods",
        nargs="+",
        choices=["dense", "sparse", "hybrid"],
        default=["dense", "sparse", "hybrid"],
    )
    evaluation.add_argument(
        "--questions", type=Path, default=Path("eval/retrieval_questions.jsonl")
    )
    evaluation.set_defaults(func=cmd_eval_retrieval)

    ask = sub.add_parser("ask", help="Pose une question au système multi-agents")
    ask.add_argument("question")
    ask.add_argument("--strategy", choices=["semantic", "recursive"])
    ask.add_argument("-v", "--verbose", action="store_true", help="Affiche les étapes du graphe")
    ask.set_defaults(func=cmd_ask)

    sub.add_parser("graph", help="Diagramme Mermaid du graphe").set_defaults(func=cmd_graph)

    args = parser.parse_args()
    logging.basicConfig(
        level=get_settings().log_level, format="%(asctime)s %(levelname)s %(name)s | %(message)s"
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    args.func(args)


if __name__ == "__main__":
    main()
