"""Ligne de commande du projet.

rag download                    # télécharge et filtre le corpus
rag ingest --strategy semantic  # ingestion -> chunking -> indexation
"""

import argparse
import json
import logging
import statistics

from rag.agents.chunking import ChunkingAgent
from rag.agents.ingestion import IngestionAgent
from rag.config import get_settings
from rag.indexing import get_vectorstore, index_chunks, save_chunks
from rag.ingestion.corpus import download_corpus
from rag.models import get_embeddings

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


def main() -> None:
    parser = argparse.ArgumentParser(prog="rag")
    sub = parser.add_subparsers(required=True)

    sub.add_parser("download", help="Télécharge le corpus").set_defaults(func=cmd_download)

    ingest = sub.add_parser("ingest", help="Ingestion, chunking et indexation")
    ingest.add_argument("--strategy", choices=["semantic", "recursive"])
    ingest.add_argument("--limit", type=int, help="Nombre maximum de fiches (debug)")
    ingest.add_argument("--no-index", action="store_true", help="Ne pas indexer dans ChromaDB")
    ingest.set_defaults(func=cmd_ingest)

    args = parser.parse_args()
    logging.basicConfig(
        level=get_settings().log_level, format="%(asctime)s %(levelname)s %(name)s | %(message)s"
    )
    args.func(args)


if __name__ == "__main__":
    main()
