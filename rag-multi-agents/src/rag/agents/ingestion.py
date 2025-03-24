"""Agent d'ingestion : charge les fiches, les convertit en Documents et les déduplique."""

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path

from langchain_core.documents import Document

from rag.ingestion.service_public import parse_fiche

logger = logging.getLogger(__name__)


@dataclass
class IngestionReport:
    files: int = 0
    documents: int = 0
    empty: int = 0
    duplicates: int = 0
    errors: list[str] = field(default_factory=list)


class IngestionAgent:
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir

    def run(self, limit: int | None = None) -> tuple[list[Document], IngestionReport]:
        paths = sorted(self.data_dir.glob("*.xml"))[:limit]
        report = IngestionReport(files=len(paths))
        documents: list[Document] = []
        seen: set[str] = set()

        for path in paths:
            try:
                doc = parse_fiche(path)
            except Exception as exc:  # une fiche malformée ne doit pas bloquer le pipeline
                logger.warning("Échec du parsing de %s : %s", path.name, exc)
                report.errors.append(path.name)
                continue
            if doc is None:
                report.empty += 1
                continue
            digest = hashlib.sha256(doc.page_content.encode()).hexdigest()
            if digest in seen:
                report.duplicates += 1
                continue
            seen.add(digest)
            documents.append(doc)

        report.documents = len(documents)
        logger.info(
            "Ingestion : %d fichiers, %d documents, %d vides, %d doublons, %d erreurs",
            report.files,
            report.documents,
            report.empty,
            report.duplicates,
            len(report.errors),
        )
        return documents, report
