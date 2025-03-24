"""Agent de chunking : découpe les documents en chunks prêts à indexer.

Deux stratégies, pour pouvoir comparer leur impact sur le retrieval :

- ``semantic`` : découpage par structure (titres Markdown), puis regroupement des blocs
  consécutifs d'une même section tant qu'ils restent sémantiquement proches. Une frontière
  est posée quand la distance cosinus entre deux blocs voisins dépasse le percentile
  ``breakpoint_percentile`` de toutes les distances du corpus, ou quand le chunk atteint
  ``max_chars``.
- ``recursive`` : baseline classique, ``RecursiveCharacterTextSplitter`` par section.

Chaque chunk est préfixé par son fil d'Ariane (titre de la fiche > chapitre > cas) pour
qu'un chunk isolé reste compréhensible, à la fois par le retriever et par le LLM.
"""

import logging
import re
from dataclasses import dataclass
from typing import Literal

import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

HEADER = re.compile(r"^(#{1,6}) (.+)$")


@dataclass
class Section:
    doc: Document
    path: list[str]
    blocks: list[str]

    @property
    def breadcrumb(self) -> str:
        return " > ".join(self.path)


def split_sections(doc: Document) -> list[Section]:
    """Découpe un document Markdown en sections selon ses titres."""
    sections: list[Section] = []
    path: list[str] = []
    current: list[str] = []

    def flush():
        if current:
            sections.append(Section(doc, list(path), list(current)))
            current.clear()

    for block in doc.page_content.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        match = HEADER.match(block)
        if match:
            flush()
            level = len(match.group(1))
            path[level - 1 :] = [match.group(2)]
        else:
            current.append(block)
    flush()
    return sections


def cosine_distances(vectors: np.ndarray) -> np.ndarray:
    """Distance cosinus entre chaque vecteur et le suivant."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    normed = vectors / np.clip(norms, 1e-12, None)
    return 1.0 - np.sum(normed[:-1] * normed[1:], axis=1)


class ChunkingAgent:
    def __init__(
        self,
        strategy: Literal["semantic", "recursive"] = "semantic",
        embeddings: Embeddings | None = None,
        max_chars: int = 1500,
        min_chars: int = 200,
        breakpoint_percentile: float = 90.0,
    ):
        if strategy == "semantic" and embeddings is None:
            raise ValueError("La stratégie 'semantic' nécessite un modèle d'embeddings")
        self.strategy = strategy
        self.embeddings = embeddings
        self.max_chars = max_chars
        self.min_chars = min_chars
        self.breakpoint_percentile = breakpoint_percentile
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=max_chars,
            chunk_overlap=max_chars // 10,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def run(self, documents: list[Document]) -> list[Document]:
        sections = [s for doc in documents for s in split_sections(doc)]
        if self.strategy == "semantic":
            groups = self._semantic_groups(sections)
        else:
            groups = [self.splitter.split_text("\n\n".join(s.blocks)) for s in sections]

        chunks: list[Document] = []
        counters: dict[str, int] = {}
        for section, texts in zip(sections, groups, strict=True):
            for text in texts:
                doc_id = section.doc.metadata["doc_id"]
                index = counters.get(doc_id, 0)
                counters[doc_id] = index + 1
                chunks.append(
                    Document(
                        page_content=f"{section.breadcrumb}\n\n{text}",
                        metadata={
                            **section.doc.metadata,
                            "section": " > ".join(section.path[1:]),
                            "chunk_id": f"{doc_id}-{index}",
                            "chunk_index": index,
                            "strategy": self.strategy,
                        },
                    )
                )
        logger.info(
            "Chunking (%s) : %d documents -> %d chunks", self.strategy, len(documents), len(chunks)
        )
        return chunks

    def _units(self, section: Section) -> list[str]:
        """Blocs de la section, les blocs trop longs étant redécoupés."""
        units = []
        for block in section.blocks:
            if len(block) > self.max_chars:
                units.extend(self.splitter.split_text(block))
            else:
                units.append(block)
        return units

    def _semantic_groups(self, sections: list[Section]) -> list[list[str]]:
        units_per_section = [self._units(s) for s in sections]
        all_units = [u for units in units_per_section for u in units]
        logger.info("Embedding de %d blocs pour le chunking sémantique", len(all_units))
        vectors = np.array(self.embeddings.embed_documents(all_units))

        # Distances entre blocs voisins d'une même section
        distances_per_section = []
        offset = 0
        for units in units_per_section:
            section_vectors = vectors[offset : offset + len(units)]
            offset += len(units)
            distances_per_section.append(
                cosine_distances(section_vectors) if len(units) > 1 else np.array([])
            )

        all_distances = np.concatenate(distances_per_section)
        threshold = (
            float(np.percentile(all_distances, self.breakpoint_percentile))
            if all_distances.size
            else 1.0
        )
        logger.info("Seuil de rupture sémantique : %.3f", threshold)

        return [
            self._group(units, distances, threshold)
            for units, distances in zip(units_per_section, distances_per_section, strict=True)
        ]

    def _group(self, units: list[str], distances: np.ndarray, threshold: float) -> list[str]:
        groups: list[list[str]] = [[units[0]]]
        for unit, distance in zip(units[1:], distances, strict=True):
            size = sum(len(u) for u in groups[-1]) + len(unit)
            if distance > threshold or size > self.max_chars:
                groups.append([unit])
            else:
                groups[-1].append(unit)

        texts = ["\n\n".join(g) for g in groups]
        # Fusionne les chunks trop petits avec le précédent si la taille le permet
        merged = [texts[0]]
        for text in texts[1:]:
            too_small = len(text) < self.min_chars or len(merged[-1]) < self.min_chars
            if too_small and len(merged[-1]) + len(text) <= self.max_chars:
                merged[-1] = f"{merged[-1]}\n\n{text}"
            else:
                merged.append(text)
        return merged
