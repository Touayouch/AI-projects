import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from rag.agents.chunking import ChunkingAgent, split_sections

TOPICS = ["congé", "salaire", "retraite"]


class TopicEmbeddings(Embeddings):
    """Embeddings factices : un axe par thème, pour contrôler les ruptures sémantiques."""

    def embed_documents(self, texts):
        return [self.embed_query(t) for t in texts]

    def embed_query(self, text):
        return [1.0 if topic in text else 0.01 for topic in TOPICS]


def make_doc(body: str) -> Document:
    return Document(page_content=body, metadata={"doc_id": "F1", "title": "Titre"})


def test_split_sections_tracks_headers():
    doc = make_doc("# Titre\n\nIntro\n\n## Chapitre\n\nA\n\n### Cas\n\nB\n\n## Autre\n\nC")
    sections = split_sections(doc)
    assert [s.breadcrumb for s in sections] == [
        "Titre",
        "Titre > Chapitre",
        "Titre > Chapitre > Cas",
        "Titre > Autre",
    ]
    assert [s.blocks for s in sections] == [["Intro"], ["A"], ["B"], ["C"]]


def test_semantic_chunking_breaks_on_topic_change():
    blocks = [f"Paragraphe {i} sur le congé payé." for i in range(3)]
    blocks += [f"Paragraphe {i} sur le salaire minimum." for i in range(3)]
    doc = make_doc("# Titre\n\n## Section\n\n" + "\n\n".join(blocks))

    agent = ChunkingAgent("semantic", TopicEmbeddings(), min_chars=0, breakpoint_percentile=50)
    chunks = agent.run([doc])

    assert len(chunks) == 2
    assert "salaire" not in chunks[0].page_content
    assert "congé" not in chunks[1].page_content
    assert all(c.page_content.startswith("Titre > Section\n\n") for c in chunks)


def test_semantic_chunking_respects_max_chars():
    blocks = ["congé " * 40] * 10  # ~240 caractères chacun, même thème
    doc = make_doc("# Titre\n\n" + "\n\n".join(blocks))

    chunks = ChunkingAgent("semantic", TopicEmbeddings(), max_chars=600, min_chars=0).run([doc])

    assert len(chunks) > 1
    assert all(len(c.page_content) <= 600 + len("Titre\n\n") for c in chunks)


def test_small_chunks_are_merged():
    doc = make_doc("# Titre\n\ncongé court\n\nsalaire court\n\nretraite court")
    chunks = ChunkingAgent("semantic", TopicEmbeddings(), min_chars=200).run([doc])
    assert len(chunks) == 1


def test_chunk_metadata():
    doc = make_doc("# Titre\n\n## A\n\ncongé\n\n## B\n\nsalaire")
    chunks = ChunkingAgent("recursive").run([doc])
    assert [c.metadata["chunk_id"] for c in chunks] == ["F1-0", "F1-1"]
    assert [c.metadata["section"] for c in chunks] == ["A", "B"]
    assert all(c.metadata["strategy"] == "recursive" for c in chunks)


def test_semantic_requires_embeddings():
    with pytest.raises(ValueError):
        ChunkingAgent("semantic")
