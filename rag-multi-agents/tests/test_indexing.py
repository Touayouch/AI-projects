from langchain_core.documents import Document

from rag.indexing import load_chunks, save_chunks


def test_jsonl_roundtrip(tmp_path):
    chunks = [Document(page_content="Congés é", metadata={"chunk_id": "F1-0", "n": 1})]
    path = tmp_path / "chunks.jsonl"
    save_chunks(chunks, path)
    assert load_chunks(path) == chunks
