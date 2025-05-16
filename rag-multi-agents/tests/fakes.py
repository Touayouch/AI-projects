"""Doublures de test partagées : LLM et retriever factices."""

from collections import defaultdict

from langchain_core.documents import Document
from langchain_core.messages import AIMessage


class FakeLLM:
    """LLM factice : renvoie des réponses préparées, par type de sortie structurée."""

    def __init__(self, structured: list, answers: list[str] | None = None):
        self.queues = defaultdict(list)
        for response in structured:
            self.queues[type(response)].append(response)
        self.answers = list(answers or [])
        self.prompts: list[str] = []

    def with_structured_output(self, schema):
        llm = self

        class Runnable:
            def invoke(self, prompt):
                llm.prompts.append(prompt)
                return llm.queues[schema].pop(0)

        return Runnable()

    def invoke(self, prompt):
        self.prompts.append(prompt)
        return AIMessage(content=self.answers.pop(0))


class FakeRetriever:
    def __init__(self):
        self.queries: list[str] = []

    def search(self, query, k=5):
        self.queries.append(query)
        return [
            (
                Document(
                    page_content=f"Chunk {i} sur les congés",
                    metadata={"chunk_id": f"F1-{i}", "doc_id": "F1", "title": "Congés", "url": "u"},
                ),
                1.0,
            )
            for i in range(min(3, k))
        ]
