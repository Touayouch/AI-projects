"""Agent de tri : garde uniquement les chunks utiles pour répondre à la question.

Les chunks sont jugés en un seul appel au LLM (et non un appel par chunk) : sur CPU,
chaque appel coûte plusieurs secondes.
"""

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, Field

GRADE_PROMPT = """Tu évalues des extraits de fiches service-public.gouv.fr.

Question : {question}

Extraits :
{documents}

Indique les numéros des extraits qui contiennent des informations utiles pour répondre à \
la question, même partiellement. Un extrait qui traite d'un sujet voisin mais différent \
(autre type de congé, autre statut, autre prestation) n'est PAS utile. Si aucun n'est \
utile, renvoie une liste vide."""


class GradeResult(BaseModel):
    relevant: list[int] = Field(description="Numéros des extraits utiles")


def format_documents(documents: list[Document], max_chars: int | None = None) -> str:
    return "\n\n".join(
        f"[{i}] {doc.page_content[:max_chars]}" for i, doc in enumerate(documents, start=1)
    )


class DocumentGrader:
    def __init__(self, llm: BaseChatModel, max_chars: int = 800):
        self.llm = llm
        self.max_chars = max_chars

    def grade(self, question: str, documents: list[Document]) -> list[Document]:
        if not documents:
            return []
        prompt = GRADE_PROMPT.format(
            question=question, documents=format_documents(documents, self.max_chars)
        )
        result = self.llm.with_structured_output(GradeResult).invoke(prompt)
        indices = sorted({i for i in result.relevant if 1 <= i <= len(documents)})
        return [documents[i - 1] for i in indices]
