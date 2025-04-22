"""Agent de génération : rédige la réponse à partir des seuls extraits retenus."""

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel

from rag.agents.grader import format_documents

GENERATE_PROMPT = """Tu es un assistant qui répond aux questions sur le droit du travail et \
la protection sociale en France, à partir d'extraits de fiches service-public.gouv.fr.

Règles :
- Réponds uniquement avec les informations des extraits ci-dessous, sans rien inventer.
- Cite tes sources avec leur numéro entre crochets, par exemple [1] ou [2][3].
- Si les extraits ne permettent pas de répondre entièrement, dis-le clairement.
- Réponds en français, de façon claire et concise (5 à 10 phrases maximum).

Extraits :
{documents}

Question : {question}
{feedback}
Réponse :"""

FEEDBACK = """
Une première réponse a été rejetée par le contrôle qualité pour la raison suivante : \
{feedback}
Corrige ce problème dans ta nouvelle réponse.
"""


class Generator:
    def __init__(self, llm: BaseChatModel):
        self.llm = llm

    def generate(
        self, question: str, documents: list[Document], feedback: str | None = None
    ) -> str:
        prompt = GENERATE_PROMPT.format(
            documents=format_documents(documents),
            question=question,
            feedback=FEEDBACK.format(feedback=feedback) if feedback else "",
        )
        return str(self.llm.invoke(prompt).content).strip()
