"""Agent de validation : vérifie la fidélité aux sources et la pertinence de la réponse."""

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, Field

from rag.agents.grader import format_documents

VALIDATE_PROMPT = """Tu es le contrôle qualité d'un assistant documentaire.

Extraits de référence :
{documents}

Question : {question}

Réponse à contrôler :
{answer}

Vérifie deux points :
1. grounded : chaque affirmation de la réponse est-elle appuyée par les extraits ? \
(false si la réponse contient une information absente ou contraire aux extraits)
2. answers_question : la réponse traite-t-elle bien la question posée, et non un sujet voisin ?
3. provides_information : true si la réponse apporte une information concrète ; false si elle \
se contente d'indiquer que les extraits ne permettent pas de répondre.
Si le point 1 ou 2 échoue, explique brièvement le problème dans feedback."""


class Validation(BaseModel):
    grounded: bool = Field(description="Réponse fidèle aux extraits")
    answers_question: bool = Field(description="Réponse pertinente pour la question")
    provides_information: bool = Field(
        default=True, description="La réponse apporte une information concrète"
    )
    feedback: str = Field(default="", description="Problème constaté, le cas échéant")

    @property
    def is_valid(self) -> bool:
        return self.grounded and self.answers_question


class AnswerValidator:
    def __init__(self, llm: BaseChatModel):
        self.llm = llm

    def validate(self, question: str, documents: list[Document], answer: str) -> Validation:
        prompt = VALIDATE_PROMPT.format(
            documents=format_documents(documents), question=question, answer=answer
        )
        return self.llm.with_structured_output(Validation).invoke(prompt)
