"""Agent de requête : filtre les questions hors périmètre et reformule en cas d'échec.

La première recherche utilise la question telle quelle : avec un petit modèle local, une
reformulation systématique introduit des dérives (ex. « pacs » reformulé en « maternité »),
alors que la recherche hybride est déjà robuste aux formulations courantes. Le LLM ne
reformule que si aucun document pertinent n'a été trouvé.
"""

from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, Field

SCOPE = (
    "droit du travail et emploi en France (contrat, congés, salaire, licenciement, démission, "
    "temps de travail, formation), chômage, retraite, santé, sécurité sociale, handicap, "
    "aides sociales, fonction publique"
)

ANALYZE_PROMPT = """Tu es l'agent de filtrage d'un assistant qui répond à partir des fiches \
officielles de service-public.gouv.fr sur : {scope}.

Question de l'utilisateur : {question}

in_scope : true si la question concerne ce périmètre, même formulée en langage familier \
(« ma boîte » = mon entreprise, « mon patron » = mon employeur, « la sécu », « le chômage »). \
false uniquement si elle n'a clairement aucun rapport (cuisine, sport, programmation, \
culture générale, météo...). En cas de doute, réponds true : la recherche documentaire \
dira ensuite si l'information existe."""

REWRITE_PROMPT = """La recherche documentaire n'a trouvé aucun document pertinent.

Question de l'utilisateur : {question}
Requête déjà essayée : {previous_query}

Propose une nouvelle requête de recherche en français, formulée différemment : synonymes, \
terme juridique officiel, sigle développé. Elle doit porter exactement sur le même sujet \
que la question, sans en changer le sens, et être différente de la requête déjà essayée."""


class QueryAnalysis(BaseModel):
    in_scope: bool = Field(description="La question relève-t-elle du périmètre ?")


class RewrittenQuery(BaseModel):
    search_query: str = Field(description="Nouvelle requête de recherche")


class QueryAgent:
    def __init__(self, llm: BaseChatModel):
        self.llm = llm

    def analyze(self, question: str) -> QueryAnalysis:
        prompt = ANALYZE_PROMPT.format(scope=SCOPE, question=question)
        return self.llm.with_structured_output(QueryAnalysis).invoke(prompt)

    def rewrite(self, question: str, previous_query: str) -> str:
        prompt = REWRITE_PROMPT.format(question=question, previous_query=previous_query)
        return self.llm.with_structured_output(RewrittenQuery).invoke(prompt).search_query
