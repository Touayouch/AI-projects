"""Interface de démonstration (Streamlit) branchée sur l'API /query/stream.

make api    # dans un terminal
make demo   # dans un autre, puis http://localhost:8501
"""

import json
import os

import httpx
import streamlit as st

API_URL = os.environ.get("RAG_API_URL", "http://localhost:8080")

EXAMPLES = [
    "J'ai droit à combien de jours de congé quand je me pacse ?",
    "Dois-je payer l'ordinateur de ma boîte que j'ai cassé ?",
    "Comment fonctionne la prime de partage de la valeur ?",
    "Quelle est la meilleure recette de crêpes ?",
]

STEP_LABELS = {
    "analyze": "Analyse de la question",
    "retrieve": "Recherche hybride (dense + BM25)",
    "grade": "Tri des extraits pertinents",
    "rewrite": "Reformulation de la requête",
    "generate": "Rédaction de la réponse",
    "validate": "Contrôle qualité de la réponse",
    "finalize": "Finalisation",
    "out_of_scope": "Question hors périmètre",
    "no_answer": "Aucune information trouvée",
}

STATUS_LABELS = {
    "answered": ("Réponse validée", "green"),
    "no_answer": ("Information absente des fiches", "orange"),
    "unverified": ("Réponse non fiable, sources proposées", "orange"),
    "out_of_scope": ("Hors périmètre", "gray"),
}


def stream_events(question: str):
    """Lit le flux Server-Sent Events de l'API : (événement, données)."""
    with httpx.stream(
        "POST", f"{API_URL}/query/stream", json={"question": question}, timeout=600
    ) as response:
        response.raise_for_status()
        event = None
        for line in response.iter_lines():
            if line.startswith("event: "):
                event = line.removeprefix("event: ")
            elif line.startswith("data: "):
                yield event, json.loads(line.removeprefix("data: "))


st.set_page_config(page_title="RAG multi-agents", page_icon="⚖️")
st.title("Assistant droit du travail et protection sociale")
st.caption(
    "Réponses sourcées à partir de 875 fiches service-public.gouv.fr — LangGraph, "
    "recherche hybride, Qwen3-4B en local."
)

columns = st.columns(2)
for i, example in enumerate(EXAMPLES):
    if columns[i % 2].button(example, use_container_width=True):
        st.session_state.question = example

question = st.text_input("Votre question", key="question")

if st.button("Poser la question", type="primary") and question.strip():
    result = None
    with st.status("Le graphe multi-agents travaille...", expanded=True) as status:
        try:
            for event, data in stream_events(question):
                if event == "step":
                    details = " — ".join(data["steps"])
                    st.write(f"**{STEP_LABELS.get(data['node'], data['node'])}** {details}")
                elif event == "result":
                    result = data
                elif event == "error":
                    st.error(data["detail"])
        except httpx.HTTPError as exc:
            st.error(f"API injoignable ({API_URL}) : {exc}")
        status.update(label="Terminé", state="complete", expanded=False)

    if result:
        label, color = STATUS_LABELS[result["status"]]
        st.markdown(f":{color}-badge[{label}] · {result['latency_s']:.0f} s")
        st.markdown(result["answer"])
        if result["sources"]:
            st.subheader("Sources")
            for source in result["sources"]:
                section = f" — {source['section']}" if source["section"] else ""
                st.markdown(f"**[{source['n']}]** [{source['title']}]({source['url']}){section}")
