"""Parsing des fiches XML service-public.gouv.fr en Markdown structuré.

La hiérarchie XML (Chapitre > SousChapitre > BlocCas > Cas, Situation...) est convertie en
titres Markdown (#, ##, ###...) afin que le chunking puisse s'appuyer sur la structure.
"""

import re
import xml.etree.ElementTree as ET
from pathlib import Path

from langchain_core.documents import Document

DC = "{http://purl.org/dc/elements/1.1/}"

# Blocs qui introduisent un niveau de titre
SECTIONS = {"Chapitre", "SousChapitre", "Situation", "Cas"}
# Encadrés : rendus en un seul paragraphe préfixé par leur libellé
ENCADRES = {
    "ANoter": "À noter",
    "ASavoir": "À savoir",
    "Attention": "Attention",
    "Exemple": "Exemple",
    "Rappel": "Rappel",
    "Complement": "Complément",
}
# Liens vers des services : on ne garde que leur titre
LIENS = {"ServiceEnLigne": "Service en ligne", "OuSAdresser": "Où s'adresser"}
# Éléments sans contenu textuel utile
IGNORES = {
    "Titre",
    "TitreRiche",
    "Condition",
    "Colonne",
    "Source",
    "RessourceWeb",
    "PivotLocal",
    "Video",
    "Infographie",
    "Credits",
}


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def element_text(element: ET.Element | None) -> str:
    if element is None:
        return ""
    return clean_text("".join(element.itertext()))


def section_title(element: ET.Element) -> str:
    return element_text(element.find("Titre")) or element_text(element.find("TitreRiche"))


def render(element: ET.Element, level: int) -> list[str]:
    """Convertit un élément XML en une liste de blocs Markdown (séparés par une ligne vide)."""
    tag = element.tag
    if tag in IGNORES:
        return []
    if tag == "Paragraphe":
        text = element_text(element)
        return [text] if text else []
    if tag == "TitreFlottant":
        return [f"**{element_text(element)}**"]
    if tag in SECTIONS:
        blocks = []
        title = section_title(element)
        if title:
            blocks.append(f"{'#' * min(level, 6)} {title}")
        for child in element:
            blocks.extend(render(child, level + 1))
        return blocks
    if tag == "Liste":
        items = [element_text(item) for item in element.findall("Item")]
        return ["\n".join(f"- {item}" for item in items if item)]
    if tag == "Tableau":
        return [render_table(element)]
    if tag in ENCADRES:
        label = section_title(element) or ENCADRES[tag]
        content = " ".join(b for child in element for b in render(child, level))
        return [f"{label} : {content}"] if content else []
    if tag in LIENS:
        title = section_title(element)
        return [f"{LIENS[tag]} : {title}"] if title else []
    # Conteneurs (Texte, Introduction, BlocCas, ListeSituations, FragmentConditionne...)
    return [block for child in element for block in render(child, level)]


def render_table(table: ET.Element) -> str:
    lines = []
    title = element_text(table.find("TitreRiche"))
    if title:
        lines.append(f"**{title}**")
    for i, row in enumerate(table.findall("Rangée")):
        cells = [element_text(cell) for cell in row.findall("Cellule")]
        lines.append("| " + " | ".join(cells) + " |")
        if i == 0 and row.get("type") == "header":
            lines.append("|" + " --- |" * len(cells))
    return "\n".join(lines)


def parse_fiche(path: Path) -> Document | None:
    """Parse une fiche XML. Renvoie None si la fiche n'a pas de contenu textuel."""
    root = ET.parse(path).getroot()
    title = element_text(root.find(f"{DC}title"))

    blocks = [f"# {title}"]
    for part in ("Introduction", "Texte", "ListeSituations"):
        element = root.find(part)
        if element is not None:
            blocks.extend(render(element, level=2))

    content = "\n\n".join(b for b in blocks if b.strip())
    if len(blocks) <= 1:
        return None

    theme = root.find("Theme/Titre")
    sous_theme = root.find("SousThemePere")
    return Document(
        page_content=content,
        metadata={
            "doc_id": root.get("ID", path.stem),
            "title": title,
            "url": root.get("spUrl", ""),
            "theme": theme.text if theme is not None else "",
            "sous_theme": element_text(sous_theme),
            "description": element_text(root.find(f"{DC}description")),
            "date_modif": (root.get("dateDerniereModificationImportante") or "")[:10],
            "type": root.get("type", ""),
            "source": path.name,
        },
    )
