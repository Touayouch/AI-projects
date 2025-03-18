"""Téléchargement du corpus : fiches pratiques service-public.gouv.fr (open data DILA).

L'archive contient ~3000 fiches XML (F*.xml) tous thèmes confondus. On ne garde que les
fiches des thèmes demandés (par défaut « Travail - Formation » et « Social - Santé »).
"""

import io
import logging
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

logger = logging.getLogger(__name__)


def fiche_theme(xml_bytes: bytes) -> str | None:
    root = ET.fromstring(xml_bytes)
    theme = root.find("Theme/Titre")
    return theme.text if theme is not None else None


def extract_fiches(archive: bytes, dest: Path, themes: list[str]) -> int:
    """Extrait dans `dest` les fiches (F*.xml) dont le thème est dans `themes`."""
    dest.mkdir(parents=True, exist_ok=True)
    count = 0
    with zipfile.ZipFile(io.BytesIO(archive)) as zf:
        for name in zf.namelist():
            filename = Path(name).name
            if not (filename.startswith("F") and filename.endswith(".xml")):
                continue
            content = zf.read(name)
            if fiche_theme(content) in themes:
                (dest / filename).write_bytes(content)
                count += 1
    return count


def download_corpus(url: str, dest: Path, themes: list[str]) -> int:
    logger.info("Téléchargement de %s", url)
    with urllib.request.urlopen(url, timeout=300) as response:
        archive = response.read()
    count = extract_fiches(archive, dest, themes)
    logger.info("%d fiches extraites dans %s", count, dest)
    return count
