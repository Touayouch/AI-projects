import shutil
import zipfile
from pathlib import Path

from rag.agents.ingestion import IngestionAgent
from rag.ingestion.corpus import extract_fiches
from rag.ingestion.service_public import parse_fiche

FIXTURE = Path(__file__).parent / "fixtures" / "F9999.xml"


def test_parse_fiche_metadata():
    doc = parse_fiche(FIXTURE)
    assert doc.metadata["doc_id"] == "F9999"
    assert doc.metadata["title"] == "Congés payés du salarié"
    assert doc.metadata["theme"] == "Travail - Formation"
    assert doc.metadata["sous_theme"] == "Congés"
    assert doc.metadata["date_modif"] == "2025-01-15"


def test_parse_fiche_structure():
    content = parse_fiche(FIXTURE).page_content
    assert content.startswith("# Congés payés du salarié")
    assert "## Combien de jours ?" in content
    assert "### Apprenti" in content
    assert "Le salarié acquiert 2,5 jours par mois." in content
    assert "- Temps plein\n- Temps partiel" in content
    assert "Attention : En cas de maladie" in content
    assert "| Cas général | 100 € |" in content
    assert "Où s'adresser : Inspection du travail" in content
    assert "ddets" not in content


def test_ingestion_deduplicates(tmp_path):
    shutil.copy(FIXTURE, tmp_path / "F9999.xml")
    shutil.copy(FIXTURE, tmp_path / "F9998.xml")
    (tmp_path / "F0000.xml").write_text("<pas du xml")

    documents, report = IngestionAgent(tmp_path).run()

    assert report.files == 3
    assert report.documents == len(documents) == 1
    assert report.duplicates == 1
    assert report.errors == ["F0000.xml"]


def test_extract_fiches_filters_by_theme(tmp_path):
    archive = tmp_path / "corpus.zip"
    other_theme = FIXTURE.read_text().replace("Travail - Formation", "Logement")
    with zipfile.ZipFile(archive, "w") as zf:
        zf.write(FIXTURE, "vosdroits/F9999.xml")
        zf.writestr("vosdroits/F1.xml", other_theme)
        zf.writestr("vosdroits/menu.xml", "<menu/>")

    count = extract_fiches(archive.read_bytes(), tmp_path / "raw", ["Travail - Formation"])

    assert count == 1
    assert [p.name for p in (tmp_path / "raw").iterdir()] == ["F9999.xml"]
