"""Chargement des benchmarks GSM8K et MATH-500.

Les fichiers sources (JSONL officiels) sont téléchargés une fois dans ``data/``. Le
sous-ensemble évalué est figé dans ``configs/subsets/`` (liste d'identifiants versionnée),
pour que toutes les expériences portent exactement sur les mêmes problèmes.
"""

import json
import logging
import random
import re
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)

DatasetName = Literal["gsm8k", "math500"]

SOURCES: dict[str, str] = {
    "gsm8k": "https://raw.githubusercontent.com/openai/grade-school-math/master/"
    "grade_school_math/data/test.jsonl",
    "math500": "https://huggingface.co/datasets/HuggingFaceH4/MATH-500/resolve/main/test.jsonl",
}


@dataclass(frozen=True)
class Problem:
    id: str
    dataset: DatasetName
    question: str
    answer: str  # réponse de référence (nombre pour GSM8K, LaTeX pour MATH)
    solution: str = ""  # solution rédigée de référence
    metadata: dict = field(default_factory=dict, hash=False, compare=False)


def download(name: DatasetName, data_dir: Path) -> Path:
    path = data_dir / f"{name}_test.jsonl"
    if not path.exists():
        data_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Téléchargement de %s", SOURCES[name])
        with urllib.request.urlopen(SOURCES[name], timeout=120) as response:
            path.write_bytes(response.read())
    return path


def gsm8k_answer(solution: str) -> str:
    """La réponse GSM8K suit '####' en fin de solution : '... #### 1,234' -> '1234'."""
    match = re.search(r"####\s*(.+?)\s*$", solution)
    if not match:
        raise ValueError(f"Solution GSM8K sans '####' : {solution[-80:]!r}")
    return match.group(1).replace(",", "").strip()


def parse_gsm8k(lines: list[dict]) -> list[Problem]:
    return [
        Problem(
            id=f"gsm8k-{i:04d}",
            dataset="gsm8k",
            question=row["question"],
            answer=gsm8k_answer(row["answer"]),
            # Retire les annotations de calcul <<16-3=13>> de la solution
            solution=re.sub(r"<<[^>]*>>", "", row["answer"]),
        )
        for i, row in enumerate(lines)
    ]


def parse_math500(lines: list[dict]) -> list[Problem]:
    return [
        Problem(
            id=row["unique_id"],
            dataset="math500",
            question=row["problem"],
            answer=row["answer"],
            solution=row["solution"],
            metadata={"subject": row["subject"], "level": row["level"]},
        )
        for row in lines
    ]


def load_problems(name: DatasetName, data_dir: Path) -> list[Problem]:
    path = download(name, data_dir)
    with path.open(encoding="utf-8") as f:
        lines = [json.loads(line) for line in f if line.strip()]
    return parse_gsm8k(lines) if name == "gsm8k" else parse_math500(lines)


def make_subset(problems: list[Problem], size: int, seed: int = 0) -> list[str]:
    """Tire un sous-ensemble reproductible d'identifiants, dans l'ordre d'origine."""
    if size >= len(problems):
        return [p.id for p in problems]
    chosen = set(random.Random(seed).sample([p.id for p in problems], size))
    return [p.id for p in problems if p.id in chosen]


def load_subset(name: DatasetName, data_dir: Path, subset_file: Path) -> list[Problem]:
    ids = [line.strip() for line in subset_file.read_text().splitlines() if line.strip()]
    by_id = {p.id: p for p in load_problems(name, data_dir)}
    missing = [i for i in ids if i not in by_id]
    if missing:
        raise ValueError(f"{len(missing)} identifiants inconnus dans {subset_file}")
    return [by_id[i] for i in ids]
