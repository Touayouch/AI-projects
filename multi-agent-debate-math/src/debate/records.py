"""Format des enregistrements : un appel au LLM, et lecture des journaux JSONL."""

import gzip
import json
from pathlib import Path

from debate.grading import equivalent, extract_answer
from debate.llm import Completion


def call_record(completion: Completion, seed: int, gold: str, dataset: str) -> dict:
    answer = extract_answer(completion.text, dataset)
    return {
        "model": completion.model,
        "seed": seed,
        "text": completion.text,
        "answer": answer,
        "correct": equivalent(gold, answer, dataset),
        "prompt_tokens": completion.prompt_tokens,
        "completion_tokens": completion.completion_tokens,
        "latency_s": completion.latency_s,
        "finish_reason": completion.finish_reason,
    }


def read_records(path: Path) -> list[dict]:
    """Lit un journal JSONL, compressé ou non (``x.jsonl`` ou ``x.jsonl.gz``)."""
    path = Path(path)
    if path.suffix != ".gz" and not path.exists():
        path = path.with_name(path.name + ".gz")  # version compressée éventuelle
    if not path.exists():
        return []
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as f:
            lines = f.readlines()
    else:
        lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def repair_jsonl(path: Path) -> int:
    """Retire une dernière ligne incomplète (processus tué pendant l'écriture) ; le problème
    correspondant sera simplement refait à la reprise. Renvoie le nombre d'octets retirés."""
    if not path.exists() or path.stat().st_size == 0:
        return 0
    data = path.read_bytes()
    if data.endswith(b"\n"):
        try:
            json.loads(data.rsplit(b"\n", 2)[-2])
            return 0
        except (ValueError, IndexError):
            pass
    lines = data.split(b"\n")
    kept = [line for line in lines if line.strip()]
    while kept:
        try:
            json.loads(kept[-1])
            break
        except ValueError:
            kept.pop()
    repaired = b"".join(line + b"\n" for line in kept)
    path.write_bytes(repaired)
    return len(data) - len(repaired)
