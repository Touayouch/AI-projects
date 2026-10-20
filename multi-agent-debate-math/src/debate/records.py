"""Format des enregistrements : un appel au LLM, et lecture des journaux JSONL."""

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
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
