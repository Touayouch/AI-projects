import json
from pathlib import Path

import pytest

from debate.data import (
    gsm8k_answer,
    load_problems,
    load_subset,
    make_subset,
    parse_gsm8k,
    parse_math500,
)

SUBSETS = Path(__file__).parents[1] / "configs" / "subsets"

GSM8K_ROWS = [
    {"question": "Q1", "answer": "16 - 3 = <<16-3=13>>13\n#### 13"},
    {"question": "Q2", "answer": "Total is 1,200.\n#### 1,200"},
]
MATH_ROWS = [
    {
        "problem": "P1",
        "solution": r"So \boxed{\frac{1}{2}}.",
        "answer": r"\frac{1}{2}",
        "subject": "Algebra",
        "level": 2,
        "unique_id": "test/algebra/1.json",
    }
]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def test_gsm8k_answer():
    assert gsm8k_answer("... #### 1,234") == "1234"
    with pytest.raises(ValueError):
        gsm8k_answer("pas de réponse")


def test_parse_gsm8k_strips_calculator_annotations():
    problems = parse_gsm8k(GSM8K_ROWS)
    assert [p.id for p in problems] == ["gsm8k-0000", "gsm8k-0001"]
    assert problems[0].answer == "13"
    assert "<<" not in problems[0].solution
    assert problems[1].answer == "1200"


def test_parse_math500_keeps_metadata():
    problem = parse_math500(MATH_ROWS)[0]
    assert problem.id == "test/algebra/1.json"
    assert problem.metadata == {"subject": "Algebra", "level": 2}


def test_make_subset_is_reproducible_and_ordered():
    problems = parse_gsm8k([{"question": f"Q{i}", "answer": "#### 1"} for i in range(100)])
    first = make_subset(problems, 10, seed=0)
    assert first == make_subset(problems, 10, seed=0)
    assert first != make_subset(problems, 10, seed=1)
    assert first == sorted(first)  # ordre d'origine conservé
    assert len(make_subset(problems, 500)) == 100


def test_load_subset_from_cached_file(tmp_path):
    write_jsonl(tmp_path / "gsm8k_test.jsonl", GSM8K_ROWS)
    subset = tmp_path / "subset.txt"
    subset.write_text("gsm8k-0001\n")
    assert [p.answer for p in load_subset("gsm8k", tmp_path, subset)] == ["1200"]
    assert len(load_problems("gsm8k", tmp_path)) == 2

    subset.write_text("gsm8k-9999\n")
    with pytest.raises(ValueError, match="inconnus"):
        load_subset("gsm8k", tmp_path, subset)


@pytest.mark.parametrize(
    ("name", "size"),
    [("gsm8k_500", 500), ("math500_500", 500), ("gsm8k_dev_30", 30), ("math500_dev_30", 30)],
)
def test_frozen_subsets(name, size):
    ids = (SUBSETS / f"{name}.txt").read_text().split()
    assert len(ids) == len(set(ids)) == size
