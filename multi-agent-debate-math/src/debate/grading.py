"""Extraction et vérification des réponses.

Toutes les méthodes (réponse unique, vote, débat) utilisent exactement ces fonctions :
une différence d'extraction entre méthodes fausserait la comparaison.

- La réponse finale est attendue dans ``\\boxed{...}`` (consigne donnée au modèle) ; on
  prend le dernier ``\\boxed`` du texte.
- GSM8K : à défaut de ``\\boxed``, on retient le dernier nombre du texte ; comparaison
  numérique.
- MATH-500 : sans ``\\boxed``, pas de réponse ; équivalence mathématique avec
  ``math-verify`` (``\\frac{1}{2}`` = ``0.5``, ``x=3`` = ``3``, intervalles, matrices...).
"""

import re
from functools import lru_cache

from math_verify import parse, verify

NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def extract_boxed(text: str) -> str | None:
    """Contenu du dernier \\boxed{...} (ou \\fbox{...}), accolades imbriquées comprises."""
    start = max(text.rfind("\\boxed"), text.rfind("\\fbox"))
    if start < 0:
        return None
    i = text.find("{", start)
    if i < 0:
        return None
    depth = 0
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[i + 1 : j].strip()
    return None  # accolade non fermée (réponse tronquée)


def last_number(text: str) -> str | None:
    matches = NUMBER.findall(text)
    return matches[-1].replace(",", "") if matches else None


def extract_answer(text: str, dataset: str) -> str | None:
    boxed = extract_boxed(text)
    if boxed is not None:
        return boxed
    if dataset == "gsm8k":
        return last_number(text)
    return None


def to_number(answer: str) -> float | None:
    """'\\$1,234.00', '18 dollars', '50\\%' -> nombre ; None si aucun nombre."""
    cleaned = re.sub(r"\\text\{([^}]*)\}", r"\1", answer)
    cleaned = cleaned.replace("\\$", "").replace("$", "").replace("\\%", "").replace(",", "")
    frac = re.fullmatch(r"\s*(-?)\\d?frac\{(\d+)\}\{(\d+)\}\s*", cleaned)
    if frac:
        sign = -1 if frac.group(1) else 1
        return sign * int(frac.group(2)) / int(frac.group(3))
    match = NUMBER.search(cleaned)
    return float(match.group()) if match else None


@lru_cache(maxsize=50_000)
def _parse_latex(answer: str):
    return parse(f"${answer}$")


def equivalent(gold: str, prediction: str, dataset: str) -> bool:
    if prediction is None:
        return False
    if gold.strip() == prediction.strip():
        return True
    if dataset == "gsm8k":
        a, b = to_number(gold), to_number(prediction)
        return a is not None and b is not None and abs(a - b) < 1e-6
    return verify(_parse_latex(gold), _parse_latex(prediction))


def is_correct(response: str, gold: str, dataset: str) -> bool:
    return equivalent(gold, extract_answer(response, dataset), dataset)
