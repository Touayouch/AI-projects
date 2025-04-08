"""Normalisation du texte français pour la recherche lexicale (BM25)."""

import re
import unicodedata
from functools import lru_cache

import snowballstemmer

# Mots vides français (liste Snowball, légèrement réduite)
STOPWORDS_TEXT = """
    a à afin ai aie aient aies ait alors as au aucun aucune aupres auquel aura aurai
    auraient aurais aurait auras aurez auriez aurions aurons auront aussi autre aux
    auxquelles auxquels avaient avais avait avant avec avez aviez avions avoir avons ayant
    ayez ayons c ça car ce ceci cela celle celles celui cependant certain certaine certaines
    certains ces cet cette ceux chaque chez ci comme comment d dans de des desquelles
    desquels dès donc dont du duquel elle elles en encore entre es est et étaient étais
    était étant été êtes étiez étions être eu eue eues eûmes eurent eus eusse eussent eut
    eux fait faut fois furent fus fut ici il ils j je jusqu jusque l la laquelle le lequel
    les lesquelles lesquels leur leurs lors lui m ma mais me même mêmes mes moi mon n ne ni
    non nos notre nous on ont ou où par parce pas pendant peu peut peuvent plus pour
    pourquoi puis qu quand que quel quelle quelles quels qui quoi s sa sans se sera serai
    seraient serais serait seras serez seriez serions serons seront ses si sien soi soient
    sois soit sommes son sont sous suis sur t ta te tes toi ton tous tout toute toutes très
    tu un une unes uns vers vos votre vous y
"""
STOPWORDS = set(STOPWORDS_TEXT.split())

ELISION = re.compile(r"\b([cdjlmnst]|qu|jusqu|lorsqu|puisqu)['’]", re.IGNORECASE)
TOKEN = re.compile(r"\w+(?:-\w+)*")

_stemmer = snowballstemmer.stemmer("french")


def strip_accents(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))


@lru_cache(maxsize=100_000)
def stem(word: str) -> str:
    return strip_accents(_stemmer.stemWord(word))


def tokenize(text: str) -> list[str]:
    """Minuscules, suppression des élisions et mots vides, racinisation, sans accents."""
    text = ELISION.sub(" ", text.lower())
    return [stem(w) for w in TOKEN.findall(text) if w not in STOPWORDS]
