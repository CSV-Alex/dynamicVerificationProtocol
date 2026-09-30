"""Independent divergence oracle for the evaluation harness.

Self-contained: it imports nothing from ``scp.py``, from ``SemanticVerifier``,
or from ``executor``. It re-implements the machine-checkable semantics of the
guarantees so that its judgement is independent of how the executor produced
the output and of how the protocol under test made its admission decision.
"""

from __future__ import annotations

import re

DATE_RE = re.compile(
    r"\b\d{4}-\d{1,2}-\d{1,2}\b"          # 2025-03-15
    r"|\b\d{1,2}/\d{1,2}/\d{4}\b"          # 15/03/2025
)

GLOSSARY: list[str] = [
    "clause", "liability", "jurisdiction", "termination",
    "indemnity", "plaintiff", "defendant",
]

_ES_STOPWORDS = {
    "el", "la", "los", "las", "de", "del", "y", "que", "en", "un", "una",
    "es", "para", "por", "con",
}
_EN_STOPWORDS = {
    "the", "a", "an", "and", "of", "to", "in", "is", "for", "with", "on",
    "this",
}


def _has_dates(output: str) -> bool:
    """Return True if the output contains at least one date token."""
    return bool(DATE_RE.search(output))


def _has_glossary_term(output: str, term: str) -> bool:
    """Return True if ``term`` appears (case-insensitive) in the output."""
    return term in output.lower()


def _detect_language(output: str) -> str:
    """Classify the output as "es" or "en" by a stopword-ratio heuristic."""
    tokens = {t.lower() for t in output.split()}
    es = len(tokens & _ES_STOPWORDS)
    en = len(tokens & _EN_STOPWORDS)
    return "es" if es > en else "en"


def diverges(output: str, required: dict) -> bool:
    """Return True if ``output`` violates any required guarantee in ``R(t)``.

    ``required`` keys: ``preserve_dates`` (bool), ``preserve_glossary`` (bool),
    ``max_tokens`` (int), ``language`` (str). Dates are checked for presence by
    regex; glossary for presence of at least one fixed term by substring match;
    token count by ``len(output.split())``; language by a stopword heuristic.
    """
    if required.get("preserve_dates") and not _has_dates(output):
        return True

    if required.get("preserve_glossary") and not any(
        _has_glossary_term(output, term) for term in GLOSSARY
    ):
        return True

    max_tokens = required.get("max_tokens")
    if max_tokens is not None and len(output.split()) > max_tokens:
        return True

    language = required.get("language")
    if language and _detect_language(output) != language:
        return True

    return False
