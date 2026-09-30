"""Self-contained scripted executor for the evaluation harness.

Independent by design: it imports nothing from ``scp.py``, from
``SemanticVerifier``, or from ``oracle``. The date formats and the glossary list
below are the shared OPERATIONAL semantics of the guarantees, re-implemented
here deliberately so that the executor and the oracle cannot share an
implementation and thereby hide a bug in the protocol under test.

The "task" is reduced to its ``source_text``: that is the only input the
executor needs, together with the provider's latent profile ``L(p)``.
"""

from __future__ import annotations

import re

# Date formats used across the harness: ISO and slash day-first.
DATE_RE = re.compile(
    r"\b\d{4}-\d{1,2}-\d{1,2}\b"          # 2025-03-15
    r"|\b\d{1,2}/\d{1,2}/\d{4}\b"          # 15/03/2025
)

# Fixed glossary of "technical" terms used across the harness.
GLOSSARY: list[str] = [
    "clause", "liability", "jurisdiction", "termination",
    "indemnity", "plaintiff", "defendant",
]

_INTRO: dict[str, str] = {
    "en": "Summary of the requested task.",
    "es": "Resumen de la tarea solicitada.",
}


def extract_dates(source_text: str) -> list[str]:
    """Return the date strings found in ``source_text`` (deterministic)."""
    return DATE_RE.findall(source_text)


def extract_glossary_terms(source_text: str) -> list[str]:
    """Return the glossary terms that appear in ``source_text``."""
    lowered = source_text.lower()
    return [term for term in GLOSSARY if term in lowered]


def execute(source_text: str, profile: dict) -> str:
    """Produce a deterministic output honoring the latent profile ``L(p)``.

    ``profile`` keys: ``preserve_dates`` (bool), ``preserve_glossary`` (bool),
    ``max_tokens`` (int), ``language`` (str, "en" or "es").

    Truncation: if the output exceeds ``max_tokens`` tokens it is truncated
    (split on whitespace); tokens are never expanded.
    """
    dates: list[str] = extract_dates(source_text) if profile.get("preserve_dates") else []
    terms: list[str] = (
        extract_glossary_terms(source_text) if profile.get("preserve_glossary") else []
    )
    language: str = profile.get("language", "en")

    parts: list[str] = [_INTRO.get(language, _INTRO["en"])]
    if dates:
        parts.append("Dates: " + ", ".join(dates) + ".")
    if terms:
        parts.append("Terms: " + ", ".join(terms) + ".")

    output: str = " ".join(parts)
    tokens = output.split()
    if len(tokens) > profile.get("max_tokens", 200):
        output = " ".join(tokens[: profile["max_tokens"]])
    return output
