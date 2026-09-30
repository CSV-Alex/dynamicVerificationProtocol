"""Reproducible (task, provider) suite generator for the evaluation harness.

Generates pairs with a fixed seed and saves them to ``data/task_suite.json``.
Each pair is classified by ``divergence_kind``: ``"semantic"`` (a guarantee the
provider does not honor, high severity), ``"operational"`` (a latency constraint
the provider exceeds, medium severity), or ``"none"`` (compatible). The required
profile R(t) and the latent profile L(p) share the keys ``preserve_dates``,
``preserve_glossary``, ``max_tokens``, ``language`` and ``max_latency_ms``.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

# Providers, each with a distinct latent profile. Every provider has at least
# one guarantee set to False so a semantic divergence can always be produced,
# and a declared max_latency_ms so an operational (medium-severity) divergence
# can be produced by lowering the requester's limit.
PROVIDERS: list[dict] = [
    {
        "id": "dates-only-en",
        "latent_profile": {
            "preserve_dates": True,
            "preserve_glossary": False,
            "max_tokens": 800,
            "language": "en",
            "max_latency_ms": 8000,
        },
    },
    {
        "id": "glossary-only-en",
        "latent_profile": {
            "preserve_dates": False,
            "preserve_glossary": True,
            "max_tokens": 300,
            "language": "en",
            "max_latency_ms": 3000,
        },
    },
    {
        "id": "lax-es",
        "latent_profile": {
            "preserve_dates": False,
            "preserve_glossary": False,
            "max_tokens": 200,
            "language": "es",
            "max_latency_ms": 2000,
        },
    },
]

SOURCES: list[str] = [
    "The plaintiff signed the agreement on 2025-03-15. "
    "The clause limits liability and jurisdiction. Termination requires written notice.",
    "The defendant filed the claim on 15/01/2025. "
    "The indemnity clause caps liability. Jurisdiction is defined in the agreement.",
    "The parties executed the contract on 2025-07-01. "
    "The termination clause and the liability limit were amended.",
]


def make_semantic_divergent(latent: dict) -> dict:
    """Return R(t) that flips a guarantee to True where L(p) has it False."""
    required = dict(latent)
    if not latent["preserve_dates"]:
        required["preserve_dates"] = True
    else:
        required["preserve_glossary"] = True
    return required


def make_operational_divergent(latent: dict) -> dict:
    """Return R(t) whose latency limit is stricter than what L(p) offers."""
    required = dict(latent)
    required["max_latency_ms"] = max(1, latent["max_latency_ms"] // 2)
    return required


def generate(
    n_pairs: int = 110,
    n_semantic: int = 30,
    n_operational: int = 30,
    seed: int = 42,
) -> list[dict]:
    """Generate the suite.

    The first ``n_semantic`` pairs diverge semantically (high severity), the next
    ``n_operational`` pairs diverge operationally (medium severity), and the
    remaining pairs are compatible.
    """
    rng = random.Random(seed)
    pairs: list[dict] = []
    for i in range(n_pairs):
        if i < n_semantic:
            kind = "semantic"
        elif i < n_semantic + n_operational:
            kind = "operational"
        else:
            kind = "none"
        provider = rng.choice(PROVIDERS)
        source = rng.choice(SOURCES)
        latent = provider["latent_profile"]
        if kind == "semantic":
            required = make_semantic_divergent(latent)
        elif kind == "operational":
            required = make_operational_divergent(latent)
        else:
            required = dict(latent)
        pairs.append(
            {
                "task_id": f"task-{i:03d}",
                "capability": "text_summarization",
                "source_text": source,
                "required_profile": required,
                "provider_id": provider["id"],
                "latent_profile": latent,
                "designed_to_diverge": kind != "none",
                "divergence_kind": kind,
            }
        )
    return pairs


def save() -> list[dict]:
    """Generate the suite and write it to ``data/task_suite.json`` (repo root)."""
    pairs = generate()
    out = Path(__file__).resolve().parents[2] / "data" / "task_suite.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(pairs, indent=2), encoding="utf-8")
    return pairs


if __name__ == "__main__":
    data = save()
    by_kind: dict[str, int] = {}
    for p in data:
        by_kind[p["divergence_kind"]] = by_kind.get(p["divergence_kind"], 0) + 1
    print(f"wrote {len(data)} pairs: {by_kind}")
