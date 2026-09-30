"""Tests for the evaluation harness executor and oracle.

Runnable as a plain script with ``python3 test_evaluation_oracle.py`` or
through a test runner that discovers ``test_*`` functions.
"""

import sys
from pathlib import Path

# Make ``src/`` importable so ``evaluation.*`` resolves.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.executor import execute  # noqa: E402
from evaluation.oracle import diverges  # noqa: E402

SOURCE = (
    "The plaintiff signed the agreement on 2025-03-15. "
    "The clause limits liability and jurisdiction. "
    "Termination requires written notice."
)


def test_dates_divergence() -> None:
    """L(p).preserve_dates=False with R(t).preserve_dates=True diverges."""
    profile = {"preserve_dates": False, "preserve_glossary": True,
               "max_tokens": 200, "language": "en"}
    required = {"preserve_dates": True, "preserve_glossary": False,
                "max_tokens": 200, "language": "en"}
    output = execute(SOURCE, profile)
    assert diverges(output, required) is True, output


def test_match_no_divergence() -> None:
    """R(t) == L(p) must not diverge."""
    profile = {"preserve_dates": True, "preserve_glossary": True,
               "max_tokens": 200, "language": "en"}
    output = execute(SOURCE, profile)
    assert diverges(output, profile) is False, output


def test_max_tokens_divergence() -> None:
    """R(t).max_tokens smaller than the output length diverges."""
    profile = {"preserve_dates": True, "preserve_glossary": True,
               "max_tokens": 200, "language": "en"}
    required = {"preserve_dates": False, "preserve_glossary": False,
                "max_tokens": 2, "language": "en"}
    output = execute(SOURCE, profile)
    assert diverges(output, required) is True, output


if __name__ == "__main__":
    test_dates_divergence()
    print("test_dates_divergence: OK")
    test_match_no_divergence()
    print("test_match_no_divergence: OK")
    test_max_tokens_divergence()
    print("test_max_tokens_divergence: OK")
    print("PASS")
