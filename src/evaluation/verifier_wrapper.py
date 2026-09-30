"""Severity-threshold wrappers around scp.py's SemanticVerifier.

This is the ONLY module in ``src/evaluation/`` that imports from ``scp.py``.
That import is necessary because the threshold variants must exercise the
actual verifier behaviour -- its divergence detection and value-compatibility
rules -- rather than a reimplementation; only the accept/reject threshold is
allowed to differ.

The wrapper re-labels divergence severities so that ``DivergenceReport.compatible``
(which rejects on any "high" divergence) applies the chosen threshold:

* permissive: only high rejects (scp.py's default),
* balanced:   high and medium reject,
* strict:     high, medium and low reject.

Because ``scp.py`` currently emits only "high" and "medium", ``strict`` and
``balanced`` behave identically in practice; methodology.tex documents that the
reference implementation emits only ``high`` and ``medium``, so ``strict`` is
kept for completeness and is equivalent to ``balanced`` in this evaluation.
"""

from __future__ import annotations

from scp import (
    CapabilityContract,
    Divergence,
    DivergenceReport,
    SemanticVerifier,
    TaskRequest,
)

_BLOCK: dict[str, set[str]] = {
    "permissive": {"high"},
    "balanced": {"high", "medium"},
    "strict": {"high", "medium", "low"},
}


class ThresholdVerifier:
    """A verifier whose accept/reject threshold can be configured."""

    def __init__(self, mode: str = "permissive", verifier: SemanticVerifier | None = None) -> None:
        if mode not in _BLOCK:
            raise ValueError(f"unknown mode {mode!r}; expected one of {sorted(_BLOCK)}")
        self.mode = mode
        self._verifier = verifier if verifier is not None else SemanticVerifier()

    def verify(self, task: TaskRequest, contract: CapabilityContract) -> DivergenceReport:
        """Run the real verifier and re-label severities for the threshold."""
        report = self._verifier.verify(task, contract)
        block = _BLOCK[self.mode]
        adjusted = [
            Divergence(
                d.kind, d.field, d.required, d.offered,
                "high" if d.severity in block else d.severity,
                d.explanation,
            )
            for d in report.divergences
        ]
        return DivergenceReport(
            report.task_id, report.contract_id, adjusted, report.textual_similarity
        )
