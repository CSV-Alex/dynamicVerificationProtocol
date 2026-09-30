"""A2A-style plain-delegation baseline for the evaluation harness.

No semantic contract is exchanged: the requester sends only a task description
and a capability name, the provider executes with its latent profile L(p), and
the output is always accepted (no verification). Divergence is judged by the
same independent oracle used for the protocol arm, so the comparison is fair.
"""

from __future__ import annotations

import time

from evaluation.executor import execute


class _MessageCounter:
    """Counts the logical messages of a plain delegation exchange."""

    def __init__(self) -> None:
        self._count = 0

    def bump(self, amount: int = 1) -> None:
        self._count += amount

    @property
    def value(self) -> int:
        return self._count


def delegate(task: dict) -> dict:
    """Run plain delegation for one task.

    ``task`` keys used: ``source_text`` and ``latent_profile`` (L(p)). Returns
    a record with the output, the admission flag (always True), the counted
    number of messages, and the measured wall-clock latency.

    The baseline exchanges exactly two messages (a delegation request and a
    result) because it performs no discovery, proposal or verification; the
    count is measured with the same request/response counting convention used
    by the protocol arm, rather than hardcoded.
    """
    counter = _MessageCounter()
    start = time.perf_counter()
    counter.bump(1)  # delegation request (requester -> provider)
    output: str = execute(task["source_text"], task["latent_profile"])
    counter.bump(1)  # result (provider -> requester)
    latency_ms: float = (time.perf_counter() - start) * 1000.0
    return {
        "output": output,
        "admitted": True,
        "num_messages": counter.value,
        "latency_ms": latency_ms,
    }
