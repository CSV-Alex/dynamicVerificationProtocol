"""Optional instrumentation layer for the Semantic Contract Protocol (SCP).

This module provides an opt-in measurement layer for the reference
implementation in ``scp.py``. For each executed task it records:

* ``num_messages`` -- the number of logical protocol messages exchanged in
  that task (a per-task delta, so a reused orchestrator does not accumulate),
* ``latency_ms``    -- wall-clock time from task delegation to completion,
* ``phase_latency_ms`` -- per-phase latency for discovery, proposal,
  verification and binding (binding is the residual of the total), and
* ``success``       -- whether the task completed according to the semantic
  contract.

The base protocol in ``scp.py`` is left untouched, so instrumentation is
disabled by default there and enabled only through this module. The classes
here wrap the protocol collaborators and delegate every call to the original
implementation, so the protocol logic and its results are unchanged.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator, Optional

from scp import (
    Agent,
    CapabilityContract,
    DivergenceReport,
    LatentProfile,
    ProtocolOrchestrator,
    Registry,
    SemanticGuarantee,
    SemanticVerifier,
    TaskRequest,
    build_registry,
)


@dataclass
class ProtocolMetrics:
    """Measurements collected during a single task execution.

    Attributes:
        num_messages:     Number of logical protocol messages in this task.
        latency_ms:       Wall-clock duration of the execution in milliseconds.
        phase_latency_ms: Per-phase latency in milliseconds.
        success:          Whether the task completed according to the contract.
        started_at:       Monotonic timestamp captured before execution.
        finished_at:      Monotonic timestamp captured after execution.
    """

    num_messages: int = 0
    latency_ms: float = 0.0
    success: bool = False
    started_at: float = 0.0
    finished_at: float = 0.0
    phase_latency_ms: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return the metrics as a plain dictionary."""
        return {
            "num_messages": self.num_messages,
            "latency_ms": self.latency_ms,
            "success": self.success,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "phase_latency_ms": dict(self.phase_latency_ms),
        }


class MeasurementContext:
    """Shared measurement state: a message counter and per-phase latencies.
    ------------
    The context is passed to the counting wrappers so that every collaborator
    call increments the same counter and records into the same phase-latency
    dictionary.
    """

    def __init__(self) -> None:
        self._count: int = 0
        self.phase_latency_ms: dict[str, float] = {}

    @property
    def value(self) -> int:
        """Return the accumulated message count."""
        return self._count

    def bump(self, amount: int = 1) -> None:
        """Increment the accumulated message count by ``amount``."""
        self._count += amount

    def reset(self) -> None:
        """Reset the accumulated message count and phase latencies to empty."""
        self._count = 0
        self.phase_latency_ms.clear()

    @contextmanager
    def timed_phase(self, name: str) -> Iterator[None]:
        """Record the elapsed time of the enclosed block under ``name`` (ms)."""
        start = time.perf_counter()
        try:
            yield
        finally:
            self.phase_latency_ms[name] = (time.perf_counter() - start) * 1000.0


class CountingRegistry(Registry):
    """A registry that counts and times the discovery exchange."""

    def __init__(self, context: MeasurementContext) -> None:
        super().__init__()
        self._context = context

    def discover(self, capability: str) -> list[Agent]:
        """Return candidates for ``capability``, counting the exchange."""
        with self._context.timed_phase("discovery"):
            self._context.bump(2)
            return super().discover(capability)


class CountingAgent(Agent):
    """An agent whose ``get_contract`` call is counted and timed."""

    def __init__(
        self,
        agent_id: str,
        framework: str,
        model: str,
        context: MeasurementContext,
        latent_profile: Optional[LatentProfile] = None,
    ) -> None:
        super().__init__(
            agent_id, framework, model, latent_profile=latent_profile
        )
        self._context = context

    def get_contract(self, name: str) -> Optional[CapabilityContract]:
        """Return the contract for ``name``, counting request and response."""
        with self._context.timed_phase("proposal"):
            self._context.bump(2)
            return super().get_contract(name)


class CountingVerifier(SemanticVerifier):
    """A verifier whose ``verify`` call is counted and timed."""

    def __init__(self, context: MeasurementContext) -> None:
        super().__init__()
        self._context = context

    def verify(
        self, task: TaskRequest, contract: CapabilityContract
    ) -> DivergenceReport:
        """Verify ``task`` against ``contract``, counting the exchange."""
        with self._context.timed_phase("verification"):
            self._context.bump(2)
            return super().verify(task, contract)


class InstrumentedProtocolOrchestrator(ProtocolOrchestrator):
    """An orchestrator that measures one task execution.

    It delegates to the base orchestrator unchanged and, around that call,
    records wall-clock time, per-task message count, per-phase latency and
    success. ``num_messages`` is computed as a per-task delta so that reusing
    the same orchestrator across tasks does not accumulate counts, and the
    counter is reset after each task.
    """

    def __init__(
        self,
        registry: Registry,
        verifier: SemanticVerifier,
        context: MeasurementContext,
    ) -> None:
        super().__init__(registry, verifier)
        self._context = context
        self.last_metrics: ProtocolMetrics = ProtocolMetrics()

    def execute(
        self, requester: Agent, task: TaskRequest, provider_id: str
    ) -> dict[str, Any]:
        """Run one task and record its metrics.

        Returns the same result as the base orchestrator while populating
        :attr:`last_metrics`.
        """
        started_at: float = time.perf_counter()
        start_count: int = self._context.value
        result: dict[str, Any] = super().execute(requester, task, provider_id)
        finished_at: float = time.perf_counter()

        status = result.get("status")
        if status == "executed":
            self._context.bump(2)  # bind message + execution result
            success = True
        elif status == "rejected":
            self._context.bump(1)  # rejection notice
            success = False
        else:  # provider_not_found
            success = False

        num_messages: int = self._context.value - start_count

        total_ms: float = (finished_at - started_at) * 1000.0
        phase_latency: dict[str, float] = dict(self._context.phase_latency_ms)
        phase_latency["binding"] = max(0.0, total_ms - sum(phase_latency.values()))

        self.last_metrics = ProtocolMetrics(
            num_messages=num_messages,
            latency_ms=total_ms,
            success=success,
            started_at=started_at,
            finished_at=finished_at,
            phase_latency_ms=phase_latency,
        )
        self._context.reset()
        return result


def build_instrumented_registry(
    context: MeasurementContext,
) -> CountingRegistry:
    """Build a registry whose discovery and agents are counted.

    Reuses the contracts defined by ``scp.build_registry`` and wraps each
    agent in a :class:`CountingAgent`.
    """
    base: Registry = build_registry()
    registry = CountingRegistry(context)
    for agent in base._agents.values():  # noqa: SLF001 - reuse existing setup
        counting_agent = CountingAgent(
            agent.agent_id, agent.framework, agent.model, context,
            latent_profile=agent.latent_profile,
        )
        for contract in agent._caps.values():  # noqa: SLF001
            counting_agent.register_capability(contract)
        registry.register(counting_agent)
    return registry


_SOURCE: str = (
    "The plaintiff signed the agreement on 2025-03-15. "
    "The clause limits liability and jurisdiction. "
    "Termination requires written notice."
)


def _build_scenarios() -> list[tuple[str, str, str, str, TaskRequest, str]]:
    """Return the four demonstration scenarios.

    Each entry is ``(name, requester_id, framework, model, task, provider_id)``.
    """
    compatible = TaskRequest(
        task_description=(
            "Summarize the legal document preserving clause references and dates."
        ),
        target_capability="text_summarization",
        required_guarantees=[
            SemanticGuarantee("preserve_technical_terms", True, "Keep technical terms."),
            SemanticGuarantee("preserve_dates", True, "Keep all dates."),
        ],
        source_text=_SOURCE,
        constraints={"max_latency_ms": 10000, "language": "es"},
    )
    divergent = TaskRequest(
        task_description="Summarize the legal document.",
        target_capability="text_summarization",
        required_guarantees=[
            SemanticGuarantee("preserve_technical_terms", True, "Keep technical terms."),
            SemanticGuarantee("preserve_dates", True, "Keep all dates."),
        ],
        source_text=_SOURCE,
        constraints={"max_latency_ms": 5000, "language": "es"},
    )
    renegotiation = TaskRequest(
        task_description="Summarize the document briefly.",
        target_capability="text_summarization",
        required_guarantees=[
            SemanticGuarantee("max_output_tokens", 200, "Summary at most 200 tokens."),
        ],
        source_text=_SOURCE,
        constraints={"max_latency_ms": 5000, "language": "es"},
    )
    not_found = TaskRequest(
        task_description="Summarize the document.",
        target_capability="text_summarization",
        required_guarantees=[],
        source_text=_SOURCE,
        constraints={"max_latency_ms": 5000, "language": "es"},
    )
    return [
        ("compatible", "legal-requester-X", "CrewAI", "gpt-4o", compatible, "legal-summarizer-B"),
        ("divergent", "legal-requester-X", "CrewAI", "gpt-4o", divergent, "summarizer-A"),
        ("renegotiation", "casual-requester-Y", "CrewAI", "gpt-4o", renegotiation, "summarizer-A"),
        ("provider_not_found", "casual-requester-Y", "CrewAI", "gpt-4o", not_found, "nonexistent-agent"),
    ]


def run_all_scenarios() -> list[tuple[str, dict[str, Any], ProtocolMetrics]]:
    """Run all four scenarios through a single reused orchestrator.

    Reusing one orchestrator exercises the per-task ``num_messages`` delta:
    each returned metrics must report an independent count rather than an
    accumulated total.
    """
    context = MeasurementContext()
    registry = build_instrumented_registry(context)
    verifier = CountingVerifier(context)
    orchestrator = InstrumentedProtocolOrchestrator(registry, verifier, context)

    results: list[tuple[str, dict[str, Any], ProtocolMetrics]] = []
    for name, rid, framework, model, task, provider_id in _build_scenarios():
        requester = Agent(rid, framework=framework, model=model)
        result = orchestrator.execute(requester, task, provider_id)
        results.append((name, result, orchestrator.last_metrics))
    return results


def run_instrumented_task() -> tuple[dict[str, Any], ProtocolMetrics]:
    """Run the compatible scenario once and return its metrics.

    Returns:
        A tuple ``(result, metrics)`` where ``result`` is the orchestrator
        result dictionary and ``metrics`` the collected measurements.
    """
    name, result, metrics = run_all_scenarios()[0]
    return result, metrics
