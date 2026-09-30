"""Basic tests for the SCP instrumentation layer.

Runnable as a plain script with ``python3 test_instrumentation.py`` or through
a test runner that discovers ``test_*`` functions.
"""

import sys
from pathlib import Path

# Make ``src/`` importable so ``scp`` and ``scp_instrumented`` resolve.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scp_instrumented import run_all_scenarios, run_instrumented_task  # noqa: E402


def test_single_task_metrics() -> None:
    """A single task must yield num_messages > 0 and latency_ms > 0."""
    result, metrics = run_instrumented_task()

    assert result["status"] == "executed", f"unexpected status: {result}"
    assert metrics.num_messages > 0, f"num_messages not positive: {metrics}"
    assert metrics.latency_ms > 0.0, f"latency_ms not positive: {metrics}"
    assert metrics.success is True, f"success should be True: {metrics}"

    print(
        f"single: status={result['status']} "
        f"num_messages={metrics.num_messages} "
        f"latency_ms={metrics.latency_ms:.4f} "
        f"success={metrics.success}"
    )


def test_no_accumulation() -> None:
    """A reused orchestrator must report independent per-task message counts.

    The three main scenarios must report 8, 7 and 8 messages respectively, and
    the missing-provider scenario 2, rather than accumulated totals such as
    8, 15, 23.
    """
    expected = {
        "compatible": 8,
        "divergent": 7,
        "renegotiation": 8,
        "provider_not_found": 2,
    }
    for name, _result, metrics in run_all_scenarios():
        assert metrics.num_messages == expected[name], (
            f"{name}: expected {expected[name]}, got {metrics.num_messages} "
            f"(accumulation?)"
        )
        print(
            f"{name}: num_messages={metrics.num_messages} "
            f"success={metrics.success} "
            f"phase_latency_ms={ {k: round(v, 4) for k, v in metrics.phase_latency_ms.items()} }"
        )


def test_latent_profile_propagation() -> None:
    """Providers with different latent profiles must yield different outputs.

    ``compatible`` executes ``legal-summarizer-B`` (preserve_dates=True) and
    ``renegotiation`` executes ``summarizer-A`` (preserve_dates=False). If the
    latent profile were not propagated, both would fall back to the default and
    produce identical, date-free outputs.
    """
    outputs: dict[str, str] = {}
    for name, result, _metrics in run_all_scenarios():
        if result.get("status") == "executed":
            outputs[name] = result["result"]["output"]

    assert "compatible" in outputs, outputs
    assert "renegotiation" in outputs, outputs
    assert outputs["compatible"] != outputs["renegotiation"], outputs
    assert "Dates:" in outputs["compatible"], outputs["compatible"]
    assert "Dates:" not in outputs["renegotiation"], outputs["renegotiation"]

    print(
        f"propagation: compatible={outputs['compatible']!r} "
        f"renegotiation={outputs['renegotiation']!r}"
    )


if __name__ == "__main__":
    test_single_task_metrics()
    test_no_accumulation()
    test_latent_profile_propagation()
    print("PASS")
