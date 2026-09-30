"""Dependent-variable computation and smoke test for the evaluation harness.

This module ties the harness pieces together: the task suite, the baseline and
protocol arms, the independent oracle, and the instrumented protocol. It is the
runner, so (unlike ``executor.py`` and ``oracle.py``) it imports from ``scp.py``
and ``scp_instrumented.py`` to exercise the real protocol and its measurement.

The dependent variables are:

* divergence_rate      = (admitted AND diverge) / admitted
* residual_divergence  = (admitted AND diverge) / all
* detection_rate       = (rejected AND diverge) / all_divergent
* false_positive_rate  = (rejected AND NOT diverge) / all_compatible
* avg_messages, avg_latency_ms  (secondary, from the instrumented protocol)
"""

from __future__ import annotations

import contextlib
import csv
import io
import json
import random
from pathlib import Path

from scp import (
    Agent,
    CapabilityContract,
    LatentProfile,
    SemanticGuarantee,
    TaskRequest,
)
from scp_instrumented import (
    CountingAgent,
    CountingRegistry,
    CountingVerifier,
    InstrumentedProtocolOrchestrator,
    MeasurementContext,
)
from evaluation.baseline import delegate
from evaluation.executor import execute as eval_execute
from evaluation.oracle import diverges as eval_diverges
from evaluation.task_suite import generate
from evaluation.verifier_wrapper import ThresholdVerifier

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def _declared_contract(provider_id: str, latent: dict) -> CapabilityContract:
    """Build a provider's declared contract from its latent profile L(p)."""
    return CapabilityContract(
        agent_id=provider_id,
        capability_name="text_summarization",
        description="Summarizes a document.",
        inputs={"document": "str"},
        outputs={"summary": "str"},
        guarantees=[
            SemanticGuarantee("preserve_dates", latent["preserve_dates"], "Dates preserved."),
            SemanticGuarantee("preserve_glossary", latent["preserve_glossary"], "Glossary preserved."),
            SemanticGuarantee("max_tokens", latent["max_tokens"], "Max output tokens."),
        ],
        constraints={
            "language": latent["language"],
            "max_latency_ms": latent["max_latency_ms"],
        },
    )


def _required_task(task: dict) -> TaskRequest:
    """Build a TaskRequest whose required guarantees derive from R(t)."""
    required = task["required_profile"]
    guarantees = [
        SemanticGuarantee(name, required[name], "")
        for name in ("preserve_dates", "preserve_glossary", "max_tokens")
        if required.get(name) is not None
    ]
    constraints = {}
    if required.get("language"):
        constraints["language"] = required["language"]
    if required.get("max_latency_ms") is not None:
        constraints["max_latency_ms"] = required["max_latency_ms"]
    return TaskRequest(
        task_description="Summarize the document.",
        target_capability="text_summarization",
        required_guarantees=guarantees,
        constraints=constraints,
        source_text=task["source_text"],
    )


def _operational_diverges(latent: dict, required: dict) -> bool:
    """Return True if the provider's latency exceeds the required limit."""
    if (
        required.get("max_latency_ms") is not None
        and latent.get("max_latency_ms") is not None
    ):
        return latent["max_latency_ms"] > required["max_latency_ms"]
    return False


def _ground_truth_diverges(task: dict) -> bool:
    """Judge divergence independently of the protocol.

    A pair diverges if its executed output violates a semantic guarantee (the
    oracle) OR if its latent profile violates an operational constraint such as
    the latency limit.
    """
    output = eval_execute(task["source_text"], task["latent_profile"])
    semantic = eval_diverges(output, task["required_profile"])
    operational = _operational_diverges(
        task["latent_profile"], task["required_profile"]
    )
    return semantic or operational


def run_baseline_pair(task: dict) -> dict:
    """Run one pair through the baseline (always admitted)."""
    record = delegate(task)
    return {
        "admitted": True,
        "diverges": _ground_truth_diverges(task),
        "num_messages": record["num_messages"],
        "latency_ms": record["latency_ms"],
    }


def run_protocol_pair(
    task: dict, mode: str, declared_profile: dict | None = None
) -> dict:
    """Run one pair through the protocol with a severity threshold.

    ``declared_profile`` is the profile the provider declares (D(p)); it
    defaults to the latent profile L(p) (honest). Execution always uses L(p),
    so a declared profile that over-claims models a deceptive provider.
    """
    latent = task["latent_profile"]
    declared = declared_profile if declared_profile is not None else latent
    context = MeasurementContext()
    registry = CountingRegistry(context)
    agent = CountingAgent(
        task["provider_id"], framework="eval", model="eval", context=context,
        latent_profile=LatentProfile(
            preserve_dates=latent["preserve_dates"],
            preserve_glossary=latent["preserve_glossary"],
            max_tokens=latent["max_tokens"],
            language=latent["language"],
        ),
    )
    agent.register_capability(_declared_contract(task["provider_id"], declared))
    registry.register(agent)

    counting_verifier = CountingVerifier(context)
    threshold_verifier = ThresholdVerifier(mode, counting_verifier)
    orchestrator = InstrumentedProtocolOrchestrator(
        registry, threshold_verifier, context
    )

    requester = Agent("requester", framework="eval", model="eval")
    result = orchestrator.execute(requester, _required_task(task), task["provider_id"])
    metrics = orchestrator.last_metrics

    return {
        "admitted": result.get("status") == "executed",
        "diverges": _ground_truth_diverges(task),
        "num_messages": metrics.num_messages,
        "latency_ms": metrics.latency_ms,
    }


def compute_metrics(records: list[dict]) -> dict:
    """Compute the dependent variables from a list of per-pair records."""
    n = len(records)
    admitted = [r for r in records if r["admitted"]]
    divergent = [r for r in records if r["diverges"]]
    compatible = [r for r in records if not r["diverges"]]
    admitted_divergent = [r for r in records if r["admitted"] and r["diverges"]]
    rejected_divergent = [r for r in records if (not r["admitted"]) and r["diverges"]]
    rejected_compatible = [r for r in records if (not r["admitted"]) and (not r["diverges"])]

    def ratio(a: int, b: int) -> float:
        return a / b if b else 0.0

    return {
        "n_pairs": n,
        "n_admitted": len(admitted),
        "n_divergent": len(divergent),
        "divergence_rate": ratio(len(admitted_divergent), len(admitted)),
        "residual_divergence": ratio(len(admitted_divergent), n),
        "detection_rate": ratio(len(rejected_divergent), len(divergent)),
        "false_positive_rate": ratio(len(rejected_compatible), len(compatible)),
        "avg_messages": sum(r["num_messages"] for r in records) / n if n else 0.0,
        "avg_latency_ms": sum(r["latency_ms"] for r in records) / n if n else 0.0,
    }


def smoke(n_pairs: int = 10, mode: str = "permissive") -> None:
    """Run ``n_pairs`` through both arms, compute DVs, and save results."""
    suite = generate()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "task_suite.json").write_text(
        json.dumps(suite, indent=2), encoding="utf-8"
    )

    subset = suite[:n_pairs]
    # Prefer a balanced mix so the smoke test exercises both rejection and
    # false positives, not only the divergent (first) pairs.
    divergent_pairs = [p for p in suite if p["designed_to_diverge"]][: n_pairs // 2]
    compatible_pairs = [p for p in suite if not p["designed_to_diverge"]][
        : n_pairs - (n_pairs // 2)
    ]
    subset = divergent_pairs + compatible_pairs
    baseline_records = [run_baseline_pair(t) for t in subset]
    protocol_records = [run_protocol_pair(t, mode) for t in subset]

    baseline_metrics = compute_metrics(baseline_records)
    protocol_metrics = compute_metrics(protocol_records)

    rows = [
        {"arm": "baseline", **baseline_metrics},
        {"arm": f"protocol_{mode}", **protocol_metrics},
    ]
    fields = [
        "arm", "n_pairs", "n_admitted", "n_divergent", "divergence_rate",
        "residual_divergence", "detection_rate", "false_positive_rate",
        "avg_messages", "avg_latency_ms",
    ]
    out = DATA_DIR / "smoke_results.csv"
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    for row in rows:
        print(
            f"{row['arm']:16s} n={row['n_pairs']} admitted={row['n_admitted']} "
            f"divergent={row['n_divergent']} "
            f"div_rate={row['divergence_rate']:.3f} "
            f"resid={row['residual_divergence']:.3f} "
            f"det_rate={row['detection_rate']:.3f} "
            f"fpr={row['false_positive_rate']:.3f} "
            f"avg_msg={row['avg_messages']:.2f} "
            f"avg_lat={row['avg_latency_ms']:.3f}"
        )
    print(f"wrote {out}")


def bootstrap_ci_diff(
    records_a: list[dict], records_b: list[dict], n_boot: int = 2000, seed: int = 0
) -> tuple[float, float]:
    """Return a 95% bootstrap CI for divergence_rate(a) - divergence_rate(b).

    Pairs are resampled jointly (paired bootstrap) so the two arms share the
    same sampled pairs, then the divergence-rate difference is recomputed.
    """
    rng = random.Random(seed)
    n = len(records_a)

    def div_rate(recs: list[dict]) -> float:
        admitted = [r for r in recs if r["admitted"]]
        admitted_divergent = [r for r in recs if r["admitted"] and r["diverges"]]
        return len(admitted_divergent) / len(admitted) if admitted else 0.0

    diffs: list[float] = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        diffs.append(
            div_rate([records_a[i] for i in idx])
            - div_rate([records_b[i] for i in idx])
        )
    diffs.sort()
    return diffs[int(0.025 * n_boot)], diffs[int(0.975 * n_boot)]


VERIFIED_COMPONENTS = (
    "protocol=scp.py (FULLY faithful); "
    "oracle=src/evaluation/oracle.py (independent); "
    "executor=src/evaluation/executor.py (independent)"
)


def run_experiment() -> None:
    """Run all four conditions over the full suite and save the results."""
    suite = json.loads((DATA_DIR / "task_suite.json").read_text(encoding="utf-8"))
    n = len(suite)
    n_deceptive = int(0.2 * n)

    with contextlib.redirect_stdout(io.StringIO()):
        baseline = [run_baseline_pair(t) for t in suite]
        balanced = [run_protocol_pair(t, "balanced") for t in suite]
        permissive = [run_protocol_pair(t, "permissive") for t in suite]
        deceptive = [
            run_protocol_pair(
                t,
                "balanced",
                declared_profile=(
                    t["required_profile"] if i < n_deceptive else t["latent_profile"]
                ),
            )
            for i, t in enumerate(suite)
        ]

    conditions = [
        ("baseline", baseline, None),
        ("protocol_balanced", balanced, baseline),
        ("protocol_permissive", permissive, baseline),
        ("protocol_balanced_deceptive", deceptive, baseline),
    ]

    fields = [
        "condition", "divergence_rate", "residual_divergence", "detection_rate",
        "false_positive_rate", "avg_messages", "avg_latency_ms", "num_pairs",
        "ci_95_diff_vs_baseline", "verified_components",
    ]
    rows = []
    for name, records, baseline_records in conditions:
        m = compute_metrics(records)
        ci = ""
        if baseline_records is not None:
            lo, hi = bootstrap_ci_diff(baseline_records, records)
            ci = f"[{lo:.4f}, {hi:.4f}]"
        rows.append({
            "condition": name,
            "divergence_rate": round(m["divergence_rate"], 4),
            "residual_divergence": round(m["residual_divergence"], 4),
            "detection_rate": round(m["detection_rate"], 4),
            "false_positive_rate": round(m["false_positive_rate"], 4),
            "avg_messages": round(m["avg_messages"], 4),
            "avg_latency_ms": round(m["avg_latency_ms"], 6),
            "num_pairs": n,
            "ci_95_diff_vs_baseline": ci,
            "verified_components": VERIFIED_COMPONENTS,
        })

    out = DATA_DIR / "evaluation_results.csv"
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    for row in rows:
        print(
            f"{row['condition']:28s} div_rate={row['divergence_rate']:.4f} "
            f"resid={row['residual_divergence']:.4f} "
            f"det_rate={row['detection_rate']:.4f} "
            f"fpr={row['false_positive_rate']:.4f} "
            f"avg_msg={row['avg_messages']:.4f} "
            f"avg_lat={row['avg_latency_ms']:.4f} "
            f"ci={row['ci_95_diff_vs_baseline']}"
        )
    print(f"wrote {out}")


if __name__ == "__main__":
    run_experiment()
