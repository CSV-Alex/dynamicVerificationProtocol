"""Collect per-scenario metrics for the demonstration scenarios.

Runs the four demonstration scenarios (compatible, divergent, renegotiation,
provider_not_found) through the instrumented orchestrator and writes one CSV
row per scenario to ``data/scenarios.csv`` with the columns ``scenario,
num_messages, latency_ms, success``.

Usage:
    python3 src/collect_scenarios.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

# Make src/ importable when the script is run from any directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[0]))

from scp_instrumented import run_all_scenarios  # noqa: E402


def main() -> None:
    """Collect the scenarios and write ``data/scenarios.csv``."""
    rows = []
    for name, _result, metrics in run_all_scenarios():
        rows.append(
            (name, metrics.num_messages, round(metrics.latency_ms, 4), metrics.success)
        )

    out = Path(__file__).resolve().parents[1] / "data" / "scenarios.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["scenario", "num_messages", "latency_ms", "success"])
        writer.writerows(rows)
    print(f"wrote {out}")
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
