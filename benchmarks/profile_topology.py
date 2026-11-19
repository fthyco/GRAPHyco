"""Profile legacy graph metric hotspots vs the identical indexed implementation.

All timings are measurements on this runner, not production performance SLAs.
The slow reference is deliberately independent: it retains original repeated
edge scans and integer formulas, and validates output exactness.
"""

from __future__ import annotations

import argparse
import cProfile
import io
import json
import pstats
import statistics
import time
from pathlib import Path

from graphyco.core.evaluation import evaluate, perturbation_profile
from graphyco.core.graph import compute_local_density
from graphyco.core.primitives import GraphState, SCALE, checked_add, checked_mul
from benchmarks.graph_scaling import graph


def original_profile(state: GraphState) -> dict:
    """Pre-optimization O(V*E + neighbor-density-repeated) definition."""
    result = {}
    for nid in state.nodes:
        d = compute_local_density(nid, state)
        primary = max(1 + d // SCALE, 1)
        neighbors = set()
        for edge in state.edges:
            if edge.source == nid:
                neighbors.add(edge.target)
            if edge.target == nid:
                neighbors.add(edge.source)
        secondary = 0
        for other in neighbors:
            if other in state.nodes:
                density = compute_local_density(other, state)
                secondary = checked_add(secondary, max(density // SCALE, 1))
        result[nid] = checked_add(primary, secondary)
    return result


def original_fan_balance(state: GraphState) -> dict:
    result = {}
    for nid in state.nodes:
        fi = sum(edge.target == nid for edge in state.edges)
        fo = sum(edge.source == nid for edge in state.edges)
        total = fi + fo
        result[nid] = {
            "fan_in": fi,
            "fan_out": fo,
            "imbalance": 0 if total == 0 else abs(fi - fo) * SCALE // total,
        }
    return result


def _sample(fn, rounds):
    samples = []
    value = None
    for _ in range(rounds):
        start = time.perf_counter()
        value = fn()
        samples.append(time.perf_counter() - start)
    return value, samples


def _profile_hotspots(state: GraphState):
    profiler = cProfile.Profile()
    profiler.enable()
    evaluate(state)
    profiler.disable()
    buffer = io.StringIO()
    pstats.Stats(profiler, stream=buffer).sort_stats("cumulative").print_stats(18)
    return buffer.getvalue()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", type=int, default=[128, 512, 1024])
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.sizes or min(args.sizes) < 3 or args.rounds < 1:
        parser.error("sizes must be >=3 and rounds >=1")
    rows = {}
    for n in args.sizes:
        state = graph(n)
        naive_prof, slow_samples = _sample(lambda: original_profile(state), args.rounds)
        fast_prof, fast_samples = _sample(lambda: perturbation_profile(state), args.rounds)
        assert naive_prof == fast_prof, "optimized profile differs from original"
        old_balance, old_samples = _sample(lambda: original_fan_balance(state), args.rounds)
        new_balance, new_samples = _sample(lambda: evaluate(state)["fan_balance"], args.rounds)
        assert old_balance == new_balance
        report = evaluate(state)
        assert report["perturbation_profile"] == naive_prof
        rows[str(n)] = {
            "nodes": n,
            "edges": len(state.edges),
            "legacy_profile_median_seconds": statistics.median(slow_samples),
            "indexed_profile_median_seconds": statistics.median(fast_samples),
            "profile_speedup": statistics.median(slow_samples) / max(
                statistics.median(fast_samples), 1e-12
            ),
            "legacy_balance_median_seconds": statistics.median(old_samples),
            "full_evaluate_median_seconds": statistics.median(new_samples),
            "legacy_profile_samples_seconds": slow_samples,
            "fast_profile_samples_seconds": fast_samples,
            "cprofile_top_cumulative": _profile_hotspots(state),
        }
    result = {
        "method": "independent legacy profile vs cached-snapshot profile; CPU",
        "notes": "Timing is machine-specific. Exact profile and balance parity asserted.",
        "sizes": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for n, r in rows.items():
        print("nodes={} old_profile={:.5f}s new_profile={:.5f}s speedup={:.2f}x".format(
            n, r["legacy_profile_median_seconds"],
            r["indexed_profile_median_seconds"], r["profile_speedup"],
        ))
    print("profile parity and instrumentation passed")


if __name__ == "__main__":
    main()
