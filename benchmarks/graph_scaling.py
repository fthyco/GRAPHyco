"""Static graph scaling probe: edge count, topology runtime and process RSS."""

from __future__ import annotations

import argparse
import json
import resource
import statistics
import time
from pathlib import Path

from graphyco.core.evaluation import evaluate, perturbation_profile
from graphyco.core.primitives import Edge, GraphState, Node


def graph(n: int) -> GraphState:
    nodes = {
        "n{}".format(i): Node(
            id="n{}".format(i), name="n{}".format(i), purpose="operation",
        )
        for i in range(n)
    }
    edges = [
        Edge("n{}".format(i), "n{}".format(i + 1))
        for i in range(n - 1)
    ]
    edges += [
        Edge("n{}".format(i), "n{}".format(i + 2))
        for i in range(n - 2)
    ]
    return GraphState(nodes=nodes, edges=edges)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[32, 128, 512])
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.sizes) < 3 or args.repetitions < 1:
        parser.error("sizes >= 3, repetitions >= 1 required")
    rows = {}
    for size in args.sizes:
        durations = []
        state = graph(size)
        for _ in range(args.repetitions):
            start = time.perf_counter()
            report = evaluate(state)
            profile = perturbation_profile(state)
            durations.append(time.perf_counter() - start)
            assert report["node_count"] == size
            assert len(profile) == size
            assert report["edge_count"] == 2 * size - 3
        rows[str(size)] = {
            "nodes": size,
            "edges": len(state.edges),
            "median_seconds": statistics.median(durations),
            "samples_seconds": durations,
            "peak_process_rss_mib": resource.getrusage(
                resource.RUSAGE_SELF
            ).ru_maxrss / 1024.0,
        }
    result = {
        "method": "single-process graph evaluation plus perturbation_profile",
        "results": rows,
        "note": (
            "Acyclic two-neighbor graphs; only CPU microbenchmarks. "
            "Runtimes depend on Python/platform; no generic scalability SLA."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
