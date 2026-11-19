"""Independent legacy-oracle tests for optimized topology accounting.

Assertions are exact integer and dictionary equality, including parallel arcs,
self-loops, dangling endpoints and mutations. Nothing here calls the new
_topology_snapshot helper in the reference calculation.
"""

from __future__ import annotations

import random

from graphyco.core.evaluation import (
    connectivity_coherence,
    evaluate,
    fan_balance,
    perturbation_profile,
    structural_perturbation_resilience,
    topological_bottleneck_ratio,
)
from graphyco.core.graph import (
    compute_local_density, detect_critical_cycles, find_isolated_nodes,
    in_degree, out_degree,
)
from graphyco.core.primitives import Edge, GraphState, Node, SCALE, checked_add, checked_mul


def oracle(state):
    ids = list(state.nodes)
    n = len(ids)
    densities = {nid: compute_local_density(nid, state) for nid in ids}
    values = list(densities.values())
    if n < 2 or not state.edges:
        coherence = 0
        bottleneck = 0
    else:
        mean = sum(values) // n
        variance = sum((x - mean) ** 2 for x in values) // n
        max_variance = (n - 1) * SCALE * SCALE // n
        if max_variance == 0:
            coherence = SCALE
        else:
            ratio = variance * SCALE // max_variance
            coherence = max(0, min(SCALE, SCALE - ratio))
        bottleneck = checked_mul(max(values), SCALE) // mean if mean else 0
    balance = {}
    for nid in ids:
        fi, fo = in_degree(nid, state.edges), out_degree(nid, state.edges)
        total = fi + fo
        balance[nid] = {
            "fan_in": fi, "fan_out": fo,
            "imbalance": 0 if not total else abs(fi - fo) * SCALE // total,
        }
    profile = {}
    for nid in ids:
        primary = max(1 + densities[nid] // SCALE, 1)
        neighbors = set()
        for edge in state.edges:
            if edge.source == nid:
                neighbors.add(edge.target)
            if edge.target == nid:
                neighbors.add(edge.source)
        secondary = 0
        for other in neighbors:
            if other in state.nodes:
                secondary = checked_add(secondary, max(densities[other] // SCALE, 1))
        profile[nid] = checked_add(primary, secondary)
    total = sum(profile.values())
    resilience = (
        0 if not profile
        else SCALE if not total
        else checked_mul(len(profile), SCALE) // total
    )
    return coherence, bottleneck, balance, profile, resilience


def assert_equal(state):
    c, b, f, p, r = oracle(state)
    assert connectivity_coherence(state) == c
    assert topological_bottleneck_ratio(state) == b
    assert fan_balance(state) == f
    assert perturbation_profile(state) == p
    assert structural_perturbation_resilience(state) == r
    report = evaluate(state)
    assert report["connectivity_coherence"] == c
    assert report["topological_bottleneck_ratio"] == b
    assert report["fan_balance"] == f
    assert report["perturbation_profile"] == p
    assert report["structural_perturbation_resilience"] == r
    assert report["isolated_nodes"] == find_isolated_nodes(state)
    assert report["critical_cycles"] == detect_critical_cycles(state)


def test_empty_and_parallel_self_loop_and_ghost_reference_cases():
    assert_equal(GraphState())
    for arcs in (
        [("n0", "n0")],
        [("n0", "n1")] * 5,
        [("n0", "ghost"), ("ghost", "n0"), ("n1", "ghost")],
        [("n0", "n0"), ("n0", "n1"), ("n0", "n1")],
        [("ghost", "ghost"), ("ghost", "n1")],
        [("n0", "n1"), ("n1", "n0"), ("n1", "n1")],
    ):
        state = GraphState(
            nodes={v: Node(v, v, "test") for v in ("n0", "n1", "n2")},
            edges=[Edge(a, b, critical=(i % 2 == 0)) for i, (a, b) in enumerate(arcs)],
        )
        assert_equal(state)


def test_randomized_exact_legacy_contract_600_multigraphs():
    rng = random.Random(10719)
    for n in range(1, 9):
        ids = ["n{}".format(i) for i in range(n)]
        endpoints = ids + ["ghost"]
        for _ in range(75):
            arcs = [
                Edge(rng.choice(endpoints), rng.choice(endpoints),
                     critical=rng.choice((True, False)))
                for _ in range(rng.randrange(0, 40))
            ]
            state = GraphState(
                nodes={nid: Node(nid, nid, "test") for nid in ids},
                edges=arcs,
            )
            assert_equal(state)


def test_repeated_evaluation_and_graph_mutation_recomputes_snapshot():
    nodes = {nid: Node(nid, nid, "test") for nid in ("a", "b", "c")}
    state = GraphState(nodes=nodes, edges=[Edge("a", "b")])
    before = evaluate(state)
    assert_equal(state)
    state.edges.append(Edge("b", "c"))
    state.edges.append(Edge("c", "c", critical=True))
    assert_equal(state)
    after = evaluate(state)
    assert after != before
    state.edges.clear()
    assert_equal(state)
    assert evaluate(state)["edge_count"] == 0
