"""Stage 2: independently specified mathematical oracle cases.

Historical metric values and serialization are frozen in this stage.
New opt-in graph-theoretic references have separate names and tests.
"""
from __future__ import annotations


import pytest
import torch

from graphyco.core.primitives import Edge, GraphState, Node, SCALE
from graphyco.core.graph import compute_graph_density, compute_local_density, detect_critical_cycles
from graphyco.core.evaluation import (
    connectivity_coherence,
    topological_bottleneck_ratio,
    fan_balance,
    dead_ratio,
    structural_perturbation_resilience,
    critical_path_density,
    evaluate,
)
from graphyco.core.metric_contracts import QUERY_METRIC_DEFINITIONS
from graphyco.core.structural_reference import (
    directed_node_betweenness,
    pairwise_reachability_ablation_loss,
)
from graphyco.observer.grad_observer import (
    DynamicNodeRecord,
    compute_tensor_stats,
)
from graphyco.observer.grad_eval import (
    compute_gradient_bottleneck,
    compute_gradient_attenuation,
    compute_forward_backward_alignment,
    compute_temporal_stability,
    detect_gradient_anomalies,
)
from graphyco.visualizer.query import QueryEngine, compute_ablation_map


def graph(vertices, arcs, critical=False):
    return GraphState(
        nodes={
            v: Node(id=v, name=v, purpose="test", attributes=["test"])
            for v in vertices
        },
        edges=[Edge(source=a, target=b, critical=critical) for a, b in arcs],
    )


def record(name, activation, gradient=None):
    act = compute_tensor_stats(torch.tensor([float(activation)]))
    grad = (compute_tensor_stats(torch.tensor([float(gradient)]))
            if gradient is not None else None)
    return DynamicNodeRecord(
        node_id=name, node_type="test",
        activation=act, activation_gradient=grad,
    )


def test_legacy_path_three_node_exact_values() -> None:
    path = graph(("A", "B", "C"), [("A", "B"), ("B", "C")], critical=True)
    result = evaluate(path)
    assert result["node_count"] == 3
    assert result["edge_count"] == 2
    assert result["graph_density"] == 3333
    assert {n: compute_local_density(n, path) for n in path.nodes} == {
        "A": 5000, "B": 10000, "C": 5000,
    }
    assert result["connectivity_coherence"] == 9167
    assert result["topological_bottleneck_ratio"] == 15001
    assert result["critical_path_density"] == 10000
    assert result["dead_ratio"] == 0
    assert result["perturbation_profile"] == {"A": 2, "B": 4, "C": 2}
    assert result["structural_perturbation_resilience"] == 3750
    assert fan_balance(path) == {
        "A": {"fan_in": 0, "fan_out": 1, "imbalance": 10000},
        "B": {"fan_in": 1, "fan_out": 1, "imbalance": 0},
        "C": {"fan_in": 1, "fan_out": 0, "imbalance": 10000},
    }


def test_empty_and_disconnected_legacy_contract() -> None:
    empty = graph((), ())
    assert compute_graph_density(empty) == 0
    assert connectivity_coherence(empty) == 0
    assert topological_bottleneck_ratio(empty) == 0
    assert structural_perturbation_resilience(empty) == 0
    disconnected = graph(("A", "B", "C"), [("A", "B")])
    assert dead_ratio(disconnected) == SCALE // 3
    assert critical_path_density(disconnected) == 0
    assert compute_local_density("C", disconnected) == 0


def test_graph_density_legacy_is_raw_edge_count_even_for_parallel_edges() -> None:
    # This is a compatibility assertion, NOT standard simple-graph density.
    parallel = graph(("A", "B"), [("A", "B"), ("A", "B"), ("A", "B")])
    assert compute_graph_density(parallel) == 15000


def test_exact_directed_betweenness_path_and_diamond() -> None:
    path = graph(("A", "B", "C"), [("A", "B"), ("B", "C")])
    assert directed_node_betweenness(path) == {
        "A": 0.0, "B": 0.5, "C": 0.0,
    }
    assert directed_node_betweenness(path, normalized=False) == {
        "A": 0.0, "B": 1.0, "C": 0.0,
    }

    diamond = graph(
        ("A", "B", "C", "D"),
        [("A", "B"), ("A", "C"), ("B", "D"), ("C", "D")],
    )
    result = directed_node_betweenness(diamond)
    assert result["A"] == result["D"] == 0
    assert result["B"] == pytest.approx(1.0 / 12.0)
    assert result["C"] == pytest.approx(1.0 / 12.0)


def test_exact_directed_betweenness_reciprocal_chain_and_duplicate_edges() -> None:
    reciprocal = graph(
        ("A", "B", "C"),
        [("A", "B"), ("B", "A"), ("B", "C"), ("C", "B")],
    )
    assert directed_node_betweenness(reciprocal)["B"] == pytest.approx(1.0)
    repeated = graph(
        ("A", "B", "C"),
        [("A", "B"), ("A", "B"), ("B", "C")],
    )
    assert directed_node_betweenness(repeated)["B"] == pytest.approx(0.5)
    assert directed_node_betweenness(graph((), ())) == {}


def test_exact_reachability_ablation_path_diamond_and_multi_root() -> None:
    path = graph(("A", "B", "C"), [("A", "B"), ("B", "C")])
    assert pairwise_reachability_ablation_loss(path) == {
        "A": 0.0, "B": 1.0, "C": 0.0,
    }
    diamond = graph(
        ("A", "B", "C", "D"),
        [("A", "B"), ("A", "C"), ("B", "D"), ("C", "D")],
    )
    assert pairwise_reachability_ablation_loss(diamond) == {
        "A": 0.0, "B": 0.0, "C": 0.0, "D": 0.0,
    }
    multi_root = graph(
        ("R1", "R2", "M", "O"),
        [("R1", "M"), ("R2", "M"), ("M", "O")],
    )
    assert pairwise_reachability_ablation_loss(multi_root)["M"] == 1.0


def test_legacy_single_source_ablation_is_distinct_and_unchanged() -> None:
    path = graph(("A", "B", "C"), [("A", "B"), ("B", "C")])
    legacy = compute_ablation_map(path)
    exact = pairwise_reachability_ablation_loss(path)
    assert legacy == {"A": 0.0, "B": pytest.approx(2 / 3), "C": pytest.approx(1 / 3)}
    assert exact["B"] == 1.0
    assert exact["C"] == 0.0


def test_dynamic_rms_concentration_and_attenuation_oracles() -> None:
    records = {"A": record("A", 1.0, 1.0), "B": record("B", 2.0, 3.0)}
    result = compute_gradient_bottleneck(records)
    assert result.mean_gradient == 2.0
    assert result.bottleneck_node == "B"
    assert result.max_concentration == pytest.approx(1.5)
    assert result.node_concentrations["A"] == pytest.approx(0.5)
    edge = compute_gradient_attenuation(
        [Edge("A", "B")], records,
    )[("A", "B")]
    assert edge.forward_ratio == pytest.approx(1.0 / 3.0)
    assert edge.backward_ratio == pytest.approx(3.0)
    assert edge.log_forward_ratio == pytest.approx(-1.09861229, abs=1e-7)

    # Legacy semantics treat a missing gradient as a measured zero.
    missing = {**records, "C": record("C", 7.0)}
    assert compute_gradient_bottleneck(missing).max_concentration == pytest.approx(2.25)


def test_dynamic_alignment_temporal_and_anomaly_oracles() -> None:
    descending = {
        "A": record("A", 1.0, 3.0),
        "B": record("B", 2.0, 2.0),
        "C": record("C", 3.0, 1.0),
    }
    alignment = compute_forward_backward_alignment(descending)
    assert alignment.spearman_rho == pytest.approx(-1.0)
    assert alignment.pearson_r == pytest.approx(-1.0)

    stable = compute_temporal_stability([
        {"A": record("A", 1.0, 1.0)},
        {"A": record("A", 3.0, 3.0)},
    ])
    assert stable["A"].mean_gradient == 2.0
    assert stable["A"].std_gradient == 1.0
    assert stable["A"].cv == pytest.approx(0.5, rel=1e-7)

    alerts = detect_gradient_anomalies({
        "low": record("low", 1.0, 1e-8),
        "high": record("high", 1.0, 200.0),
        "normal": record("normal", 1.0, 1.0),
    })
    assert alerts.vanishing_nodes == ["low"]
    assert alerts.exploding_nodes == ["high"]


def test_public_query_descriptions_are_truthful_and_legacy_keys_stable() -> None:
    values = {
        "density": 0.2,
        "coherence": 0.5,
        "topological_bottleneck": 1.6,
        "resilience": 0.3,
        "grad_bottleneck_gmax": 2.4,
        "bottleneck_node": "B",
        "spearman_act_grad": 0.1,
        "mean_stability_cv": 0.4,
    }
    before = dict(values)
    engine = QueryEngine({"architectures": {"Toy": {"summary": values}}})
    frame = engine.describe()
    assert "Incident-Edge Uniformity (C)" in frame.index
    assert "Degree-Concentration Proxy (B)" in frame.index
    assert "Structural Cost Proxy (R)" in frame.index
    assert "not node betweenness" in frame.loc["Degree-Concentration Proxy (B)", "Description"]
    assert "not measured reachability" in frame.loc["Structural Cost Proxy (R)", "Description"]
    assert values == before
    assert {key for key, _, _ in QUERY_METRIC_DEFINITIONS} == set(values)


def test_all_three_vertex_directed_graphs_against_independent_oracles() -> None:
    """Enumerate all 64 simple directed 3-node graphs (incl. cycles/shortcuts)."""
    vertices = ("A", "B", "C")
    possible = [(a, b) for a in vertices for b in vertices if a != b]

    def all_simple_paths(adjacency, source, destination, trail):
        if source == destination:
            return [tuple(trail)]
        paths = []
        for neighbor in adjacency[source]:
            if neighbor not in trail:
                paths.extend(all_simple_paths(
                    adjacency, neighbor, destination, trail + [neighbor]
                ))
        return paths

    def independent_reachability(adjacency):
        # Floyd-Warshall oracle, deliberately different from production BFS.
        reachable = {(a, b) for a in vertices for b in adjacency.get(a, ())}
        for middle in vertices:
            for source in vertices:
                for target in vertices:
                    if (source, middle) in reachable and (middle, target) in reachable:
                        reachable.add((source, target))
        return {pair for pair in reachable if pair[0] != pair[1]}

    for mask in range(1 << len(possible)):
        arcs = [arc for bit, arc in enumerate(possible) if mask & (1 << bit)]
        state = graph(vertices, arcs)
        adjacency = {v: set() for v in vertices}
        for a, b in arcs:
            adjacency[a].add(b)

        shortest_credit = {v: 0.0 for v in vertices}
        for source in vertices:
            for destination in vertices:
                if source == destination:
                    continue
                routes = all_simple_paths(adjacency, source, destination, [source])
                if not routes:
                    continue
                shortest_len = min(len(p) for p in routes)
                best_routes = [p for p in routes if len(p) == shortest_len]
                for route in best_routes:
                    for intermediate in route[1:-1]:
                        shortest_credit[intermediate] += 1 / len(best_routes)

        computed_bc = directed_node_betweenness(state, normalized=False)
        for v in vertices:
            assert computed_bc[v] == pytest.approx(shortest_credit[v]), (mask, v)

        original_pairs = independent_reachability(adjacency)
        calculated_loss = pairwise_reachability_ablation_loss(state)
        for removed in vertices:
            baseline = {p for p in original_pairs if removed not in p}
            updated_adjacency = {
                v: {t for t in adjacency[v] if t != removed}
                for v in vertices if v != removed
            }
            # All surviving pairs are checked by an independent DFS.
            remaining_reachable = {
                (s, t)
                for s, t in baseline
                if all_simple_paths(updated_adjacency, s, t, [s])
            }
            oracle_loss = (
                (len(baseline) - len(remaining_reachable)) / len(baseline)
                if baseline else 0.0
            )
            assert calculated_loss[removed] == pytest.approx(oracle_loss), (
                mask, removed,
            )


def test_cycle_witnesses_follow_real_directed_critical_edges() -> None:
    cycle = graph(
        ("A", "B", "C"),
        [("A", "B"), ("B", "C"), ("C", "A")],
        critical=True,
    )
    assert detect_critical_cycles(cycle) == [["A", "B", "C", "A"]]
    assert evaluate(cycle)["cycle_pressure"] == 1

    # Exhaustive 3-node directed graphs: every reported witness is a
    # closed walk using only true, forward-directed critical edges.
    verts = ("A", "B", "C")
    possible = [(a, b) for a in verts for b in verts if a != b]
    for mask in range(1 << len(possible)):
        arcs = [arc for i, arc in enumerate(possible) if mask & (1 << i)]
        state = graph(verts, arcs, critical=True)
        edge_set = set(arcs)
        for witness in detect_critical_cycles(state):
            assert witness[0] == witness[-1], (mask, witness)
            assert len(witness) >= 3, (mask, witness)
            assert all(
                (a, b) in edge_set
                for a, b in zip(witness, witness[1:])
            ), (mask, witness)
