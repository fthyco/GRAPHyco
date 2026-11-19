"""Legacy static graph metrics with shared linear-time topology accounting.

These are deliberately the historical fixed-point / proxy definitions, not
new graph-theoretic estimators. Edge multiplicity and self-loop semantics
remain unchanged; callers get identical integer results and key schemas.
"""
from __future__ import annotations

from typing import Dict, Mapping, Set, Tuple

from .primitives import GraphState, SCALE, checked_add, checked_mul
from .graph import compute_graph_density, find_isolated_nodes, detect_critical_cycles


def _topology_snapshot(
    state: GraphState,
) -> Tuple[Dict[str, int], Dict[str, int], Dict[str, int], Dict[str, Set[str]]]:
    """Single pass over edges, keeping three *distinct* degree conventions.

    Legacy density counts a self-loop once at its endpoint, while fan_in and
    fan_out each count it once. Parallel directed edges count repeatedly.
    Unknown endpoint ids do not become new graph nodes. Neighbor sets are
    undirected and de-duplicated, matching perturbation_profile's set logic.
    """
    incident = dict.fromkeys(state.nodes, 0)
    fan_in = dict.fromkeys(state.nodes, 0)
    fan_out = dict.fromkeys(state.nodes, 0)
    neighbors: Dict[str, Set[str]] = {nid: set() for nid in state.nodes}
    for edge in state.edges:
        a, b = edge.source, edge.target
        if a in incident:
            incident[a] += 1
            fan_out[a] += 1
            neighbors[a].add(b)
        if b in incident:
            if b != a:
                incident[b] += 1
            fan_in[b] += 1
            neighbors[b].add(a)
    return incident, fan_in, fan_out, neighbors


def _densities(state: GraphState, incident: Mapping[str, int]) -> Dict[str, int]:
    total = len(state.edges)
    if not total:
        return dict.fromkeys(state.nodes, 0)
    return {nid: checked_mul(count, SCALE) // total for nid, count in incident.items()}


def _coherence(state: GraphState, densities: Mapping[str, int]) -> int:
    n = len(state.nodes)
    if n < 2 or not state.edges:
        return 0
    values = list(densities.values())
    mean_d = sum(values) // n
    variance = sum((d - mean_d) ** 2 for d in values) // n
    max_variance = (n - 1) * SCALE * SCALE // n
    if max_variance == 0:
        return SCALE
    ratio = variance * SCALE // max_variance
    return max(0, min(SCALE, SCALE - ratio))


def _bottleneck(state: GraphState, densities: Mapping[str, int]) -> int:
    if len(state.nodes) < 2 or not state.edges:
        return 0
    values = list(densities.values())
    mean_d = sum(values) // len(values)
    if mean_d == 0:
        return 0
    return checked_mul(max(values), SCALE) // mean_d


def _fan_balance(
    state: GraphState, fan_in: Mapping[str, int], fan_out: Mapping[str, int],
) -> Dict[str, Dict[str, int]]:
    result = {}
    for nid in state.nodes:
        fi, fo = fan_in[nid], fan_out[nid]
        total = fi + fo
        imbalance = 0 if total == 0 else abs(fi - fo) * SCALE // total
        result[nid] = {"fan_in": fi, "fan_out": fo, "imbalance": imbalance}
    return result


def _profile(
    state: GraphState,
    densities: Mapping[str, int],
    neighbors: Mapping[str, Set[str]],
) -> Dict[str, int]:
    result = {}
    for nid in state.nodes:
        primary = max(1 + densities[nid] // SCALE, 1)
        secondary = 0
        for cid in neighbors[nid]:
            if cid in state.nodes:
                secondary = checked_add(secondary, max(densities[cid] // SCALE, 1))
        result[nid] = checked_add(primary, secondary)
    return result


def _resilience(profile: Mapping[str, int]) -> int:
    if not profile:
        return 0
    total_cost = sum(profile.values())
    if total_cost == 0:
        return SCALE
    return checked_mul(len(profile), SCALE) // total_cost


def connectivity_coherence(state: GraphState) -> int:
    incident, _, _, _ = _topology_snapshot(state)
    return _coherence(state, _densities(state, incident))


def topological_bottleneck_ratio(state: GraphState) -> int:
    incident, _, _, _ = _topology_snapshot(state)
    return _bottleneck(state, _densities(state, incident))


def fan_balance(state: GraphState) -> Dict[str, Dict[str, int]]:
    _, fan_in, fan_out, _ = _topology_snapshot(state)
    return _fan_balance(state, fan_in, fan_out)


def dead_ratio(state: GraphState) -> int:
    if not state.nodes:
        return 0
    isolated = find_isolated_nodes(state)
    return checked_mul(len(isolated), SCALE) // len(state.nodes)


def cycle_pressure(state: GraphState) -> int:
    return len(detect_critical_cycles(state))


def perturbation_profile(state: GraphState) -> Dict[str, int]:
    incident, _, _, neighbors = _topology_snapshot(state)
    return _profile(state, _densities(state, incident), neighbors)


def structural_perturbation_resilience(state: GraphState) -> int:
    return _resilience(perturbation_profile(state))


def critical_path_density(state: GraphState) -> int:
    if not state.edges:
        return 0
    critical_count = sum(1 for e in state.edges if e.critical)
    return checked_mul(critical_count, SCALE) // len(state.edges)


def evaluate(state: GraphState) -> Dict:
    """Reuse one topology snapshot and one profile without mutating GraphState."""
    incident, fan_in, fan_out, neighbors = _topology_snapshot(state)
    densities = _densities(state, incident)
    profile = _profile(state, densities, neighbors)
    isolated = find_isolated_nodes(state)
    cycles = detect_critical_cycles(state)
    return {
        "node_count": len(state.nodes),
        "edge_count": len(state.edges),
        "graph_density": compute_graph_density(state),
        "structural_debt": state.structural_debt,
        "phase": state.phase,
        "connectivity_coherence": _coherence(state, densities),
        "topological_bottleneck_ratio": _bottleneck(state, densities),
        "dead_ratio": (
            checked_mul(len(isolated), SCALE) // len(state.nodes)
            if state.nodes else 0
        ),
        "cycle_pressure": len(cycles),
        "structural_perturbation_resilience": _resilience(profile),
        "critical_path_density": critical_path_density(state),
        "isolated_nodes": isolated,
        "critical_cycles": cycles,
        "perturbation_profile": profile,
        "fan_balance": _fan_balance(state, fan_in, fan_out),
    }
