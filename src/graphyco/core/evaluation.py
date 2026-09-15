from __future__ import annotations
from typing import Dict, List
from .primitives import GraphState, SCALE, checked_add, checked_mul
from .graph import compute_graph_density, compute_local_density, find_isolated_nodes, detect_critical_cycles, in_degree, out_degree

def connectivity_coherence(state: GraphState) -> int:
    n = len(state.nodes)
    if n < 2 or not state.edges:
        return 0
    densities = [compute_local_density(nid, state) for nid in state.nodes]
    mean_d = sum(densities) // n
    variance = sum(((d - mean_d) ** 2 for d in densities)) // n
    max_variance = (n - 1) * SCALE * SCALE // n
    if max_variance == 0:
        return SCALE
    ratio = variance * SCALE // max_variance
    coherence = SCALE - ratio
    return max(0, min(SCALE, coherence))

def topological_bottleneck_ratio(state: GraphState) -> int:
    if len(state.nodes) < 2 or not state.edges:
        return 0
    densities = [compute_local_density(nid, state) for nid in state.nodes]
    max_d = max(densities)
    mean_d = sum(densities) // len(densities)
    if mean_d == 0:
        return 0
    return checked_mul(max_d, SCALE) // mean_d

def fan_balance(state: GraphState) -> Dict[str, Dict[str, int]]:
    result = {}
    for nid in state.nodes:
        fi = in_degree(nid, state.edges)
        fo = out_degree(nid, state.edges)
        total = fi + fo
        imbalance = 0 if total == 0 else abs(fi - fo) * SCALE // total
        result[nid] = {'fan_in': fi, 'fan_out': fo, 'imbalance': imbalance}
    return result

def dead_ratio(state: GraphState) -> int:
    if not state.nodes:
        return 0
    isolated = find_isolated_nodes(state)
    return checked_mul(len(isolated), SCALE) // len(state.nodes)

def cycle_pressure(state: GraphState) -> int:
    return len(detect_critical_cycles(state))

def perturbation_profile(state: GraphState) -> Dict[str, int]:
    result = {}
    for nid in state.nodes:
        d = compute_local_density(nid, state)
        primary = max(1 + d // SCALE, 1)
        neighbors = set()
        for e in state.edges:
            if e.source == nid:
                neighbors.add(e.target)
            if e.target == nid:
                neighbors.add(e.source)
        secondary = 0
        for cid in neighbors:
            if cid in state.nodes:
                cd = compute_local_density(cid, state)
                secondary = checked_add(secondary, max(cd // SCALE, 1))
        result[nid] = checked_add(primary, secondary)
    return result

def structural_perturbation_resilience(state: GraphState) -> int:
    profile = perturbation_profile(state)
    if not profile:
        return 0
    total_cost = sum(profile.values())
    n = len(profile)
    if total_cost == 0:
        return SCALE
    return checked_mul(n, SCALE) // total_cost

def critical_path_density(state: GraphState) -> int:
    if not state.edges:
        return 0
    critical_count = sum((1 for e in state.edges if e.critical))
    return checked_mul(critical_count, SCALE) // len(state.edges)

def evaluate(state: GraphState) -> Dict:
    return {'node_count': len(state.nodes), 'edge_count': len(state.edges), 'graph_density': compute_graph_density(state), 'structural_debt': state.structural_debt, 'phase': state.phase, 'connectivity_coherence': connectivity_coherence(state), 'topological_bottleneck_ratio': topological_bottleneck_ratio(state), 'dead_ratio': dead_ratio(state), 'cycle_pressure': cycle_pressure(state), 'structural_perturbation_resilience': structural_perturbation_resilience(state), 'critical_path_density': critical_path_density(state), 'isolated_nodes': find_isolated_nodes(state), 'critical_cycles': detect_critical_cycles(state), 'perturbation_profile': perturbation_profile(state), 'fan_balance': fan_balance(state)}
