"""Exact structural reference metrics, distinct from legacy degree-based proxies.

These functions are opt-in and do not alter historical graphyco metric keys.
They operate on a directed, unweighted *simple* graph: repeated edges are
collapsed and self-loops are ignored for shortest-path/reachability analysis.
"""
from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional, Set

from .primitives import GraphState


def _adjacency(state: GraphState, excluded: Optional[str] = None) -> Dict[str, Set[str]]:
    nodes = set(state.nodes)
    nodes.discard(excluded)
    adjacency: Dict[str, Set[str]] = {node: set() for node in nodes}
    for edge in state.edges:
        if edge.source in nodes and edge.target in nodes and edge.source != edge.target:
            adjacency[edge.source].add(edge.target)
    return adjacency


def directed_node_betweenness(state: GraphState, normalized: bool = True) -> Dict[str, float]:
    """Exact directed unweighted Brandes node betweenness, excluding endpoints.

    Multiple shortest paths share credit equally. For a directed graph with N>2,
    normalization is 1/((N-1)(N-2)); unlike undirected betweenness, there is
    no additional factor of 1/2. A disconnected graph is supported.
    """
    graph = _adjacency(state)
    vertices = sorted(graph)
    scores: Dict[str, float] = {v: 0.0 for v in vertices}

    for source in vertices:
        predecessors: Dict[str, List[str]] = {v: [] for v in vertices}
        path_counts: Dict[str, float] = {v: 0.0 for v in vertices}
        distances: Dict[str, int] = {v: -1 for v in vertices}
        path_counts[source] = 1.0
        distances[source] = 0
        queue = deque([source])
        stack: List[str] = []

        while queue:
            vertex = queue.popleft()
            stack.append(vertex)
            for neighbor in sorted(graph[vertex]):
                if distances[neighbor] < 0:
                    distances[neighbor] = distances[vertex] + 1
                    queue.append(neighbor)
                if distances[neighbor] == distances[vertex] + 1:
                    path_counts[neighbor] += path_counts[vertex]
                    predecessors[neighbor].append(vertex)

        dependency: Dict[str, float] = {v: 0.0 for v in vertices}
        while stack:
            node = stack.pop()
            for predecessor in predecessors[node]:
                dependency[predecessor] += (
                    path_counts[predecessor] / path_counts[node]
                ) * (1.0 + dependency[node])
            if node != source:
                scores[node] += dependency[node]

    if normalized and len(vertices) > 2:
        factor = 1.0 / ((len(vertices) - 1) * (len(vertices) - 2))
        scores = {v: value * factor for v, value in scores.items()}
    return scores


def _reachable_pairs(adjacency: Dict[str, Set[str]]) -> Set[tuple[str, str]]:
    pairs: Set[tuple[str, str]] = set()
    for source in adjacency:
        seen = {source}
        queue = deque([source])
        while queue:
            node = queue.popleft()
            for neighbor in adjacency[node]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    queue.append(neighbor)
        pairs.update((source, target) for target in seen if target != source)
    return pairs


def pairwise_reachability_ablation_loss(state: GraphState) -> Dict[str, float]:
    """Fraction of surviving-node ordered reachable pairs lost by deleting v.

    For each v, compare the graph before and after removal, considering only
    ordered pairs (source, target) whose endpoints are both different from v.
    The denominator is the number of those pairs reachable *before* removal;
    zero-denominator cases return 0.0. This avoids counting endpoints lost
    just because v itself is gone, and avoids arbitrary single-root choice.

    It measures topological reachability change, not predictive performance,
    causal importance for ML, or gradient-flow resilience.
    """
    full_adjacency = _adjacency(state)
    all_pairs = _reachable_pairs(full_adjacency)
    loss: Dict[str, float] = {}
    for removed in sorted(full_adjacency):
        baseline = {
            pair for pair in all_pairs if removed not in pair
        }
        if not baseline:
            loss[removed] = 0.0
            continue
        surviving = _reachable_pairs(_adjacency(state, excluded=removed))
        loss[removed] = float(len(baseline - surviving) / len(baseline))
    return loss
