from __future__ import annotations
from typing import Dict, List, Set, Tuple
from .primitives import Edge, GraphState, SCALE, checked_mul

def build_adjacency_map(edges: List[Edge]) -> Dict[str, List[str]]:
    adj: Dict[str, List[str]] = {}
    for edge in edges:
        adj.setdefault(edge.source, []).append(edge.target)
    return adj

def compute_graph_density(state: GraphState) -> int:
    n = len(state.nodes)
    if n < 2:
        return 0
    max_edges = n * (n - 1)
    if max_edges == 0:
        return 0
    return checked_mul(len(state.edges), SCALE) // max_edges

def compute_local_density(node_id: str, state: GraphState) -> int:
    if not state.edges:
        return 0
    count = sum((1 for e in state.edges if e.source == node_id or e.target == node_id))
    total = len(state.edges)
    if total == 0:
        return 0
    return checked_mul(count, SCALE) // total

def find_isolated_nodes(state: GraphState) -> List[str]:
    connected: Set[str] = set()
    for edge in state.edges:
        connected.add(edge.source)
        connected.add(edge.target)
    return sorted((nid for nid in state.nodes if nid not in connected))

def in_degree(node_id: str, edges: List[Edge]) -> int:
    return sum((1 for e in edges if e.target == node_id))

def out_degree(node_id: str, edges: List[Edge]) -> int:
    return sum((1 for e in edges if e.source == node_id))

def detect_critical_cycles(state: GraphState) -> List[List[str]]:
    critical_adj: Dict[str, List[str]] = {}
    for edge in state.edges:
        if edge.critical:
            critical_adj.setdefault(edge.source, []).append(edge.target)
    WHITE, GREY, BLACK = (0, 1, 2)
    colour: Dict[str, int] = {nid: WHITE for nid in sorted(state.nodes)}
    parent: Dict[str, str | None] = {}
    cycles: List[List[str]] = []

    def _dfs(start: str) -> None:
        stack: List[Tuple[str, int]] = [(start, 0)]
        colour[start] = GREY
        parent[start] = None
        while stack:
            node, idx = stack[-1]
            neighbours = sorted(critical_adj.get(node, []))
            if idx < len(neighbours):
                stack[-1] = (node, idx + 1)
                nbr = neighbours[idx]
                if colour.get(nbr, WHITE) == GREY:
                    cycle = [nbr]
                    for sn, _ in reversed(stack):
                        cycle.append(sn)
                        if sn == nbr:
                            break
                    cycles.append(cycle)
                elif colour.get(nbr, WHITE) == WHITE:
                    colour[nbr] = GREY
                    parent[nbr] = node
                    stack.append((nbr, 0))
            else:
                colour[node] = BLACK
                stack.pop()
    for nid in sorted(state.nodes):
        if colour.get(nid, WHITE) == WHITE:
            _dfs(nid)
    return cycles
