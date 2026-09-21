from __future__ import annotations
import re
from typing import List
from .primitives import GraphState, NODE_ID_PATTERN
from .graph import detect_critical_cycles

class InvariantViolationError(Exception):

    def __init__(self, rule: str, detail: str) -> None:
        self.rule = rule
        self.detail = detail
        super().__init__(f'[INVARIANT:{rule}] {detail}')

def validate_invariants(state: GraphState) -> None:
    _check_node_id_format(state)
    _check_edge_integrity(state)
    _check_port_connectivity(state)
    _check_node_uniqueness(state)
    _check_graph_liveness(state)
    _check_attributes_nonempty(state)
    _check_acyclic_critical_subgraph(state)

def _check_node_id_format(state: GraphState) -> None:
    for nid in state.nodes:
        if not NODE_ID_PATTERN.match(nid):
            raise InvariantViolationError('node_id_format', f'Node ID {nid!r} contains invalid characters — must match [a-zA-Z0-9_-]+')

def _check_edge_integrity(state: GraphState) -> None:
    for edge in state.edges:
        if edge.source not in state.nodes:
            raise InvariantViolationError('edge_integrity', f'Edge source={edge.source!r} does not exist in nodes')
        if edge.target not in state.nodes:
            raise InvariantViolationError('edge_integrity', f'Edge target={edge.target!r} does not exist in nodes')

def _check_port_connectivity(state: GraphState) -> None:
    all_inputs: set[str] = set()
    for node in state.nodes.values():
        all_inputs.update(node.in_ports)
    for node in state.nodes.values():
        for output in node.out_ports:
            if output not in all_inputs:
                raise InvariantViolationError('port_connectivity', f'Node {node.id!r} produces output {output!r} that no node consumes as in_port')

def _check_node_uniqueness(state: GraphState) -> None:
    ids = list(state.nodes.keys())
    if len(ids) != len(set(ids)):
        raise InvariantViolationError('node_uniqueness', 'Duplicate node IDs detected')

def _check_graph_liveness(state: GraphState) -> None:
    if not state.nodes:
        return
    if not any((n.active for n in state.nodes.values())):
        raise InvariantViolationError('graph_liveness', 'No active nodes remain in the graph')

def _check_attributes_nonempty(state: GraphState) -> None:
    for node in state.nodes.values():
        if not node.attributes:
            raise InvariantViolationError('attributes_nonempty', f'Node {node.id!r} has zero attributes')

def _check_acyclic_critical_subgraph(state: GraphState) -> None:
    cycles = detect_critical_cycles(state)
    if cycles:
        cycle_str = ' -> '.join(cycles[0])
        raise InvariantViolationError('acyclic_critical_subgraph', f'Critical edge cycle detected: {cycle_str}')
