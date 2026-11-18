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
    """Validate FX-style port references in both directions.

    For a dependency src -> dst, src.out_ports stores dst.id and
    dst.in_ports stores src.id. The old check incorrectly looked for
    dst.id among all input ports; it would reject ordinary sink edges.
    This invariant checks the reciprocal reference, not merely the
    presence of a matching identifier anywhere in the graph.
    """
    for node in state.nodes.values():
        for consumer_id in node.out_ports:
            consumer = state.nodes.get(consumer_id)
            if consumer is None or node.id not in consumer.in_ports:
                raise InvariantViolationError(
                    'port_connectivity',
                    f'Node {node.id!r} lists consumer {consumer_id!r} '
                    'without reciprocal in_port reference',
                )
        for producer_id in node.in_ports:
            producer = state.nodes.get(producer_id)
            if producer is None or node.id not in producer.out_ports:
                raise InvariantViolationError(
                    'port_connectivity',
                    f'Node {node.id!r} lists producer {producer_id!r} '
                    'without reciprocal out_port reference',
                )

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
