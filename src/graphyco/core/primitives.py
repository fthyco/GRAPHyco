from __future__ import annotations
import copy
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional
SCALE: int = 10000
NODE_ID_PATTERN = re.compile('^[a-zA-Z0-9_-]+$')

def validate_node_id(node_id: str) -> None:
    if not NODE_ID_PATTERN.match(node_id):
        raise ValueError(f'Invalid node ID {node_id!r}: must match [a-zA-Z0-9_-]+')
_INT64_MIN = -2 ** 63
_INT64_MAX = 2 ** 63 - 1

def checked_add(a: int, b: int) -> int:
    result = a + b
    if result < _INT64_MIN or result > _INT64_MAX:
        raise OverflowError(f'Integer overflow: {a} + {b} = {result}')
    return result

def checked_mul(a: int, b: int) -> int:
    result = a * b
    if result < _INT64_MIN or result > _INT64_MAX:
        raise OverflowError(f'Integer overflow: {a} * {b} = {result}')
    return result

@dataclass
class Node:
    id: str
    name: str
    purpose: str
    attributes: List[str] = field(default_factory=list)
    in_ports: List[str] = field(default_factory=list)
    out_ports: List[str] = field(default_factory=list)
    phase: str = 'nascent'
    active: bool = True

    def to_dict(self) -> dict:
        return {'id': self.id, 'name': self.name, 'purpose': self.purpose, 'attributes': sorted(self.attributes), 'in_ports': sorted(self.in_ports), 'out_ports': sorted(self.out_ports), 'phase': self.phase, 'active': self.active}

@dataclass
class Edge:
    source: str
    target: str
    edge_type: str = 'operational'
    critical: bool = False

    def to_dict(self) -> dict:
        return {'source': self.source, 'target': self.target, 'edge_type': self.edge_type, 'critical': self.critical}

@dataclass
class CapacityVector:
    axes: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return dict(sorted(self.axes.items()))

    def capacity_index(self) -> int:
        if not self.axes:
            return 0
        total = 0
        for v in self.axes.values():
            total = checked_add(total, v)
        return total // len(self.axes)

@dataclass(frozen=True)
class DomainConstants:
    fission_threshold: int = 3
    fission_min_capacity: int = 60000
    fusion_max_attributes: int = 5
    cascade_failure_threshold: int = 8
    cascade_debt_base_multiplier: int = 1
    deferred_split_debt_increment: int = 1

    def to_dict(self) -> dict:
        return {'cascade_debt_base_multiplier': self.cascade_debt_base_multiplier, 'cascade_failure_threshold': self.cascade_failure_threshold, 'deferred_split_debt_increment': self.deferred_split_debt_increment, 'fission_min_capacity': self.fission_min_capacity, 'fission_threshold': self.fission_threshold, 'fusion_max_attributes': self.fusion_max_attributes}

@dataclass(frozen=True)
class TransitionResult:
    event_type: str = ''
    success: bool = True
    split_executed: bool = False
    split_deferred: bool = False
    split_skipped: bool = False
    merge_executed: bool = False
    node_failed: bool = False
    reason: str = ''
    primary_debt: int = 0
    secondary_debt: int = 0
    target_density: int = 0
    cascade_target: str = ''
    magnitude: int = 0

@dataclass
class GraphState:
    nodes: Dict[str, Node] = field(default_factory=dict)
    edges: List[Edge] = field(default_factory=list)
    capacity_vector: CapacityVector = field(default_factory=CapacityVector)
    constants: DomainConstants = field(default_factory=DomainConstants)
    phase: str = 'nascent'
    structural_debt: int = 0
    event_history: List[dict] = field(default_factory=list)

    def copy(self) -> 'GraphState':
        return copy.deepcopy(self)

    def to_dict(self) -> dict:
        return {'nodes': {nid: n.to_dict() for nid, n in sorted(self.nodes.items())}, 'edges': [e.to_dict() for e in sorted(self.edges, key=lambda e: (e.source, e.target, e.edge_type))], 'capacity_vector': self.capacity_vector.to_dict(), 'phase': self.phase, 'structural_debt': self.structural_debt, 'event_count': len(self.event_history)}
