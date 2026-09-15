from __future__ import annotations
import re
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import torch
import torch.nn as nn
from torch.fx import symbolic_trace, Graph, Node as FxNode
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.primitives import Edge, GraphState, Node, CapacityVector, DomainConstants
from core.graph import compute_graph_density, compute_local_density
from core.evaluation import evaluate
_ID_CLEAN = re.compile('[^a-zA-Z0-9_-]')

def _clean_id(name: str) -> str:
    cleaned = _ID_CLEAN.sub('_', name)
    return cleaned if cleaned else 'unnamed'

def _module_attributes(module: nn.Module) -> List[str]:
    attrs = [type(module).__name__]
    if isinstance(module, nn.Linear):
        attrs.append(f'in={module.in_features}')
        attrs.append(f'out={module.out_features}')
        if module.bias is not None:
            attrs.append('bias')
    elif isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        attrs.append(f'in_ch={module.in_channels}')
        attrs.append(f'out_ch={module.out_channels}')
        attrs.append(f'kernel={module.kernel_size}')
    elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
        attrs.append(f'features={module.num_features}')
    elif isinstance(module, nn.MultiheadAttention):
        attrs.append(f'embed={module.embed_dim}')
        attrs.append(f'heads={module.num_heads}')
    elif isinstance(module, (nn.LSTM, nn.GRU, nn.RNN)):
        attrs.append(f'input={module.input_size}')
        attrs.append(f'hidden={module.hidden_size}')
        attrs.append(f'layers={module.num_layers}')
    elif isinstance(module, nn.Dropout):
        attrs.append(f'p={module.p}')
    elif isinstance(module, nn.LayerNorm):
        attrs.append(f'shape={module.normalized_shape}')
    elif isinstance(module, nn.Embedding):
        attrs.append(f'num={module.num_embeddings}')
        attrs.append(f'dim={module.embedding_dim}')
    return sorted(set(attrs))

def _param_count(module: nn.Module) -> int:
    return sum((p.numel() for p in module.parameters(recurse=False)))

def module_to_graph(model: nn.Module, include_containers: bool=False, min_params: int=0) -> GraphState:
    nodes: Dict[str, Node] = {}
    edges: List[Edge] = []
    param_counts: Dict[str, int] = {}
    for name, module in model.named_modules():
        if not name:
            name = 'root'
        is_leaf = len(list(module.children())) == 0
        if not include_containers and (not is_leaf):
            continue
        pc = _param_count(module)
        if pc < min_params and is_leaf:
            continue
        nid = _clean_id(name)
        attrs = _module_attributes(module)
        nodes[nid] = Node(id=nid, name=name, purpose=type(module).__name__, attributes=attrs, in_ports=[], out_ports=[], phase='stable', active=True)
        param_counts[nid] = pc
    node_ids = sorted(nodes.keys())
    for nid in node_ids:
        parts = nid.rsplit('_', 1)
        if len(parts) > 1:
            parent_id = parts[0]
            if parent_id in nodes:
                edges.append(Edge(source=parent_id, target=nid, edge_type='operational', critical=False))
    parent_children: Dict[str, List[str]] = {}
    for nid in node_ids:
        parts = nodes[nid].name.rsplit('.', 1)
        parent = parts[0] if len(parts) > 1 else 'root'
        parent_children.setdefault(parent, []).append(nid)
    for parent, children in parent_children.items():
        for i in range(len(children) - 1):
            edges.append(Edge(source=children[i], target=children[i + 1], edge_type='operational', critical=True))
    total_params = sum(param_counts.values())
    capacity_axes = {'parameters': total_params, 'layer_count': len(nodes)}
    return GraphState(nodes=nodes, edges=edges, capacity_vector=CapacityVector(axes=capacity_axes), constants=DomainConstants(), phase='stable', structural_debt=0, event_history=[])

def trace_to_graph(model: nn.Module, concrete_args: Optional[Dict]=None) -> GraphState:
    traced = symbolic_trace(model, concrete_args=concrete_args)
    fx_graph: Graph = traced.graph
    nodes: Dict[str, Node] = {}
    edges: List[Edge] = []
    for fx_node in fx_graph.nodes:
        nid = _clean_id(fx_node.name)
        attrs = [fx_node.op]
        if fx_node.op == 'call_module':
            target_module = traced.get_submodule(fx_node.target)
            attrs.extend(_module_attributes(target_module))
        elif fx_node.op == 'call_function':
            attrs.append(getattr(fx_node.target, '__name__', str(fx_node.target)))
        elif fx_node.op == 'call_method':
            attrs.append(str(fx_node.target))
        def _extract_fx_nodes(arg):
            if isinstance(arg, FxNode):
                return [arg]
            elif isinstance(arg, (list, tuple, set)):
                res = []
                for x in arg:
                    res.extend(_extract_fx_nodes(x))
                return res
            elif isinstance(arg, dict):
                res = []
                for x in arg.values():
                    res.extend(_extract_fx_nodes(x))
                return res
            return []

        in_nodes = []
        for arg in fx_node.args:
            in_nodes.extend(_extract_fx_nodes(arg))
        in_ports = [_clean_id(n.name) for n in in_nodes]
        out_ports = [_clean_id(u.name) for u in fx_node.users.keys()]
        nodes[nid] = Node(id=nid, name=fx_node.name, purpose=fx_node.op, attributes=sorted(set(attrs)), in_ports=sorted(in_ports), out_ports=sorted(out_ports), phase='stable', active=True)
    for fx_node in fx_graph.nodes:
        target_id = _clean_id(fx_node.name)
        for arg in fx_node.args:
            for n in _extract_fx_nodes(arg):
                source_id = _clean_id(n.name)
                if source_id in nodes and target_id in nodes:
                    edges.append(Edge(source=source_id, target=target_id, edge_type='operational', critical=True))
        for v in fx_node.kwargs.values():
            for n in _extract_fx_nodes(v):
                source_id = _clean_id(n.name)
                if source_id in nodes and target_id in nodes:
                    edges.append(Edge(source=source_id, target=target_id, edge_type='informational', critical=False))
    total_params = sum((p.numel() for p in model.parameters()))
    capacity_axes = {'parameters': total_params, 'op_count': len(nodes)}
    return GraphState(nodes=nodes, edges=edges, capacity_vector=CapacityVector(axes=capacity_axes), constants=DomainConstants(), phase='stable', structural_debt=0, event_history=[])

def evaluate_model(model: nn.Module, mode: str='trace', concrete_args: Optional[Dict]=None) -> Dict:
    if mode == 'trace':
        try:
            state = trace_to_graph(model, concrete_args)
        except Exception:
            state = module_to_graph(model)
    else:
        state = module_to_graph(model)
    eval_result = evaluate(state)
    total_params = sum((p.numel() for p in model.parameters()))
    trainable = sum((p.numel() for p in model.parameters() if p.requires_grad))
    layer_types = {}
    for m in model.modules():
        t = type(m).__name__
        layer_types[t] = layer_types.get(t, 0) + 1
    return {'graph_state': state, 'evaluation': eval_result, 'model_info': {'total_parameters': total_params, 'trainable_parameters': trainable, 'layer_type_counts': layer_types}}


def evaluate_model_dynamic(
    model: nn.Module,
    inputs: torch.Tensor,
    loss_fn: Optional[Callable] = None,
    mode: str = "trace",
    concrete_args: Optional[Dict] = None,
) -> Dict[str, Any]:
    """Evaluates both static topology and dynamic forward/backward execution flow.

    Returns the unified representation G_t = (V, E, X_t, A_t, G_t) with
    observational runtime metrics alongside deterministic static invariants.
    """
    from .grad_bridge import DynamicExecutionBridge

    bridge = DynamicExecutionBridge(model, mode=mode, concrete_args=concrete_args)
    dyn_state = bridge.step(inputs=inputs, loss_fn=loss_fn)

    total_params = sum((p.numel() for p in model.parameters()))
    trainable = sum((p.numel() for p in model.parameters() if p.requires_grad))
    layer_types: Dict[str, int] = {}
    for m in model.modules():
        t = type(m).__name__
        layer_types[t] = layer_types.get(t, 0) + 1

    return {
        "dynamic_graph_state": dyn_state,
        "graph_state": dyn_state.graph_state,
        "static_evaluation": dyn_state.static_evaluation,
        "dynamic_evaluation": dyn_state.dynamic_evaluation,
        "model_info": {
            "total_parameters": total_params,
            "trainable_parameters": trainable,
            "layer_type_counts": layer_types,
        },
    }

