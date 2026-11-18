from __future__ import annotations
import re
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import torch
import torch.nn as nn
from torch.fx import symbolic_trace, Graph, GraphModule, Node as FxNode
from ..core.primitives import Edge, GraphState, Node, CapacityVector, DomainConstants
from ..core.graph import compute_graph_density, compute_local_density
from ..core.evaluation import evaluate
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

def module_to_graph(
    model: nn.Module,
    include_containers: bool = True,
    min_params: int = 0,
) -> GraphState:
    """Inspect the registered nn.Module hierarchy, NOT execution dataflow.

    A hierarchy edge means "contains". It is neither a tensor dependency
    nor an assumed sequential link. In leaf-only mode there are no inferred
    edges. This deliberately changes the old, misleading sibling-order edges.
    """
    nodes: Dict[str, Node] = {}
    edges: List[Edge] = []
    path_ids: Dict[str, str] = {}
    param_counts: Dict[str, int] = {}

    for path, module in model.named_modules():
        is_leaf = not any(module.children())
        if not include_containers and not is_leaf:
            continue
        pc = _param_count(module)
        if is_leaf and pc < min_params:
            continue

        name = path or "root"
        nid = _clean_id(name)
        if nid in nodes:
            # Qualified module names may collide after punctuation cleaning.
            base, index = nid, 2
            while nid in nodes:
                nid = f"{base}_{index}"
                index += 1

        nodes[nid] = Node(
            id=nid, name=name, purpose=type(module).__name__,
            attributes=_module_attributes(module),
            in_ports=[], out_ports=[], phase="stable", active=True,
        )
        path_ids[path] = nid
        param_counts[nid] = pc

    if include_containers:
        # Only real parent-child membership edges; never infer execution from
        # registration or alphabetical order.
        for path, child_id in path_ids.items():
            if not path:
                continue
            parent_path = path.rsplit(".", 1)[0] if "." in path else ""
            parent_id = path_ids.get(parent_path)
            if parent_id is not None:
                edges.append(Edge(
                    source=parent_id, target=child_id,
                    edge_type="contains", critical=False,
                ))

    return GraphState(
        nodes=nodes, edges=edges,
        capacity_vector=CapacityVector(axes={
            "parameters": sum(param_counts.values()),
            "layer_count": len(nodes),
        }),
        constants=DomainConstants(), phase="stable",
        structural_debt=0, event_history=[],
        provenance={
            "graph_kind": "module_hierarchy" if include_containers else "module_leaf_inventory",
            "edge_semantics": "containment_only" if include_containers else "no_edges",
            "source": "named_modules",
            "execution_verified": False,
            "dataflow_verified": False,
            "trace_status": "not_attempted",
        },
    )


def _fx_to_graph(model: nn.Module, traced: GraphModule) -> GraphState:
    """Convert a single PyTorch FX trace into explicit dependency edges."""
    fx_graph: Graph = traced.graph
    nodes: Dict[str, Node] = {}
    edges: List[Edge] = []

    def _input_nodes(value: Any) -> List[FxNode]:
        result: List[FxNode] = []
        seen: Set[FxNode] = set()

        def visit(node: FxNode) -> FxNode:
            if node not in seen:
                seen.add(node)
                result.append(node)
            return node

        torch.fx.map_arg(value, visit)
        return result

    for fx_node in fx_graph.nodes:
        nid = _clean_id(fx_node.name)
        if nid in nodes:
            raise ValueError(f"Colliding FX node ID after normalization: {nid}")
        attrs = [fx_node.op]
        if fx_node.op == "call_module":
            target_module = traced.get_submodule(str(fx_node.target))
            attrs.extend(_module_attributes(target_module))
        elif fx_node.op == "call_function":
            attrs.append(getattr(fx_node.target, "__name__", str(fx_node.target)))
        elif fx_node.op == "call_method":
            attrs.append(str(fx_node.target))

        # FX all_input_nodes covers both positional and keyword tensor refs.
        in_ports = sorted({_clean_id(n.name) for n in fx_node.all_input_nodes})
        out_ports = sorted({_clean_id(n.name) for n in fx_node.users})
        nodes[nid] = Node(
            id=nid, name=fx_node.name, purpose=fx_node.op,
            attributes=sorted(set(attrs)),
            in_ports=in_ports, out_ports=out_ports, phase="stable", active=True,
        )

    for fx_node in fx_graph.nodes:
        target_id = _clean_id(fx_node.name)
        positional = set(_input_nodes(fx_node.args))
        for source_node in fx_node.all_input_nodes:
            source_id = _clean_id(source_node.name)
            edges.append(Edge(
                source=source_id, target=target_id,
                # Keyword arguments still represent actual FX dependencies;
                # 'informational' only describes parameter position, not a
                # merely inferred relationship.
                edge_type="operational" if source_node in positional else "informational",
                critical=True,
            ))

    total_params = sum(p.numel() for p in model.parameters())
    return GraphState(
        nodes=nodes, edges=edges,
        capacity_vector=CapacityVector(axes={
            "parameters": total_params, "op_count": len(nodes),
        }),
        constants=DomainConstants(), phase="stable",
        structural_debt=0, event_history=[],
        provenance={
            "graph_kind": "fx_symbolic_trace",
            "edge_semantics": "fx_dependencies",
            "source": "torch.fx.symbolic_trace",
            "trace_status": "success",
            # FX is a symbolic blueprint, NOT a verified runtime execution.
            "execution_verified": False,
            "dataflow_verified": "symbolic_only",
        },
    )


def trace_to_graph(
    model: nn.Module,
    concrete_args: Optional[Dict] = None,
) -> GraphState:
    return _fx_to_graph(model, symbolic_trace(model, concrete_args=concrete_args))


def evaluate_model(
    model: nn.Module,
    mode: str = "trace",
    concrete_args: Optional[Dict] = None,
    strict_trace: bool = False,
) -> Dict:
    """Evaluate FX dependencies or explicit module containment.

    FX failure can fall back to an honestly labeled hierarchy (default).
    Set strict_trace=True to surface the original exception instead.
    """
    if mode not in ("trace", "module"):
        raise ValueError(f"Unknown extraction mode: {mode!r}")

    if mode == "trace":
        try:
            state = trace_to_graph(model, concrete_args)
        except Exception as exc:
            if strict_trace:
                raise
            state = module_to_graph(model)
            state.provenance.update({
                "trace_status": "failed",
                "fallback_from": "fx_symbolic_trace",
                "fallback_reason_type": type(exc).__name__,
                "fallback_reason": str(exc)[:500],
            })
    else:
        state = module_to_graph(model)

    eval_result = evaluate(state)
    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    layer_types: Dict[str, int] = {}
    for module in model.modules():
        name = type(module).__name__
        layer_types[name] = layer_types.get(name, 0) + 1

    return {
        "graph_state": state,
        "graph_provenance": dict(state.provenance),
        "evaluation": eval_result,
        "model_info": {
            "total_parameters": total_params,
            "trainable_parameters": trainable,
            "layer_type_counts": layer_types,
        },
    }


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
        "graph_provenance": dict(dyn_state.graph_state.provenance),
        "static_evaluation": dyn_state.static_evaluation,
        "dynamic_evaluation": dyn_state.dynamic_evaluation,
        "model_info": {
            "total_parameters": total_params,
            "trainable_parameters": trainable,
            "layer_type_counts": layer_types,
        },
    }

