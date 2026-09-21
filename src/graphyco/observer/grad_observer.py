from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import torch
import torch.nn as nn
from torch.fx import GraphModule, Interpreter, Node as FxNode

from core.primitives import SCALE


# ══════════════════════════════════════════════════════════════
#  TENSOR STATISTICS ENGINE
# ══════════════════════════════════════════════════════════════

@dataclass
class TensorStats:
    """Summary statistics for an activation or gradient tensor.

    All metrics are computed on detached float64 representations to prevent
    numerical overflow while leaving model autograd graphs completely untouched.
    """
    numel: int
    mean: float
    std: float
    min_val: float
    max_val: float
    l1_norm: float
    l2_norm: float
    rms: float
    zero_fraction: float
    non_finite_fraction: float
    normalized_magnitude: float  # RMS = L2 / sqrt(N)
    fixed_point_rms: int         # Normalized RMS scaled to int64 fixed-point

    def to_dict(self) -> Dict[str, Any]:
        return {
            "numel": self.numel,
            "mean": self.mean,
            "std": self.std,
            "min": self.min_val,
            "max": self.max_val,
            "l1": self.l1_norm,
            "l2": self.l2_norm,
            "rms": self.rms,
            "zero_fraction": self.zero_fraction,
            "non_finite_fraction": self.non_finite_fraction,
            "normalized_magnitude": self.normalized_magnitude,
            "fixed_point_rms": self.fixed_point_rms,
        }


def compute_tensor_stats(t: Optional[torch.Tensor], scale: int = SCALE) -> Optional[TensorStats]:
    """Computes full mathematical statistics for a tensor in an observational manner."""
    if t is None:
        return None

    if not isinstance(t, torch.Tensor):
        return None

    numel = t.numel()
    if numel == 0:
        return TensorStats(
            numel=0,
            mean=0.0,
            std=0.0,
            min_val=0.0,
            max_val=0.0,
            l1_norm=0.0,
            l2_norm=0.0,
            rms=0.0,
            zero_fraction=0.0,
            non_finite_fraction=0.0,
            normalized_magnitude=0.0,
            fixed_point_rms=0,
        )

    with torch.no_grad():
        # Detach and convert to double on CPU for numerical stability
        flat = t.detach().flatten().to(dtype=torch.float64, device="cpu")

        # Non-finite elements check
        finite_mask = torch.isfinite(flat)
        non_finite_count = int((~finite_mask).sum().item())
        non_finite_fraction = float(non_finite_count) / float(numel)

        # Zero elements check
        zero_count = int((flat == 0.0).sum().item())
        zero_fraction = float(zero_count) / float(numel)

        # Replace non-finite values for statistical aggregation
        safe_flat = torch.nan_to_num(flat, nan=0.0, posinf=1e12, neginf=-1e12)

        mean_val = float(safe_flat.mean().item())
        std_val = float(safe_flat.std(unbiased=False).item()) if numel > 1 else 0.0
        min_val = float(safe_flat.min().item())
        max_val = float(safe_flat.max().item())

        l1_norm = float(torch.linalg.norm(safe_flat, ord=1).item())
        l2_norm = float(torch.linalg.norm(safe_flat, ord=2).item())

        # RMS = sqrt(1/N * sum(x^2)) = L2 / sqrt(N)
        rms = float(l2_norm / math.sqrt(numel))
        normalized_mag = rms

        fp_rms = int(round(normalized_mag * scale))

        return TensorStats(
            numel=numel,
            mean=mean_val,
            std=std_val,
            min_val=min_val,
            max_val=max_val,
            l1_norm=l1_norm,
            l2_norm=l2_norm,
            rms=rms,
            zero_fraction=zero_fraction,
            non_finite_fraction=non_finite_fraction,
            normalized_magnitude=normalized_mag,
            fixed_point_rms=fp_rms,
        )


# ══════════════════════════════════════════════════════════════
#  EDGE-LEVEL AUTOGRAD HOOK
# ══════════════════════════════════════════════════════════════

class EdgeAutogradHook(torch.autograd.Function):
    """Observational autograd function inserted along graph edges.

    In the forward pass, this acts as an exact identity operation: tensor.view_as(tensor).
    In the backward pass, it intercepts the gradient passing specifically along
    the directed dependency (source_node -> target_node) before it accumulates
    at source_node.
    """

    @staticmethod
    def forward(ctx, tensor: torch.Tensor, edge_key: Tuple[str, str], sink: Dict[Tuple[str, str], torch.Tensor]) -> torch.Tensor:
        ctx.edge_key = edge_key
        ctx.sink = sink
        return tensor.view_as(tensor)

    @staticmethod
    def backward(ctx, grad_output: Optional[torch.Tensor]):
        if grad_output is not None and ctx.sink is not None:
            ctx.sink[ctx.edge_key] = grad_output.detach().clone()
        return grad_output, None, None


# ══════════════════════════════════════════════════════════════
#  DYNAMIC NODE & EXECUTION RECORD
# ══════════════════════════════════════════════════════════════

@dataclass
class DynamicNodeRecord:
    """Dynamic execution record for a single computational graph node at step t."""
    node_id: str
    node_type: str
    activation: Optional[TensorStats] = None
    activation_gradient: Optional[TensorStats] = None
    parameter_gradient: Optional[TensorStats] = None
    ratio_grad_act: Optional[float] = None
    has_parameters: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "node_type": self.node_type,
            "activation": self.activation.to_dict() if self.activation else None,
            "activation_gradient": self.activation_gradient.to_dict() if self.activation_gradient else None,
            "parameter_gradient": self.parameter_gradient.to_dict() if self.parameter_gradient else None,
            "ratio_grad_act": self.ratio_grad_act,
            "has_parameters": self.has_parameters,
        }


# ══════════════════════════════════════════════════════════════
#  MODE A: FX GRAPH DYNAMIC MONITOR
# ══════════════════════════════════════════════════════════════

class FXGradObserver(Interpreter):
    """Observational runtime monitor for PyTorch FX-traced computational graphs.

    Subclasses torch.fx.Interpreter to execute the graph node-by-node, recording:
    1. Forward activations for every node (call_module, call_function, call_method, placeholder, output).
    2. Backward activation gradients dL/dx_i attached to node outputs.
    3. Directed edge-level backward gradients dL/de intercepted via EdgeAutogradHook.
    4. Parameter gradients dL/dW_i for modules invoked by call_module nodes.
    """

    def __init__(self, module: GraphModule, record_edges: bool = True):
        super().__init__(module)
        self.record_edges = record_edges
        self.reset()

    def reset(self) -> None:
        """Clears captured runtime tensors."""
        self.raw_activations: Dict[str, torch.Tensor] = {}
        self.raw_activation_grads: Dict[str, torch.Tensor] = {}
        self.raw_edge_grads: Dict[Tuple[str, str], torch.Tensor] = {}
        self.raw_param_grads: Dict[str, torch.Tensor] = {}
        self.node_types: Dict[str, str] = {}
        self.node_modules: Dict[str, nn.Module] = {}

    def _attach_tensor_hook(self, name: str, t: torch.Tensor) -> None:
        if t.requires_grad:
            def hook(grad: torch.Tensor):
                if grad is not None:
                    self.raw_activation_grads[name] = grad.detach().clone()
            t.register_hook(hook)

    def run_node(self, n: FxNode) -> Any:
        self.node_types[n.name] = n.op

        # Track target submodule if this is a call_module
        if n.op == "call_module":
            try:
                submod = self.module.get_submodule(str(n.target))
                self.node_modules[n.name] = submod
            except Exception:
                pass

        # Optional edge-level hooking for inputs
        if self.record_edges:
            def map_arg(arg: Any) -> Any:
                if isinstance(arg, FxNode):
                    val = self.env.get(arg)
                    if isinstance(val, torch.Tensor) and val.requires_grad:
                        edge_key = (arg.name, n.name)
                        return EdgeAutogradHook.apply(val, edge_key, self.raw_edge_grads)
                    return val
                return arg

            args = torch.fx.map_arg(n.args, map_arg)
            kwargs = torch.fx.map_arg(n.kwargs, map_arg)
        else:
            args = torch.fx.map_arg(n.args, lambda a: self.env[a] if isinstance(a, FxNode) else a)
            kwargs = torch.fx.map_arg(n.kwargs, lambda a: self.env[a] if isinstance(a, FxNode) else a)

        # Execute node operation
        out = getattr(self, n.op)(n.target, args, kwargs)
        self.env[n] = out

        # Record activation and attach backward hook
        if isinstance(out, torch.Tensor):
            self.raw_activations[n.name] = out.detach().clone()
            self._attach_tensor_hook(n.name, out)
        elif isinstance(out, (tuple, list)):
            # If multiple outputs, record the first tensor or concatenate
            tensor_items = [item for item in out if isinstance(item, torch.Tensor)]
            if tensor_items:
                self.raw_activations[n.name] = tensor_items[0].detach().clone()
                for item in tensor_items:
                    self._attach_tensor_hook(n.name, item)

        return out

    def capture_parameter_gradients(self) -> None:
        """Collects parameter gradients for all call_module nodes."""
        for name, mod in self.node_modules.items():
            param_tensors = [p.grad for p in mod.parameters(recurse=False) if p.grad is not None]
            if param_tensors:
                flat_grads = torch.cat([p.flatten().to(dtype=torch.float64, device="cpu") for p in param_tensors])
                self.raw_param_grads[name] = flat_grads

    def extract_records(self, eps: float = 1e-8) -> Tuple[Dict[str, DynamicNodeRecord], Dict[Tuple[str, str], TensorStats]]:
        """Converts raw captured tensors into statistical DynamicNodeRecord structures."""
        node_records: Dict[str, DynamicNodeRecord] = {}

        all_node_names = set(self.raw_activations.keys()) | set(self.raw_activation_grads.keys()) | set(self.node_types.keys())

        for name in sorted(all_node_names):
            act_stats = compute_tensor_stats(self.raw_activations.get(name))
            grad_stats = compute_tensor_stats(self.raw_activation_grads.get(name))
            param_stats = compute_tensor_stats(self.raw_param_grads.get(name))

            ratio = None
            if act_stats is not None and grad_stats is not None:
                ratio = grad_stats.rms / (act_stats.rms + eps)

            has_params = name in self.node_modules and any(p.numel() > 0 for p in self.node_modules[name].parameters())

            node_records[name] = DynamicNodeRecord(
                node_id=name,
                node_type=self.node_types.get(name, "unknown"),
                activation=act_stats,
                activation_gradient=grad_stats,
                parameter_gradient=param_stats,
                ratio_grad_act=ratio,
                has_parameters=has_params,
            )

        edge_records: Dict[Tuple[str, str], TensorStats] = {}
        for edge_key, grad_t in sorted(self.raw_edge_grads.items()):
            e_stats = compute_tensor_stats(grad_t)
            if e_stats is not None:
                edge_records[edge_key] = e_stats

        return node_records, edge_records


# ══════════════════════════════════════════════════════════════
#  MODE B: MODULE HIERARCHY DYNAMIC MONITOR
# ══════════════════════════════════════════════════════════════

class ModuleGradObserver:
    """Observational runtime monitor for PyTorch nn.Module hierarchies.

    Uses register_forward_hook and register_full_backward_hook to monitor
    module-level activations and output gradients without symbolic tracing.
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self.handles: List[Any] = []
        self.reset()
        self._register_hooks()

    def reset(self) -> None:
        self.raw_activations: Dict[str, torch.Tensor] = {}
        self.raw_activation_grads: Dict[str, torch.Tensor] = {}
        self.raw_param_grads: Dict[str, torch.Tensor] = {}
        self.module_types: Dict[str, str] = {}

    def _register_hooks(self) -> None:
        for name, module in self.model.named_modules():
            if not name:
                name = "root"
            # Only monitor leaf modules
            if len(list(module.children())) == 0:
                self.module_types[name] = type(module).__name__
                h1 = module.register_forward_hook(self._make_fwd_hook(name))
                h2 = module.register_full_backward_hook(self._make_bwd_hook(name))
                self.handles.extend([h1, h2])

    def _make_fwd_hook(self, name: str) -> Callable:
        def hook(module: nn.Module, inp: Any, outp: Any) -> None:
            if isinstance(outp, torch.Tensor):
                self.raw_activations[name] = outp.detach().clone()
            elif isinstance(outp, (tuple, list)) and len(outp) > 0 and isinstance(outp[0], torch.Tensor):
                self.raw_activations[name] = outp[0].detach().clone()
        return hook

    def _make_bwd_hook(self, name: str) -> Callable:
        def hook(module: nn.Module, grad_input: Any, grad_output: Any) -> None:
            if grad_output and len(grad_output) > 0 and isinstance(grad_output[0], torch.Tensor):
                self.raw_activation_grads[name] = grad_output[0].detach().clone()
        return hook

    def capture_parameter_gradients(self) -> None:
        for name, module in self.model.named_modules():
            if not name:
                name = "root"
            param_tensors = [p.grad for p in module.parameters(recurse=False) if p.grad is not None]
            if param_tensors:
                flat_grads = torch.cat([p.flatten().to(dtype=torch.float64, device="cpu") for p in param_tensors])
                self.raw_param_grads[name] = flat_grads

    def remove_hooks(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles.clear()

    def extract_records(self, eps: float = 1e-8) -> Dict[str, DynamicNodeRecord]:
        records: Dict[str, DynamicNodeRecord] = {}
        all_names = set(self.raw_activations.keys()) | set(self.raw_activation_grads.keys()) | set(self.module_types.keys())
        for name in sorted(all_names):
            act_stats = compute_tensor_stats(self.raw_activations.get(name))
            grad_stats = compute_tensor_stats(self.raw_activation_grads.get(name))
            param_stats = compute_tensor_stats(self.raw_param_grads.get(name))

            ratio = None
            if act_stats is not None and grad_stats is not None:
                ratio = grad_stats.rms / (act_stats.rms + eps)

            records[name] = DynamicNodeRecord(
                node_id=name,
                node_type=self.module_types.get(name, "Module"),
                activation=act_stats,
                activation_gradient=grad_stats,
                parameter_gradient=param_stats,
                ratio_grad_act=ratio,
                has_parameters=param_stats is not None,
            )
        return records
