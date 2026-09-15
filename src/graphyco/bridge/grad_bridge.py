from __future__ import annotations

import csv
from contextlib import contextmanager
import io
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.fx import GraphModule, symbolic_trace

from core.evaluation import evaluate, perturbation_profile
from core.primitives import Edge, GraphState, SCALE
from observer.grad_eval import evaluate_dynamic_flow
from observer.grad_observer import (
    DynamicNodeRecord,
    FXGradObserver,
    ModuleGradObserver,
    TensorStats,
    compute_tensor_stats,
)
from .torch_bridge import module_to_graph, trace_to_graph


# ══════════════════════════════════════════════════════════════
#  DYNAMIC GRAPH STATE REPRESENTATION: G_t = (V, E, X_t, A_t, G_t)
# ══════════════════════════════════════════════════════════════

@dataclass
class DynamicGraphState:
    """Represents the complete state G_t = (V, E, X_t, A_t, G_t).

    Preserves the structural representation G = (V, E) while attaching runtime
    activation state A_t and backward gradient statistics G_t.
    """
    graph_state: GraphState
    step: int
    node_records: Dict[str, DynamicNodeRecord]
    edge_records: Dict[Tuple[str, str], TensorStats] = field(default_factory=dict)
    static_evaluation: Dict[str, Any] = field(default_factory=dict)
    dynamic_evaluation: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "static_graph": self.graph_state.to_dict(),
            "static_evaluation": self.static_evaluation,
            "dynamic_evaluation": self.dynamic_evaluation,
            "nodes": {nid: rec.to_dict() for nid, rec in sorted(self.node_records.items())},
            "edges": {
                f"{k[0]}->{k[1]}": stats.to_dict()
                for k, stats in sorted(self.edge_records.items())
            },
        }

    def to_json(self, path: Optional[str] = None, indent: int = 2) -> str:
        data = self.to_dict()
        s = json.dumps(data, indent=indent)
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(s)
        return s

    def to_csv_nodes(self, path: Optional[str] = None) -> str:
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow([
            "node_id", "node_type", "has_params",
            "act_numel", "act_rms", "act_mean", "act_std", "act_zero_frac",
            "grad_numel", "grad_rms", "grad_mean", "grad_std", "grad_zero_frac",
            "ratio_grad_act", "category"
        ])

        categories = self.dynamic_evaluation.get("correspondence", {}).get("node_categories", {})

        for nid, rec in sorted(self.node_records.items()):
            act = rec.activation
            grad = rec.activation_gradient
            writer.writerow([
                nid,
                rec.node_type,
                rec.has_parameters,
                act.numel if act else 0,
                f"{act.rms:.6f}" if act else "",
                f"{act.mean:.6f}" if act else "",
                f"{act.std:.6f}" if act else "",
                f"{act.zero_fraction:.4f}" if act else "",
                grad.numel if grad else 0,
                f"{grad.rms:.6f}" if grad else "",
                f"{grad.mean:.6f}" if grad else "",
                f"{grad.std:.6f}" if grad else "",
                f"{grad.zero_fraction:.4f}" if grad else "",
                f"{rec.ratio_grad_act:.6f}" if rec.ratio_grad_act is not None else "",
                categories.get(nid, "Unknown"),
            ])

        csv_str = output.getvalue()
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(csv_str)
        return csv_str


# ══════════════════════════════════════════════════════════════
#  DYNAMIC EXECUTION BRIDGE
# ══════════════════════════════════════════════════════════════

class DynamicExecutionBridge:
    """Manages observational forward and backward graph execution.

    Coordinates between PyTorch model evaluation, static graph extraction,
    and dynamic runtime monitoring.
    """

    def __init__(
        self,
        model: nn.Module,
        mode: str = "trace",
        concrete_args: Optional[Dict] = None,
        record_edges: bool = True,
    ):
        self.model = model
        self.mode = mode
        self.concrete_args = concrete_args
        self.record_edges = record_edges

        # Extract static graph representation
        if self.mode == "trace":
            try:
                self.static_state = trace_to_graph(model, concrete_args=concrete_args)
                self.traced_module = symbolic_trace(model, concrete_args=concrete_args)
                self.observer = FXGradObserver(self.traced_module, record_edges=record_edges)
            except Exception:
                # Fallback to module mode if FX symbolic trace fails
                self.mode = "module"
                self.static_state = module_to_graph(model)
                self.traced_module = None
                self.observer = ModuleGradObserver(model)
        else:
            self.static_state = module_to_graph(model)
            self.traced_module = None
            self.observer = ModuleGradObserver(model)

        self.static_evaluation = evaluate(self.static_state)
        self.static_profile = perturbation_profile(self.static_state)
        self.trajectory: List[DynamicGraphState] = []
        self.step_counter = 0

    def step(
        self,
        inputs: Union[torch.Tensor, Tuple, Dict],
        target: Optional[torch.Tensor] = None,
        loss_fn: Optional[Callable] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ) -> DynamicGraphState:
        """Executes a single observational monitoring step.

        1. Forward pass (recording activations)
        2. Loss evaluation
        3. Backward pass (recording activation and edge gradients)
        4. Parameter gradient capture
        5. Flow statistics and metric evaluation
        """
        self.step_counter += 1
        self.observer.reset()

        # Prepare inputs with autograd enabled
        if isinstance(inputs, torch.Tensor):
            if not inputs.requires_grad and inputs.is_floating_point():
                inputs = inputs.clone().detach().requires_grad_(True)
            args = (inputs,)
            kwargs = {}
        elif isinstance(inputs, tuple):
            args = inputs
            kwargs = {}
        elif isinstance(inputs, dict):
            args = ()
            kwargs = inputs
        else:
            args = (inputs,)
            kwargs = {}

        # 1. Forward pass
        if isinstance(self.observer, FXGradObserver):
            output = self.observer.run(*args, **kwargs)
        else:
            output = self.model(*args, **kwargs)

        # 2. Loss computation
        if loss_fn is not None:
            if target is not None:
                loss = loss_fn(output, target)
            else:
                loss = loss_fn(output)
        else:
            if isinstance(output, torch.Tensor):
                loss = output.sum()
            elif isinstance(output, (tuple, list)) and len(output) > 0 and isinstance(output[0], torch.Tensor):
                loss = output[0].sum()
            else:
                raise ValueError("Model output is not a tensor; cannot compute default loss sum.")

        # 3. Backward pass
        if optimizer is not None:
            optimizer.zero_grad()
        loss.backward()

        # 4. Parameter gradient capture
        self.observer.capture_parameter_gradients()

        # 5. Extract records
        if isinstance(self.observer, FXGradObserver):
            node_records, edge_records = self.observer.extract_records()
        else:
            node_records = self.observer.extract_records()
            edge_records = {}

        # Build trajectory for temporal analysis
        trajectory_records = [s.node_records for s in self.trajectory] + [node_records]

        # 6. Evaluate dynamic metrics
        dynamic_eval = evaluate_dynamic_flow(
            graph_state=self.static_state,
            node_records=node_records,
            edge_records=edge_records,
            static_perturbation_profile=self.static_profile,
            trajectory=trajectory_records,
        )

        state = DynamicGraphState(
            graph_state=self.static_state,
            step=self.step_counter,
            node_records=node_records,
            edge_records=edge_records,
            static_evaluation=self.static_evaluation,
            dynamic_evaluation=dynamic_eval,
        )
        self.trajectory.append(state)

        # 7. Optimizer step if requested
        if optimizer is not None:
            optimizer.step()

        return state

    def run_training_trajectory(
        self,
        batches: Union[List[torch.Tensor], List[Tuple[torch.Tensor, torch.Tensor]], Iterator],
        loss_fn: Optional[Callable] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        steps: Optional[int] = None,
    ) -> List[DynamicGraphState]:
        """Executes a multi-step training trajectory, recording temporal statistics."""
        count = 0
        for batch in batches:
            if steps is not None and count >= steps:
                break

            if isinstance(batch, (tuple, list)) and len(batch) == 2:
                inp, tgt = batch
                self.step(inputs=inp, target=tgt, loss_fn=loss_fn, optimizer=optimizer)
            else:
                self.step(inputs=batch, target=None, loss_fn=loss_fn, optimizer=optimizer)

            count += 1

        return self.trajectory


# ══════════════════════════════════════════════════════════════
#  HIGH-LEVEL CONVENIENCE FUNCTION
# ══════════════════════════════════════════════════════════════

def monitor_model(
    model: nn.Module,
    inputs: torch.Tensor,
    loss_fn: Optional[Callable] = None,
    mode: str = "trace",
    concrete_args: Optional[Dict] = None,
) -> DynamicGraphState:
    """One-line convenience function to monitor a forward and backward pass on a PyTorch model."""
    bridge = DynamicExecutionBridge(model, mode=mode, concrete_args=concrete_args)
    return bridge.step(inputs=inputs, loss_fn=loss_fn)


# ══════════════════════════════════════════════════════════════
#  LIVE TRAINING MONITOR (FOR ACTIVE RUNNING TRAINING LOOPS)
# ══════════════════════════════════════════════════════════════

class LiveTrainingMonitor:
    """Non-intrusive live monitor for running PyTorch training loops.

    Attaches directly to any custom training loop via context manager:
        monitor = LiveTrainingMonitor(model, log_interval=10)

        for step, (x, y) in enumerate(dataloader):
            with monitor.observe(step):
                optimizer.zero_grad()
                out = model(x)
                loss = criterion(out, y)
                loss.backward()
                optimizer.step()

            # Retrieve dynamic gradient flow metrics in real time!
            if monitor.has_new_data():
                state = monitor.get_latest_state()
                bneck = monitor.get_latest_bottleneck()
                anomalies = monitor.get_live_anomalies()
                stability = monitor.get_temporal_stability()
    """

    def __init__(
        self,
        model: nn.Module,
        mode: str = "trace",
        concrete_args: Optional[Dict] = None,
        log_interval: int = 1,
        record_edges: bool = True,
    ):
        self.model = model
        self.mode = mode
        self.concrete_args = concrete_args
        self.log_interval = max(1, log_interval)
        self.record_edges = record_edges

        if self.mode == "trace":
            try:
                self.static_state = trace_to_graph(model, concrete_args=concrete_args)
                self.traced_module = symbolic_trace(model, concrete_args=concrete_args)
                self.observer = FXGradObserver(self.traced_module, record_edges=record_edges)
            except Exception:
                self.mode = "module"
                self.static_state = module_to_graph(model)
                self.traced_module = None
                self.observer = ModuleGradObserver(model)
        else:
            self.static_state = module_to_graph(model)
            self.traced_module = None
            self.observer = ModuleGradObserver(model)

        self.static_evaluation = evaluate(self.static_state)
        self.static_profile = perturbation_profile(self.static_state)
        self.trajectory: List[DynamicGraphState] = []
        self.latest_state: Optional[DynamicGraphState] = None
        self._new_data: bool = False
        self._orig_forward: Optional[Callable] = None

    def has_new_data(self) -> bool:
        """Returns True if the most recent step was observed and metrics were computed."""
        return self._new_data

    def get_latest_state(self) -> Optional[DynamicGraphState]:
        """Returns the most recent DynamicGraphState snapshot G_t."""
        return self.latest_state

    def get_latest_bottleneck(self) -> Dict[str, Any]:
        """Returns the current dynamic gradient bottleneck G_max and choke node."""
        if not self.latest_state:
            return {}
        return self.latest_state.dynamic_evaluation.get("bottleneck", {})

    def get_live_anomalies(self) -> Dict[str, Any]:
        """Returns any vanishing or exploding gradient nodes detected on the latest step."""
        if not self.latest_state:
            return {}
        return self.latest_state.dynamic_evaluation.get("anomalies", {})

    def get_temporal_stability(self) -> Dict[str, Any]:
        """Returns temporal CV and gradient stability metrics across all observed steps."""
        if not self.latest_state:
            return {}
        return self.latest_state.dynamic_evaluation.get("temporal", {})

    def export_live_json(self, path: str, indent: int = 2) -> None:
        """Exports the entire observed training trajectory to a JSON file for live visualization."""
        data = {
            "steps": [s.to_dict() for s in self.trajectory],
            "latest_step": self.latest_state.step if self.latest_state else 0,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=indent)

    @contextmanager
    def observe(self, step: int):
        """Context manager wrapping forward, loss, backward, and optimizer execution."""
        active = (step % self.log_interval == 0)
        self._new_data = False
        if active:
            self.observer.reset()
            if self.mode == "trace":
                self._orig_forward = self.model.forward
                self.model.forward = lambda *args, **kwargs: self.observer.run(*args, **kwargs)
        try:
            yield self
        finally:
            if active:
                if self.mode == "trace" and self._orig_forward is not None:
                    self.model.forward = self._orig_forward
                    self._orig_forward = None

                self.observer.capture_parameter_gradients()

                if hasattr(self.observer, "extract_records") and self.mode == "trace":
                    node_recs, edge_recs = self.observer.extract_records()
                else:
                    node_recs = self.observer.extract_records()
                    edge_recs = {}

                traj_recs = [s.node_records for s in self.trajectory] + [node_recs]
                dyn_eval = evaluate_dynamic_flow(
                    self.static_state,
                    node_recs,
                    edge_records=edge_recs,
                    static_perturbation_profile=self.static_profile,
                    trajectory=traj_recs,
                )

                self.latest_state = DynamicGraphState(
                    graph_state=self.static_state,
                    step=step,
                    node_records=node_recs,
                    edge_records=edge_recs,
                    static_evaluation=self.static_evaluation,
                    dynamic_evaluation=dyn_eval,
                )
                self.trajectory.append(self.latest_state)
                self._new_data = True

