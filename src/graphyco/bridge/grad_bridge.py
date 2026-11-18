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

from ..core.evaluation import evaluate, perturbation_profile
from ..core.primitives import Edge, GraphState, SCALE
from ..observer.grad_eval import evaluate_dynamic_flow
from ..observer.grad_observer import (
    DynamicNodeRecord,
    FXGradObserver,
    ModuleGradObserver,
    TensorStats,
    compute_tensor_stats,
)
from .torch_bridge import module_to_graph, _fx_to_graph


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
    """Executes instrumented forward/backward analysis and optional training.

    With no optimizer, parameter gradients are restored after telemetry
    extraction, but forward buffers and RNG may still change. With an
    optimizer, this method explicitly applies optimizer updates.
    """

    def __init__(
        self,
        model: nn.Module,
        mode: str = "trace",
        concrete_args: Optional[Dict] = None,
        record_edges: bool = True,
        strict_trace: bool = False,
        max_history_steps: Optional[int] = 128,
        stats_backend: str = "cpu",
    ):
        self.model = model
        if mode not in ("trace", "module"):
            raise ValueError(f"Unknown extraction mode: {mode!r}")
        self.mode = mode
        self.concrete_args = concrete_args
        self.record_edges = record_edges
        if stats_backend not in ("cpu", "device"):
            raise ValueError("stats_backend must be 'cpu' or 'device'")
        self.stats_backend = stats_backend
        if max_history_steps is not None and max_history_steps < 1:
            raise ValueError("max_history_steps must be >= 1 or None")
        self.max_history_steps = max_history_steps
        self._closed = False

        # A single FX trace feeds both the graph and the interpreter:
        # previously they were traced independently and could diverge.
        if self.mode == "trace":
            try:
                self.traced_module = symbolic_trace(model, concrete_args=concrete_args)
                self.static_state = _fx_to_graph(model, self.traced_module)
                self.observer = FXGradObserver(self.traced_module, record_edges=record_edges, stats_backend=stats_backend)
            except Exception as exc:
                if strict_trace:
                    raise
                self.mode = "module"
                self.static_state = module_to_graph(model)
                self.static_state.provenance.update({
                    "trace_status": "failed",
                    "fallback_from": "fx_symbolic_trace",
                    "fallback_reason_type": type(exc).__name__,
                    "fallback_reason": str(exc)[:500],
                })
                self.traced_module = None
                self.observer = ModuleGradObserver(model, stats_backend=stats_backend)
        else:
            self.static_state = module_to_graph(model)
            self.traced_module = None
            self.observer = ModuleGradObserver(model, stats_backend=stats_backend)

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
        if self._closed:
            raise RuntimeError("DynamicExecutionBridge has been closed")
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
        # A diagnostic call without an optimizer must not contaminate
        # parameter .grad values retained by the caller. The backward pass
        # still runs and model forward/stateful buffers may change.
        prior_grads = None
        if optimizer is None:
            prior_grads = [
                (parameter, None if parameter.grad is None else parameter.grad.detach().clone())
                for parameter in self.model.parameters()
            ]
        try:
            loss.backward()

            # 4. Parameter gradient capture
            self.observer.capture_parameter_gradients()

            # 5. Extract records
            if isinstance(self.observer, FXGradObserver):
                node_records, edge_records = self.observer.extract_records()
            else:
                node_records = self.observer.extract_records()
                edge_records = {}

            # Window-aware temporal analysis; keep only a bounded number of
            # full metric snapshots and node-record dictionaries in memory.
            previous = self.trajectory
            if self.max_history_steps is not None:
                previous = previous[-max(0, self.max_history_steps - 1):] if self.max_history_steps > 1 else []
            trajectory_records = [s.node_records for s in previous] + [node_records]

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
            if self.max_history_steps is not None:
                del self.trajectory[:-self.max_history_steps]

            # 7. Optimizer step if requested
            if optimizer is not None:
                optimizer.step()

            return state

        finally:
            if prior_grads is not None:
                for parameter, grad in prior_grads:
                    parameter.grad = grad

    def close(self) -> None:
        """Release persistent hooks in module fallback mode."""
        if self._closed:
            return
        if isinstance(self.observer, ModuleGradObserver):
            self.observer.enabled = False
            self.observer.remove_hooks()
        self._closed = True

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
    try:
        return bridge.step(inputs=inputs, loss_fn=loss_fn)
    finally:
        bridge.close()


# ══════════════════════════════════════════════════════════════
#  LIVE TRAINING MONITOR (FOR ACTIVE RUNNING TRAINING LOOPS)
# ══════════════════════════════════════════════════════════════

class LiveTrainingMonitor:
    """Live training monitor with explicit instrumentation tradeoffs.

    FX mode temporarily replaces model.forward with an interpreter; module
    mode executes the original forward and uses output-tensor hooks.
    Neither mode promises parity for arbitrary custom PyTorch models.

    Attaches to training loops via context manager:
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
        strict_trace: bool = False,
        max_history_steps: Optional[int] = 128,
        stats_backend: str = "cpu",
    ):
        self.model = model
        if mode not in ("trace", "module"):
            raise ValueError(f"Unknown extraction mode: {mode!r}")
        self.mode = mode
        self.concrete_args = concrete_args
        self.log_interval = max(1, log_interval)
        self.record_edges = record_edges
        if stats_backend not in ("cpu", "device"):
            raise ValueError("stats_backend must be 'cpu' or 'device'")
        self.stats_backend = stats_backend
        if max_history_steps is not None and max_history_steps < 1:
            raise ValueError("max_history_steps must be >= 1 or None")
        self.max_history_steps = max_history_steps
        self._closed = False
        self._observing = False

        if self.mode == "trace":
            try:
                self.traced_module = symbolic_trace(model, concrete_args=concrete_args)
                self.static_state = _fx_to_graph(model, self.traced_module)
                self.observer = FXGradObserver(self.traced_module, record_edges=record_edges, stats_backend=stats_backend)
            except Exception as exc:
                if strict_trace:
                    raise
                self.mode = "module"
                self.static_state = module_to_graph(model)
                self.static_state.provenance.update({
                    "trace_status": "failed",
                    "fallback_from": "fx_symbolic_trace",
                    "fallback_reason_type": type(exc).__name__,
                    "fallback_reason": str(exc)[:500],
                })
                self.traced_module = None
                self.observer = ModuleGradObserver(model, stats_backend=stats_backend)
        else:
            self.static_state = module_to_graph(model)
            self.traced_module = None
            self.observer = ModuleGradObserver(model, stats_backend=stats_backend)

        self.static_evaluation = evaluate(self.static_state)
        self.static_profile = perturbation_profile(self.static_state)
        self.trajectory: List[DynamicGraphState] = []
        self.latest_state: Optional[DynamicGraphState] = None
        self._new_data: bool = False
        self._orig_forward: Optional[Callable] = None
        if isinstance(self.observer, ModuleGradObserver):
            # No forward/backward capture on unselected training steps.
            self.observer.enabled = False

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
        """Capture an active training step and always restore model state.

        FX mode temporarily substitutes the model forward with an FX
        Interpreter; this is an explicit runtime intervention and may not
        preserve custom Python forward side effects. Use mode="module"
        to execute the original model.forward without replacement.
        """
        if self._closed:
            raise RuntimeError("LiveTrainingMonitor has been closed")
        if self._observing:
            raise RuntimeError("LiveTrainingMonitor observation already active")

        self._observing = True
        active = (step % self.log_interval == 0)
        self._new_data = False
        had_instance_forward = "forward" in self.model.__dict__
        instance_forward = self.model.__dict__.get("forward")

        try:
            self.observer.reset()
            if isinstance(self.observer, ModuleGradObserver):
                self.observer.enabled = active
            if active and self.mode == "trace":
                self.model.forward = lambda *args, **kwargs: self.observer.run(
                    *args, **kwargs
                )

            yield self

            # Do not commit an incomplete telemetry snapshot when the
            # caller's forward/backward/optimizer block raises.
            if active:
                self.observer.capture_parameter_gradients()
                if isinstance(self.observer, FXGradObserver):
                    node_recs, edge_recs = self.observer.extract_records()
                else:
                    node_recs = self.observer.extract_records()
                    edge_recs = {}

                previous = self.trajectory
                if self.max_history_steps is not None:
                    previous = (
                        previous[-(self.max_history_steps - 1):]
                        if self.max_history_steps > 1 else []
                    )
                traj_recs = [s.node_records for s in previous] + [node_recs]
                dyn_eval = evaluate_dynamic_flow(
                    self.static_state,
                    node_recs,
                    edge_records=edge_recs,
                    static_perturbation_profile=self.static_profile,
                    trajectory=traj_recs,
                )

                state = DynamicGraphState(
                    graph_state=self.static_state,
                    step=step,
                    node_records=node_recs,
                    edge_records=edge_recs,
                    static_evaluation=self.static_evaluation,
                    dynamic_evaluation=dyn_eval,
                )
                self.latest_state = state
                self.trajectory.append(state)
                if self.max_history_steps is not None:
                    del self.trajectory[:-self.max_history_steps]
                self._new_data = True
        finally:
            if active and self.mode == "trace":
                # Preserve the exact original instance attribute (or absence),
                # not a new bound method stored in the instance dictionary.
                if had_instance_forward:
                    self.model.forward = instance_forward
                else:
                    self.model.__dict__.pop("forward", None)
            if isinstance(self.observer, ModuleGradObserver):
                self.observer.enabled = False
            self._observing = False

    def close(self) -> None:
        """Detach persistent module hooks; idempotent and safe to repeat."""
        if self._observing:
            raise RuntimeError("Cannot close LiveTrainingMonitor while observing")
        if self._closed:
            return
        if isinstance(self.observer, ModuleGradObserver):
            self.observer.enabled = False
            self.observer.remove_hooks()
        self._closed = True
