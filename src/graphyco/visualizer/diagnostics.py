from __future__ import annotations

from contextlib import contextmanager
import math
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from bridge.grad_bridge import DynamicGraphState, LiveTrainingMonitor
from observer.grad_observer import compute_tensor_stats, TensorStats


class GradientHealthStatus:
    HEALTHY = "HEALTHY"
    WARNING_VANISHING = "WARNING: Vanishing Gradients"
    WARNING_EXPLODING = "WARNING: Exploding Gradients"
    CRITICAL_NAN_INF = "CRITICAL: NaN or Inf Detected"
    WARNING_DEAD_NEURONS = "WARNING: High Dead Neuron Ratio"


class LiveTrainingDiagnostics:
    """Direct runtime diagnostic monitor for active PyTorch training loops.

    Provides in-memory, real-time access to gradient health, layer bottlenecks,
    activation distributions, and structural invariants as batches execute.
    """

    def __init__(
        self,
        model: nn.Module,
        dummy_input: Optional[torch.Tensor] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        loss_fn: Optional[Callable] = None,
        vanishing_threshold: float = 1e-6,
        exploding_threshold: float = 50.0,
        dead_neuron_threshold: float = 0.5,
        history_len: int = 100,
    ):
        self.model = model
        self.dummy_input = dummy_input
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.vanishing_threshold = vanishing_threshold
        self.exploding_threshold = exploding_threshold
        self.dead_neuron_threshold = dead_neuron_threshold
        self.history_len = history_len

        self._lock = threading.Lock()
        self._current_step = 0
        self._loss_history: List[float] = []
        self._step_times: List[float] = []
        self._layer_grad_norms: Dict[str, List[float]] = {}
        self._layer_act_norms: Dict[str, List[float]] = {}
        self._dead_neuron_ratios: Dict[str, float] = {}
        self._active_bottleneck: Optional[str] = None
        self._gmax_value: float = 1.0
        self._health_status: str = GradientHealthStatus.HEALTHY
        self._alerts: List[Dict[str, Any]] = []

        # Internal hook handles
        self._hook_handles: List[Any] = []
        self._current_step_grads: Dict[str, torch.Tensor] = {}
        self._current_step_acts: Dict[str, torch.Tensor] = {}
        self._setup_hooks()

    def _setup_hooks(self) -> None:
        """Attaches lightweight forward and backward hooks to all named modules."""
        for name, module in self.model.named_modules():
            if len(list(module.children())) > 0:
                continue

            def forward_hook(mod, inp, out, mod_name=name):
                if isinstance(out, torch.Tensor):
                    with torch.no_grad():
                        flat = out.detach().view(-1)
                        if flat.numel() > 0:
                            zeros = (flat == 0).float().mean().item()
                            self._current_step_acts[mod_name] = flat
                            self._dead_neuron_ratios[mod_name] = zeros

            def backward_hook(mod, grad_in, grad_out, mod_name=name):
                if grad_out and len(grad_out) > 0 and isinstance(grad_out[0], torch.Tensor):
                    with torch.no_grad():
                        g = grad_out[0].detach()
                        self._current_step_grads[mod_name] = g

            h_fwd = module.register_forward_hook(forward_hook)
            h_bwd = module.register_full_backward_hook(backward_hook)
            self._hook_handles.extend([h_fwd, h_bwd])

    def close(self) -> None:
        """Removes all registered hooks."""
        for h in self._hook_handles:
            try:
                h.remove()
            except Exception:
                pass
        self._hook_handles.clear()

    def __del__(self):
        self.close()

    @contextmanager
    def observe_step(self, step: int, loss: Optional[float] = None):
        """Context manager wrapping forward, backward, and optimizer steps."""
        t0 = time.time()
        self.current_loss = loss
        try:
            yield self
        finally:
            dt = time.time() - t0
            self._process_step(step, loss=self.current_loss, duration=dt)

    def _process_step(self, step: int, loss: Optional[float], duration: float) -> None:
        with self._lock:
            self._current_step = step
            if loss is not None:
                self._loss_history.append(float(loss))
                if len(self._loss_history) > self.history_len:
                    self._loss_history.pop(0)

            self._step_times.append(duration)
            if len(self._step_times) > self.history_len:
                self._step_times.pop(0)

            # Analyze layer gradients
            grad_rms_map: Dict[str, float] = {}
            has_nan_inf = False
            has_vanishing = False
            has_exploding = False

            for name, g in self._current_step_grads.items():
                rms = float(torch.sqrt(torch.mean(g ** 2)).item()) if g.numel() > 0 else 0.0
                if math.isnan(rms) or math.isinf(rms):
                    has_nan_inf = True
                    rms = 0.0
                grad_rms_map[name] = rms

                if rms < self.vanishing_threshold and rms > 0.0:
                    has_vanishing = True
                elif rms > self.exploding_threshold:
                    has_exploding = True

                self._layer_grad_norms.setdefault(name, []).append(rms)
                if len(self._layer_grad_norms[name]) > self.history_len:
                    self._layer_grad_norms[name].pop(0)

            # Analyze layer activations
            for name, a in self._current_step_acts.items():
                rms = float(torch.sqrt(torch.mean(a ** 2)).item()) if a.numel() > 0 else 0.0
                self._layer_act_norms.setdefault(name, []).append(rms)
                if len(self._layer_act_norms[name]) > self.history_len:
                    self._layer_act_norms[name].pop(0)

            # Check bottleneck
            if grad_rms_map:
                mean_grad = sum(grad_rms_map.values()) / max(1, len(grad_rms_map))
                max_node = max(grad_rms_map, key=grad_rms_map.get)
                max_grad = grad_rms_map[max_node]
                self._active_bottleneck = max_node
                self._gmax_value = (max_grad / mean_grad) if mean_grad > 1e-12 else 1.0
            else:
                self._active_bottleneck = None
                self._gmax_value = 1.0

            # Check dead neurons
            dead_nodes = [
                n for n, r in self._dead_neuron_ratios.items()
                if r > self.dead_neuron_threshold
            ]

            # Update overall health status
            if has_nan_inf:
                self._health_status = GradientHealthStatus.CRITICAL_NAN_INF
                self._add_alert(step, "CRITICAL", "NaN or Inf gradient encountered in backward pass.")
            elif has_exploding:
                self._health_status = GradientHealthStatus.WARNING_EXPLODING
                self._add_alert(step, "WARNING", f"Exploding gradient detected (peak {self._gmax_value:.2f}x mean).")
            elif has_vanishing:
                self._health_status = GradientHealthStatus.WARNING_VANISHING
                self._add_alert(step, "WARNING", f"Vanishing gradient (< {self.vanishing_threshold}) in layer {self._active_bottleneck}.")
            elif dead_nodes:
                self._health_status = GradientHealthStatus.WARNING_DEAD_NEURONS
                self._add_alert(step, "WARNING", f"{len(dead_nodes)} layers have > {int(self.dead_neuron_threshold * 100)}% dead activations: {dead_nodes[:3]}")
            else:
                self._health_status = GradientHealthStatus.HEALTHY

            self._current_step_grads.clear()
            self._current_step_acts.clear()

    def _add_alert(self, step: int, level: str, message: str) -> None:
        alert = {
            "step": step,
            "level": level,
            "message": message,
            "timestamp": time.strftime("%H:%M:%S"),
        }
        # Prevent spamming identical consecutive messages
        if not self._alerts or self._alerts[-1]["message"] != message:
            self._alerts.append(alert)
            if len(self._alerts) > 50:
                self._alerts.pop(0)

    def get_snapshot(self) -> Dict[str, Any]:
        """Thread-safe snapshot of the current training diagnostic state."""
        with self._lock:
            latest_loss = self._loss_history[-1] if self._loss_history else None
            avg_time = (sum(self._step_times) / len(self._step_times)) if self._step_times else 0.0

            layer_summary = {}
            for name in set(self._layer_grad_norms.keys()) | set(self._layer_act_norms.keys()):
                g_hist = self._layer_grad_norms.get(name, [])
                a_hist = self._layer_act_norms.get(name, [])
                layer_summary[name] = {
                    "latest_grad_rms": g_hist[-1] if g_hist else 0.0,
                    "latest_act_rms": a_hist[-1] if a_hist else 0.0,
                    "dead_neuron_ratio": self._dead_neuron_ratios.get(name, 0.0),
                }

            return {
                "step": self._current_step,
                "latest_loss": latest_loss,
                "loss_history": list(self._loss_history),
                "avg_step_time_ms": avg_time * 1000.0,
                "health_status": self._health_status,
                "active_bottleneck_layer": self._active_bottleneck,
                "gmax": round(self._gmax_value, 4),
                "layers": layer_summary,
                "alerts": list(self._alerts),
            }

