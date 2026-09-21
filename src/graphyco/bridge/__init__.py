from __future__ import annotations

from .torch_bridge import module_to_graph, trace_to_graph, evaluate_model, evaluate_model_dynamic
from .grad_bridge import DynamicGraphState, DynamicExecutionBridge, monitor_model, LiveTrainingMonitor

def visualize(*args, **kwargs):
    from visualizer import visualize as _vis
    return _vis(*args, **kwargs)

def launch_app(*args, **kwargs):
    from visualizer import launch_app as _launch
    return _launch(*args, **kwargs)

def extract_benchmark_json(*args, **kwargs):
    from visualizer import extract_benchmark_json as _extract
    return _extract(*args, **kwargs)

__all__ = [
    "module_to_graph",
    "trace_to_graph",
    "evaluate_model",
    "evaluate_model_dynamic",
    "DynamicGraphState",
    "DynamicExecutionBridge",
    "monitor_model",
    "LiveTrainingMonitor",
    "visualize",
    "launch_app",
    "extract_benchmark_json",
]


