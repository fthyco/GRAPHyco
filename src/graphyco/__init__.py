"""Graphyco: Computational graph topology and dynamic gradient-flow monitoring for PyTorch."""

from __future__ import annotations

from .core.primitives import Edge, GraphState, Node, SCALE
from .core.evaluation import evaluate
from .core.invariants import validate_invariants, InvariantViolationError
from .bridge.torch_bridge import trace_to_graph, evaluate_model
from .bridge.grad_bridge import DynamicExecutionBridge, DynamicGraphState, LiveTrainingMonitor
from .visualizer import visualize, extract_benchmark_json
from .visualizer.query import QueryEngine, query_metrics
from .visualizer.diagnostics import LiveTrainingDiagnostics, GradientHealthStatus

try:
    from .visualizer.gui import VisualizerDesktopApp, launch_desktop_app
except ImportError:
    VisualizerDesktopApp = None  # type: ignore
    launch_desktop_app = None  # type: ignore

__version__ = "0.1.0"
__name__ = "graphyco"

__all__ = [
    "Edge",
    "GraphState",
    "Node",
    "SCALE",
    "evaluate",
    "validate_invariants",
    "InvariantViolationError",
    "trace_to_graph",
    "evaluate_model",
    "DynamicExecutionBridge",
    "DynamicGraphState",
    "LiveTrainingMonitor",
    "visualize",
    "extract_benchmark_json",
    "QueryEngine",
    "query_metrics",
    "LiveTrainingDiagnostics",
    "GradientHealthStatus",
    "VisualizerDesktopApp",
    "launch_desktop_app",
]
