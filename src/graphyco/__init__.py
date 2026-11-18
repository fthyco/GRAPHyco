"""Graphyco: Computational graph topology and dynamic gradient-flow monitoring for PyTorch."""

from __future__ import annotations

from .core.primitives import Edge, GraphState, Node, SCALE
from .core.evaluation import evaluate
from .core.invariants import validate_invariants, InvariantViolationError
from .bridge.torch_bridge import trace_to_graph, evaluate_model
from .bridge.grad_bridge import DynamicExecutionBridge, DynamicGraphState, LiveTrainingMonitor
from .visualizer import visualize, extract_benchmark_json, launch_desktop_app
from .visualizer.query import QueryEngine, query_metrics
from .visualizer.diagnostics import LiveTrainingDiagnostics, GradientHealthStatus

def __getattr__(name: str):
    """Load the optional GUI class only when explicitly requested."""
    if name == "VisualizerDesktopApp":
        try:
            from .visualizer import VisualizerDesktopApp
            return VisualizerDesktopApp
        except ImportError as exc:
            if "graphyco[gui]" in str(exc):
                return None
            raise
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__version__ = "0.1.2"
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
