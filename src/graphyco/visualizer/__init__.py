"""Public visualization API with optional desktop dependencies.

Core graph analysis, query and HTML rendering can be imported without PySide6.
The Qt GUI is loaded only when a desktop visualization is requested.
"""
from __future__ import annotations

from .query import compute_ablation_map, extract_benchmark_json, get_canonical_models, QueryEngine, query_metrics
from .diagnostics import GradientHealthStatus, LiveTrainingDiagnostics
from .renderer import Renderer


def _gui_entrypoint(name: str):
    try:
        from . import gui
    except ModuleNotFoundError as exc:
        if exc.name and (exc.name == "PySide6" or exc.name.startswith("PySide6.")):
            raise ImportError(
                "Graphyco desktop visualization requires PySide6. "
                "Install it with: pip install 'graphyco[gui]'"
            ) from exc
        raise
    return getattr(gui, name)


def visualize(*args, **kwargs):
    return _gui_entrypoint("visualize")(*args, **kwargs)


def visualize_inline(*args, **kwargs):
    return _gui_entrypoint("visualize_inline")(*args, **kwargs)


def launch_app(*args, **kwargs):
    return _gui_entrypoint("launch_app")(*args, **kwargs)


def launch_desktop_app(*args, **kwargs):
    return _gui_entrypoint("launch_desktop_app")(*args, **kwargs)


def __getattr__(name: str):
    if name == "VisualizerDesktopApp":
        return _gui_entrypoint(name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "visualize",
    "visualize_inline",
    "launch_app",
    "launch_desktop_app",
    "VisualizerDesktopApp",
    "LiveTrainingDiagnostics",
    "GradientHealthStatus",
    "extract_benchmark_json",
    "QueryEngine",
    "query_metrics",
    "compute_ablation_map",
    "get_canonical_models",
    "Renderer",
]
