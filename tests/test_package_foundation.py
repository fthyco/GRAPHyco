"""Stage 1: package integrity and optional-GUI regression checks.

Run after installing the package into a fresh environment:
    python -m pip install -e '.[dev]'
    python -m pytest -q
"""
from __future__ import annotations

from importlib import metadata, resources
import subprocess
import sys

import torch
from torch import nn

import graphyco
from graphyco.bridge.torch_bridge import evaluate_model, module_to_graph, trace_to_graph
from graphyco.core.primitives import GraphState
from graphyco.visualizer.renderer import Renderer


def test_declared_runtime_dependencies_include_scipy() -> None:
    assert graphyco.__name__ == "graphyco"
    requirements = metadata.requires("graphyco") or []
    assert any(req.lower().startswith("scipy") for req in requirements)


def test_trace_and_module_graph_have_canonical_class_identity() -> None:
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
    traced = trace_to_graph(model)
    declared = module_to_graph(model)
    evaluated = evaluate_model(model)
    assert type(traced) is GraphState
    assert type(declared) is GraphState
    assert type(evaluated["graph_state"]) is GraphState
    assert len(traced.nodes) > 0
    assert len(traced.edges) > 0
    assert "core" not in sys.modules
    assert "core.primitives" not in sys.modules
    assert "observer" not in sys.modules


def test_dynamic_bridge_produces_canonical_graph_state() -> None:
    from graphyco.bridge.grad_bridge import DynamicExecutionBridge

    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
    bridge = DynamicExecutionBridge(model)
    state = bridge.step(inputs=torch.randn(3, 4))
    assert type(state.graph_state) is GraphState
    assert state.step == 1
    assert state.node_records


def test_renderer_static_assets_are_packaged() -> None:
    root = resources.files("graphyco.visualizer")
    for name in ("visualizer.html", "visualizer.css", "visualizer.js"):
        asset = root.joinpath("static", name)
        assert asset.is_file(), str(asset)
        assert asset.read_text(encoding="utf-8")
    html = Renderer().assemble({"architectures": {}, "step_trajectories": {}})
    assert 'id="embedded-data"' in html
    assert '"architectures": {}' in html
    assert "<style>" in html


def test_import_and_query_cli_do_not_require_qt(tmp_path) -> None:
    # A fresh interpreter is necessary because imports are cached within pytest.
    script = r"""
import importlib.abc
import runpy
import sys

class ForbidQt(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "PySide6" or fullname.startswith("PySide6."):
            raise ModuleNotFoundError("No module named 'PySide6'", name="PySide6")
        return None

sys.meta_path.insert(0, ForbidQt())

import graphyco
from graphyco import evaluate_model, QueryEngine, visualize
from graphyco.visualizer import Renderer

assert callable(evaluate_model)
assert callable(visualize)
assert callable(Renderer)

try:
    visualize()
except ImportError as exc:
    assert "graphyco[gui]" in str(exc)
else:
    raise AssertionError("Expected a clear optional-GUI dependency error")

# CLI --help must succeed without importing PySide6.
sys.argv = ["graphyco", "--help"]
runpy.run_module("graphyco.visualizer", run_name="__main__")
"""
    process = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert process.returncode == 0, process.stdout + "\n" + process.stderr
    assert "Graphyco" in process.stdout
