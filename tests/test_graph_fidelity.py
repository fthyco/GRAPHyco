"""Stage 3 — graph semantics, exact FX dependency parity and fallback safety."""
from __future__ import annotations

import json
import subprocess
import sys

import pytest
import torch
from torch import nn
from torch.fx import symbolic_trace

from graphyco.bridge.torch_bridge import evaluate_model, evaluate_model_dynamic, module_to_graph, trace_to_graph
from graphyco.bridge.grad_bridge import DynamicExecutionBridge, LiveTrainingMonitor
from graphyco.core.invariants import validate_invariants, InvariantViolationError
from graphyco.visualizer.query import extract_benchmark_json, get_canonical_models


class BranchWithResidual(nn.Module):
    def __init__(self):
        super().__init__()
        self.left = nn.Linear(4, 4)
        self.right = nn.Linear(4, 4)
        self.post = nn.Linear(4, 4)

    def forward(self, x):
        return self.post(self.left(x) + self.right(x)) + x


class KeywordDependencies(nn.Module):
    def __init__(self):
        super().__init__()
        self.left = nn.Linear(4, 4)
        self.right = nn.Linear(4, 4)

    def forward(self, x):
        a = self.left(x)
        b = self.right(x)
        return torch.add(input=a, other=b)


class DynamicBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.positive = nn.Linear(4, 4)
        self.negative = nn.Linear(4, 4)

    def forward(self, x):
        if x.sum() > 0:
            return self.positive(x)
        return self.negative(x)


class CountTrace(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4)
        self.trace_calls = 0

    def forward(self, x):
        self.trace_calls += 1
        return self.linear(x)


def expected_fx_edges(traced):
    # Independent PyTorch FX oracle includes kwargs and multi-consumer edges.
    return {
        (dep.name, node.name)
        for node in traced.graph.nodes
        for dep in node.all_input_nodes
    }


def extracted_edges(state):
    return {(e.source, e.target) for e in state.edges}


@pytest.mark.parametrize("model", [BranchWithResidual(), KeywordDependencies()])
def test_fx_dependencies_match_pytorch_exactly(model):
    model.eval()
    torch.manual_seed(123)
    x = torch.randn(2, 4)
    fx = symbolic_trace(model)
    state = trace_to_graph(model)
    assert extracted_edges(state) == expected_fx_edges(fx)
    assert len(state.edges) == len(extracted_edges(state))
    assert state.provenance["graph_kind"] == "fx_symbolic_trace"
    assert state.provenance["dataflow_verified"] == "symbolic_only"
    assert state.provenance["execution_verified"] is False
    assert state.to_dict()["provenance"] == state.provenance
    validate_invariants(state)
    torch.testing.assert_close(model(x), fx(x))


def test_fx_keyword_dependency_ports_are_complete():
    state = trace_to_graph(KeywordDependencies())
    add_nodes = [node for node in state.nodes.values() if "add" in node.name]
    assert len(add_nodes) == 1
    add = add_nodes[0]
    assert set(add.in_ports) == {"left", "right"}
    incoming = {(e.source, e.target) for e in state.edges if e.target == add.id}
    assert incoming == {("left", add.id), ("right", add.id)}
    assert all(e.critical for e in state.edges)



def test_port_connectivity_rejects_nonreciprocal_edge_reference():
    state = trace_to_graph(KeywordDependencies())
    assert "add" in state.nodes["left"].out_ports
    state.nodes["left"].out_ports.remove("add")
    with pytest.raises(InvariantViolationError, match="port_connectivity"):
        validate_invariants(state)

    state = trace_to_graph(KeywordDependencies())
    state.nodes["add"].in_ports.append("ghost")
    with pytest.raises(InvariantViolationError, match="port_connectivity"):
        validate_invariants(state)

    state = trace_to_graph(KeywordDependencies())
    state.edges = [
        edge for edge in state.edges
        if (edge.source, edge.target) != ("left", "add")
    ]
    with pytest.raises(InvariantViolationError, match="port_connectivity"):
        validate_invariants(state)


def test_module_fallback_is_real_containment_not_registration_order():
    model = BranchWithResidual()
    state = module_to_graph(model)
    assert state.provenance["graph_kind"] == "module_hierarchy"
    assert state.provenance["edge_semantics"] == "containment_only"
    assert state.provenance["execution_verified"] is False
    assert extracted_edges(state) == {
        ("root", "left"), ("root", "right"), ("root", "post")
    }
    assert all(e.edge_type == "contains" and not e.critical for e in state.edges)
    assert ("left", "right") not in extracted_edges(state)

    leaf = module_to_graph(model, include_containers=False)
    assert set(leaf.nodes) == {"left", "right", "post"}
    assert leaf.edges == []
    assert leaf.provenance["edge_semantics"] == "no_edges"


def test_nested_module_containment_uses_true_parent_not_sibling_order():
    model = nn.Sequential(
        nn.Sequential(nn.Linear(4, 4), nn.ReLU()),
        nn.Sequential(nn.Linear(4, 4)),
    )
    state = module_to_graph(model)
    assert extracted_edges(state) == {
        ("root", "0"), ("root", "1"), ("0", "0_0"),
        ("0", "0_1"), ("1", "1_0"),
    }
    assert all(e.edge_type == "contains" for e in state.edges)


def test_cleaned_module_name_collision_is_not_silent():
    model = nn.Module()
    model.add_module("a_b", nn.Linear(4, 4))
    nested = nn.Module()
    nested.add_module("b", nn.Linear(4, 4))
    model.add_module("a", nested)
    state = module_to_graph(model)
    assert len(state.nodes) == len(set(state.nodes))
    by_name = {n.name: n.id for n in state.nodes.values()}
    assert "a_b" in by_name and "a.b" in by_name
    assert by_name["a_b"] != by_name["a.b"]


def test_explicit_fallback_provenance_and_strict_trace():
    model = DynamicBranch()
    fallback = evaluate_model(model, mode="trace")
    state = fallback["graph_state"]
    assert state.provenance["graph_kind"] == "module_hierarchy"
    assert state.provenance["trace_status"] == "failed"
    assert state.provenance["fallback_from"] == "fx_symbolic_trace"
    assert "fallback_reason_type" in state.provenance
    assert fallback["graph_provenance"] == state.provenance
    assert all(e.edge_type == "contains" for e in state.edges)
    with pytest.raises(Exception):
        evaluate_model(model, mode="trace", strict_trace=True)
    with pytest.raises(ValueError):
        evaluate_model(model, mode="unknown")


def test_dynamic_fallback_preserves_graph_provenance():
    bridge = DynamicExecutionBridge(DynamicBranch(), strict_trace=False)
    assert bridge.mode == "module"
    assert bridge.static_state.provenance["trace_status"] == "failed"
    state = bridge.step(torch.ones(2, 4))
    assert state.graph_state.provenance["graph_kind"] == "module_hierarchy"
    assert state.to_dict()["static_graph"]["provenance"]["trace_status"] == "failed"


def test_dynamic_evaluate_wrapper_exposes_strict_tracing_and_provenance():
    model = DynamicBranch()
    x = torch.ones(2, 4)
    with pytest.raises(Exception):
        evaluate_model_dynamic(model, x, strict_trace=True)
    result = evaluate_model_dynamic(model, x, strict_trace=False)
    assert result["graph_provenance"]["graph_kind"] == "module_hierarchy"
    assert result["graph_provenance"]["trace_status"] == "failed"
    assert result["dynamic_graph_state"].graph_state is result["graph_state"]



def test_bridges_symbolically_trace_only_once():
    model = CountTrace()
    DynamicExecutionBridge(model, mode="trace", strict_trace=True)
    assert model.trace_calls == 1
    model2 = CountTrace()
    LiveTrainingMonitor(model2, mode="trace", strict_trace=True)
    assert model2.trace_calls == 1


def test_canonical_models_are_self_contained_traceable_and_deterministic():
    names = {
        "Sequential MLP", "ResNet Block", "DenseBlock",
        "U-Net Toy", "Self-Attention Toy",
    }
    before = torch.get_rng_state().clone()
    models = get_canonical_models()
    after = torch.get_rng_state()
    assert torch.equal(before, after)
    assert set(models) == names
    for name, (model, inputs) in models.items():
        model.eval()
        with torch.no_grad():
            output = model(inputs)
        assert isinstance(output, torch.Tensor), name
        assert output.shape == inputs.shape, name
        assert torch.isfinite(output).all(), name
        state = trace_to_graph(model)
        fx = symbolic_trace(model)
        assert extracted_edges(state) == expected_fx_edges(fx), name
        validate_invariants(state)
    models2 = get_canonical_models()
    for name in names:
        m1, x1 = models[name]
        m2, x2 = models2[name]
        torch.testing.assert_close(x1, x2)
        for p1, p2 in zip(m1.parameters(), m2.parameters()):
            torch.testing.assert_close(p1, p2)


def test_all_canonical_benchmark_and_serialized_provenance():
    data = extract_benchmark_json("all", steps=1)
    assert set(data["architectures"]) == {
        "Sequential MLP", "ResNet Block", "DenseBlock",
        "U-Net Toy", "Self-Attention Toy",
    }
    for name, architecture in data["architectures"].items():
        provenance = architecture["static_graph"]["provenance"]
        assert provenance["graph_kind"] == "fx_symbolic_trace", (name, provenance)
        assert provenance["dataflow_verified"] == "symbolic_only"
        assert data["step_trajectories"][name]
        assert isinstance(architecture["dynamic_evaluation"], dict)
    json.dumps(data)


def test_cli_canonical_query_without_gui(tmp_path):
    run = subprocess.run(
        [sys.executable, "-m", "graphyco.visualizer", "--model",
         "resnet", "--steps", "1", "--query", "summary"],
        cwd=tmp_path, capture_output=True, text=True,
        timeout=120, check=False,
    )
    assert run.returncode == 0, run.stdout + "\n" + run.stderr
    assert "ResNet Block" in run.stdout or "topological_bottleneck" in run.stdout
