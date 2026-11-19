"""Pre-merge resource and numerical regression tests (CPU, deterministic inputs).

Time and RSS fluctuate on shared runners: test hard invariants here and collect
latency/RSS evidence separately in benchmarks/perf_matrix.py.
"""

from __future__ import annotations

import copy
import gc
import math
import tracemalloc

import pytest
import torch
from torch import nn

from graphyco.bridge.grad_bridge import DynamicExecutionBridge, LiveTrainingMonitor
from graphyco.core.primitives import GraphState, Node
from graphyco.observer import (
    AnalysisPolicy,
    NodeIdentity,
    NodeIdentityRegistry,
    compare_analysis_policies,
)
from graphyco.observer.grad_observer import DynamicNodeRecord, compute_tensor_stats


def model_with_width(width: int) -> nn.Module:
    return nn.Sequential(
        nn.Linear(8, width),
        nn.Tanh(),
        nn.Linear(width, width),
        nn.ReLU(),
        nn.Linear(width, 3),
    )


@pytest.mark.parametrize("mode", ["trace", "module"])
def test_bounded_180_step_monitoring_does_not_retain_all_snapshots(mode):
    torch.manual_seed(719)
    model = model_with_width(16)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.002)
    monitor = LiveTrainingMonitor(
        model, mode=mode, max_history_steps=8, log_interval=1,
    )
    x = torch.randn(4, 8)
    y = torch.randn(4, 3)
    try:
        for step in range(180):
            with monitor.observe(step):
                optimizer.zero_grad(set_to_none=True)
                loss = (model(x) - y).square().mean()
                loss.backward()
                optimizer.step()
            assert len(monitor.trajectory) <= 8
        assert [entry.step for entry in monitor.trajectory] == list(range(172, 180))
        assert all(torch.isfinite(p).all() for p in model.parameters())
    finally:
        monitor.close()
    assert not any(m._forward_hooks for m in model.modules())
    assert not any(m._backward_hooks for m in model.modules())
    assert not any(m._forward_pre_hooks for m in model.modules())


@pytest.mark.parametrize("mode", ["trace", "module"])
def test_sparse_monitoring_only_retains_logged_steps(mode):
    torch.manual_seed(194)
    model = model_with_width(16)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    monitor = LiveTrainingMonitor(
        model, mode=mode, log_interval=8, max_history_steps=12,
    )
    x = torch.randn(3, 8)
    try:
        seen = []
        for step in range(96):
            with monitor.observe(step):
                optimizer.zero_grad(set_to_none=True)
                model(x).square().mean().backward()
                optimizer.step()
            if monitor.has_new_data():
                seen.append(monitor.get_latest_state().step)
        assert seen == list(range(0, 96, 8))
        assert [state.step for state in monitor.trajectory] == seen
    finally:
        monitor.close()


@pytest.mark.parametrize("width", [16, 64, 256])
@pytest.mark.parametrize("mode", ["module", "trace"])
def test_training_parameter_and_gradient_parity_across_model_sizes(width, mode):
    torch.manual_seed(924)
    base = model_with_width(width)
    tracked = copy.deepcopy(base)
    a = torch.optim.Adam(base.parameters(), lr=0.001)
    b = torch.optim.Adam(tracked.parameters(), lr=0.001)
    monitor = LiveTrainingMonitor(tracked, mode=mode, max_history_steps=2)
    inputs = torch.randn(7, 8)
    labels = torch.randn(7, 3)
    try:
        for step in range(3):
            a.zero_grad(set_to_none=True)
            loss_a = (base(inputs) - labels).square().mean()
            loss_a.backward()
            grad_a = [p.grad.detach().clone() for p in base.parameters()]
            a.step()

            with monitor.observe(step):
                b.zero_grad(set_to_none=True)
                loss_b = (tracked(inputs) - labels).square().mean()
                loss_b.backward()
                grad_b = [p.grad.detach().clone() for p in tracked.parameters()]
                b.step()

            torch.testing.assert_close(loss_a, loss_b, rtol=1e-6, atol=1e-7)
            for left, right in zip(grad_a, grad_b):
                torch.testing.assert_close(left, right, rtol=1e-6, atol=1e-7)
            for left, right in zip(base.parameters(), tracked.parameters()):
                torch.testing.assert_close(left, right, rtol=1e-6, atol=1e-7)
        assert len(monitor.trajectory) == 2
    finally:
        monitor.close()


@pytest.mark.parametrize("mode", ["trace", "module"])
def test_one_shot_bridge_hook_cleanup_after_repeated_calls(mode):
    torch.manual_seed(16)
    model = model_with_width(32)
    x = torch.randn(4, 8)
    for _ in range(24):
        bridge = DynamicExecutionBridge(model, mode=mode, max_history_steps=2)
        try:
            state = bridge.step(x)
            assert state.dynamic_evaluation["summary"]["node_count"] >= 1
        finally:
            bridge.close()
        assert not any(m._forward_hooks for m in model.modules())
        assert not any(m._backward_hooks for m in model.modules())


def test_registry_audit_is_bounded_under_4000_model_events():
    registry = NodeIdentityRegistry(max_events=32)
    for i in range(64):
        registry.register(NodeIdentity(
            stable_id="n-{}".format(i),
            namespace="hidden",
            physical_index=i,
        ))
    for step in range(4000):
        key = "n-{}".format(step % 64)
        registry.transition(
            key,
            lifecycle="inactive" if step % 2 else "active",
            execution="not_executed" if step % 2 else "executed",
        )
        assert len(registry.history()) <= 32
    assert registry.version == 4064
    assert len(registry.snapshot()) == 64
    assert registry.history()[-1]["version"] == 4064


def test_policy_comparison_long_trajectory_preserves_source_observations():
    g = GraphState(nodes={
        "node-{}".format(i): Node(
            id="node-{}".format(i), name="node-{}".format(i), purpose="hidden",
        )
        for i in range(32)
    })
    measured = {
        "node-{}".format(i): DynamicNodeRecord(
            node_id="node-{}".format(i),
            node_type="call_module",
            activation=compute_tensor_stats(torch.tensor([float(i + 1)])),
            activation_gradient=(
                compute_tensor_stats(torch.tensor([float(i % 4)]))
                if i % 3 else None
            ),
        )
        for i in range(32)
    }
    registry = NodeIdentityRegistry(max_events=16)
    for i in range(32):
        registry.register(NodeIdentity(
            "node-{}".format(i), "hidden",
            physical_index=i,
            lifecycle="inactive" if i >= 28 else "active",
        ))
    before = copy.deepcopy(measured)
    trajectory = [measured] * 128
    policies = {
        "legacy": AnalysisPolicy(gradient_policy="legacy_zero_fill"),
        "research": AnalysisPolicy(gradient_policy="observed_only"),
        "active": AnalysisPolicy(
            graph_scope="active", gradient_policy="observed_only",
        ),
    }
    for _ in range(3):
        output = compare_analysis_policies(
            g, measured, policies, trajectory=trajectory,
            identity_registry=registry,
        )
        assert output["research"]["coverage"]["eligible_node_count"] == 28
        assert output["active"]["coverage"]["selected_node_count"] == 28
        assert output["research"]["temporal"]["node-0"]["mean_gradient"] is None
        assert math.isfinite(output["research"]["bottleneck"]["max_concentration"])
    assert measured == before


def test_python_snapshot_heap_growth_bounded_under_fixed_history():
    """Coarse guard for accidental unbounded Python telemetry retention.

    This is not a CUDA allocator or total-RSS test. Torch caching and CI noise
    require a conservative bound, while trajectory length is asserted exactly.
    """
    torch.manual_seed(804)
    model = model_with_width(16)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    monitor = LiveTrainingMonitor(
        model, mode="module", max_history_steps=4, log_interval=1,
    )
    x = torch.randn(3, 8)
    tracemalloc.start()
    try:
        for step in range(200):
            with monitor.observe(step):
                optimizer.zero_grad(set_to_none=True)
                model(x).square().mean().backward()
                optimizer.step()
            if step == 99:
                gc.collect()
                midpoint, _ = tracemalloc.get_traced_memory()
        gc.collect()
        endpoint, _ = tracemalloc.get_traced_memory()
        assert len(monitor.trajectory) == 4
        assert endpoint - midpoint < 8 * 1024 * 1024
    finally:
        monitor.close()
        tracemalloc.stop()
