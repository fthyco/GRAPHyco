"""End-to-end policy pass-through for existing monitoring entry points."""

import pytest
import torch
import torch.nn as nn

from graphyco.bridge.grad_bridge import (
    DynamicExecutionBridge,
    LiveTrainingMonitor,
    monitor_model,
)
from graphyco.observer import AnalysisPolicy, NodeIdentity, NodeIdentityRegistry


def sample_model():
    return nn.Sequential(nn.Linear(3, 5), nn.ReLU(), nn.Linear(5, 2))


@pytest.mark.parametrize("mode", ["trace", "module"])
def test_bridge_opt_in_policy_retains_identity_without_fabricating_gradient(mode):
    torch.manual_seed(123)
    model = sample_model()
    registry = NodeIdentityRegistry()
    registry.register(NodeIdentity(
        "external/inactive", "research-layer", physical_index=4,
        lifecycle="inactive", execution="not_executed",
    ))
    policy = AnalysisPolicy(graph_scope="full", gradient_policy="observed_only")
    bridge = DynamicExecutionBridge(
        model, mode=mode, analysis_policy=policy, identity_registry=registry,
    )
    try:
        state = bridge.step(torch.randn(4, 3))
        report = state.dynamic_evaluation
        assert report["policy"] == policy.to_dict()
        assert report["observations"]["external/inactive"]["lifecycle"] == "inactive"
        assert report["observations"]["external/inactive"]["gradient_status"] == "not_observed"
        assert "external/inactive" not in report["metric_populations"]["gradient"]
        assert report["coverage"]["valid_gradient_count"] >= 1
    finally:
        bridge.close()


@pytest.mark.parametrize("mode", ["trace", "module"])
def test_live_monitor_opt_in_policy_is_visible_in_latest_state(mode):
    torch.manual_seed(456)
    model = sample_model()
    policy = AnalysisPolicy(graph_scope="active", gradient_policy="observed_only")
    monitor = LiveTrainingMonitor(
        model, mode=mode, analysis_policy=policy, log_interval=1,
    )
    try:
        with monitor.observe(0):
            model.zero_grad(set_to_none=True)
            loss = model(torch.randn(4, 3)).square().mean()
            loss.backward()
        report = monitor.get_latest_state().dynamic_evaluation
        assert report["policy"] == policy.to_dict()
        assert report["coverage"]["selected_node_count"] > 0
    finally:
        monitor.close()


def test_one_shot_accepts_policy_but_default_still_legacy():
    torch.manual_seed(789)
    model = sample_model()
    policy = AnalysisPolicy(gradient_policy="observed_only")
    data = torch.randn(4, 3)
    new_state = monitor_model(model, data, mode="module", analysis_policy=policy)
    old_state = monitor_model(model, data, mode="module")
    assert new_state.dynamic_evaluation["policy"] == policy.to_dict()
    assert "policy" not in old_state.dynamic_evaluation
    assert "coverage" not in old_state.dynamic_evaluation


def test_visualizer_summary_propagates_undefined_policy_metrics():
    from graphyco.core.primitives import GraphState, Node
    from graphyco.observer.grad_eval import evaluate_dynamic_flow
    from graphyco.observer.grad_observer import DynamicNodeRecord
    from graphyco.visualizer.query import _build_summary

    g = GraphState(nodes={"A": Node("A", "A", "hidden")})
    missing = {"A": DynamicNodeRecord(node_id="A", node_type="call_module")}
    report = evaluate_dynamic_flow(
        g, missing, analysis_policy=AnalysisPolicy(
            gradient_policy="observed_only",
        ),
    )
    summary = _build_summary("Missing gradient", {}, report, missing)
    assert summary["grad_bottleneck_gmax"] is None
    assert summary["bottleneck_node"] is None
    assert summary["spearman_act_grad"] is None
    assert summary["mean_stability_cv"] is None
    assert summary["mean_grad_flow"] is None


def test_visualizer_summary_skips_missing_temporal_cv():
    from graphyco.core.primitives import GraphState, Node
    from graphyco.observer.grad_eval import evaluate_dynamic_flow
    from graphyco.observer.grad_observer import DynamicNodeRecord
    from graphyco.visualizer.query import _build_summary

    g = GraphState(nodes={"A": Node("A", "A", "hidden")})
    rec = DynamicNodeRecord(node_id="A", node_type="call_module")
    report = evaluate_dynamic_flow(
        g, {"A": rec},
        trajectory=[{"A": rec}, {"A": rec}],
        analysis_policy=AnalysisPolicy(gradient_policy="observed_only"),
    )
    assert report["temporal"]["A"]["cv"] is None
    summary = _build_summary("Sparse trajectory", {}, report, {"A": rec})
    assert summary["mean_stability_cv"] is None
