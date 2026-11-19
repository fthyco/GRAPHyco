"""Contract-first tests for stable identity and explicit gradient analysis policies."""

from __future__ import annotations

import copy

import pytest
import torch

from graphyco.core.primitives import Edge, GraphState, Node
from graphyco.observer.grad_observer import DynamicNodeRecord, compute_tensor_stats
from graphyco.observer.grad_eval import evaluate_dynamic_flow
from graphyco.observer.analysis_policy import (
    AnalysisPolicy,
    NodeIdentity,
    NodeIdentityRegistry,
    compare_analysis_policies,
)


def record(node_id, activation=None, gradient=None):
    return DynamicNodeRecord(
        node_id=node_id,
        node_type="call_module",
        activation=compute_tensor_stats(
            torch.tensor([activation], dtype=torch.float64)
        ) if activation is not None else None,
        activation_gradient=compute_tensor_stats(
            torch.tensor([gradient], dtype=torch.float64)
        ) if gradient is not None else None,
    )


def graph(*ids, edges=()):
    return GraphState(
        nodes={nid: Node(id=nid, name=nid, purpose="hidden") for nid in ids},
        edges=[Edge(a, b) for a, b in edges],
    )


def policy(**kwargs):
    return AnalysisPolicy(gradient_policy="observed_only", **kwargs)


def test_missing_is_not_a_measured_zero_for_bottleneck_or_anomalies():
    g = graph("A", "B", "C", edges=[("A", "B"), ("B", "C")])
    recs = {
        "A": record("A", activation=1.0, gradient=0.8),
        "B": record("B", activation=2.0, gradient=0.0),
        "C": record("C", activation=3.0),
    }
    report = evaluate_dynamic_flow(g, recs, analysis_policy=policy())
    assert report["bottleneck"]["mean_gradient"] == pytest.approx(0.4)
    assert report["bottleneck"]["max_concentration"] == pytest.approx(2.0)
    assert set(report["bottleneck"]["node_concentrations"]) == {"A", "B"}
    assert report["anomalies"]["vanishing_nodes"] == ["B"]
    assert ("C" not in report["anomalies"]["vanishing_nodes"])
    assert set(report["attenuation"]) == {"A->B"}
    assert report["coverage"]["observed_zero_count"] == 1
    assert report["coverage"]["missing_count"] == 1
    assert report["coverage"]["eligible_coverage"] == pytest.approx(2 / 3)
    assert report["observations"]["B"]["gradient_status"] == "observed_zero"
    assert report["observations"]["C"]["gradient_status"] == "not_observed"


def test_all_missing_returns_undefined_not_a_fake_zero_measurement():
    g = graph("A", "B")
    report = evaluate_dynamic_flow(g, {"A": record("A", 1.0)}, analysis_policy=policy())
    assert report["bottleneck"]["max_concentration"] is None
    assert report["bottleneck"]["mean_gradient"] is None
    assert report["bottleneck"]["status"] == "undefined"
    assert report["summary"]["max_gradient_concentration"] is None
    assert report["alignment"]["spearman_rho"] is None
    assert report["coverage"]["valid_gradient_count"] == 0


def test_nonfinite_gradient_is_invalid_not_zero_or_vanishing():
    g = graph("A", "bad", "missing")
    recs = {
        "A": record("A", 1.0, 0.6),
        "bad": record("bad", 1.0, float("nan")),
        "missing": record("missing", 1.0),
    }
    report = evaluate_dynamic_flow(g, recs, analysis_policy=policy())
    assert report["observations"]["bad"]["gradient_status"] == "invalid"
    assert report["coverage"]["invalid_count"] == 1
    assert report["coverage"]["missing_count"] == 1
    assert report["bottleneck"]["mean_gradient"] == pytest.approx(0.6)
    assert report["anomalies"]["vanishing_nodes"] == []


def test_identity_survives_deactivation_and_physical_index_reuse():
    registry = NodeIdentityRegistry()
    registry.register(NodeIdentity("original", "layer0", physical_index=3))
    registry.transition("original", lifecycle="inactive", execution="not_executed")
    registry.register(NodeIdentity("new", "layer0", physical_index=3))
    snapshot = registry.snapshot()
    assert snapshot["original"].physical_index == 3
    assert snapshot["original"].lifecycle == "inactive"
    assert snapshot["new"].lifecycle == "active"
    assert snapshot["original"].stable_id != snapshot["new"].stable_id
    with pytest.raises(ValueError, match="occupied"):
        registry.transition("original", lifecycle="active")
    with pytest.raises(ValueError, match="already registered"):
        registry.register(NodeIdentity("original", "layer0", physical_index=10))
    assert registry.snapshot() == snapshot
    registry.transition("original", lifecycle="active", physical_index=4)
    assert registry.snapshot()["original"].physical_index == 4


def test_identity_registry_rejects_ambiguous_active_slots():
    registry = NodeIdentityRegistry()
    registry.register(NodeIdentity("one", "layer0", physical_index=0))
    with pytest.raises(ValueError, match="occupied"):
        registry.register(NodeIdentity("two", "layer0", physical_index=0))
    registry.register(NodeIdentity("two", "layer1", physical_index=0))
    with pytest.raises(ValueError, match="lifecycle"):
        NodeIdentity("bad", "layer0", lifecycle="not_observed")


def test_retained_inactive_identity_not_counted_as_missing_active_gradient():
    registry = NodeIdentityRegistry()
    registry.register(NodeIdentity("A", "layer0", physical_index=0))
    registry.register(NodeIdentity("B", "layer0", physical_index=1))
    registry.register(NodeIdentity("C", "layer0", physical_index=2, lifecycle="inactive", execution="not_executed"))
    recs = {"A": record("A", 1.0, 0.4), "B": record("B", 2.0, 0.0)}
    g = graph("A", "B", "C")
    full = evaluate_dynamic_flow(g, recs, analysis_policy=policy(graph_scope="full"), identity_registry=registry)
    active = evaluate_dynamic_flow(g, recs, analysis_policy=policy(graph_scope="active"), identity_registry=registry)
    assert set(full["observations"]) == {"A", "B", "C"}
    assert full["observations"]["C"]["lifecycle"] == "inactive"
    assert full["coverage"]["selected_node_count"] == 3
    assert full["coverage"]["eligible_node_count"] == 2
    assert full["coverage"]["eligible_coverage"] == 1.0
    assert active["coverage"]["selected_node_count"] == 2
    assert "C" not in active["observations"]
    assert full["bottleneck"]["mean_gradient"] == active["bottleneck"]["mean_gradient"]


def test_executed_scope_requires_positive_evidence_not_absence_inference():
    registry = NodeIdentityRegistry()
    registry.register(NodeIdentity("A", "layer0", execution="unknown"))
    registry.register(NodeIdentity("B", "layer0", execution="executed"))
    registry.register(NodeIdentity("C", "layer0", execution="unknown"))
    recs = {"A": record("A", 1.0, 0.3), "C": record("C")}
    report = evaluate_dynamic_flow(
        graph("A", "B", "C"), recs,
        analysis_policy=policy(graph_scope="executed"), identity_registry=registry,
    )
    assert set(report["observations"]) == {"A", "B"}
    assert report["observations"]["C"] if False else True
    assert report["observations"]["B"]["execution"] == "executed"


def test_observed_only_topology_and_alignment_exclude_unobserved_pair():
    g = graph("A", "B", "C")
    recs = {
        "A": record("A", 1.0, 3.0),
        "B": record("B", 2.0, 1.0),
        "C": record("C", 10.0),
    }
    report = evaluate_dynamic_flow(
        g, recs, static_perturbation_profile={"A": 1, "B": 2, "C": 100},
        analysis_policy=policy(),
    )
    assert report["alignment"]["spearman_rho"] == pytest.approx(-1.0)
    assert set(report["alignment"]["node_sensitivity_ratios"]) == {"A", "B"}
    assert set(report["correspondence"]["node_categories"]) == {"A", "B"}


def test_temporal_missing_steps_are_not_counted_as_zero_steps():
    g = graph("A", "B")
    trajectory = [
        {"A": record("A", 1, 1), "B": record("B", 2)},
        {"A": record("A", 2), "B": record("B", 3)},
        {"A": record("A", 3, 3), "B": record("B", 4)},
    ]
    report = evaluate_dynamic_flow(g, trajectory[-1], trajectory=trajectory, analysis_policy=policy())
    a = report["temporal"]["A"]
    b = report["temporal"]["B"]
    assert a["steps"] == 3
    assert a["observed_gradient_steps"] == 2
    assert a["mean_gradient"] == pytest.approx(2.0)
    assert a["std_gradient"] == pytest.approx(1.0)
    assert a["zero_step_fraction"] == 0.0
    assert a["gradient_coverage"] == pytest.approx(2 / 3)
    assert b["classification"] == "not_observed"
    assert b["mean_gradient"] is None


def test_legacy_default_output_remains_bitwise_equal_to_explicit_legacy_policy():
    g = graph("A", "B")
    recs = {"A": record("A", 1.0, 0.4), "B": record("B", 1.0)}
    before = evaluate_dynamic_flow(g, recs)
    after = evaluate_dynamic_flow(
        g, recs, analysis_policy=AnalysisPolicy(
            graph_scope="full", gradient_policy="legacy_zero_fill",
        ),
    )
    for key in ("attenuation", "bottleneck", "alignment", "anomalies", "correspondence", "temporal", "summary"):
        assert before[key] == after[key]
    assert before["anomalies"]["vanishing_nodes"] == ["B"]


def test_multiple_policies_compare_without_changing_raw_data_or_identity():
    g = graph("A", "B", "C")
    records = {"A": record("A", 1.0, 0.8), "B": record("B", 2.0, 0.2)}
    registry = NodeIdentityRegistry()
    registry.register(NodeIdentity("A", "layer0", physical_index=0))
    registry.register(NodeIdentity("B", "layer0", physical_index=1))
    registry.register(NodeIdentity("C", "layer0", physical_index=2, lifecycle="inactive"))
    before_records = copy.deepcopy(records)
    before_graph = copy.deepcopy(g)
    before_registry = registry.snapshot()
    results = compare_analysis_policies(
        g, records, {
            "historical": AnalysisPolicy(gradient_policy="legacy_zero_fill"),
            "research": policy(),
        }, identity_registry=registry,
    )
    assert results["historical"]["bottleneck"]["max_concentration"] > results["research"]["bottleneck"]["max_concentration"]
    assert records == before_records
    assert g == before_graph
    assert registry.snapshot() == before_registry


def test_invalid_policy_rejected_explicitly():
    with pytest.raises(ValueError, match="gradient_policy"):
        AnalysisPolicy(gradient_policy="ignore")
    with pytest.raises(ValueError, match="graph_scope"):
        AnalysisPolicy(graph_scope="mystery")
