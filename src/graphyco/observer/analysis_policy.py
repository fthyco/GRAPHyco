"""Opt-in, provenance-aware analysis over stable node identities and raw observations.

No policy changes an observer record or manufactures a measured gradient. An
external model/adapter must declare identity changes; physical indices alone do
not establish identity across pruning, regeneration, or reordering.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np

from ..core.primitives import GraphState
from .grad_observer import DynamicNodeRecord, TensorStats


_LIFECYCLES = frozenset(("active", "inactive", "retired"))
_EXECUTIONS = frozenset(("unknown", "executed", "not_executed"))
_SCOPES = frozenset(("full", "active", "executed"))
_GRADIENT_POLICIES = frozenset(("observed_only", "legacy_zero_fill"))


@dataclass(frozen=True)
class NodeIdentity:
    """Caller-declared stable identity; physical_index is only the last known slot.

    Statuses describe *model-declared* lifecycle/execution, not gradient
    observations. Reusing a physical slot never reuses a stable identity.
    """

    stable_id: str
    namespace: str
    physical_index: Optional[int] = None
    lifecycle: str = "active"
    execution: str = "unknown"

    def __post_init__(self) -> None:
        if not isinstance(self.stable_id, str) or not self.stable_id.strip():
            raise ValueError("stable_id must be non-empty")
        if not isinstance(self.namespace, str) or not self.namespace.strip():
            raise ValueError("namespace must be non-empty")
        if self.physical_index is not None and (
            type(self.physical_index) is not int or self.physical_index < 0
        ):
            raise ValueError("physical_index must be a non-negative integer or None")
        if self.lifecycle not in _LIFECYCLES:
            raise ValueError("invalid lifecycle: %s" % self.lifecycle)
        if self.execution not in _EXECUTIONS:
            raise ValueError("invalid execution: %s" % self.execution)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stable_id": self.stable_id,
            "namespace": self.namespace,
            "physical_index": self.physical_index,
            "lifecycle": self.lifecycle,
            "execution": self.execution,
        }


_UNCHANGED = object()


class NodeIdentityRegistry:
    """Explicit registry with bounded transition audit and active-slot validation.

    Inactive identities can retain their *last* physical index even if an active
    new identity occupies that same slot. This never implies parameter reuse.
    """

    def __init__(self, max_events: int = 256):
        if type(max_events) is not int or max_events < 0:
            raise ValueError("max_events must be a non-negative integer")
        self._nodes: Dict[str, NodeIdentity] = {}
        self._events: list = []
        self._max_events = max_events
        self._version = 0

    def _validate_slot(self, candidate: NodeIdentity) -> None:
        if candidate.lifecycle != "active" or candidate.physical_index is None:
            return
        for other in self._nodes.values():
            if other.stable_id != candidate.stable_id and (
                other.lifecycle == "active"
                and other.namespace == candidate.namespace
                and other.physical_index == candidate.physical_index
            ):
                raise ValueError(
                    "active physical slot occupied by %s" % other.stable_id
                )

    def _record_event(self, old: Optional[NodeIdentity], new: NodeIdentity) -> None:
        self._version += 1
        if self._max_events:
            self._events.append({
                "version": self._version,
                "stable_id": new.stable_id,
                "previous": old.to_dict() if old is not None else None,
                "current": new.to_dict(),
            })
            if len(self._events) > self._max_events:
                del self._events[:-self._max_events]

    def register(self, identity: NodeIdentity) -> None:
        if not isinstance(identity, NodeIdentity):
            raise TypeError("register requires NodeIdentity")
        if identity.stable_id in self._nodes:
            raise ValueError("identity already registered: %s" % identity.stable_id)
        self._validate_slot(identity)
        self._nodes[identity.stable_id] = identity
        self._record_event(None, identity)

    def transition(
        self,
        stable_id: str,
        *,
        lifecycle: Optional[str] = None,
        execution: Optional[str] = None,
        physical_index: Any = _UNCHANGED,
    ) -> NodeIdentity:
        old = self._nodes[stable_id]
        replacement = replace(
            old,
            lifecycle=old.lifecycle if lifecycle is None else lifecycle,
            execution=old.execution if execution is None else execution,
            physical_index=(
                old.physical_index if physical_index is _UNCHANGED
                else physical_index
            ),
        )
        self._validate_slot(replacement)
        self._nodes[stable_id] = replacement
        self._record_event(old, replacement)
        return replacement

    def snapshot(self) -> Dict[str, NodeIdentity]:
        """Detached mapping of frozen records; later transitions cannot change it."""
        return dict(self._nodes)

    def history(self) -> Tuple[Dict[str, Any], ...]:
        """Bounded, detached audit snapshots. A zero bound disables storage."""
        return tuple({
            "version": event["version"],
            "stable_id": event["stable_id"],
            "previous": (
                dict(event["previous"]) if event["previous"] is not None else None
            ),
            "current": dict(event["current"]),
        } for event in self._events)

    @property
    def version(self) -> int:
        return self._version


@dataclass(frozen=True)
class AnalysisPolicy:
    """An explicit post-observation interpretation choice.

    The default public evaluator keeps its historical behavior *unless* the
    caller supplies a policy or registry. In policy mode, full scope is the
    union of registry identities, static nodes, and recorded runtime nodes.
    """

    graph_scope: str = "full"
    gradient_policy: str = "legacy_zero_fill"

    def __post_init__(self) -> None:
        if self.graph_scope not in _SCOPES:
            raise ValueError("invalid graph_scope: %s" % self.graph_scope)
        if self.gradient_policy not in _GRADIENT_POLICIES:
            raise ValueError("invalid gradient_policy: %s" % self.gradient_policy)

    def to_dict(self) -> Dict[str, str]:
        return {
            "graph_scope": self.graph_scope,
            "gradient_policy": self.gradient_policy,
        }


def _valid_stats(stats: Optional[TensorStats]) -> bool:
    """Non-finite tensors are invalid even when their sanitized RMS is finite."""
    return (
        stats is not None
        and stats.numel > 0
        and stats.non_finite_fraction == 0.0
        and math.isfinite(stats.rms)
        and math.isfinite(stats.l2_norm)
        and stats.rms >= 0.0
        and stats.l2_norm >= 0.0
    )


def _gradient_status(rec: Optional[DynamicNodeRecord]) -> str:
    stats = rec.activation_gradient if rec is not None else None
    if stats is None:
        return "not_observed"
    if not _valid_stats(stats):
        return "invalid"
    return "observed_zero" if stats.rms == 0.0 else "observed_nonzero"


def _identity_metadata(
    nid: str,
    graph_state: GraphState,
    identities: Mapping[str, NodeIdentity],
    rec: Optional[DynamicNodeRecord],
) -> Dict[str, Any]:
    declared = identities.get(nid)
    if declared is not None:
        lifecycle = declared.lifecycle
        execution = declared.execution
        identity_source = "model_declared"
        namespace = declared.namespace
        physical_index = declared.physical_index
    elif nid in graph_state.nodes:
        lifecycle = "active" if graph_state.nodes[nid].active else "inactive"
        execution = "unknown"
        identity_source = "graph_declared"
        namespace = None
        physical_index = None
    else:
        lifecycle = "unknown"
        execution = "unknown"
        identity_source = "runtime_record"
        namespace = None
        physical_index = None

    activation_seen = rec is not None and rec.activation is not None
    conflicts = []
    if activation_seen and execution == "not_executed":
        conflicts.append("activation_observed_but_declared_not_executed")
    if _gradient_status(rec).startswith("observed_") and lifecycle in ("inactive", "retired"):
        conflicts.append("gradient_observed_but_declared_inactive")
    if execution == "unknown" and activation_seen:
        execution = "executed"
        execution_source = "observed_activation"
    else:
        execution_source = "model_declared" if declared is not None and declared.execution != "unknown" else "unknown"

    return {
        "stable_id": nid,
        "namespace": namespace,
        "physical_index": physical_index,
        "lifecycle": lifecycle,
        "lifecycle_source": identity_source,
        "execution": execution,
        "execution_source": execution_source,
        "gradient_status": _gradient_status(rec),
        "gradient_rms": (
            float(rec.activation_gradient.rms)
            if rec is not None and _valid_stats(rec.activation_gradient)
            else None
        ),
        "conflicts": conflicts,
    }


def _select_node_ids(
    policy: AnalysisPolicy, metadata: Mapping[str, Dict[str, Any]]
) -> Tuple[str, ...]:
    if policy.graph_scope == "full":
        return tuple(sorted(metadata))
    if policy.graph_scope == "active":
        return tuple(sorted(
            nid for nid, info in metadata.items()
            if info["lifecycle"] == "active"
        ))
    return tuple(sorted(
        nid for nid, info in metadata.items()
        if info["execution"] == "executed"
        or info["execution_source"] == "observed_activation"
    ))


def _coverage(
    selected: Tuple[str, ...],
    metadata: Mapping[str, Dict[str, Any]],
) -> Dict[str, Any]:
    records = [metadata[nid] for nid in selected]
    # Eligibility is model-declared active (or runtime-only unknown), not
    # explicitly inactive/retired/not-executed. This is a denominator convention,
    # never an assertion that an unobserved gradient mathematically equals zero.
    eligible = [
        info for info in records
        if info["lifecycle"] not in ("inactive", "retired")
        and info["execution"] != "not_executed"
    ]
    valid = ("observed_zero", "observed_nonzero")
    eligible_valid = sum(1 for info in eligible if info["gradient_status"] in valid)
    return {
        "selected_node_count": len(records),
        "eligible_node_count": len(eligible),
        "valid_gradient_count": sum(1 for info in records if info["gradient_status"] in valid),
        "observed_zero_count": sum(1 for info in records if info["gradient_status"] == "observed_zero"),
        "missing_count": sum(1 for info in records if info["gradient_status"] == "not_observed"),
        "invalid_count": sum(1 for info in records if info["gradient_status"] == "invalid"),
        "inactive_count": sum(1 for info in records if info["lifecycle"] in ("inactive", "retired")),
        "selected_coverage": (
            sum(1 for info in records if info["gradient_status"] in valid) / len(records)
            if records else None
        ),
        "eligible_coverage": eligible_valid / len(eligible) if eligible else None,
        "status_conflicts": {
            info["stable_id"]: info["conflicts"]
            for info in records if info["conflicts"]
        },
    }


def _observed_temporal(
    trajectory: list,
    selected: Tuple[str, ...],
    cv_unstable_threshold: float = 0.5,
    eps: float = 1e-8,
) -> Dict[str, Dict[str, Any]]:
    """Missing/invalid steps never become zero samples or vanishing evidence."""
    steps = len(trajectory)
    values_by_id = {}
    activations_by_id = {}
    invalid_by_id = {}
    for nid in selected:
        grads, activations, invalid_count = [], [], 0
        for step in trajectory:
            rec = step.get(nid)
            if rec is None:
                continue
            if _valid_stats(rec.activation):
                activations.append(float(rec.activation.rms))
            grad_status = _gradient_status(rec)
            if grad_status in ("observed_zero", "observed_nonzero"):
                grads.append(float(rec.activation_gradient.rms))
            elif grad_status == "invalid":
                invalid_count += 1
        values_by_id[nid] = grads
        activations_by_id[nid] = activations
        invalid_by_id[nid] = invalid_count

    measured_means = [float(np.mean(g)) for g in values_by_id.values() if g]
    global_median = float(np.median(measured_means)) if measured_means else None
    output: Dict[str, Dict[str, Any]] = {}
    for nid in selected:
        grads = values_by_id[nid]
        acts = activations_by_id[nid]
        count = len(grads)
        if not count:
            mu_g = sd_g = cv_g = zero_frac = None
            classification = "not_observed"
        else:
            mu_g = float(np.mean(grads))
            sd_g = float(np.std(grads, ddof=0))
            cv_g = float(sd_g / (abs(mu_g) + eps))
            zero_frac = sum(g == 0.0 for g in grads) / count
            if count < 2:
                classification = "insufficient_observations"
            elif zero_frac > 0.3:
                classification = "intermittent"
            elif cv_g > cv_unstable_threshold:
                classification = "unstable"
            elif global_median is not None and mu_g >= global_median:
                classification = "consistently_strong"
            else:
                classification = "consistently_weak"
        output[nid] = {
            "node_id": nid,
            "steps": steps,
            "observed_gradient_steps": count,
            "invalid_gradient_steps": invalid_by_id[nid],
            "missing_gradient_steps": steps - count - invalid_by_id[nid],
            "gradient_coverage": count / steps if steps else None,
            "mean_gradient": mu_g,
            "std_gradient": sd_g,
            "cv": cv_g,
            "mean_activation": float(np.mean(acts)) if acts else None,
            "std_activation": float(np.std(acts, ddof=0)) if acts else None,
            "zero_step_fraction": zero_frac,
            "classification": classification,
        }
    return output


def evaluate_with_policy(
    graph_state: GraphState,
    node_records: Mapping[str, DynamicNodeRecord],
    *,
    policy: AnalysisPolicy,
    identity_registry: Optional[NodeIdentityRegistry] = None,
    edge_records: Optional[Mapping[Tuple[str, str], TensorStats]] = None,
    static_perturbation_profile: Optional[Mapping[str, int]] = None,
    trajectory: Optional[list] = None,
) -> Dict[str, Any]:
    """Pure policy view over raw records: no mutations and no fabricated gradient."""
    # Late import: public grad_eval delegates here only for explicit policy mode.
    from .grad_eval import (
        compute_gradient_attenuation,
        compute_gradient_bottleneck,
        compute_forward_backward_alignment,
        compute_topology_gradient_correspondence,
        compute_temporal_stability,
        detect_gradient_anomalies,
    )
    if not isinstance(policy, AnalysisPolicy):
        raise TypeError("policy must be AnalysisPolicy")
    identities = identity_registry.snapshot() if identity_registry is not None else {}
    all_ids = sorted(set(graph_state.nodes) | set(node_records) | set(identities))
    metadata = {
        nid: _identity_metadata(nid, graph_state, identities, node_records.get(nid))
        for nid in all_ids
    }
    selected = _select_node_ids(policy, metadata)
    observations = {nid: metadata[nid] for nid in selected}
    coverage = _coverage(selected, metadata)

    # Synthetic *records* are only for explicit legacy zero-imputation, never
    # written to raw observations and never marked as observed in provenance.
    scoped_records = {
        nid: node_records.get(nid) or DynamicNodeRecord(
            node_id=nid, node_type="unobserved"
        )
        for nid in selected
    }
    eligible_ids = set(
        nid for nid in selected
        if metadata[nid]["lifecycle"] not in ("inactive", "retired")
        and metadata[nid]["execution"] != "not_executed"
    )
    grad_records = {
        nid: node_records[nid]
        for nid in selected if nid in eligible_ids
        and nid in node_records
        and metadata[nid]["gradient_status"] in ("observed_zero", "observed_nonzero")
    }
    joint_records = {
        nid: rec for nid, rec in grad_records.items() if _valid_stats(rec.activation)
    }
    valid_edges = [
        edge for edge in graph_state.edges
        if edge.source in selected and edge.target in selected
    ]
    if policy.gradient_policy == "observed_only":
        bottleneck_input = grad_records
        anomaly_input = grad_records
        alignment_input = joint_records
        correspondence_input = grad_records
        attenuation_edges = [
            edge for edge in valid_edges
            if edge.source in grad_records and edge.target in grad_records
        ]
        attenuation_records = grad_records
    else:
        bottleneck_input = anomaly_input = scoped_records
        alignment_input = correspondence_input = scoped_records
        attenuation_edges = valid_edges
        attenuation_records = scoped_records

    attenuation = compute_gradient_attenuation(
        attenuation_edges, attenuation_records, edge_records
    )
    bottleneck = compute_gradient_bottleneck(bottleneck_input).to_dict()
    alignment = compute_forward_backward_alignment(alignment_input).to_dict()
    anomalies = detect_gradient_anomalies(anomaly_input).to_dict()
    correspondence = compute_topology_gradient_correspondence(
        dict(static_perturbation_profile or {}), correspondence_input
    ).to_dict()

    if policy.gradient_policy == "observed_only":
        if not grad_records:
            bottleneck.update({
                "status": "undefined",
                "node_concentrations": {},
                "max_concentration": None,
                "bottleneck_node": None,
                "mean_gradient": None,
                "median_gradient": None,
                "total_gradient": None,
                "fixed_point_max_concentration": None,
            })
            anomalies["status"] = "undefined"
        else:
            bottleneck["status"] = "defined"
            anomalies["status"] = "defined"

        # Correlations are undefined for insufficient pairs or constant inputs.
        act_vals = [float(rec.activation.rms) for rec in joint_records.values()]
        grad_vals = [float(rec.activation_gradient.rms) for rec in joint_records.values()]
        if len(joint_records) < 2 or len(set(act_vals)) < 2 or len(set(grad_vals)) < 2:
            alignment.update({
                "status": "undefined",
                "spearman_rho": None,
                "spearman_pvalue": None,
                "pearson_r": None,
                "pearson_pvalue": None,
                "fixed_point_spearman": None,
                "fixed_point_pearson": None,
            })
        else:
            alignment["status"] = "defined"
        common_ids = set(static_perturbation_profile or {}) & set(grad_records)
        topo_vals = [float(static_perturbation_profile[nid]) for nid in common_ids]
        grad_vals_topo = [float(grad_records[nid].activation_gradient.rms) for nid in common_ids]
        if len(common_ids) < 2 or len(set(topo_vals)) < 2 or len(set(grad_vals_topo)) < 2:
            correspondence.update({
                "status": "undefined",
                "spearman_topo_grad": None,
                "spearman_pvalue": None,
            })
        else:
            correspondence["status"] = "defined"

    temporal = None
    if trajectory and len(trajectory) > 1:
        if policy.gradient_policy == "observed_only":
            temporal = _observed_temporal(trajectory, selected)
        else:
            scoped_trajectory = [
                {nid: rec for nid, rec in step.items() if nid in selected}
                for step in trajectory
            ]
            stats = compute_temporal_stability(scoped_trajectory)
            temporal = {nid: rec.to_dict() for nid, rec in stats.items()}

    return {
        "attenuation": {
            "%s->%s" % (k[0], k[1]): v.to_dict()
            for k, v in attenuation.items()
        },
        "bottleneck": bottleneck,
        "alignment": alignment,
        "anomalies": anomalies,
        "correspondence": correspondence,
        "temporal": temporal,
        "summary": {
            "node_count": len(scoped_records),
            "edge_count": len(valid_edges),
            "max_gradient_concentration": bottleneck["max_concentration"],
            "bottleneck_node": bottleneck["bottleneck_node"],
            "spearman_act_grad": alignment["spearman_rho"],
            "pearson_act_grad": alignment["pearson_r"],
            "spearman_topo_grad": correspondence["spearman_topo_grad"],
            "type_counts": correspondence["counts"],
        },
        "policy": policy.to_dict(),
        "observations": observations,
        "coverage": coverage,
        "metric_populations": {
            "gradient": sorted(bottleneck_input),
            "alignment": sorted(alignment_input),
            "correspondence": sorted(set(correspondence_input) & set(static_perturbation_profile or {})),
            "attenuation_edges": len(attenuation_edges),
        },
    }


def compare_analysis_policies(
    graph_state: GraphState,
    node_records: Mapping[str, DynamicNodeRecord],
    policies: Mapping[str, AnalysisPolicy],
    *,
    identity_registry: Optional[NodeIdentityRegistry] = None,
    edge_records: Optional[Mapping[Tuple[str, str], TensorStats]] = None,
    static_perturbation_profile: Optional[Mapping[str, int]] = None,
    trajectory: Optional[list] = None,
) -> Dict[str, Dict[str, Any]]:
    """Evaluate several explicitly labelled policies against identical raw data."""
    if not policies:
        raise ValueError("policies must be non-empty")
    return {
        label: evaluate_with_policy(
            graph_state, node_records,
            policy=policy,
            identity_registry=identity_registry,
            edge_records=edge_records,
            static_perturbation_profile=static_perturbation_profile,
            trajectory=trajectory,
        )
        for label, policy in policies.items()
    }
