from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from scipy.stats import pearsonr, spearmanr

from core.primitives import Edge, GraphState, SCALE, checked_mul
from .grad_observer import DynamicNodeRecord, TensorStats


# ══════════════════════════════════════════════════════════════
#  GRADIENT ATTENUATION ANALYSIS
# ══════════════════════════════════════════════════════════════

@dataclass
class EdgeAttenuation:
    """Attenuation statistics across a directed dependency edge (source -> target)."""
    source: str
    target: str
    source_grad_l2: float
    target_grad_l2: float
    source_grad_rms: float
    target_grad_rms: float
    forward_ratio: float      # ||g_source|| / (||g_target|| + eps)
    backward_ratio: float     # ||g_target|| / (||g_source|| + eps)
    log_forward_ratio: float  # log((||g_source|| + eps) / (||g_target|| + eps))
    log_backward_ratio: float # log((||g_target|| + eps) / (||g_source|| + eps))
    edge_grad_rms: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "source_grad_l2": self.source_grad_l2,
            "target_grad_l2": self.target_grad_l2,
            "source_grad_rms": self.source_grad_rms,
            "target_grad_rms": self.target_grad_rms,
            "forward_ratio": self.forward_ratio,
            "backward_ratio": self.backward_ratio,
            "log_forward_ratio": self.log_forward_ratio,
            "log_backward_ratio": self.log_backward_ratio,
            "edge_grad_rms": self.edge_grad_rms,
        }


def compute_gradient_attenuation(
    edges: List[Edge],
    node_records: Dict[str, DynamicNodeRecord],
    edge_records: Optional[Dict[Tuple[str, str], TensorStats]] = None,
    eps: float = 1e-8,
) -> Dict[Tuple[str, str], EdgeAttenuation]:
    """Computes gradient attenuation across all graph edges.

    For a dependency edge e = (source -> target):
    - Forward ratio alpha = ||g_source||_2 / (||g_target||_2 + eps)
    - Backward ratio beta = ||g_target||_2 / (||g_source||_2 + eps)
    - Delta = log((||g_source||_2 + eps) / (||g_target||_2 + eps))
    """
    attenuations: Dict[Tuple[str, str], EdgeAttenuation] = {}

    for edge in edges:
        src = edge.source
        tgt = edge.target
        key = (src, tgt)

        src_rec = node_records.get(src)
        tgt_rec = node_records.get(tgt)

        src_l2 = src_rec.activation_gradient.l2_norm if (src_rec and src_rec.activation_gradient) else 0.0
        tgt_l2 = tgt_rec.activation_gradient.l2_norm if (tgt_rec and tgt_rec.activation_gradient) else 0.0
        src_rms = src_rec.activation_gradient.rms if (src_rec and src_rec.activation_gradient) else 0.0
        tgt_rms = tgt_rec.activation_gradient.rms if (tgt_rec and tgt_rec.activation_gradient) else 0.0

        fwd_ratio = float(src_l2 / (tgt_l2 + eps))
        bwd_ratio = float(tgt_l2 / (src_l2 + eps))
        log_fwd = float(math.log((src_l2 + eps) / (tgt_l2 + eps)))
        log_bwd = float(math.log((tgt_l2 + eps) / (src_l2 + eps)))

        edge_rms = None
        if edge_records and key in edge_records:
            edge_rms = edge_records[key].rms

        attenuations[key] = EdgeAttenuation(
            source=src,
            target=tgt,
            source_grad_l2=src_l2,
            target_grad_l2=tgt_l2,
            source_grad_rms=src_rms,
            target_grad_rms=tgt_rms,
            forward_ratio=fwd_ratio,
            backward_ratio=bwd_ratio,
            log_forward_ratio=log_fwd,
            log_backward_ratio=log_bwd,
            edge_grad_rms=edge_rms,
        )

    return attenuations


# ══════════════════════════════════════════════════════════════
#  DYNAMIC GRADIENT BOTTLENECK ANALYSIS
# ══════════════════════════════════════════════════════════════

@dataclass
class GradientBottleneckResult:
    """Dynamic gradient concentration measures across computational graph nodes."""
    node_concentrations: Dict[str, float]  # q_i = g_i / (mean(g) + eps)
    max_concentration: float              # G_max = max_i q_i
    bottleneck_node: str                  # argmax_i q_i
    mean_gradient: float                  # mean(g)
    median_gradient: float
    total_gradient: float
    fixed_point_max_concentration: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_concentrations": self.node_concentrations,
            "max_concentration": self.max_concentration,
            "bottleneck_node": self.bottleneck_node,
            "mean_gradient": self.mean_gradient,
            "median_gradient": self.median_gradient,
            "total_gradient": self.total_gradient,
            "fixed_point_max_concentration": self.fixed_point_max_concentration,
        }


def compute_gradient_bottleneck(
    node_records: Dict[str, DynamicNodeRecord],
    eps: float = 1e-8,
    scale: int = SCALE,
) -> GradientBottleneckResult:
    """Computes dynamic gradient concentration independently of static graph topology.

    q_i = g_i / ( 1/|V| * sum_j(g_j) + eps )
    where g_i is the normalized gradient magnitude (RMS).
    """
    grad_magnitudes: Dict[str, float] = {}
    for nid, rec in node_records.items():
        val = rec.activation_gradient.rms if rec.activation_gradient else 0.0
        grad_magnitudes[nid] = float(val)

    values = list(grad_magnitudes.values())
    if not values or len(values) == 0:
        return GradientBottleneckResult(
            node_concentrations={},
            max_concentration=0.0,
            bottleneck_node="",
            mean_gradient=0.0,
            median_gradient=0.0,
            total_gradient=0.0,
            fixed_point_max_concentration=0,
        )

    mean_g = float(np.mean(values))
    median_g = float(np.median(values))
    total_g = float(np.sum(values))

    denom = mean_g + eps
    concentrations = {nid: float(val / denom) for nid, val in grad_magnitudes.items()}

    b_node = max(concentrations, key=lambda k: concentrations[k])
    max_c = concentrations[b_node]
    fp_max_c = int(round(max_c * scale))

    return GradientBottleneckResult(
        node_concentrations=concentrations,
        max_concentration=max_c,
        bottleneck_node=b_node,
        mean_gradient=mean_g,
        median_gradient=median_g,
        total_gradient=total_g,
        fixed_point_max_concentration=fp_max_c,
    )


# ══════════════════════════════════════════════════════════════
#  FORWARD-BACKWARD ALIGNMENT ANALYSIS
# ══════════════════════════════════════════════════════════════

@dataclass
class ForwardBackwardAlignment:
    """Correlation and local ratio analysis between forward activations and backward gradients."""
    node_sensitivity_ratios: Dict[str, float]  # r_i = g_i / (a_i + eps)
    spearman_rho: float
    spearman_pvalue: float
    pearson_r: float
    pearson_pvalue: float
    fixed_point_spearman: int
    fixed_point_pearson: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_sensitivity_ratios": self.node_sensitivity_ratios,
            "spearman_rho": self.spearman_rho,
            "spearman_pvalue": self.spearman_pvalue,
            "pearson_r": self.pearson_r,
            "pearson_pvalue": self.pearson_pvalue,
            "fixed_point_spearman": self.fixed_point_spearman,
            "fixed_point_pearson": self.fixed_point_pearson,
        }


def compute_forward_backward_alignment(
    node_records: Dict[str, DynamicNodeRecord],
    eps: float = 1e-8,
    scale: int = SCALE,
) -> ForwardBackwardAlignment:
    """Evaluates alignment between forward activation magnitude and backward gradient magnitude."""
    ratios: Dict[str, float] = {}
    acts: List[float] = []
    grads: List[float] = []

    for nid in sorted(node_records.keys()):
        rec = node_records[nid]
        a = rec.activation.rms if rec.activation else 0.0
        g = rec.activation_gradient.rms if rec.activation_gradient else 0.0

        ratio = float(g / (a + eps))
        ratios[nid] = ratio
        acts.append(float(a))
        grads.append(float(g))

    if len(acts) < 2 or np.all(np.array(acts) == acts[0]) or np.all(np.array(grads) == grads[0]):
        rho, p_rho = 0.0, 1.0
        r, p_r = 0.0, 1.0
    else:
        try:
            s_res = spearmanr(acts, grads)
            rho = float(s_res.statistic) if not np.isnan(s_res.statistic) else 0.0
            p_rho = float(s_res.pvalue) if not np.isnan(s_res.pvalue) else 1.0
        except Exception:
            rho, p_rho = 0.0, 1.0

        try:
            p_res = pearsonr(acts, grads)
            r = float(p_res.statistic) if not np.isnan(p_res.statistic) else 0.0
            p_r = float(p_res.pvalue) if not np.isnan(p_res.pvalue) else 1.0
        except Exception:
            r, p_r = 0.0, 1.0

    return ForwardBackwardAlignment(
        node_sensitivity_ratios=ratios,
        spearman_rho=rho,
        spearman_pvalue=p_rho,
        pearson_r=r,
        pearson_pvalue=p_r,
        fixed_point_spearman=int(round(rho * scale)),
        fixed_point_pearson=int(round(r * scale)),
    )


# ══════════════════════════════════════════════════════════════
#  TOPOLOGY-GRADIENT CORRESPONDENCE (TYPES I - IV)
# ══════════════════════════════════════════════════════════════

@dataclass
class TopologyGradientClassification:
    """Scientific categorization of nodes based on structural vs. dynamic importance.

    Type I:   High Topology Importance + High Gradient
    Type II:  High Topology Importance + Low Gradient  (Structural bottleneck with weak gradient)
    Type III: Low Topology Importance  + High Gradient (Structurally ordinary but dynamically critical)
    Type IV:  Low Topology Importance  + Low Gradient
    """
    node_categories: Dict[str, str]
    type_i_nodes: List[str]
    type_ii_nodes: List[str]
    type_iii_nodes: List[str]
    type_iv_nodes: List[str]
    topology_threshold: float
    gradient_threshold: float
    spearman_topo_grad: float
    spearman_pvalue: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_categories": self.node_categories,
            "counts": {
                "Type I": len(self.type_i_nodes),
                "Type II": len(self.type_ii_nodes),
                "Type III": len(self.type_iii_nodes),
                "Type IV": len(self.type_iv_nodes),
            },
            "type_i_nodes": self.type_i_nodes,
            "type_ii_nodes": self.type_ii_nodes,
            "type_iii_nodes": self.type_iii_nodes,
            "type_iv_nodes": self.type_iv_nodes,
            "topology_threshold": self.topology_threshold,
            "gradient_threshold": self.gradient_threshold,
            "spearman_topo_grad": self.spearman_topo_grad,
            "spearman_pvalue": self.spearman_pvalue,
        }


def compute_topology_gradient_correspondence(
    static_profile: Dict[str, int],
    node_records: Dict[str, DynamicNodeRecord],
    topo_threshold: Optional[float] = None,
    grad_threshold: Optional[float] = None,
) -> TopologyGradientClassification:
    """Classifies nodes into Types I-IV using median splits or user thresholds."""
    common_nodes = sorted(set(static_profile.keys()) & set(node_records.keys()))
    if not common_nodes:
        return TopologyGradientClassification(
            node_categories={},
            type_i_nodes=[],
            type_ii_nodes=[],
            type_iii_nodes=[],
            type_iv_nodes=[],
            topology_threshold=0.0,
            gradient_threshold=0.0,
            spearman_topo_grad=0.0,
            spearman_pvalue=1.0,
        )

    topo_vals = [float(static_profile[nid]) for nid in common_nodes]
    grad_vals = [
        float(node_records[nid].activation_gradient.rms if node_records[nid].activation_gradient else 0.0)
        for nid in common_nodes
    ]

    t_thresh = float(np.median(topo_vals)) if topo_threshold is None else topo_threshold
    g_thresh = float(np.median(grad_vals)) if grad_threshold is None else grad_threshold

    categories: Dict[str, str] = {}
    t_i: List[str] = []
    t_ii: List[str] = []
    t_iii: List[str] = []
    t_iv: List[str] = []

    for nid in common_nodes:
        t = float(static_profile[nid])
        g = float(node_records[nid].activation_gradient.rms if node_records[nid].activation_gradient else 0.0)

        high_t = t >= t_thresh
        high_g = g >= g_thresh

        if high_t and high_g:
            cat = "Type I"
            t_i.append(nid)
        elif high_t and not high_g:
            cat = "Type II"
            t_ii.append(nid)
        elif not high_t and high_g:
            cat = "Type III"
            t_iii.append(nid)
        else:
            cat = "Type IV"
            t_iv.append(nid)

        categories[nid] = cat

    # Rank correlation between static topology importance and dynamic gradient magnitude
    if len(topo_vals) >= 2 and not (np.all(np.array(topo_vals) == topo_vals[0]) or np.all(np.array(grad_vals) == grad_vals[0])):
        s_res = spearmanr(topo_vals, grad_vals)
        s_rho = float(s_res.statistic) if not np.isnan(s_res.statistic) else 0.0
        s_pval = float(s_res.pvalue) if not np.isnan(s_res.pvalue) else 1.0
    else:
        s_rho, s_pval = 0.0, 1.0

    return TopologyGradientClassification(
        node_categories=categories,
        type_i_nodes=t_i,
        type_ii_nodes=t_ii,
        type_iii_nodes=t_iii,
        type_iv_nodes=t_iv,
        topology_threshold=t_thresh,
        gradient_threshold=g_thresh,
        spearman_topo_grad=s_rho,
        spearman_pvalue=s_pval,
    )


# ══════════════════════════════════════════════════════════════
#  TEMPORAL STABILITY & DYNAMICS
# ══════════════════════════════════════════════════════════════

@dataclass
class TemporalNodeStats:
    """Temporal statistics of a node over execution steps t = 1, ..., T."""
    node_id: str
    steps: int
    mean_gradient: float
    std_gradient: float
    cv: float                  # sigma / (|mu| + eps)
    mean_activation: float
    std_activation: float
    zero_step_fraction: float
    classification: str        # 'consistently_strong', 'consistently_weak', 'unstable', 'intermittent'

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "steps": self.steps,
            "mean_gradient": self.mean_gradient,
            "std_gradient": self.std_gradient,
            "cv": self.cv,
            "mean_activation": self.mean_activation,
            "std_activation": self.std_activation,
            "zero_step_fraction": self.zero_step_fraction,
            "classification": self.classification,
        }


def compute_temporal_stability(
    trajectory: List[Dict[str, DynamicNodeRecord]],
    cv_unstable_threshold: float = 0.5,
    eps: float = 1e-8,
) -> Dict[str, TemporalNodeStats]:
    """Computes temporal gradient mean, variance, and CV across multiple execution steps."""
    if not trajectory:
        return {}

    steps = len(trajectory)
    all_nodes: Set[str] = set()
    for step_rec in trajectory:
        all_nodes.update(step_rec.keys())

    # Collect trajectory arrays per node
    node_grads: Dict[str, List[float]] = {nid: [] for nid in all_nodes}
    node_acts: Dict[str, List[float]] = {nid: [] for nid in all_nodes}

    for step_rec in trajectory:
        for nid in all_nodes:
            rec = step_rec.get(nid)
            g = rec.activation_gradient.rms if (rec and rec.activation_gradient) else 0.0
            a = rec.activation.rms if (rec and rec.activation) else 0.0
            node_grads[nid].append(float(g))
            node_acts[nid].append(float(a))

    all_means = [float(np.mean(g_list)) for g_list in node_grads.values()]
    global_median_mean = float(np.median(all_means)) if all_means else 0.0

    results: Dict[str, TemporalNodeStats] = {}
    for nid in sorted(all_nodes):
        g_arr = np.array(node_grads[nid], dtype=np.float64)
        a_arr = np.array(node_acts[nid], dtype=np.float64)

        mu_g = float(g_arr.mean())
        std_g = float(g_arr.std(ddof=0)) if len(g_arr) > 1 else 0.0
        cv_g = float(std_g / (abs(mu_g) + eps))

        mu_a = float(a_arr.mean())
        std_a = float(a_arr.std(ddof=0)) if len(a_arr) > 1 else 0.0

        zero_fraction = float(np.sum(g_arr == 0.0) / float(steps))

        # Classification
        if zero_fraction > 0.3:
            cls = "intermittent"
        elif cv_g > cv_unstable_threshold:
            cls = "unstable"
        elif mu_g >= global_median_mean:
            cls = "consistently_strong"
        else:
            cls = "consistently_weak"

        results[nid] = TemporalNodeStats(
            node_id=nid,
            steps=steps,
            mean_gradient=mu_g,
            std_gradient=std_g,
            cv=cv_g,
            mean_activation=mu_a,
            std_activation=std_a,
            zero_step_fraction=zero_fraction,
            classification=cls,
        )

    return results


# ══════════════════════════════════════════════════════════════
#  VANISHING & EXPLODING GRADIENT DETECTION
# ══════════════════════════════════════════════════════════════

@dataclass
class AnomalyDetectionResult:
    """Detection of vanishing and exploding gradient nodes."""
    vanishing_nodes: List[str]
    exploding_nodes: List[str]
    low_threshold: float
    high_threshold: float
    is_relative: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "vanishing_nodes": self.vanishing_nodes,
            "exploding_nodes": self.exploding_nodes,
            "low_threshold": self.low_threshold,
            "high_threshold": self.high_threshold,
            "is_relative": self.is_relative,
        }


def detect_gradient_anomalies(
    node_records: Dict[str, DynamicNodeRecord],
    tau_low: Optional[float] = None,
    tau_high: Optional[float] = None,
    relative: bool = False,
) -> AnomalyDetectionResult:
    """Identifies nodes exhibiting vanishing or exploding gradients.

    Supports both absolute thresholds and relative percentile/distributional thresholds.
    """
    magnitudes = {
        nid: float(rec.activation_gradient.rms if rec.activation_gradient else 0.0)
        for nid, rec in node_records.items()
    }
    values = list(magnitudes.values())
    if not values:
        return AnomalyDetectionResult([], [], 0.0, 0.0, relative)

    if relative:
        med = float(np.median(values))
        t_low = 0.01 * med if tau_low is None else tau_low
        t_high = 100.0 * med if tau_high is None else tau_high
    else:
        t_low = 1e-6 if tau_low is None else tau_low
        t_high = 100.0 if tau_high is None else tau_high

    vanishing = [nid for nid, val in magnitudes.items() if val < t_low]
    exploding = [nid for nid, val in magnitudes.items() if val > t_high]

    return AnomalyDetectionResult(
        vanishing_nodes=sorted(vanishing),
        exploding_nodes=sorted(exploding),
        low_threshold=t_low,
        high_threshold=t_high,
        is_relative=relative,
    )


# ══════════════════════════════════════════════════════════════
#  UNIFIED COMPREHENSIVE FLOW EVALUATION
# ══════════════════════════════════════════════════════════════

def evaluate_dynamic_flow(
    graph_state: GraphState,
    node_records: Dict[str, DynamicNodeRecord],
    edge_records: Optional[Dict[Tuple[str, str], TensorStats]] = None,
    static_perturbation_profile: Optional[Dict[str, int]] = None,
    trajectory: Optional[List[Dict[str, DynamicNodeRecord]]] = None,
) -> Dict[str, Any]:
    """Runs the complete suite of dynamic flow evaluations over a monitored execution."""
    attenuation = compute_gradient_attenuation(graph_state.edges, node_records, edge_records)
    bottleneck = compute_gradient_bottleneck(node_records)
    alignment = compute_forward_backward_alignment(node_records)
    anomalies = detect_gradient_anomalies(node_records)

    # Topology-gradient correspondence
    topo_profile = static_perturbation_profile or {}
    correspondence = compute_topology_gradient_correspondence(topo_profile, node_records)

    # Temporal stability if trajectory provided
    temporal = None
    if trajectory and len(trajectory) > 1:
        temporal = compute_temporal_stability(trajectory)

    return {
        "attenuation": {f"{k[0]}->{k[1]}": v.to_dict() for k, v in attenuation.items()},
        "bottleneck": bottleneck.to_dict(),
        "alignment": alignment.to_dict(),
        "anomalies": anomalies.to_dict(),
        "correspondence": correspondence.to_dict(),
        "temporal": {k: v.to_dict() for k, v in temporal.items()} if temporal else None,
        "summary": {
            "node_count": len(node_records),
            "edge_count": len(graph_state.edges),
            "max_gradient_concentration": bottleneck.max_concentration,
            "bottleneck_node": bottleneck.bottleneck_node,
            "spearman_act_grad": alignment.spearman_rho,
            "pearson_act_grad": alignment.pearson_r,
            "spearman_topo_grad": correspondence.spearman_topo_grad,
            "type_counts": correspondence.to_dict()["counts"],
        }
    }
