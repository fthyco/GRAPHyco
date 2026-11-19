"""Audited human-readable contracts for the legacy summary keys.

Do not change the legacy serialized keys or their numeric implementations here.
The names and descriptions explicitly distinguish structural proxies, gradient
observations, and true graph-theoretic measurements.
"""
from __future__ import annotations

# (legacy summary key, accurate user-facing label, mathematical meaning)
QUERY_METRIC_DEFINITIONS = (
    (
        "density",
        "Directed Edge-Count Density (D)",
        "Raw |E|/(|V|(|V|-1)) on directed edges; may exceed 1 for multigraphs or self-loops.",
    ),
    (
        "coherence",
        "Incident-Edge Uniformity (C)",
        "1 minus normalized variance of incident-edge shares; not gradient-flow conservation.",
    ),
    (
        "topological_bottleneck",
        "Degree-Concentration Proxy (B)",
        "Maximum incident-edge share divided by its mean; not node betweenness centrality.",
    ),
    (
        "resilience",
        "Structural Cost Proxy (R)",
        "Inverse aggregate neighbor-cost heuristic; not measured reachability under node removal.",
    ),
    (
        "grad_bottleneck_gmax",
        "Gradient-RMS Concentration (G_max)",
        "Largest node gradient RMS divided by mean node gradient RMS (+eps); unobserved is treated as zero.",
    ),
    (
        "bottleneck_node",
        "Peak Gradient-RMS Node",
        "Node selected by highest recorded gradient RMS; not proof of a causal learning bottleneck.",
    ),
    (
        "spearman_act_grad",
        "Activation-Gradient Spearman rho",
        "Spearman correlation across nodes of activation RMS and gradient RMS; missing values are zero-filled.",
    ),
    (
        "mean_stability_cv",
        "Mean Temporal Gradient CV",
        "Mean of per-node temporal gradient coefficients of variation across recorded steps.",
    ),
)

# The reference metrics are opt-in; historical summary keys remain unchanged.
REFERENCE_METRIC_DEFINITIONS = {
    "directed_node_betweenness": (
        "Directed unweighted shortest-path betweenness, endpoint-excluded, "
        "Brandes algorithm; normalized over (N-1)(N-2)."
    ),
    "pairwise_reachability_ablation_loss": (
        "Fraction of surviving-node ordered reachable pairs eliminated by "
        "deleting each node; baseline excludes pairs incident to removed node."
    ),
}
