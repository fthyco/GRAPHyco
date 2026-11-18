# GRAPHyco mathematical contracts — Stage 2

Status: **audited source definitions**. This document describes existing legacy
functions as implemented and two opt-in graph-theoretic reference functions.
It does not assert causal neural-network importance or predictive benefit.
See `tests/test_mathematical_contracts.py` for independent hand-checkable oracles.

## Compatibility

- Every legacy `core.evaluation.evaluate()` key, numeric computation, scale
  (`SCALE=10000`), and `DynamicGraphState.to_dict()` key is unchanged.
- `QueryEngine.describe()` receives corrected display labels/descriptions
  imported from `core.metric_contracts.QUERY_METRIC_DEFINITIONS`.
  **Its DataFrame row labels change**; integrations should use stable serialized
  keys instead of human-readable DataFrame labels.
- The exact reference functions are opt-in and are not automatically inserted
  into historical benchmark JSON or renamed over old values.

## 1. Legacy static values (units: fixed-point integer)

Let `N=|V|`, `M=|E|`, and `S=10000`. The edge list may contain
duplicates and self-loops unless the caller filters it.

| Key | Actual computation | Interpretable as | Not equivalent to |
| --- | --- | --- | --- |
| `graph_density` | `floor(M*S / (N*(N-1)))` for `N>1` | Raw directed edge-count ratio | Simple directed graph density when duplicate edges or self-loops exist |
| `compute_local_density(i)` | `floor(S*(# edges incident to i)/M)` for `M>0` | Incident-edge **share**, not normalized by all possible local neighbors | Standard local subgraph density |
| `connectivity_coherence` | `clip(S-floor(variance(d)*S/max_variance),0,S)` where `d_i` is incident-edge share | Degree-share uniformity heuristic | Conservation of backpropagated gradient |
| `topological_bottleneck_ratio` | `floor(max(d)*S/floor(mean(d)))`, or `0` if integer mean vanishes | Degree concentration proxy | Shortest-path betweenness / bottleneck causality |
| `perturbation_profile(i)` | Local cost based on integer-thresholded `d_i` and unique undirected neighbors | Local neighbor-cost heuristic | Actual deletion perturbation |
| `structural_perturbation_resilience` | `floor(N*S / sum_i cost_i)` when cost is nonzero | Inverse aggregate structural cost | Reachability retention, damage or model accuracy under ablation |
| `fan_balance(i)` | `floor(S*abs(in_i-out_i)/(in_i+out_i))`, or `0` when isolated | Directional degree imbalance | Actual gradient-flow conservation |
| `dead_ratio` | `floor(S*(number of isolated nodes)/N)`, or `0` when empty | Isolated-node fraction | Dead ReLU fraction |
| `critical_path_density` | `floor(S*(number of critical-flagged edges)/M)` | Fraction of flagged edges | Critical-path length or longest-path concentration |
| `cycle_pressure` | Number of DFS back-edge cycles returned by `detect_critical_cycles` | Detector count | Enumeration of every directed simple cycle |

All formulas above describe existing code; no normalization range beyond what
is actually enforced is promised. In particular raw graph density can exceed
`S` on a directed multigraph. The use of integer floors can cause numerical
degeneracy for extremely sparse, very large graphs; the legacy result is
preserved in Stage 2 for report compatibility.

### Hand-checkable legacy example

For the directed path `A -> B -> C`, two critical edges:

| Quantity | Legacy result |
| --- | ---: |
| `graph_density` | 3333 |
| Incident-edge shares `A, B, C` | 5000, 10000, 5000 |
| `connectivity_coherence` | 9167 |
| `topological_bottleneck_ratio` | 15001 |
| `perturbation_profile` | A=2, B=4, C=2 |
| `structural_perturbation_resilience` | 3750 |
| `critical_path_density` | 10000 |

These oracles guard historical behavior; the numerical values are not evidence
that the middle node is a causal optimizer bottleneck.

## 2. Exact opt-in graph-theoretic references (unscaled float)

`graphyco.core.directed_node_betweenness(state, normalized=True)`:
Directed **unweighted** shortest-path betweenness computed using Brandes'
algorithm with endpoints excluded:

`C_B(v)=sum_(s,t: s!=v!=t, s!=t) sigma_st(v)/sigma_st`

where unreachable pairs contribute zero. With `N>2`, normalized by
`(N-1)(N-2)`; with `N<=2`, values are zero. Parallel arcs are deduplicated
and self-loops ignored.

`graphyco.core.pairwise_reachability_ablation_loss(state)`:
For each removed node `v`, let `P_v` be the set of **ordered reachable**
`(s,t)` pairs in the original graph, with `s!=t`, `s!=v`, `t!=v`.
Let `P'_v` be those same pairs reachable after deleting `v` and its
incident edges. Define

`L_v = |P_v - P'_v| / |P_v|` if `|P_v|>0`, otherwise `0`.

This excludes pairs involving the removed node: it measures loss of
connectivity **among surviving vertices**, not lost service from deleting the
node itself. This is a graph-theoretic ablation result, not a neural model
ablation or an optimizer outcome. It does not assume a unique input node.

### Exact example

On `A -> B -> C`, normalized directed betweenness is `B=0.5`,
`A=C=0`; pairwise reachability ablation loss is `B=1.0`, `A=C=0`.
On diamond `A -> {B,C} -> D`, `B=C=1/12` betweenness and both
`B=C=0` ablation loss because each branch has an alternate path.

The historical `visualizer.query.compute_ablation_map()` has different
semantics: it chooses the first zero-indegree node (or first node), treats the
initial source's removal as `0`, and measures a decrease in visited vertices
from that one source. Do not compare its results numerically with the new
pairwise metric without specifying their different denominators.

## 3. Legacy dynamic values (floating-point)

- **Gradient concentration `G_max`:** `g_i = RMS(dL/da_i)` if recorded,
  otherwise `0`; `q_i=g_i/(mean_j g_j + 1e-8)`;
  `G_max=max_i q_i`. This is a concentration statistic, not a proof that the
  maximizer obstructs optimization. A missing gradient is **not** semantically
  the same as a measured zero; the legacy function currently conflates them.
- **Edge attenuation:** uses `L2` norms of **node activation gradients**
  on both endpoints, not necessarily the intercepted per-edge gradient:
  `alpha=||g_source||_2/(||g_target||_2+eps)`. Norms depend on tensor size.
  The separately recorded per-edge RMS is a distinct diagnostic observation.
- **Activation-gradient correlation:** Spearman/Pearson are over per-node
  activation RMS and gradient RMS, filling missing values with zero.
  Missing and constant arrays result in `rho=0, p=1`.
- **Topology-gradient correspondence:** classifies by median splits on
  the current *structural cost proxy* and gradient RMS. Median ties are
  counted in the high groups via `>=`; labels I–IV are heuristic quadrants.
- **Temporal stability:** population standard deviation divided by
  `abs(mean gradient)+eps` across observed steps; missing observations are
  zero-filled in the legacy computation. The query summary reports the mean
  of these per-node CVs.
- **Anomaly thresholds:** default absolute RMS values `1e-6` and `100`
  are heuristics, not universal thresholds for healthy training.

No claims of causal importance, reliable neuron ranking, optimization gains,
or universal diagnostic thresholds can be derived from these definitions
without controlled experiments and independent calibration.

## 4. Scope of future correction

Any change to legacy metric numerical results (missing-value handling,
floating-point normalization, graph-type constraints, or cycle counting) must
be implemented as a versioned `v2` contract or an explicit migration.
Stage 2 only corrects descriptions and adds opt-in exact structural references.
Graph provenance, fallback fidelity and runtime intervention are separate stages.
