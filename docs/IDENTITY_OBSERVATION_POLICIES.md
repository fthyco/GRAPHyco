# Stable node identity and gradient-observation policies

Stage 8 implementation, stacked on [PR #7](https://github.com/fthyco/GRAPHyco/pull/7).

## Contract

Separate **model-declared lifecycle**, **runtime evidence**, and **metric
interpretation**:

- Identity is a caller-declared stable ID; it is **not inferred** from a tensor
  offset, layer path, PyTorch module/FX node name, or gradient presence.
- A physical index is the **last known slot**, not a persistent identity.
  Deactivation may preserve that slot; activating a *new* neuron in the same
  slot must use a **different stable_id**. Active slot collisions within a
  namespace are rejected.
- Lifecycle values: `active`, `inactive`, `retired`. Execution declarations:
  `executed`, `not_executed`, `unknown`. A monitored activation supplies
  positive evidence of execution. The absence of an activation or hook does
  **not** prove non-execution.
- Gradient statuses are derived independently from captured TensorStats:
  `observed_zero`, `observed_nonzero`, `not_observed`, `invalid`.
  Nonfinite tensor content (even with finite sanitized RMS) is invalid and
  excluded from missing-aware numerical metrics. Empty tensors are invalid.
- The original `DynamicNodeRecord` remains a raw observation; policy evaluation
  is a pure projection and **does not mutate input records or rewrite missing
  observations as measured zeros**.
- A registry supplies explicit identity mappings from an instrumented
  model/adapter. Creating a registry does **not** install hooks or infer neuron
  identities. GRAPHyco's existing FX and module observers commonly represent
  operations/modules, not individual neurons. Granularity must be declared by
  the caller.

## Policy axes

`AnalysisPolicy(graph_scope="full"|"active"|"executed",
gradient_policy="observed_only"|"legacy_zero_fill")`

`full` covers the union of GraphState nodes, observer records, and registered
identities. `active` uses declared lifecycle (GraphState.node.active only when
no explicit registry identity exists). `executed` uses positive execution
declarations or monitored activation, never inference from absence.

`observed_only` excludes missing, invalid, explicitly inactive, and explicitly
non-executed nodes from **gradient** calculations. A measured zero is included.
Pairwise attenuation requires valid gradients at both endpoints; alignment
requires a valid activation and gradient; topology/gradient correlation requires
valid gradients plus a structural profile. Missing temporal samples do not
become zeros, and never-observed temporal nodes remain visible as
`classification="not_observed"`, with null gradient summary fields.

`legacy_zero_fill` explicitly reproduces the original missing-as-zero
interpretation over the selected population. Its zeros are *imputed*, not
measured. This mode is necessary for historical comparison; it is **not
recommended for new scientific inferences** about vanishing gradients.

**Compatibility:** `evaluate_dynamic_flow(graph, records)` with no new
arguments is unchanged (same keys and legacy meaning). Adding
`analysis_policy=` opts into new schema fields `policy`, `observations`,
`coverage`, and `metric_populations`, without changing the legacy callable
signatures. Explicit `legacy_zero_fill` on `full` scope may include registry or
graph nodes lacking DynamicNodeRecords and therefore can differ from an old
report made from only the observed record map; compare on the same population
and note that difference.

## Example

```python
from graphyco.observer import (
    AnalysisPolicy, NodeIdentity, NodeIdentityRegistry,
    evaluate_dynamic_flow, compare_analysis_policies,
)

identities = NodeIdentityRegistry(max_events=256)
identities.register(NodeIdentity("layer0/neuron-a", "layer0", physical_index=3))
identities.transition(
    "layer0/neuron-a", lifecycle="inactive", execution="not_executed"
)
# An entirely new neuron may now occupy physical slot 3:
identities.register(NodeIdentity("layer0/neuron-b", "layer0", physical_index=3))

reports = compare_analysis_policies(
    graph_state, node_records,
    {
        "legacy": AnalysisPolicy(
            graph_scope="full", gradient_policy="legacy_zero_fill"
        ),
        "research": AnalysisPolicy(
            graph_scope="full", gradient_policy="observed_only"
        ),
    },
    identity_registry=identities,
)
print(reports["research"]["coverage"])
```

The example assumes `graph_state` and `node_records` are already available
and keyed by the same **stable identities**. An integration adapter must map FX
or module node names to these stable IDs explicitly; this stage does not
automatically remap existing observer output.

## Denominators and limitations

Policy coverage reports the number of selected identities and the subset
eligible for gradient statistics. Eligible means not explicitly inactive,
retired, or non-executed; with no explicit lifecycle evidence, a runtime-only
identity is provisionally eligible. `eligible_coverage` is
valid-gradient-count-among-eligible / eligible-count; it is `None` when the
eligible denominator is zero. `selected_coverage` is a separate statistic.
A low coverage value does **not** itself prove vanishing gradients.

The policy view reports null correlation coefficients for insufficient or
constant samples and null Gmax/mean for no valid gradient observations.
Historical metrics are intentionally not retroactively redefined.

Identity registry history is bounded by `max_events` (default 256); it is
not a complete indefinitely retained event store. Inactive/retired entries
remain registered until the user discards the registry. Underlying dynamic
model pruning/regrowth is **not** automatically instrumented here.

Temporal policy uses the **current** scope across the supplied trajectory.
For model lifecycle changes during that window, per-step registry snapshots
are future integration work; do not interpret coverage as a fully
activation-aware per-step denominator without those snapshots.

The policy evaluator is a post-processing facility. No claim is made here
that CUDA AMP, multi-rank NCCL DDP or torch.compile/Inductor are validated.
