# Graph fidelity and provenance — Stage 3

This stage separates **symbolic operation dependencies** from **registered
module hierarchy**. A source-derived FX dependency graph is not equivalent to
observed runtime execution, and neither is equivalent to module containment.

## Extraction contracts

### 1. FX dependency graph

`trace_to_graph(model)` invokes `torch.fx.symbolic_trace(model)` once.
Each FX node becomes a graph node. For each `node.all_input_nodes` dependency
`source -> node`, exactly one directed edge is emitted, including references
passed in keyword arguments and repeated uses of the same input.

- `edge_type="operational"` means it appeared in positional args.
- `edge_type="informational"` means the dependency appeared only in kwargs.
  **Both are real symbolic dependencies**, and both have `critical=True`.
- `in_ports` and `out_ports` include both positional and keyword links.
- The provenance object has `graph_kind="fx_symbolic_trace"`,
  `edge_semantics="fx_dependencies"`, `trace_status="success"`,
  `execution_verified=False`, `dataflow_verified="symbolic_only"`.

This is source-structure evidence. FX may specialize Python control flow,
ignore untraceable data-dependent branches, or treat external operators as
atomic. It does not prove an actual tensor execution trace.

### 2. Module containment graph

`module_to_graph(model, include_containers=True)` enumerates
`model.named_modules()`, including the root and containers. Each actual
registered parent-child relationship becomes a `contains` edge with
`critical=False`.

**It does not infer sequential dataflow from the registration order.**
A branching model with `left` and `right` modules has containment edges
`root -> left`, `root -> right`, never `left -> right` without real
dataflow evidence. `include_containers=False` creates a leaf inventory
with no speculative edges.

- `graph_kind="module_hierarchy"` or `"module_leaf_inventory"`.
- `edge_semantics="containment_only"` or `"no_edges"`.
- `execution_verified=False`, `dataflow_verified=False`.

**Compatibility change:** `module_to_graph()` defaults to
`include_containers=True` rather than `False`. Its nodes and edges
(and thus its topology metric values) change intentionally. Historical FX
metric calculations are not modified, but new hierarchy metrics describe
containment only and must **not** be compared with FX operation metrics as
if the same graph semantics were used.

### 3. Fallback and failure reporting

`evaluate_model(model, mode="trace")` attempts FX by default and,
if tracing fails, returns a **labeled containment graph**, with:

- `trace_status="failed"`
- `fallback_from="fx_symbolic_trace"`
- `fallback_reason_type` and a truncated diagnostic string

Call `evaluate_model(..., strict_trace=True)` to propagate the FX
exception instead of accepting fallback. An invalid `mode` raises
`ValueError`, not silent module inspection.

`DynamicExecutionBridge` and `LiveTrainingMonitor` use the same
provenance contract. Each performs **one** FX trace and uses the resulting
GraphModule both to extract the static graph and initialize the interpreter.
This eliminates possible differences from tracing twice. Their
`strict_trace=True` option propagates trace failures.

The graph `provenance` is included in `GraphState.to_dict()` only if
nonempty; manually constructed legacy GraphState objects remain schema-
compatible. `evaluate_model` additionally returns a top-level
`graph_provenance` for convenience. Dynamic graph snapshots include it
in `static_graph.provenance`.

If a graph has `edge_semantics="containment_only"`, no exported
visualization or metric may claim its edges represent tensor dependencies.

## Canonical benchmark fixtures

The five self-contained bundled examples live in
`graphyco.visualizer.canonical_models`:

| Canonical name | Graph motif | Input shape |
| --- | --- | --- |
| Sequential MLP | 4-layer vector MLP with ReLUs | `(8,64)` |
| ResNet Block | Vector residual add; **not** torchvision ResNet | `(8,64)` |
| DenseBlock | Dense concatenate paths; **not** DenseNet | `(8,32)` |
| U-Net Toy | Vector encoder/decoder skip; **not** spatial U-Net | `(8,32)` |
| Self-Attention Toy | Multihead self-attention + residual norm | `(8,4,64)` |

Initial model weights and inputs use deterministic local RNG context
without advancing the caller's global CPU random state. The examples
require no `validate_framework.py`, no repository checkout, and no
`sys.path` modification.

Both Python `extract_benchmark_json("all", steps=1)` and CLI
`python -m graphyco.visualizer --model resnet --query summary --steps 1`
are covered in CI without installing PySide6.

## Acceptance evidence

See `tests/test_graph_fidelity.py`:

1. Exact FX edge parity with PyTorch's independent `all_input_nodes`
   oracle on residual branches and keyword dependencies.
2. Node port integrity, no duplicate dependency edges, invariant checks.
3. Containment-only fallback; no sibling registration-order inference.
4. Nested containment and sanitized-name collision handling.
5. Explicit failure provenance and strict trace behavior on a dynamic
   data-dependent branch.
6. Single FX trace per DynamicExecutionBridge/LiveTrainingMonitor.
7. All five bundled toy models: forward finite, FX trace, correct output
   shapes, invariant validation, deterministic weights and inputs.
8. End-to-end JSON export and CLI query without Qt.

These tests establish agreement with **symbolic FX dependencies** on
covered motifs. They do not prove execution-instance fidelity for
dynamic PyTorch models, or training/gradient equivalence, which require
separate independent runtime instrumentation (Stage 4).
