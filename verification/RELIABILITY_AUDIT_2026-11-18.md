# GRAPHyco reliability audit — 2026-11-18

**Baseline:** `main` at `e76f3785a53dc18d615adc5c80bd23d5843fcff8`.  
**Working branch:** `audit/packaging-foundation`.  
**Method:** source-level inspection of the complete repository tree and the core, bridge, observer, visualizer, packaging, and docs modules. No independent benchmark results are asserted by this audit.

## Observations and priorities

| Priority | Finding | Primary evidence | Planned gate |
| --- | --- | --- | --- |
| P0 | `core`, `observer`, and `bridge` imported as unintended top-level packages; `torch_bridge.py` mutates `sys.path` | `bridge/torch_bridge.py`, `bridge/grad_bridge.py`, `observer/*.py`, `visualizer/query.py`, `visualizer/diagnostics.py` | Installed-package import and Python type identity |
| P0 | `scipy.stats` imported by runtime modules but SciPy not declared | `pyproject.toml`, `observer/grad_eval.py`, `visualizer/query.py` | Fresh core install |
| P0 | PySide6 declared optional but visualizer package eagerly imports GUI | `visualizer/__init__.py`, `gui.py`, top-level `__init__.py` | Import core and query without Qt |
| P1 | No published pytest suite or GitHub Actions on baseline | Git tree and Actions run list on 2026-11-18 | Automated smoke and reproducibility checks |
| P1 | Canonical example loader imports absent `validate_framework` and changes `sys.path` | `visualizer/query.py:get_canonical_models` | Packaged canonical models; CLI integration |
| P1 | Module-inspection fallback creates sibling links based on registration order, not observed tensor dependency | `bridge/torch_bridge.py:module_to_graph` | Separate hierarchy and executed dataflow modes |
| P1 | `evaluate_model` catches any trace exception and silently reports a different graph type | `bridge/torch_bridge.py:evaluate_model` | Explicit provenance and fallback diagnostics |
| P1 | `topological_bottleneck_ratio` is degree concentration, not betweenness centrality claimed by the query display | `core/evaluation.py`, `visualizer/query.py:describe` | Mathematical oracle tests and corrected names |
| P1 | `connectivity_coherence` measures degree-density uniformity, not conservation of gradients | Same | Mathematical oracle tests and corrected names |
| P1 | `structural_perturbation_resilience` is a degree-based cost proxy, not reachability loss under node deletion | `core/evaluation.py`, `visualizer/query.py:compute_ablation_map` | Distinct metrics with validated definitions |
| P1 | Live monitor temporarily overwrites `model.forward`; telemetry keeps snapshots for every monitored step | `bridge/grad_bridge.py:LiveTrainingMonitor` | Behavioral parity, exception safety, bounded resource use |
| P1 | Edge and node observers clone/copy large tensors and `compute_tensor_stats` copies to CPU float64 | `observer/grad_observer.py` | Runtime/memory overhead measurements |
| P2 | GUI sets Qt sandbox-disabling environment variables when imported | `visualizer/gui.py` | Secure GUI defaults |
| P2 | Export embeds JSON text in HTML script without escaping `</script>` sequences | `visualizer/renderer.py` | Adversarial HTML-output test |
| P2 | Static resources relied on packaging discovery instead of explicit inclusion | `pyproject.toml`, `visualizer/static/` | Verify packaged wheel contents |

## Stages and acceptance criteria

### Stage 1 — Packaging foundation (this branch)
- Normalize internal imports; eliminate `sys.path` modifications from `torch_bridge.py`.
- Declare SciPy in core requirements; keep Qt optional via deferred GUI loading.
- Make non-GUI CLI commands import without Qt.
- Explicitly package static viewer resources.
- Add a regression suite and CI for clean environment import, graph-state class identity, a minimal dynamic step, and wheel inspection.

**Exit:** all Stage 1 checks pass for Python 3.11 and 3.12 on GitHub Actions; `main` unchanged until a reviewed merge.

### Stage 2 — Mathematical correctness
- Establish independent reference computations on hand-checkable DAGs.
- Rename or re-implement mislabeled metrics without silently altering historical serialized values.
- Define precisely whether metrics are structural proxies, causal interventions, or empirical observations.

### Stage 3 — Graph fidelity and bundled examples
- Correct the registration-order fallback and report graph provenance.
- Restore standalone canonical examples, including functioning `--model` CLI commands.
- Validate FX traces and observed runtime dependencies for residual/branching and failure cases.

### Stage 4 — Gradient monitoring
- Verify forward, loss, gradient and optimizer parity with monitoring on/off.
- Reduce tensor copies and bound trajectory memory.
- Add checks for repeated modules, tuple outputs, hooks, exception handling, and CPU/GPU compatibility.

### Stage 5 — Visualizer, security and distribution
- Remove unsafe default sandbox environment flags.
- Escape embedded JSON correctly, test untrusted labels/data.
- Verify wheel/SDist, desktop optional extras, query path, and offline HTML export.

### Stage 6 — Evidence and release gate
- Independent benchmark suite across at least MLP, residual CNN, branching graphs and attention.
- Measure latency, CPU/GPU memory, gradient agreement and fidelity.
- Produce reproducible versioned artifacts and release notes.

## Scope control
Stage 1 intentionally does not revise mathematical formulas, gradient diagnostic thresholds, graph fallback semantics, benchmark data or results. Later stages must explicitly document any public metric or API incompatibility.
