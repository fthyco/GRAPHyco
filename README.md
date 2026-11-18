# Graphyco

Computational graph topology and dynamic gradient-flow monitoring for PyTorch.

---

## What is Graphyco

Graphyco analyzes PyTorch models with FX symbolic dependencies when tracing succeeds, or with an explicitly labeled module-hierarchy fallback when tracing fails. The latter does not claim actual tensor dataflow. Gradient monitoring is available in the runtime bridge.

- **Static topology** — Directed edge-count density, degree-distribution uniformity, degree-concentration and structural-cost proxies, plus 7 structural invariants in deterministic fixed-point arithmetic.
- **Dynamic gradient-flow** — Per-node activation/gradient RMS, edge attenuation, bottleneck concentration ($G_{\max}$), forward-backward alignment, temporal stability.
- **Live diagnostics** — Context managers for training loops with automated vanishing/exploding gradient and dead neuron alerts.
- **Export & query** — Serialize telemetry to JSON, compare architectures via `QueryEngine` or REST API.
- **Desktop GUI** — PySide6 real-time visualization.

### Install

```bash
pip install graphyco

# with desktop GUI
pip install graphyco[gui]
```

> **Full technical documentation**: [docs/DOCUMENTATION.md](docs/DOCUMENTATION.md)  
> **Metric meanings and mathematical validation**: [docs/METRIC_CONTRACTS.md](docs/METRIC_CONTRACTS.md). Existing 'bottleneck' and 'resilience' summary values are heuristic structural/gradient indicators, not causal neural-network bottleneck or measured ablation effects.  
> **Graph fidelity, hierarchy fallback, and provenance**: [docs/GRAPH_FIDELITY.md](docs/GRAPH_FIDELITY.md).  
> **Gradient monitoring verification and limitations**: [docs/GRADIENT_MONITORING_VERIFICATION.md](docs/GRADIENT_MONITORING_VERIFICATION.md).

---

## How to Use It

### 1. One-Line Profiling & Visualization

```python
import torch
import torch.nn as nn
from graphyco import visualize

model = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 10))

# Launch desktop GUI visualizer
visualize(model, inputs=torch.randn(8, 64), steps=5)

# Or in Jupyter notebooks:
from graphyco.visualizer.gui import visualize_inline
visualize_inline(model, inputs=torch.randn(8, 64), steps=5)
```

### 2. Static Topology Evaluation

```python
import torch
import torch.nn as nn
from graphyco import evaluate_model, validate_invariants, SCALE

model = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 10))
result = evaluate_model(model, mode="trace")  # or mode="module"

ev = result["evaluation"]
print("Graph provenance:", result["graph_provenance"])
print(f"Nodes: {ev['node_count']}, Edges: {ev['edge_count']}")
print(f"Degree-Concentration Proxy: {ev['topological_bottleneck_ratio'] / SCALE:.4f}")
print(f"Structural Cost Proxy: {ev['structural_perturbation_resilience'] / SCALE:.4f}")

# Verify 7 structural invariants on module graph
module_result = evaluate_model(model, mode="module")
validate_invariants(module_result["graph_state"])
```

### 2a. Exact Graph-Theoretic References (Optional)

The historical topology summary metrics remain unchanged. For genuinely graph-theoretic shortest-path importance and reachability ablation, use separately named opt-in functions:

```python
from graphyco.core import directed_node_betweenness, pairwise_reachability_ablation_loss

graph = result["graph_state"]
betweenness = directed_node_betweenness(graph)
ablation_loss = pairwise_reachability_ablation_loss(graph)
```

These are **structural** graph calculations only; they do not measure prediction loss or establish causal neuron importance. See [the mathematical contracts](docs/METRIC_CONTRACTS.md) for precise denominators and known limitations.

### 3. Live Training Monitoring

```python
import torch
import torch.nn as nn
from graphyco import LiveTrainingMonitor

model = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 10))
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()
monitor = LiveTrainingMonitor(model, mode="module", log_interval=5, max_history_steps=128)

for step in range(100):
    x = torch.randn(8, 64)
    y = torch.randint(0, 10, (8,))
    with monitor.observe(step):
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()

    if monitor.has_new_data():
        info = monitor.get_latest_bottleneck()
        print(f"Step {step}: peak_grad_node={info['bottleneck_node']} G_max={info['max_concentration']:.2f}")

monitor.close()  # release persistent module hooks
```

With `mode="module"`, the original `model.forward` executes normally and
per-module outputs are monitored. The default `mode="trace"` executes the
FX Interpreter during observed steps, which may bypass custom Python forward
side effects. Both modes were checked on paired CPU training tests, **not**
proven equivalent for arbitrary models. Historical telemetry is bounded to
the most recent 128 steps unless `max_history_steps=None` is specified.
Always call `close()` in a `finally` block in long-running applications.

### 4. Health Diagnostics

```python
import torch
import torch.nn as nn
from graphyco import LiveTrainingDiagnostics

model = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 10))
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()
diag = LiveTrainingDiagnostics(model)

for step in range(50):
    x = torch.randn(8, 64)
    y = torch.randint(0, 10, (8,))
    with diag.observe_step(step):
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()

snapshot = diag.get_snapshot()
print(f"Health Status    : {snapshot['health_status']}")
print(f"Active Bottleneck: {snapshot['active_bottleneck_layer']} (G_max={snapshot['gmax']:.2f})")
print(f"Avg Step Time    : {snapshot['avg_step_time_ms']:.2f} ms")
print(f"Alerts           : {snapshot['alerts']}")
diag.close()
```

### 5. Export & Query

```python
import torch
import torch.nn as nn
from graphyco import extract_benchmark_json, QueryEngine

model = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 10))
x = torch.randn(8, 64)

extract_benchmark_json(model, inputs=x, steps=10, arch_name="MyModel", export_path="benchmark.json")

engine = QueryEngine("benchmark.json")
engine.describe()                     # single-model summary DataFrame
engine.describe(siblings=True)        # cross-architecture comparison
engine.get_bottlenecks()              # ranked bottleneck nodes
engine.get_neighbors("_2")            # DAG predecessors, successors, siblings
```

### 5a. Bundled Graph-Fidelity Examples

The `--model` CLI and `get_canonical_models()` now ship with five **small toy
models**: vector MLP, vector residual block, densely connected vector block,
vector U-Net-style skip, and multi-head self-attention. They demonstrate
graph motifs; they are not full benchmark implementations or pretrained
ResNet/DenseNet/U-Net checkpoints.

```python
from graphyco.visualizer.query import get_canonical_models
from graphyco import evaluate_model

model, x = get_canonical_models()["ResNet Block"]
result = evaluate_model(model, strict_trace=True)
print(result["graph_provenance"])  # fx_symbolic_trace, symbolic_only
```

When FX fails, `evaluate_model(..., strict_trace=True)` raises instead of
silently substituting a hierarchy. Graphs of different kinds must not be
compared as equivalent topologies.

### 6. CLI

```bash
python -m graphyco.visualizer --json benchmark.json
python -m graphyco.visualizer --json benchmark.json --query bottlenecks

python -m graphyco.visualizer --model resnet --steps 10 --export resnet_benchmark.json
```

---

## Tutorials

Jupyter notebooks in [`notebooks/tutorials/`](notebooks/tutorials/):

| Notebook | Purpose |
|----------|---------|
| [Quickstart & Model Evaluation](notebooks/tutorials/quickstart_and_model_evaluation.ipynb) | Static graph extraction, topological metrics, invariant validation, perturbation analysis, scale invariance. |
| [Live Training Diagnostics](notebooks/tutorials/live_training_diagnostics.ipynb) | `LiveTrainingMonitor` and `LiveTrainingDiagnostics` in training loops, gradient bottleneck tracking, dead neuron detection, anomaly alerts. |
| [Benchmark Export & Querying](notebooks/tutorials/query_engine_and_visualization.ipynb) | Self-contained benchmark generation, `QueryEngine` summaries, cross-architecture comparison, DAG queries. |
| [Live GUI & Inline Visualization](notebooks/tutorials/live_gui_visualization.ipynb) | Interactive in-notebook visualizer (`visualize_inline`), PySide6 desktop GUI, real-time training telemetry, CLI usage. |
| [Dynamic Gradient Monitoring](notebooks/tutorials/dynamic_gradient_monitoring.ipynb) | Directed edge attenuation ($\beta, \Delta$), structural-functional quadrant classification (Types I–IV), temporal stability. |
| [Full Tutorial & JSON Export](notebooks/tutorials/full_tutorial_and_json_export.ipynb) | FX tracing rules vs module fallback, active training monitoring, JSON serialization, and offline QueryEngine exploration. |


---

## License

MIT
