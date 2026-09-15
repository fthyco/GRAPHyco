# Graphyco

Computational graph topology and dynamic gradient-flow monitoring for PyTorch.

---

## What is Graphyco

Graphyco extracts a formal computational graph from any PyTorch model and monitors gradient flow during training.

- **Static topology** — Graph density, connectivity coherence, bottleneck ratio, perturbation resilience, and 7 structural invariants in deterministic fixed-point arithmetic.
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

---

## How to Use It

### 1. One-Line Profiling

```python
import torch.nn as nn
from graphyco import visualize

model = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 10))
visualize(model, inputs=torch.randn(8, 64), steps=5, port=8000)
```

### 2. Static Topology Evaluation

```python
from graphyco import evaluate_model, validate_invariants

result = evaluate_model(model, mode="trace")  # or mode="module"

print(f"Nodes: {result['num_nodes']}, Edges: {result['num_edges']}")
print(f"Bottleneck Ratio: {result['topological_bottleneck_ratio']}")
print(f"Resilience: {result['structural_perturbation_resilience']}")

validate_invariants(result['graph_state'])
```

### 3. Live Training Monitoring

```python
from graphyco import LiveTrainingMonitor

monitor = LiveTrainingMonitor(model, log_interval=5)

for step in range(100):
    with monitor.observe(step):
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()

    if monitor.has_new_data():
        info = monitor.get_latest_bottleneck()
        print(f"Step {step}: choke={info['bottleneck_node']} G_max={info['max_concentration']:.2f}")
```

### 4. Health Diagnostics

```python
from graphyco import LiveTrainingDiagnostics

diag = LiveTrainingDiagnostics(model, inputs=torch.randn(16, 64))

for step in range(50):
    with diag.observe_step(step):
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()

health = diag.get_health_snapshot()
print(health.status)
print(health.bottleneck)
print(health.layer_telemetry)
```

### 5. Export & Query

```python
from graphyco import extract_benchmark_json, QueryEngine

extract_benchmark_json(model, inputs=x, steps=10, arch_name="MyModel", export_path="benchmark.json")

engine = QueryEngine("benchmark.json")
engine.describe()                     # single-model summary
engine.describe(siblings=True)        # cross-architecture comparison
engine.get_bottlenecks()              # ranked bottleneck nodes
engine.get_neighbors("layer_3")       # DAG predecessors, successors, siblings
```

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
| [Benchmark Export & Querying](notebooks/tutorials/query_engine_and_visualization.ipynb) | FX tracing vs module fallback, `extract_benchmark_json`, `QueryEngine` summaries, cross-architecture comparison, DAG queries. |
| [Live GUI Visualization](notebooks/tutorials/live_gui_visualization.ipynb) | PySide6 desktop GUI, real-time training visualization, `%gui qt` integration, CLI usage. |

---

## Verification

```bash
pytest
python validation/validate_framework.py
python validation/validate_dynamic_flow.py
```

---

## License

MIT
