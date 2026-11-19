# Static analysis

```python
import torch
from torch import nn
from graphyco import evaluate_model, validate_invariants, SCALE

model = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 10))
report = evaluate_model(model, mode="trace", strict_trace=True)
validate_invariants(report["graph_state"])

metrics = report["evaluation"]
print(report["graph_provenance"])
print(metrics["node_count"], metrics["edge_count"])
print(metrics["topological_bottleneck_ratio"] / SCALE)
```

`mode="trace"` extracts symbolic FX dependencies. `mode="module"` extracts module hierarchy; those edges are **not** tensor dependencies. With `strict_trace=True`, tracing errors raise instead of using fallback.

## Exact structural references

```python
from graphyco.core import directed_node_betweenness, pairwise_reachability_ablation_loss

graph = report["graph_state"]
betweenness = directed_node_betweenness(graph)
reachability_loss = pairwise_reachability_ablation_loss(graph)
```

These scores describe graph structure, not measured model accuracy or neuron importance.

[Metric contracts](../docs/METRIC_CONTRACTS.md)
