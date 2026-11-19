# Visualization and export

## Export and query

```python
import torch
from torch import nn
from graphyco import extract_benchmark_json, QueryEngine

model = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 10))
inputs = torch.randn(8, 64)
extract_benchmark_json(
    model, inputs=inputs, steps=10,
    arch_name="Example", export_path="benchmark.json",
)
engine = QueryEngine("benchmark.json")
print(engine.describe())
print(engine.get_bottlenecks())
```

## Desktop GUI

```bash
pip install "graphyco[gui]"
graphyco --json benchmark.json
```

## CLI

```bash
graphyco --model resnet --steps 1 --query summary
graphyco --json benchmark.json --query bottlenecks
graphyco --model resnet --steps 10 --export benchmark.json
```

Canonical `--model` presets are **toy graph fixtures**, not full pretrained reference architectures. Real ResNet50 tests live in the verification suite.

[Graph fidelity](../verification/GRAPH_FIDELITY.md)
