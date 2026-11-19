# Training monitoring

## Live gradient telemetry

```python
import torch
from torch import nn
from graphyco import LiveTrainingMonitor

model = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 10))
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
monitor = LiveTrainingMonitor(
    model, mode="module", log_interval=4, max_history_steps=16
)

try:
    for step in range(100):
        x = torch.randn(8, 64)
        y = torch.randint(0, 10, (8,))
        with monitor.observe(step):
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(x), y)
            loss.backward()
            optimizer.step()
        if monitor.has_new_data():
            print(monitor.get_latest_bottleneck())
finally:
    monitor.close()
```

`mode="module"` runs the model's original forward. `mode="trace"` instruments FX execution and may not preserve Python forward side effects. `log_interval` reduces observation frequency; `max_history_steps` limits retained snapshots. Monitoring adds compute and memory overhead.

## Diagnostic summary

```python
from graphyco import LiveTrainingDiagnostics

diagnostics = LiveTrainingDiagnostics(model)
try:
    with diagnostics.observe_step(0):
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.cross_entropy(model(x), y)
        loss.backward()
        optimizer.step()
    print(diagnostics.get_snapshot())
finally:
    diagnostics.close()
```

## Missing-gradient policies

```python
from graphyco.observer import AnalysisPolicy

policy = AnalysisPolicy(
    graph_scope="full",
    gradient_policy="observed_only",
)
monitor = LiveTrainingMonitor(
    model, mode="module", analysis_policy=policy
)
```

`observed_only` excludes missing/invalid gradients from numerical gradient metrics; recorded zeros remain valid. `legacy_zero_fill` reproduces historical zero-imputation and is intended for compatibility.

For explicit neuron lifecycle tracking, supply a `NodeIdentityRegistry` with model-declared stable IDs. A physical index alone is not a stable identity across pruning and regrowth. Module/FX identifiers are not automatically neuron IDs.

[Identity and policy contract](../docs/IDENTITY_OBSERVATION_POLICIES.md)
