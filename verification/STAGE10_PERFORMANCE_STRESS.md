# Stage 10 — Performance optimization and stress validation

Status: merged into `main` through [PR #10](https://github.com/fthyco/GRAPHyco/pull/10) as part of the 0.2.0 source tree. This stage addresses CPU graph-metric scalability and longer runtime telemetry tests; it is **not** GPU certification.

## 1. Root cause and mathematical preservation

The legacy static `perturbation_profile()` computed local density separately
for every node and every undirected neighbor by repeatedly scanning the entire
edge list. `fan_balance()` also scanned all edges twice per node, while
`evaluate()` recomputed a full profile separately for resilience and for its
own return dictionary. In sparse graphs with E proportional to V, the
repeated work grows approximately quadratically in V.

`src/graphyco/core/evaluation.py` now obtains one transient indexed snapshot
of incident count, incoming/outgoing multiplicity, and undirected neighbor
identity sets. It computes integer densities once, reuses the profile once
inside `evaluate()`, and preserves the historical `SCALE=10000` fixed-point
rounding and `checked_add/checked_mul` arithmetic. This is a speed
optimization, **not** a claim that legacy topology proxies are true graph
betweenness or ablation sensitivity.

Important contracts preserved:
- Parallel directed edges contribute repeatedly to counts.
- A self-loop counts only once toward local incident density, but once each
  toward fan-in and fan-out.
- Edges pointing to undeclared nodes do not implicitly register those nodes.
- Profile secondary contributions are over unique undirected neighbors,
  with the old integer truncation and overflow behavior.
- `GraphState` is never changed or globally cached. Subsequent graph
  mutation is reflected immediately.

`tests/test_topology_fast_oracle.py` asserts exact integer and dictionary
equality against independent old-style edge scans on **600 deterministic
random multigraphs** plus adversarial empty, parallel, self-loop, unknown-node
and mutation cases. Existing Stage 2 mathematical oracle tests still run.

## 2. Measured optimized CPU performance

Same CI runner / same samples for original-vs-indexed
`perturbation_profile`:
 
| Nodes | Original profile (ms) | Indexed profile (ms) | Relative speedup |
|---:|---:|---:|---:|
| 128 | 4.62 | 0.20 | 22.76x |
| 512 | 69.63 | 0.84 | 82.95x |
| 1024 | 279.67 | 1.67 | 167.23x |

Numbers from [profiling workflow 44439347342](https://github.com/fthyco/GRAPHyco/actions/runs/44439347342);
the full cProfile and raw repeated measurements are in the `graph-topology-profiler`
CI artifact. Those speedups describe **only the static profile calculation**.
They do not imply speedups of this magnitude for forward/backward hooks or
complete model training.

Independent optimized full static `evaluate()+perturbation_profile()`
benchmark from [44438921225](https://github.com/fthyco/GRAPHyco/actions/runs/44438921225):
- 32 nodes / 61 edges: 0.146 ms.
- 128 nodes / 253 edges: 0.564 ms.
- 512 nodes / 1,021 edges: 2.070 ms.
- 1,024 nodes / 2,045 edges: 4.247 ms.

Those are sparse DAG fixtures, not dense-graph or GPU scaling claims.
Microbenchmarks are noisy across runners. No production throughput SLA is
established by these measurements.

## 3. Long-run monitoring and real model families

[Dedicated stress CI 44439347342](https://github.com/fthyco/GRAPHyco/actions/runs/44439347342)
exercises actual monitored forward/backward and checks optimization parity
against an independently deep-copied unmonitored model. Gradient tensors,
losses, outputs, model parameter values and registered buffers must match.

| Test | Shape / steps | Observed result |
|---|---|---|
| Long-run MLP | 2,000 training steps; log every 8; bounded history=16 | 250 captured observations, final 16 snapshots, RSS growth from step 500 about +0.055 MiB, tracemalloc Python heap growth about +0.012 MiB |
| torchvision ResNet50 | Batch 2, 3x64x64, 32 steps | Training parity passed; monitor overhead around 3.87x; RSS growth from step 8 to 32 about +35.14 MiB |
| PyTorch TransformerEncoder | 8 layers, width 128, batch 2, 24 tokens, 64 steps | Training parity passed; monitor overhead around 3.15x; RSS growth from step 16 to 64 about +0.85 MiB |

For both real large-model cases, a model-level `LiveTrainingMonitor`
uses `mode="module"`, keeps no more than three telemetry snapshots,
requires recorded gradients, and checks for leaked forward/backward hooks
after closing. Memory thresholds are intentionally conservative fail-fast
checks; the tests **do not prove absence of leaks over arbitrary durations**.
ResNet50's initial process RSS also includes the model, optimizer state,
gradient buffers and allocator warmup; a quarter-to-final comparison is more
informative than start-to-end RSS for this purpose.

The earlier PR #9 CPU matrix, GUI security, package integrity, actual
2-process Gloo allreduce and CPU Inductor output-boundary validation
remain mandatory. Separate physical GPU AMP/Inductor/NCCL tests remain
**UNEXECUTED** without a suitable runner.

## 4. Unresolved performance engineering

- Module/FX observation overhead for *small* networks remains substantial.
  Optimizing static graph topology does not remove activation cloning,
  observer hooks, tensor-stat reductions, or CPU/GPU synchronization.
- The dynamic metric stage needs component-level flamegraphs and a selective
  instrumentation strategy before setting a production latency budget.
- The GPU memory allocator, non-CPU autograd, multi-rank NCCL and GPU Inductor
  require independent hardware tests; CPU-only CI is no substitute.
- Graph-scale benchmarking should eventually include dense, cyclic and
  multi-million-edge graphs and more varied tensor models. Current synthetic
  scaling tests are primarily sparse directed graphs.
- Larger models were monitored for dozens, not thousands, of steps. The
  2,000-step plateau test uses a smaller MLP; no extrapolation is claimed.

Production GPU certification and performance acceptance remain outstanding; this source merge does not establish them.
