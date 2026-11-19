# Verification

Scope: tested source behavior. **Passed tests do not establish universal PyTorch or GPU compatibility.**

## Verified on CPU

| Area | Evidence | Result |
|---|---|---|
| Python 3.9–3.13 | [Package CI](https://github.com/fthyco/GRAPHyco/actions/runs/44442378429) | 141 passed, 4 skipped per version |
| GUI security | Same CI | 3 passed |
| Ruff, Bandit, packaging | [Quality CI](https://github.com/fthyco/GRAPHyco/actions/runs/44442378379) | Passed |
| CPU Inductor | Quality CI | Output-boundary telemetry, training parity passed; internal fused operations not guaranteed |
| Gloo DDP | Quality CI | Two-process training and all-reduce parity passed |
| Static topology | [Stress CI](https://github.com/fthyco/GRAPHyco/actions/runs/44442378476) | 600 deterministic randomized multigraphs plus edge cases matched the legacy formulas |
| Long-run telemetry | Stress CI | 2,000 training steps, 250 observed, 16 retained |
| ResNet50 / TransformerEncoder | Stress CI | 32 / 64 CPU training steps, gradient and parameter parity passed |

CI links identify the tested repository revision and its machine-generated evidence.

## CPU performance

| Test | Measured |
|---|---:|
| Static perturbation profile, 128 nodes | 4.62 → 0.20 ms |
| Static perturbation profile, 512 nodes | 69.63 → 0.84 ms |
| Static perturbation profile, 1,024 nodes | 279.67 → 1.67 ms |
| ResNet50, 32 steps | ~3.87× monitoring overhead |
| TransformerEncoder, 64 steps | ~3.15× monitoring overhead |

Benchmarks are runner-specific and do not establish production latency limits. Static profile speedup is **not** whole-training speedup.

[Detailed stress measurements](STAGE10_PERFORMANCE_STRESS.md)

## Unverified

- [CUDA AMP, GPU Inductor and multi-GPU NCCL](CUDA.md)
- GPU memory growth and sustained GPU throughput
- Arbitrarily large or dense graph scaling; production performance thresholds
- Full per-operation telemetry inside Inductor-fused kernels
- Automatic stable neuron identity through pruning/regrowth; per-step temporal lifecycle snapshots

## Audit records

- [Graph fidelity](GRAPH_FIDELITY.md)
- [Gradient monitoring](GRADIENT_MONITORING_VERIFICATION.md)
- [Telemetry integrity](TELEMETRY_INTEGRITY_REVIEW.md)
- [Security and distribution](SECURITY_PERFORMANCE_DISTRIBUTION.md)
- [Reliability audit](RELIABILITY_AUDIT_2026-11-18.md)
- [Release integration gates](RELEASE_INTEGRATION_GATE_2026-11-18.md)
- [Historical empirical results — not revalidated](HISTORICAL_EMPIRICAL_RESULTS.md)
