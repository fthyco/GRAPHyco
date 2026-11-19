# GRAPHyco

PyTorch graph topology and gradient-flow telemetry.

FX mode extracts symbolic dependencies. Module fallback describes containment, not tensor dataflow. Runtime hooks record activations and gradients. Legacy structural scores are proxies, not causal importance estimates.

## Repository

- [Usage](usage/README.md) — installation, monitoring, analysis, export and CLI
- [Technical documentation](docs/DOCUMENTATION.md) — architecture, API and metrics
- [Verification](verification/README.md) — tested behavior, performance and limits
- [CUDA status](verification/CUDA.md) — unverified GPU paths and required tests

## Status

The audit changes are under review. CUDA AMP, GPU Inductor and multi-GPU NCCL are not certified.

License: MIT.
