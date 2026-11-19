# GRAPHyco

PyTorch graph topology and gradient-flow telemetry.

FX mode extracts symbolic dependencies. Module fallback describes containment, not tensor dataflow. Runtime hooks record activations and gradients. Legacy structural scores are proxies, not causal importance estimates.

## Repository

- [Usage](usage/README.md) — installation, monitoring, analysis, export and CLI
- [Technical documentation](docs/DOCUMENTATION.md) — architecture, API and metrics
- [Verification](verification/README.md) — tested behavior, performance and limits
- [CUDA status](verification/CUDA.md) — unverified GPU paths and required tests
- [Changelog](CHANGELOG.md) — release notes and migration

## Status

Version 0.2.0 includes CPU-validated audit changes. CUDA AMP, GPU Inductor, multi-GPU NCCL and production GPU performance remain unverified.

License: MIT.
