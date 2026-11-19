# CUDA verification

**Status: NOT RUN. No GPU compatibility certification.**

## Required hardware

- One NVIDIA CUDA GPU for CUDA AMP and GPU Inductor.
- Two CUDA GPUs with NCCL for distributed training.

## Required checks

| Gate | Required observation | Status |
|---|---|---|
| CUDA FP16 AMP | Forward, loss, gradients, optimizer and GradScaler parity | Not run |
| CUDA stats backend | Numeric agreement, zero/missing/nonfinite telemetry | Not run |
| GPU Inductor | Compiled output-boundary telemetry and training parity | Not run |
| Two-rank NCCL | Cross-rank all-reduce and monitored/unmonitored parity | Not run |
| GPU resource stress | Allocated/reserved memory, sustained throughput, hook cleanup | Not run |
| CUDA correctness | Repeat across supported GPU/PyTorch configurations | Not run |

## Validation entry points

`benchmarks/gpu_hardware_gates.py` contains single-GPU and two-GPU tests.

The manual workflow `.github/workflows/physical-gpu-gates.yml` requires labeled self-hosted CUDA runners. CPU-only CI cannot satisfy these gates.

## Acceptance

Record GPU model, driver, CUDA version, PyTorch version, exact Git SHA, raw test results, and memory/latency artifacts. Mark each gate passed only after it runs on the required hardware.

**No merge under the current GPU-required release policy until the required hardware checks pass.**
