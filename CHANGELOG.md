# Changelog

## 0.2.0 — 2026-11-19

Major audit and compatibility release. CPU-tested changes from audit stages 1–10.

### Changed

- Distinguish FX tensor-dataflow from module containment; avoid fabricated sibling dependencies.
- Add explicit identity and observation policies, preserving inactive identities without reporting their gradients as observed.
- Introduce mathematical and graph-fidelity regression oracles and stronger gradient/telemetry lifecycle checks.
- Bound monitoring history by default (128 snapshots); pass `max_history_steps=None` for unbounded history.
- Reuse transient indexed topology accounting in `evaluate()` and `perturbation_profile()`, preserving historical fixed-point integer outputs.
- Harden JSON renderer, desktop GUI sandbox behavior, CLI packaging, and supplied visualizer assets.
- Remove tracked tutorials notebooks; usage docs live under `usage/`, technical docs under `docs/`, test evidence and limitations under `verification/`.

### Compatibility / migration

- Module-mode graph topology represents containment; results from older sibling-order fallback are not directly comparable.
- `DynamicExecutionBridge.step` without an optimizer preserves the caller's prior parameter gradients.
- Historic unbounded monitoring trajectories now require `max_history_steps=None`.
- Renderer input validation is stricter; nonfinite telemetry is serialized as JSON null.
- Qt GUI no longer silently disables Chromium's sandbox; deployments must satisfy sandbox requirements.

### Verified

- Python 3.9–3.13: 141 passed and 4 skipped in each CPU core test matrix.
- Qt security: 3 passed. CPU Inductor, two-process Gloo DDP, packaging, code-quality gates and static topology oracle passed.
- 2,000-step bounded-history monitoring, 32-step ResNet50 and 64-step TransformerEncoder CPU test runs passed.
- Final-head evidence: [Package CI](https://github.com/fthyco/GRAPHyco/actions/runs/44442930548), [Quality / CPU](https://github.com/fthyco/GRAPHyco/actions/runs/44442930603), [Profiling / stress](https://github.com/fthyco/GRAPHyco/actions/runs/44442930988).

### Not verified / not claimed

- CUDA AMP, GPU Inductor, multi-GPU NCCL, GPU resource/throughput compatibility and production latency targets have **not** been certified.
- Static topology microbenchmark speedups are not training speedups. See [Verification](verification/README.md).
