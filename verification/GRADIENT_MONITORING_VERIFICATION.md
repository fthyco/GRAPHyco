# GRAPHyco — Stage 4 gradient monitoring verification

**Scope:** independent CPU training equivalence on explicitly tested model
families, monitoring lifecycle safety, bounded telemetry, and an opt-in
performance probe. This is **not** a claim of universal non-interference.

## Observed problems in the Stage 3 implementation

1. `DynamicExecutionBridge.step(..., optimizer=None)` ran
   `loss.backward()` without restoring existing `parameter.grad`.
   The supposedly diagnostic step accumulated into the user's gradients.
2. `LiveTrainingMonitor.observe()` temporarily assigned
   `model.forward = lambda ...`. It restored a bound method as an
   *instance attribute*, even if no such attribute originally existed;
   exception paths still committed partial telemetry snapshots.
3. `ModuleGradObserver` installed `register_full_backward_hook` on
   all leaf modules, including inactive log-interval steps. This has
   PyTorch view semantics and emitted warnings for inputs without gradients.
4. `FXGradObserver` overwrote a tuple node's first output gradient with
   the last output gradient to finish its backward callback; repeated
   source->target arguments also overwrote edge gradient contributions.
5. Both live and bridge trajectories retained all historical snapshots
   with no maximum. One-shot helpers could leave module hooks attached.
6. `LiveTrainingDiagnostics.observe_step` recorded failed steps, and
   used full module backward hooks with the same view/warning concerns.

## Changes implemented

### Independent forward/backward and optimizer parity

`tests/test_gradient_monitoring.py` pairs models with identical initial
weights, identical inputs, loss functions and optimizers:

- Three-step SGD and Adam runs in both `mode="trace"` and
  `mode="module"`: compare forward outputs, losses, per-parameter
  gradients, final parameters and optimizer state tensors.
- One-step `DynamicExecutionBridge.step(..., optimizer=...)`
  compared against a manually run, equivalent training step.
- Dropout training compares seeded RNG evolution, outputs, gradients and
  updated parameters for both monitoring modes.
- All numeric comparisons use explicit small tolerances, not rounded
  printed examples.

This establishes parity for the tested small CPU sequential models under
these configurations. Tests do **not** assert equivalence for `torch.compile`,
DDP, AMP, reentrant/custom autograd, model attribute side effects, in-place
operations or CUDA. Running `mode="trace"` temporarily substitutes the
original forward: for a Python-side-effectful forward, FX behavior may
differ even if tensor arithmetic in a conventional model matches.

### Parameter gradients

For `DynamicExecutionBridge.step(..., optimizer=None)`, pre-existing
parameter `.grad` tensors (including `None`) are snapshotted before
the backward pass and restored even if metric collection raises.
Recorded diagnostic parameter-gradient metrics are computed **before**
the restore.

This deliberately changes earlier behavior: callers must no longer rely
on `bridge.step` without an optimizer to accumulate gradients for their
training loop. The call still performs forward/backward, may advance RNG,
and may mutate module buffers if the forward is stateful (BatchNorm,
mutable Python attributes, etc.). It is **not** a read-only model probe.

When an optimizer is passed, the bridge remains an explicitly mutating
training step: `optimizer.zero_grad()`, `loss.backward()`,
`optimizer.step()`.

### Hook and context lifecycle

- Module mode now uses **forward hooks + output tensor gradient hooks**.
  It does not install `register_full_backward_hook`, avoiding the
  module input/output view wrappers and the corresponding warning.
- Inactive `log_interval` steps skip module activation/gradient copying.
- Live monitor re-entry (nested `observe`) is rejected.
- The original `model.forward` instance attribute is restored exactly:
  if absent before observing, it is absent after observing.
- A failed caller training block does not produce a completed
  `DynamicGraphState`; cleanup executes on exception paths.
- Both `DynamicExecutionBridge` and `LiveTrainingMonitor` have
  idempotent `close()` methods. The one-shot `monitor_model`,
  `evaluate_model_dynamic` and internally created benchmark-export
  bridges release their module hooks in `finally` blocks.
- The high-level `LiveTrainingDiagnostics` uses output tensor hooks
  and refuses to commit telemetry for failed steps.

**Important:** A shared `nn.Module` called multiple times still keeps
only the most recent invocation's public per-module summary. It does
**not** yet provide invocation-indexed output statistics. Multi-output
module/FX nodes report the first tensor's summary in existing public
fields; FX additionally retains component-specific raw output
gradients in `raw_output_component_grads`. Do not interpret these as
complete per-output coverage.

Repeated identical FX edge uses (such as `torch.add(x, x)`) now
aggregate their intercepted gradient contributions instead of silently
keeping the last one. The FX graph still deduplicates that symbolic
source-target edge.

### Bounded snapshots

`DynamicExecutionBridge` and `LiveTrainingMonitor` now accept:

`max_history_steps: Optional[int] = 128`

Only the most recent 128 full metric snapshots are retained by default;
`None` explicitly opts into the old unbounded behavior. Values less
than 1 are rejected. Temporal CV uses the retained window. A bounded
`trajectory` is consequently a **recent window**, not a record of the
full training run; `export_live_json` exports only that window.
Callers needing a complete run should persist snapshots incrementally.

This bounds full historical snapshots but **does not eliminate** the
per-step detached activation/gradient copies or GPU synchronization from
`compute_tensor_stats`. A true low-overhead streaming estimator is future
work and must be benchmarked independently.

## Performance probe

Use the separately seeded, isolated-process CPU script:

```bash
python benchmarks/monitor_overhead.py --steps 30 --warmup 5 --width 64 \
    --json /tmp/graphyco-overhead.json
```

The script measures per-step training wall time for `plain`,
`trace` and `module` with identical workloads, and reports the
**process-level peak RSS**. Initial FX tracing is excluded from timed
steps. Peak RSS includes startup and PyTorch allocations, so the
differences do not represent an accurate per-layer memory breakdown.
There is **no universal performance budget asserted** and no
hard-coded timing threshold in CI, due to runner noise.

Run on the target accelerator/environment to collect meaningful
latency/memory numbers. The included script is CPU-only; its results
must not be extrapolated to GPU.

## Acceptance and remaining gaps

`tests/test_gradient_monitoring.py` covers the parity and cleanup
behaviors above; GitHub CI runs on Python 3.11 and 3.12 CPU.
An additional short performance-probe smoke runs on Python 3.11,
without pass/fail assertions about speed.

Stage 4 does **not** certify arbitrary model training fidelity.
Further release gates should explicitly test GPU/AMP, forward
mutation, shared/repeated modules, in-place tensor operations,
exceptions inside hooks, optimizer gradient accumulation,
long-running memory, `torch.compile` and DDP. The provenance
contract from Stage 3 remains essential.
