# Stage 7 — Telemetry Integrity Review

Scope: targeted independent code review of the stacked GRAPHyco change set after PR #6. This is not a comprehensive independent security audit or a substitute for human reviewer sign-off.

## Confirmed defect 1 — pre-existing parameter gradients contaminated diagnostics

Affected code: DynamicExecutionBridge.step(..., optimizer=None), both FX and native module modes.

Prior implementation saved clones of existing parameter .grad tensors before backward, but **did not isolate them from the diagnostic backward**. PyTorch accumulated new gradients into the pre-existing buffers. Consequently, recorded per-parameter gradient L2/RMS included old gradient contributions even though the public diagnostic call restored the previous values afterward. Additionally, restoring a clone silently replaced the caller's original gradient tensor object.

The new implementation retains original parameter.grad objects (including None), temporarily assigns None for the diagnostic backward, collects telemetry from **only that backward**, then restores the exact original gradient objects in a finally block, including when backward fails.

The change does not make the whole diagnostic call read-only: the model forward may still affect RNG, buffers or Python-side effects. If an optimizer is supplied, the existing training behavior is preserved.

Regression tests compare identical models with initially None vs filled gradient buffers, using three starting values (0.0, +0.25, -3.5) in both modes, checking telemetry invariance and original object identity after success. Exception tests verify the same identity and value on invalid backward.

**Before fix:** run 44410984475 failed 8 tests on Python 3.11 and 3.12, including 0.302272... vs 0.211333... gradient RMS and lost tensor identity. **After fix:** run 44411035720 succeeded on the prior focused test set.

## Confirmed defect 2 — differentiable parameter snapshots retained autograd edges

Affected code: FXGradObserver.capture_parameter_gradients and ModuleGradObserver.capture_parameter_gradients.

Prior code concatenated tensors with .to(float64) without first detaching. When parameter.grad requires_grad (possible in higher-order/custom gradient workflows), the resulting raw_param_grads value retained a grad_fn and a connection to the autograd graph, contradicting observational sampling and increasing memory-lifetime risk.

New code detaches each parameter-gradient tensor before the float64 CPU copy/concatenation, without changing the numeric values or user-owned gradient buffers. The test directly supplies differentiable gradients and verifies that the captured snapshots have requires_grad=False and grad_fn=None in both observer modes.

**Before fix:** run 44411168647 failed 2 tests on Python 3.9 (and other CPU matrix variants); the test proved captured tensors still required gradients.

## Remaining review and release blockers

- FX trace instrumentation inserts identity-view autograd nodes and can differ from custom Python forwards, in-place operators, wrappers and some stateful models.
- Legacy analysis treats missing/unknown gradients as measured zero in several public metrics (documented historical metric semantics). This can create misleading vanishing-gradient interpretations; changing it requires a deliberate metric-contract version/migration, not an undocumented patch.
- Stage 6 only tests single-rank Gloo DDP and torch.compile eager backend. GPU AMP, multi-rank NCCL and compiled inductor remain unverified.
- This audit was included in the 0.2.0 source merge; its CPU CI results do not establish physical GPU compatibility or confirm PyPI publication.

Reference code: src/graphyco/bridge/grad_bridge.py, src/graphyco/observer/grad_observer.py, tests/test_telemetry_integrity.py.
