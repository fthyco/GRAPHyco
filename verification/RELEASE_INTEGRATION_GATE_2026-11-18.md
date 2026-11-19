# GRAPHyco — Release Integration Gate

Date: 2026-11-18. Scope: review stacked audit PRs #1–#5 and verify additional runtime cases before any merge or release.

## Current review/integration state

| PR | Branch | Base | Stage |
|---|---|---|---|
| #1 | audit/packaging-foundation | main | Packaging and CI |
| #2 | audit/mathematical-correctness | audit/packaging-foundation | Metric semantics and mathematical oracles |
| #3 | audit/graph-fidelity | audit/mathematical-correctness | FX graph semantics and provenance |
| #4 | audit/gradient-monitoring | audit/graph-fidelity | Training parity and hook lifecycle |
| #5 | audit/security-performance-distribution | audit/gradient-monitoring | Security, performance controls, distribution |

As inspected during this gate, all five remained open draft PRs, mergeable, with no submitted review approvals. Main had not changed from e76f3785a53dc18d615adc5c80bd23d5843fcff8. No enforced rulesets were listed; the branch-protection API could not be accessed by this connection (403), so the absence of a ruleset should not be taken as proof that all organizational review policy is satisfied.

**Do not accidentally merge a stacked PR into its predecessor branch and assume main is updated.** Upon review approval, land PR #1 into main first, then retarget PR #2 to main, verify its new base diff and CI, merge, and continue through #5 in sequence. Prefer merge commits when preservation of branch history is important; verify exact head SHA and CI before each merge. Never force push main. A final release version, changelog, signed tag and PyPI publication are separate decisions not made by these PRs.

## Stage 6 — runtime compatibility acceptance

A new isolated regression module, tests/test_runtime_compatibility.py, checks:

- CPU bfloat16 autocast with identical initial parameters/inputs, forward outputs, losses, gradients and Adam updates, in both trace and module monitoring.
- torch.compile with backend='eager' and native module observation, testing forward, loss and parameter parity on the compiled wrapper. This is not a guarantee for the inductor/backend native compiler, generated kernels or GPU.
- One-process Gloo DistributedDataParallel (world_size=1) with native module observation. This is not equivalent to multi-process DDP, gradient all-reduce correctness or NCCL.
- A repeatedly invoked Linear module combined with in-place ReLU under native module observation, checking training parity. Public telemetry still retains only the most recent invocation of the shared module.
- Optional CUDA float16 autocast for both monitoring modes. It is skipped when no GPU exists.

The Python matrix now tests **3.9, 3.10, 3.11, 3.12 and 3.13**, matching package metadata. CI pins Ubuntu 24.04 for core jobs because Ubuntu 26.04 GitHub runners did not provide a Python 3.9 toolchain in the current setup-python manifest. A separate Qt WebEngine job checks GUI sandbox and injection constraints.

At workflow [44405521745](https://github.com/fthyco/GRAPHyco/actions/runs/44405521745), all six jobs succeeded:
- Each Python version: **79 passed, 4 skipped** (optional CUDA and GUI fixtures in core-only jobs).
- Qt security suite: **3 passed** under Xvfb with PySide6 installed.
- Each core job: distributable wheel and sdist built, packaged assets verified, installed headless CLI verified outside source checkout.

The tests establish parity on these CPU configurations only. CUDA is **not verified** by the GitHub CPU runner; production GPU AMP, compiled inductor, multiple distributed ranks/NCCL, stateful Python forwards, long-running memory and arbitrary models remain unresolved release gates.

## Compatibility and migration concerns

The earlier stacked changes are not a patch-only change:
1. Module graphs no longer invent sibling-order dataflow; module_to_graph now defaults to including containment nodes. Graph metrics for module mode change, and must be compared only with graphs of the same provenance and semantics.
2. Calling DynamicExecutionBridge.step with no optimizer restores prior parameter gradients, instead of leaving training gradient accumulation.
3. Monitors retain 128 historical full snapshots by default; old unbounded trajectory behavior is opt-in using max_history_steps=None.
4. The JSON HTML renderer rejects invalid JSON string input and maps nonfinite telemetry to JSON null.
5. The Qt GUI no longer automatically disables the Chromium sandbox; insecure root-only GUI launches may require secure deployment configuration.

These behaviors require a semver decision and documented migration notes before the next public release; the existing pyproject.toml still declares version 0.1.2. No PyPI upload, tag or version bump has been made.

## Decisions still required before declaring production ready

Human code/security review of the stacked changes; a source-controlled change log and version decision; full multi-rank GPU/AMP/DDP/compiler compatibility on real hardware; repeatable profiling and memory accounting; and a published upgrade guide. Passing a CPU test matrix is a necessary gate, not a certification of universal training fidelity.
