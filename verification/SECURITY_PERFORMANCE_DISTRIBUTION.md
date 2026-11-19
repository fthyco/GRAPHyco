# Stage 5 — Security, Performance and Distribution Audit

Stage 5 audit evidence recorded on 2026-11-18; the changes were subsequently included in the 0.2.0 source merge. This document describes tested fixes and limitations. It is not a third-party security audit, an unconditional claim of training parity, or a performance SLA.

## Renderer and browser security

The previous Renderer inserted raw JSON inside an HTML script element. A malicious model or node label containing a closing script tag could break out of the JSON element and execute HTML. The renderer now parses JSON string inputs, requires an object, escapes HTML delimiters in serialized JSON, and maps nonfinite float telemetry to JSON null. Its script payload can be parsed back to the original data structure without executing an injected tag. Invalid raw strings are rejected.

The generated HTML includes a Content Security Policy blocking external scripts, arbitrary network connections, object embeds and forms. Packaged scripts and CSS are still allowed inline. This means CSP is defense in depth; it is not itself a guarantee against all HTML-injection vulnerabilities. URL query parameters for view, mode and step are validated against explicit allowed values.

The GUI no longer sets the Chromium sandbox-disable environment variable or sandbox bypass command flags. Embedded documents use an about:blank origin rather than a privileged local-file origin. Qt WebEngine disallows local and remote file access from the page, JS-created windows, and http/https/ftp/file/javascript navigations. Actual QWebEngine restrictions are exercised in the isolated GUI CI job under Xvfb. See tests/test_qt_webengine_optional.py and tests/test_renderer_security.py.

Limitations: these tests do not constitute fuzzing, live exploit testing, or a security evaluation of all REST/API and Qt deployment configurations. A host that cannot run sandboxed WebEngine must configure its own secure environment; the library no longer silently bypasses protection.

## Statistics and runtime performance

The historic CPU float64 TensorStats computation remains the default. An optional stats_backend="device" in DynamicExecutionBridge and LiveTrainingMonitor performs the full float64 reduction on the source device and transfers packed scalar results. It avoids full-tensor CPU transfers on accelerators but retains activation/gradient cloning, observer hooks and synchronization. CI tests verify numerical field parity on CPU, with an optional CUDA check skipped on CPU runners. No universal speedup is claimed.

A reproducible smoke measurement used PyTorch CPU, width 16, batch size 8, single-thread execution, SGD, 12 timed steps after 4 warmups, and separate seeded processes per mode. One GitHub Actions run, 44398772287, reported:

| Mode | Step time (ms) | Relative to plain |
| --- | ---: | ---: |
| Plain training | 0.22698 | 1.00x |
| FX monitoring, CPU stats | 2.72903 | 12.02x |
| Module monitoring, CPU stats | 1.90187 | 8.38x |
| FX monitoring, device stats | 2.95717 | 13.03x |
| Module monitoring, device stats | 2.17813 | 9.60x |
| FX monitoring, every fourth step | 0.90532 | 3.99x |
| Module monitoring, every fourth step | 0.76131 | 3.35x |

On-device reductions were slower on this CPU fixture. Sparse logging reduces average overhead but reduces observation coverage; it is not an equivalent full-telemetry comparison. These numbers are indicative only, not a statistically supported benchmark, and must not be extrapolated to GPU or large models. Process RSS includes PyTorch and Python startup, not just monitoring allocations.

To reproduce: python benchmarks/monitor_overhead.py --steps 100 --warmup 10 --width 64 --json /tmp/graphyco-perf.json

For reduced overhead while keeping the original Python forward execution, prefer LiveTrainingMonitor(model, mode="module", log_interval=4). The module mode does not include operation-level FX edge interception. For maximum temporal coverage use log_interval=1 and accept the overhead.

Future optimization needs production-size model profiling, calibrated GPU benchmarks, selective/streaming statistics, and AMP/DDP/compile compatibility investigations.

## Release and distribution gates

The CI builds both wheels and source distributions, verifies packaged metadata and visualizer HTML/CSS/JS assets, installs both from outside the repository, runs an installed headless CLI query, and tests core behavior on Python 3.11/3.12 without PySide6. A separate GUI job verifies Qt WebEngine restrictions with PySide6 installed.

At the time of this Stage 5 audit, the version was unchanged. The changes were later consolidated into `main` with version 0.2.0 and release notes. PyPI publication and physical GPU certification are separate from this audit and are not verified here.

## Compatibility and remaining limits

- The existing TensorStats default output and numeric metric functions are unchanged.
- The new device reductions path is opt-in.
- The Renderer intentionally rejects strings that are not valid JSON objects.
- Nonfinite floats in HTML-export data are represented by JSON null.
- The GUI no longer disables the Chromium sandbox automatically; root/container environments may need proper sandbox infrastructure.
- The HTML sidebar now uses honest Stage 2 metric labels instead of claims about causality or measured resilience.
- The current tests do not certify all GPUs, large models, custom autograd, in-place operations, arbitrary Python forward side effects, or distributed training.
