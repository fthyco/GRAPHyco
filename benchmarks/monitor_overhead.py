"""Reproducible opt-in CPU performance probe; not a release benchmark.

Run:
    python benchmarks/monitor_overhead.py --steps 30 --warmup 5
    python benchmarks/monitor_overhead.py --steps 30 --json /tmp/graphyco-perf.json

Each mode gets an isolated Python process and seeded initial weights/inputs.
Peak RSS is Linux process-level maximum RSS, not per-tensor allocation,
not a native/torch GPU memory measure. CI only smoke-checks this script.
Do not compare the ratios across machines or PyTorch releases.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path

import torch
from torch import nn


def _train(case: str, steps: int, warmup: int, width: int) -> dict:
    from graphyco.bridge.grad_bridge import LiveTrainingMonitor

    torch.set_num_threads(1)
    torch.manual_seed(5478)
    model = nn.Sequential(
        nn.Linear(width, width),
        nn.Tanh(),
        nn.Linear(width, width),
        nn.ReLU(),
        nn.Linear(width, width),
    )
    opt = torch.optim.SGD(model.parameters(), lr=0.002)
    inputs = [torch.randn(8, width) for _ in range(warmup + steps)]
    labels = [torch.randn(8, width) for _ in range(warmup + steps)]
    monitoring_mode = case.replace("_device", "")
    monitor = (
        LiveTrainingMonitor(
            model, mode=monitoring_mode, log_interval=1,
            record_edges=True, max_history_steps=32,
            stats_backend="device" if case.endswith("_device") else "cpu",
        )
        if case != "plain" else None
    )

    # Warm up graph construction/allocator outside measured region.
    for step in range(warmup + steps):
        if monitor is None:
            opt.zero_grad(set_to_none=True)
            loss = torch.nn.functional.mse_loss(model(inputs[step]), labels[step])
            loss.backward()
            opt.step()
        else:
            with monitor.observe(step):
                opt.zero_grad(set_to_none=True)
                loss = torch.nn.functional.mse_loss(
                    model(inputs[step]), labels[step]
                )
                loss.backward()
                opt.step()

        if step == warmup - 1:
            start = time.perf_counter()

    elapsed_ms = (time.perf_counter() - start) * 1000 / steps
    retained = len(monitor.trajectory) if monitor is not None else 0
    if monitor:
        monitor.close()

    return {
        "mode": case,
        "mean_step_ms": round(elapsed_ms, 5),
        "peak_process_rss_mib": round(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 2
        ),
        "retained_steps": retained,
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "steps": steps,
        "warmup": warmup,
        "width": width,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--json", dest="json_path")
    parser.add_argument("--_case", choices=(
        "plain", "trace", "module", "trace_device", "module_device"
    ))
    args = parser.parse_args()
    if args.steps < 1 or args.warmup < 1 or args.width < 2:
        parser.error("--steps>=1, --warmup>=1 and --width>=2 required")

    if args._case is not None:
        print(json.dumps(_train(args._case, args.steps, args.warmup, args.width)))
        return

    cases = {}
    for case in ("plain", "trace", "module", "trace_device", "module_device"):
        command = [
            sys.executable, str(Path(__file__).resolve()),
            "--_case", case,
            "--steps", str(args.steps),
            "--warmup", str(args.warmup),
            "--width", str(args.width),
        ]
        proc = subprocess.run(command, check=True, capture_output=True, text=True)
        cases[case] = json.loads(proc.stdout.strip())

    baseline = cases["plain"]["mean_step_ms"]
    result = {
        "method": "isolated-process CPU training wall time; initial FX tracing excluded",
        "caveat": "Process peak RSS includes Python/PyTorch startup; not incremental tensor memory.",
        "results": cases,
        "overhead_factor": {
            case: round(cases[case]["mean_step_ms"] / baseline, 3)
            for case in ("trace", "module", "trace_device", "module_device")
        },
    }
    serialized = json.dumps(result, indent=2)
    print(serialized)
    if args.json_path:
        dest = Path(args.json_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(serialized + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
