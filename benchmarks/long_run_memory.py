"""2,000-step bounded-history CPU monitoring and heap/RSS plateau gate.

Peak RSS is reported separately; VmRSS checkpoints are coarse and require
repeatable hardware to interpret. A tight log_interval ensures this probes
real forward/backward observation across 250 captured steps, not no-op loops.
"""

from __future__ import annotations

import argparse
import gc
import json
import resource
import time
import tracemalloc
from pathlib import Path

import torch
from torch import nn

from graphyco.bridge.grad_bridge import LiveTrainingMonitor


def rss_mib() -> float:
    with open("/proc/self/status", "r", encoding="utf-8") as stream:
        for line in stream:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024.0
    raise RuntimeError("requires Linux /proc")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--interval", type=int, default=8)
    parser.add_argument("--max-history", type=int, default=16)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.steps < 500 or args.interval < 1 or args.max_history < 1:
        parser.error("steps >= 500, positive interval and max history required")

    torch.set_num_threads(1)
    torch.manual_seed(451)
    model = nn.Sequential(
        nn.Linear(32, 96), nn.Tanh(),
        nn.Linear(96, 96), nn.ReLU(),
        nn.Linear(96, 8),
    )
    inputs = torch.randn(8, 32)
    targets = torch.randn(8, 8)
    opt = torch.optim.Adam(model.parameters(), lr=0.0003)
    monitor = LiveTrainingMonitor(
        model, mode="module", max_history_steps=args.max_history,
        log_interval=args.interval,
    )
    tracemalloc.start()
    checkpoints = {}
    began = time.perf_counter()
    monitored_count = 0
    try:
        for step in range(args.steps):
            with monitor.observe(step):
                opt.zero_grad(set_to_none=True)
                loss = nn.functional.mse_loss(model(inputs), targets)
                loss.backward()
                opt.step()
            if step % args.interval == 0:
                monitored_count += 1
                assert monitor.get_latest_state() is not None
                assert monitor.get_latest_state().node_records
                assert len(monitor.trajectory) <= args.max_history
            if step in (
                args.steps // 4 - 1, args.steps // 2 - 1, args.steps - 1,
            ):
                gc.collect()
                current, peak = tracemalloc.get_traced_memory()
                checkpoints[str(step + 1)] = {
                    "rss_mib": rss_mib(),
                    "python_heap_mib": current / 1048576.0,
                    "python_heap_peak_mib": peak / 1048576.0,
                    "retained_steps": len(monitor.trajectory),
                }
        assert all(torch.isfinite(p).all() for p in model.parameters())
        first = checkpoints[str(args.steps // 4)]
        last = checkpoints[str(args.steps)]
        growth_rss = last["rss_mib"] - first["rss_mib"]
        growth_heap = last["python_heap_mib"] - first["python_heap_mib"]
        # Conservative fail-fast guard against unbounded retention. This is
        # neither a GPU memory check nor proof of zero Python allocation growth.
        assert growth_heap < 32.0, "unbounded Python telemetry growth"
        assert growth_rss < 96.0, "unbounded process RSS growth"
        result = {
            "steps": args.steps,
            "captured_steps": monitored_count,
            "interval": args.interval,
            "max_history_steps": args.max_history,
            "final_retained_snapshots": len(monitor.trajectory),
            "checkpoints": checkpoints,
            "rss_growth_mib_from_quarter": growth_rss,
            "python_heap_growth_mib_from_quarter": growth_heap,
            "peak_process_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
            "elapsed_seconds": time.perf_counter() - began,
            "status": "passed",
        }
    finally:
        monitor.close()
        tracemalloc.stop()
    assert not any(m._forward_hooks for m in model.modules())
    assert not any(m._backward_hooks for m in model.modules())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
