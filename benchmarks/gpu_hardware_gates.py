"""Mandatory *physical GPU* compatibility and performance gates.

Run on a real CUDA runner, never on a CPU runner with silent skips:
  python benchmarks/gpu_hardware_gates.py --mode single --output /tmp/gpu.json
  torchrun --standalone --nproc-per-node=2 benchmarks/gpu_hardware_gates.py --mode distributed
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from datetime import timedelta
from pathlib import Path

import torch
from torch import nn

from graphyco.bridge.grad_bridge import LiveTrainingMonitor
from graphyco.observer.grad_observer import compute_tensor_stats


def network(width: int = 32) -> nn.Module:
    return nn.Sequential(
        nn.Linear(16, width),
        nn.GELU(),
        nn.Linear(width, width),
        nn.ReLU(),
        nn.Linear(width, 4),
    )


def assert_pair_parameters(first: nn.Module, second: nn.Module, *, atol: float) -> None:
    for left, right in zip(first.parameters(), second.parameters()):
        torch.testing.assert_close(left.detach(), right.detach(), atol=atol, rtol=1e-3)
        assert (left.grad is None) == (right.grad is None)
        if left.grad is not None:
            torch.testing.assert_close(
                left.grad.detach(), right.grad.detach(), atol=atol, rtol=1e-3,
            )


def _step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    inputs: torch.Tensor,
    targets: torch.Tensor,
) -> tuple:
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.float16):
        result = model(inputs)
        loss = (result.float() - targets.float()).square().mean()
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    return result.detach().clone(), loss.detach().clone()


def single_gpu() -> dict:
    assert torch.cuda.is_available(), "A CUDA GPU is required; CPU is not a pass"
    device = torch.device("cuda:0")
    result = {}
    for mode in ("module", "trace"):
        torch.manual_seed(410)
        plain = network().to(device)
        tracked = copy.deepcopy(plain)
        plain_opt = torch.optim.Adam(plain.parameters(), lr=0.001)
        monitored_opt = torch.optim.Adam(tracked.parameters(), lr=0.001)
        plain_scaler = torch.amp.GradScaler("cuda")
        tracked_scaler = torch.amp.GradScaler("cuda")
        monitor = LiveTrainingMonitor(
            tracked, mode=mode, stats_backend="device", max_history_steps=8,
            strict_trace=(mode == "trace"),
        )
        torch.cuda.reset_peak_memory_stats()
        try:
            for step in range(6):
                gen = torch.Generator().manual_seed(710 + step)
                x = torch.randn(12, 16, generator=gen).to(device)
                y = torch.randn(12, 4, generator=gen).to(device)
                a, la = _step(plain, plain_opt, plain_scaler, x, y)
                with monitor.observe(step):
                    b, lb = _step(tracked, monitored_opt, tracked_scaler, x, y)
                torch.cuda.synchronize()
                torch.testing.assert_close(a, b, atol=2e-3, rtol=2e-3)
                torch.testing.assert_close(la, lb, atol=1e-3, rtol=1e-3)
                assert_pair_parameters(plain, tracked, atol=2e-3)
                assert plain_scaler.get_scale() == tracked_scaler.get_scale()
                assert monitor.get_latest_state().node_records
            # Nonfinite/zero metrics must preserve their observation provenance.
            probe = torch.randn(256, device=device)
            a_stats = compute_tensor_stats(probe, backend="cpu")
            b_stats = compute_tensor_stats(probe, backend="device")
            for key in a_stats.to_dict():
                a = a_stats.to_dict()[key]
                b = b_stats.to_dict()[key]
                if isinstance(a, float):
                    assert abs(a - b) <= 1e-9 + 1e-9 * abs(a), key
                else:
                    assert a == b, key
            result[mode] = {
                "status": "passed",
                "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                "observed_steps": len(monitor.trajectory),
            }
        finally:
            monitor.close()

    # Compiled GPU training: test the actual inductor backend, not eager.
    assert hasattr(torch, "compile")
    torch.manual_seed(902)
    base = network(16).to(device)
    tracked = copy.deepcopy(base)
    compiled_a = torch.compile(base, backend="inductor")
    compiled_b = torch.compile(tracked, backend="inductor")
    opt_a = torch.optim.SGD(compiled_a.parameters(), lr=0.002)
    opt_b = torch.optim.SGD(compiled_b.parameters(), lr=0.002)
    monitor = LiveTrainingMonitor(
        compiled_b, mode="module", stats_backend="device",
    )
    try:
        for step in range(3):
            gen = torch.Generator().manual_seed(1220 + step)
            x = torch.randn(8, 16, generator=gen).to(device)
            y = torch.randn(8, 4, generator=gen).to(device)
            opt_a.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                out_a = compiled_a(x)
                loss_a = (out_a.float() - y).square().mean()
            loss_a.backward()
            opt_a.step()
            with monitor.observe(step):
                opt_b.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.float16):
                    out_b = compiled_b(x)
                    loss_b = (out_b.float() - y).square().mean()
                loss_b.backward()
                opt_b.step()
            torch.cuda.synchronize()
            torch.testing.assert_close(out_a, out_b, atol=2e-3, rtol=2e-3)
            torch.testing.assert_close(loss_a, loss_b, atol=1e-3, rtol=1e-3)
            assert_pair_parameters(base, tracked, atol=2e-3)
        result["cuda_inductor"] = {"status": "passed", "steps": 3}
    finally:
        monitor.close()

    return {
        "status": "passed",
        "device": torch.cuda.get_device_name(0),
        "cuda_device_count": torch.cuda.device_count(),
        "torch": torch.__version__,
        "checks": result,
    }


def distributed_gpu() -> dict:
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP

    assert torch.cuda.is_available(), "CUDA must be available"
    assert dist.is_available() and dist.is_nccl_available(), "NCCL required"
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    assert world_size >= 2 and torch.cuda.device_count() >= 2, (
        "At least two CUDA devices and ranks required"
    )
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", timeout=timedelta(seconds=120))
    monitor = None
    try:
        torch.manual_seed(190)
        base = network().to(local_rank)
        tracked = copy.deepcopy(base)
        ddp_a = DDP(base, device_ids=[local_rank])
        ddp_b = DDP(tracked, device_ids=[local_rank])
        opt_a = torch.optim.SGD(ddp_a.parameters(), lr=0.001)
        opt_b = torch.optim.SGD(ddp_b.parameters(), lr=0.001)
        monitor = LiveTrainingMonitor(
            ddp_b, mode="module", stats_backend="device",
            max_history_steps=3,
        )
        for step in range(3):
            gen = torch.Generator().manual_seed(301 + dist.get_rank() * 13 + step)
            x = torch.randn(6, 16, generator=gen).to(local_rank)
            y = torch.randn(6, 4, generator=gen).to(local_rank)
            opt_a.zero_grad(set_to_none=True)
            a = ddp_a(x)
            loss_a = (a - y).square().mean()
            loss_a.backward()
            grad_a = [p.grad.clone() for p in ddp_a.parameters()]
            opt_a.step()
            with monitor.observe(step):
                opt_b.zero_grad(set_to_none=True)
                b = ddp_b(x)
                loss_b = (b - y).square().mean()
                loss_b.backward()
                grad_b = [p.grad.clone() for p in ddp_b.parameters()]
                opt_b.step()
            torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-5)
            torch.testing.assert_close(loss_a, loss_b, rtol=1e-4, atol=1e-5)
            for left, right in zip(grad_a, grad_b):
                torch.testing.assert_close(left, right, atol=1e-5, rtol=1e-4)
            assert_pair_parameters(ddp_a, ddp_b, atol=1e-5)
            gathered = [torch.zeros_like(grad_b[0]) for _ in range(world_size)]
            dist.all_gather(gathered, grad_b[0])
            for peer in gathered:
                torch.testing.assert_close(peer, grad_b[0], atol=1e-5, rtol=1e-4)
        dist.barrier()
        return {
            "status": "passed",
            "backend": "nccl",
            "rank": dist.get_rank(),
            "world_size": world_size,
            "steps": 3,
        }
    finally:
        if monitor is not None:
            monitor.close()
        dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("single", "distributed"), required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = single_gpu() if args.mode == "single" else distributed_gpu()
    serialized = json.dumps(result, indent=2)
    print(serialized, flush=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
