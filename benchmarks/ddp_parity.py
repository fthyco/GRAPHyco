"""Real two-process CPU DDP parity gate; execute under torchrun (no skips).

    python -m torch.distributed.run --standalone --nproc_per_node=2 benchmarks/ddp_parity.py

Each process owns the same two distributed replicas and distinct rank-local
input data. Compare monitored vs unmonitored forward, loss, parameter gradients,
and optimizer updates. Validate that gradients were actually all-reduced across
both ranks, not merely locally equal to the plain model.
"""

from __future__ import annotations

import copy
import json
import os
from datetime import timedelta

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

from graphyco.bridge.grad_bridge import LiveTrainingMonitor


def model() -> nn.Module:
    return nn.Sequential(
        nn.Linear(8, 32),
        nn.Tanh(),
        nn.Linear(32, 16),
        nn.ReLU(),
        nn.Linear(16, 2),
    )


def run() -> None:
    assert dist.is_available() and dist.is_gloo_available(), "Gloo required"
    assert int(os.environ["WORLD_SIZE"]) >= 2, "requires at least two ranks"
    torch.set_num_threads(1)
    rank = int(os.environ["RANK"])
    dist.init_process_group(backend="gloo", timeout=timedelta(seconds=90))
    monitor = None
    try:
        torch.manual_seed(5482)
        base = model()
        tracked = copy.deepcopy(base)
        ddp_plain = DDP(base)
        ddp_tracked = DDP(tracked)
        opt_plain = torch.optim.Adam(ddp_plain.parameters(), lr=0.001)
        opt_mon = torch.optim.Adam(ddp_tracked.parameters(), lr=0.001)
        monitor = LiveTrainingMonitor(
            ddp_tracked, mode="module", log_interval=1, max_history_steps=3,
        )
        for step in range(3):
            gen = torch.Generator().manual_seed(3170 + rank * 100 + step)
            x = torch.randn(5, 8, generator=gen)
            y = torch.randn(5, 2, generator=gen)

            opt_plain.zero_grad(set_to_none=True)
            out_a = ddp_plain(x)
            loss_a = (out_a - y).square().mean()
            loss_a.backward()
            grads_a = [p.grad.detach().clone() for p in ddp_plain.parameters()]
            opt_plain.step()

            with monitor.observe(step):
                opt_mon.zero_grad(set_to_none=True)
                out_b = ddp_tracked(x)
                loss_b = (out_b - y).square().mean()
                loss_b.backward()
                grads_b = [p.grad.detach().clone() for p in ddp_tracked.parameters()]
                opt_mon.step()

            torch.testing.assert_close(out_a, out_b, rtol=1e-6, atol=1e-7)
            torch.testing.assert_close(loss_a, loss_b, rtol=1e-6, atol=1e-7)
            for ga, gb in zip(grads_a, grads_b):
                torch.testing.assert_close(ga, gb, rtol=1e-6, atol=1e-7)
            for pa, pb in zip(ddp_plain.parameters(), ddp_tracked.parameters()):
                torch.testing.assert_close(pa, pb, rtol=1e-6, atol=1e-7)

            # Explicitly check that allreduce happened between distinct ranks:
            # the all-reduced tensor should be the same on every rank.
            reduced_gradient = grads_b[0]
            peers = [torch.zeros_like(reduced_gradient) for _ in range(dist.get_world_size())]
            dist.all_gather(peers, reduced_gradient)
            for peer in peers:
                torch.testing.assert_close(peer, reduced_gradient, rtol=1e-6, atol=1e-7)

            assert monitor.latest_state is not None
            assert monitor.latest_state.node_records
            assert monitor.latest_state.step == step
        assert len(monitor.trajectory) == 3
        dist.barrier()
        print(json.dumps({
            "rank": rank,
            "world_size": dist.get_world_size(),
            "backend": "gloo",
            "steps": 3,
            "monitor_mode": "module",
            "status": "passed",
        }), flush=True)
    finally:
        if monitor is not None:
            monitor.close()
        dist.destroy_process_group()


if __name__ == "__main__":
    run()
