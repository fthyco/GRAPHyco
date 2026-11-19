"""CPU torch.compile Inductor integration (real code generation, not eager)."""

from __future__ import annotations

import copy

import pytest
import torch
from torch import nn

from graphyco.bridge.grad_bridge import LiveTrainingMonitor


def model() -> nn.Module:
    return nn.Sequential(
        nn.Linear(8, 16),
        nn.Tanh(),
        nn.Linear(16, 8),
        nn.ReLU(),
        nn.Linear(8, 2),
    )


def test_cpu_inductor_forward_backward_optimizer_and_monitor_parity():
    assert hasattr(torch, "compile"), "Torch compiler required for gate"
    torch.manual_seed(178)
    plain = model()
    tracked = copy.deepcopy(plain)
    plain_compiled = torch.compile(plain, backend="inductor")
    tracked_compiled = torch.compile(tracked, backend="inductor")
    optimizer_a = torch.optim.Adam(plain_compiled.parameters(), lr=0.001)
    optimizer_b = torch.optim.Adam(tracked_compiled.parameters(), lr=0.001)
    monitor = LiveTrainingMonitor(
        tracked_compiled, mode="module", max_history_steps=3,
    )
    try:
        for step in range(3):
            gen = torch.Generator().manual_seed(817 + step)
            x = torch.randn(4, 8, generator=gen)
            y = torch.randn(4, 2, generator=gen)

            optimizer_a.zero_grad(set_to_none=True)
            out_a = plain_compiled(x)
            loss_a = torch.nn.functional.mse_loss(out_a, y)
            loss_a.backward()
            grad_a = [p.grad.detach().clone() for p in plain.parameters()]
            optimizer_a.step()

            with monitor.observe(step):
                optimizer_b.zero_grad(set_to_none=True)
                out_b = tracked_compiled(x)
                loss_b = torch.nn.functional.mse_loss(out_b, y)
                loss_b.backward()
                grad_b = [p.grad.detach().clone() for p in tracked.parameters()]
                optimizer_b.step()

            torch.testing.assert_close(out_a, out_b, atol=1e-6, rtol=1e-6)
            torch.testing.assert_close(loss_a, loss_b, atol=1e-6, rtol=1e-6)
            for left, right in zip(grad_a, grad_b):
                torch.testing.assert_close(left, right, atol=1e-6, rtol=1e-6)
            for left, right in zip(plain.parameters(), tracked.parameters()):
                torch.testing.assert_close(left, right, atol=1e-6, rtol=1e-6)
            assert monitor.get_latest_state() is not None
            assert monitor.get_latest_state().node_records
    finally:
        monitor.close()
    assert not any(module._forward_hooks for module in tracked_compiled.modules())


@pytest.mark.parametrize("numel", [1, 256, 8192])
def test_compiler_independent_tensor_stats_parity(numel):
    from graphyco.observer.grad_observer import compute_tensor_stats

    torch.manual_seed(numel)
    x = torch.randn(numel)
    cpu = compute_tensor_stats(x, backend="cpu")
    device = compute_tensor_stats(x, backend="device")
    for key in cpu.to_dict():
        a = cpu.to_dict()[key]
        b = device.to_dict()[key]
        if isinstance(a, float):
            assert b == pytest.approx(a, rel=1e-10, abs=1e-10)
        else:
            assert b == a
