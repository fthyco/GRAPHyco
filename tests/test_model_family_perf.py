"""Training parity on three architecture families under module and FX observation."""

from __future__ import annotations

import copy

import pytest
import torch
from torch import nn

from benchmarks.model_families import fixture
from graphyco.bridge.grad_bridge import LiveTrainingMonitor


@pytest.mark.parametrize("family", ("mlp", "residual_cnn", "attention"))
@pytest.mark.parametrize("mode", ("module", "trace"))
def test_architecture_family_monitoring_training_parity(family, mode):
    torch.set_num_threads(1)
    plain, inputs, targets = fixture(family)
    tracked = copy.deepcopy(plain)
    opt_a = torch.optim.SGD(plain.parameters(), lr=0.001)
    opt_b = torch.optim.SGD(tracked.parameters(), lr=0.001)
    monitor = LiveTrainingMonitor(
        tracked, mode=mode, strict_trace=(mode == "trace"),
        max_history_steps=2,
    )
    try:
        for step in range(2):
            opt_a.zero_grad(set_to_none=True)
            a = plain(inputs)
            loss_a = nn.functional.cross_entropy(a, targets)
            loss_a.backward()
            grads_a = [p.grad.detach().clone() for p in plain.parameters()]
            opt_a.step()

            with monitor.observe(step):
                opt_b.zero_grad(set_to_none=True)
                b = tracked(inputs)
                loss_b = nn.functional.cross_entropy(b, targets)
                loss_b.backward()
                grads_b = [p.grad.detach().clone() for p in tracked.parameters()]
                opt_b.step()

            torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)
            torch.testing.assert_close(loss_a, loss_b, rtol=1e-5, atol=1e-6)
            for x, y in zip(grads_a, grads_b):
                torch.testing.assert_close(x, y, rtol=1e-5, atol=1e-6)
            for x, y in zip(plain.parameters(), tracked.parameters()):
                torch.testing.assert_close(x, y, rtol=1e-5, atol=1e-6)
            for x, y in zip(plain.buffers(), tracked.buffers()):
                torch.testing.assert_close(x, y, rtol=1e-5, atol=1e-6)
            assert monitor.get_latest_state().node_records
    finally:
        monitor.close()
