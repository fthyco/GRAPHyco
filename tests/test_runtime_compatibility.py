"""Pre-release runtime compatibility gates.

Positive results support explicitly exercised configurations only:
CPU AMP (bfloat16), eager backend of torch.compile, gloo single-process DDP,
shared modules, and native in-place activation. CUDA is optional and skipped
on CPU-only CI. This suite does not certify production-scale/multi-rank DDP.
"""
from __future__ import annotations

import copy
import pathlib

import pytest
import torch
from torch import nn

from graphyco.bridge.grad_bridge import LiveTrainingMonitor


def _model():
    torch.manual_seed(192)
    return nn.Sequential(
        nn.Linear(4, 8),
        nn.ReLU(),
        nn.Linear(8, 2),
    )


def _batch(device="cpu"):
    gen = torch.Generator().manual_seed(371)
    return (
        torch.randn(6, 4, generator=gen).to(device),
        torch.randn(6, 2, generator=gen).to(device),
    )


def _parameters_equal(a, b, *, atol=1e-6, rtol=1e-6):
    left, right = list(a.parameters()), list(b.parameters())
    assert len(left) == len(right)
    for pa, pb in zip(left, right):
        torch.testing.assert_close(pa.detach(), pb.detach(), atol=atol, rtol=rtol)
        assert (pa.grad is None) == (pb.grad is None)
        if pa.grad is not None:
            torch.testing.assert_close(pa.grad.detach(), pb.grad.detach(), atol=atol, rtol=rtol)


def _step(model, optimizer, x, y, *, amp_device=None):
    optimizer.zero_grad(set_to_none=True)
    if amp_device is not None:
        with torch.autocast(device_type=amp_device, dtype=(
            torch.bfloat16 if amp_device == "cpu" else torch.float16
        )):
            output = model(x)
            loss = torch.nn.functional.mse_loss(output.float(), y.float())
    else:
        output = model(x)
        loss = torch.nn.functional.mse_loss(output, y)
    loss.backward()
    optimizer.step()
    return output.detach().clone(), loss.detach().clone()


@pytest.mark.parametrize("mode", ["trace", "module"])
def test_cpu_bfloat16_autocast_training_parity(mode):
    base = _model()
    monitored = copy.deepcopy(base)
    opt_plain = torch.optim.Adam(base.parameters(), lr=0.002)
    opt_mon = torch.optim.Adam(monitored.parameters(), lr=0.002)
    monitor = LiveTrainingMonitor(monitored, mode=mode, strict_trace=True)
    x, y = _batch()
    try:
        for i in range(3):
            out1, loss1 = _step(base, opt_plain, x, y, amp_device="cpu")
            with monitor.observe(i):
                out2, loss2 = _step(monitored, opt_mon, x, y, amp_device="cpu")
            torch.testing.assert_close(out1.float(), out2.float(), atol=1e-5, rtol=1e-5)
            torch.testing.assert_close(loss1, loss2, atol=1e-6, rtol=1e-6)
            _parameters_equal(base, monitored, atol=1e-5, rtol=1e-5)
            assert monitor.get_latest_state() is not None
            assert monitor.get_latest_state().node_records
    finally:
        monitor.close()


@pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile unavailable")
def test_compiled_eager_backend_with_native_module_observer():
    original = _model()
    monitored = copy.deepcopy(original)
    # Eager backend verifies torch.compile wrapper compatibility without
    # expensive/dependency-sensitive native code generation in CI.
    compiled_plain = torch.compile(original, backend="eager")
    compiled_mon = torch.compile(monitored, backend="eager")
    opt_plain = torch.optim.SGD(compiled_plain.parameters(), lr=0.01)
    opt_mon = torch.optim.SGD(compiled_mon.parameters(), lr=0.01)
    monitor = LiveTrainingMonitor(compiled_mon, mode="module")
    x, y = _batch()
    try:
        for step in range(2):
            a, loss_a = _step(compiled_plain, opt_plain, x, y)
            with monitor.observe(step):
                b, loss_b = _step(compiled_mon, opt_mon, x, y)
            torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-6)
            torch.testing.assert_close(loss_a, loss_b, atol=1e-6, rtol=1e-6)
            _parameters_equal(original, monitored)
    finally:
        monitor.close()
    assert not any(m._forward_hooks for m in compiled_mon.modules())


def test_single_rank_gloo_ddp_with_native_module_observer(tmp_path):
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP

    if not dist.is_available() or not dist.is_gloo_available():
        pytest.skip("Gloo process group unavailable")
    if dist.is_initialized():
        pytest.skip("A caller already owns the process group")

    init_file = pathlib.Path(tmp_path) / "init"
    dist.init_process_group(
        backend="gloo",
        init_method=init_file.resolve().as_uri(),
        world_size=1, rank=0,
    )
    monitor = None
    try:
        base = _model()
        tracked = copy.deepcopy(base)
        plain_ddp = DDP(base)
        tracked_ddp = DDP(tracked)
        opt_a = torch.optim.SGD(plain_ddp.parameters(), lr=0.015)
        opt_b = torch.optim.SGD(tracked_ddp.parameters(), lr=0.015)
        monitor = LiveTrainingMonitor(tracked_ddp, mode="module")
        x, y = _batch()
        for step in range(2):
            out_a, loss_a = _step(plain_ddp, opt_a, x, y)
            with monitor.observe(step):
                out_b, loss_b = _step(tracked_ddp, opt_b, x, y)
            torch.testing.assert_close(out_a, out_b, atol=1e-6, rtol=1e-6)
            torch.testing.assert_close(loss_a, loss_b, atol=1e-6, rtol=1e-6)
            _parameters_equal(plain_ddp, tracked_ddp)
    finally:
        if monitor:
            monitor.close()
        dist.destroy_process_group()


class _SharedAndInplace(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(4, 4)
        self.relu = nn.ReLU(inplace=True)
        self.head = nn.Linear(4, 2)

    def forward(self, x):
        # This module is invoked twice and has a functional merge.
        left = self.projection(x)
        right = self.projection(x * 0.5)
        merged = left + right
        return self.head(self.relu(merged))


def test_shared_module_and_inplace_activation_native_training_parity():
    torch.manual_seed(28)
    plain = _SharedAndInplace()
    tracked = copy.deepcopy(plain)
    opt1 = torch.optim.Adam(plain.parameters(), lr=0.006)
    opt2 = torch.optim.Adam(tracked.parameters(), lr=0.006)
    monitor = LiveTrainingMonitor(tracked, mode="module")
    x, y = _batch()
    try:
        for step in range(2):
            a, loss_a = _step(plain, opt1, x, y)
            with monitor.observe(step):
                b, loss_b = _step(tracked, opt2, x, y)
            torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-6)
            torch.testing.assert_close(loss_a, loss_b, atol=1e-6, rtol=1e-6)
            _parameters_equal(plain, tracked)
            assert "projection" in monitor.get_latest_state().node_records
            assert "relu" in monitor.get_latest_state().node_records
    finally:
        monitor.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA runner unavailable")
@pytest.mark.parametrize("mode", ["trace", "module"])
def test_optional_cuda_amp_monitoring_parity(mode):
    plain = _model().cuda()
    tracked = copy.deepcopy(plain)
    opt1 = torch.optim.SGD(plain.parameters(), lr=0.005)
    opt2 = torch.optim.SGD(tracked.parameters(), lr=0.005)
    x, y = _batch("cuda")
    monitor = LiveTrainingMonitor(tracked, mode=mode, stats_backend="device")
    try:
        a, loss_a = _step(plain, opt1, x, y, amp_device="cuda")
        with monitor.observe(0):
            b, loss_b = _step(tracked, opt2, x, y, amp_device="cuda")
        torch.cuda.synchronize()
        torch.testing.assert_close(a, b, atol=1e-3, rtol=1e-3)
        torch.testing.assert_close(loss_a, loss_b, atol=1e-3, rtol=1e-3)
        _parameters_equal(plain, tracked, atol=1e-3, rtol=1e-3)
    finally:
        monitor.close()
