"""On-device tensor statistics should retain the full historical contract."""
from __future__ import annotations

import copy

import pytest
import torch
from torch import nn

from graphyco.observer.grad_observer import compute_tensor_stats, TensorStats
from graphyco.bridge.grad_bridge import DynamicExecutionBridge, LiveTrainingMonitor


@pytest.mark.parametrize("data", [
    [],
    [0],
    [1.0],
    [-1.0, 2.5, 0.0, -4.0, 4.0],
    [1e-8, -1e-8, 1e6],
    [float("nan"), float("inf"), float("-inf"), 0.0, 5.0],
])
def test_device_stats_equal_legacy_cpu_reference(data):
    tensor = torch.tensor(data, dtype=torch.float32)
    reference = compute_tensor_stats(tensor, backend="cpu")
    optimized = compute_tensor_stats(tensor, backend="device")
    assert isinstance(reference, TensorStats)
    assert isinstance(optimized, TensorStats)
    for key in reference.to_dict():
        a, b = reference.to_dict()[key], optimized.to_dict()[key]
        if isinstance(a, float):
            assert b == pytest.approx(a, rel=1e-11, abs=1e-11), (key, data)
        else:
            assert a == b, (key, data)


def test_stats_backend_rejects_invalid_options():
    with pytest.raises(ValueError):
        compute_tensor_stats(torch.ones(2), backend="fast-but-wrong")
    with pytest.raises(ValueError):
        DynamicExecutionBridge(nn.Linear(2, 2), stats_backend="invalid")
    with pytest.raises(ValueError):
        LiveTrainingMonitor(nn.Linear(2, 2), stats_backend="invalid")


@pytest.mark.parametrize("mode", ["trace", "module"])
def test_backend_parameter_parity_and_telemetry(mode):
    torch.manual_seed(7)
    base = nn.Sequential(nn.Linear(4, 8), nn.Tanh(), nn.Linear(8, 2))
    model_a, model_b = copy.deepcopy(base), copy.deepcopy(base)
    optimizer_a = torch.optim.Adam(model_a.parameters(), lr=0.01)
    optimizer_b = torch.optim.Adam(model_b.parameters(), lr=0.01)
    a = DynamicExecutionBridge(model_a, mode=mode, stats_backend="cpu")
    b = DynamicExecutionBridge(model_b, mode=mode, stats_backend="device")
    x = torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.2, -0.1, 0.5, 0.0]])
    y = torch.ones(2, 2)
    fn = torch.nn.MSELoss()
    record_a = a.step(x, target=y, loss_fn=fn, optimizer=optimizer_a)
    record_b = b.step(x, target=y, loss_fn=fn, optimizer=optimizer_b)
    for pa, pb in zip(model_a.parameters(), model_b.parameters()):
        torch.testing.assert_close(pa, pb, rtol=1e-7, atol=1e-8)
        torch.testing.assert_close(pa.grad, pb.grad, rtol=1e-7, atol=1e-8)
    assert record_a.node_records.keys() == record_b.node_records.keys()
    for name in record_a.node_records:
        left, right = record_a.node_records[name], record_b.node_records[name]
        for field in ("activation", "activation_gradient", "parameter_gradient"):
            stats_a, stats_b = getattr(left, field), getattr(right, field)
            assert (stats_a is None) == (stats_b is None)
            if stats_a is not None:
                for key in stats_a.to_dict():
                    val_a, val_b = stats_a.to_dict()[key], stats_b.to_dict()[key]
                    if isinstance(val_a, float):
                        assert val_b == pytest.approx(val_a, rel=1e-10, abs=1e-10)
                    else:
                        assert val_a == val_b
    a.close()
    b.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not present")
def test_cuda_device_backend_matches_cpu_reference():
    t = torch.randn(256, device="cuda", dtype=torch.float32)
    a = compute_tensor_stats(t, backend="cpu")
    b = compute_tensor_stats(t, backend="device")
    for key in a.to_dict():
        av, bv = a.to_dict()[key], b.to_dict()[key]
        if isinstance(av, float):
            assert bv == pytest.approx(av, rel=1e-10, abs=1e-10)
        else:
            assert av == bv
