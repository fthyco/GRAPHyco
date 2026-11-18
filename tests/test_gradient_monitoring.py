"""Stage 4: paired training parity, side effects and hook lifecycle.

These tests deliberately use deterministic CPU models with no dropout, batchnorm
or stateful forward path; they establish local parity, not universal equivalence.
"""
from __future__ import annotations

import copy

import pytest
import torch
from torch import nn

from graphyco.bridge.grad_bridge import DynamicExecutionBridge, LiveTrainingMonitor
from graphyco.observer.grad_observer import FXGradObserver


def fresh_model():
    torch.manual_seed(1234)
    return nn.Sequential(nn.Linear(4, 8), nn.Tanh(), nn.Linear(8, 2))


def batches():
    g = torch.Generator().manual_seed(2468)
    return [
        (torch.randn(3, 4, generator=g), torch.randn(3, 2, generator=g))
        for _ in range(5)
    ]


def check_optimizer_state_close(left, right):
    a, b = left.state_dict(), right.state_dict()
    assert a["param_groups"] == b["param_groups"]
    assert a["state"].keys() == b["state"].keys()
    for key in a["state"]:
        assert a["state"][key].keys() == b["state"][key].keys()
        for field in a["state"][key]:
            av, bv = a["state"][key][field], b["state"][key][field]
            if isinstance(av, torch.Tensor):
                torch.testing.assert_close(av, bv, rtol=1e-6, atol=1e-7)
            else:
                assert av == bv


def check_model_equal(first, second):
    for pa, pb in zip(first.parameters(), second.parameters()):
        torch.testing.assert_close(pa, pb, rtol=1e-6, atol=1e-7)
        assert (pa.grad is None) == (pb.grad is None)
        if pa.grad is not None:
            torch.testing.assert_close(pa.grad, pb.grad, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("mode", ["trace", "module"])
@pytest.mark.parametrize("opt_name", ["SGD", "Adam"])
def test_live_training_forward_backward_optimizer_parity(mode, opt_name):
    base = fresh_model()
    monitored = copy.deepcopy(base)
    g = batches()
    opt_cls = getattr(torch.optim, opt_name)
    opt1 = opt_cls(base.parameters(), lr=0.005)
    opt2 = opt_cls(monitored.parameters(), lr=0.005)
    monitor = LiveTrainingMonitor(monitored, mode=mode, log_interval=1)

    for step, (x, y) in enumerate(g[:3]):
        opt1.zero_grad(set_to_none=True)
        plain_out = base(x)
        plain_loss = torch.nn.functional.mse_loss(plain_out, y)
        plain_loss.backward()
        plain_grads = {n: p.grad.detach().clone() for n, p in base.named_parameters()}
        opt1.step()

        with monitor.observe(step):
            opt2.zero_grad(set_to_none=True)
            monitored_out = monitored(x.clone())
            monitored_loss = torch.nn.functional.mse_loss(monitored_out, y)
            monitored_loss.backward()
            observed_grads = {n: p.grad.detach().clone() for n, p in monitored.named_parameters()}
            opt2.step()

        torch.testing.assert_close(monitored_out, plain_out, rtol=1e-6, atol=1e-7)
        torch.testing.assert_close(monitored_loss, plain_loss, rtol=1e-6, atol=1e-7)
        for name in plain_grads:
            torch.testing.assert_close(observed_grads[name], plain_grads[name], rtol=1e-6, atol=1e-7)
        check_model_equal(base, monitored)
        check_optimizer_state_close(opt1, opt2)
        assert monitor.latest_state is not None
        assert monitor.latest_state.step == step
        assert monitor.has_new_data()
    monitor.close()


@pytest.mark.parametrize("mode", ["trace", "module"])
def test_bridge_optimizer_step_matches_plain_training(mode):
    base = fresh_model()
    monitored = copy.deepcopy(base)
    x, y = batches()[0]
    opt1 = torch.optim.Adam(base.parameters(), lr=0.003)
    opt2 = torch.optim.Adam(monitored.parameters(), lr=0.003)
    loss_fn = torch.nn.MSELoss()

    opt1.zero_grad()
    plain_loss = loss_fn(base(x.requires_grad_(True)), y)
    plain_loss.backward()
    opt1.step()

    bridge = DynamicExecutionBridge(monitored, mode=mode)
    state = bridge.step(x.clone(), y, loss_fn, optimizer=opt2)
    check_model_equal(base, monitored)
    check_optimizer_state_close(opt1, opt2)
    assert state.node_records
    bridge.close()


@pytest.mark.parametrize("initial_grad", [None, 0.25])
@pytest.mark.parametrize("mode", ["trace", "module"])
def test_bridge_without_optimizer_preserves_preexisting_gradients(mode, initial_grad):
    model = fresh_model()
    before_params = [p.detach().clone() for p in model.parameters()]
    for p in model.parameters():
        p.grad = None if initial_grad is None else torch.full_like(p, initial_grad)
    original = [p.grad.detach().clone() if p.grad is not None else None for p in model.parameters()]
    bridge = DynamicExecutionBridge(model, mode=mode)
    state = bridge.step(batches()[0][0])
    assert state.node_records
    for p, previous, before in zip(model.parameters(), original, before_params):
        torch.testing.assert_close(p, before)
        assert (p.grad is None) == (previous is None)
        if previous is not None:
            torch.testing.assert_close(p.grad, previous)
    bridge.close()


@pytest.mark.parametrize("mode", ["trace", "module"])
def test_live_monitor_exception_does_not_emit_incomplete_step(mode):
    model = fresh_model()
    monitor = LiveTrainingMonitor(model, mode=mode)
    original_instance_forward = model.__dict__.get("forward", None)
    had_instance_forward = "forward" in model.__dict__
    with pytest.raises(ValueError, match="intentional"):
        with monitor.observe(0):
            model(batches()[0][0]).sum().backward()
            raise ValueError("intentional")
    assert monitor.latest_state is None
    assert monitor.has_new_data() is False
    assert ("forward" in model.__dict__) == had_instance_forward
    assert model.__dict__.get("forward", None) is original_instance_forward
    assert not getattr(monitor, "_observing", False)
    monitor.close()


def test_live_monitor_rejects_nested_observation_and_restores_forward():
    model = fresh_model()
    monitor = LiveTrainingMonitor(model, mode="trace")
    with monitor.observe(0):
        with pytest.raises(RuntimeError, match="already active"):
            with monitor.observe(1):
                pass
        out = model(batches()[0][0])
        out.sum().backward()
    assert "forward" not in model.__dict__
    monitor.close()


def test_monitor_inactive_module_steps_do_not_clone_activations():
    model = fresh_model()
    monitor = LiveTrainingMonitor(model, mode="module", log_interval=5)
    with monitor.observe(1):
        x = model(batches()[0][0])
        x.sum().backward()
    assert monitor.trajectory == []
    assert monitor.observer.raw_activations == {}
    assert monitor.observer.raw_activation_grads == {}
    monitor.close()


@pytest.mark.parametrize("monitor_type", ["bridge", "live"])
def test_bounded_trajectory_memory_and_lifecycle(monitor_type):
    model = fresh_model()
    if monitor_type == "bridge":
        monitor = DynamicExecutionBridge(model, mode="module", max_history_steps=3)
        for _ in range(8):
            monitor.step(batches()[0][0])
    else:
        monitor = LiveTrainingMonitor(model, mode="module", max_history_steps=3)
        for step in range(8):
            with monitor.observe(step):
                model.zero_grad(set_to_none=True)
                model(batches()[0][0]).sum().backward()
        assert monitor.latest_state.step == 7
    assert len(monitor.trajectory) == 3
    assert [s.step for s in monitor.trajectory] == (
        [6, 7, 8] if monitor_type == "bridge" else [5, 6, 7]
    )
    assert monitor.trajectory[-1].dynamic_evaluation["temporal"] is not None
    assert any(module._forward_hooks for module in model.modules())
    monitor.close()
    assert not any(module._forward_hooks for module in model.modules())
    assert not any(module._backward_hooks for module in model.modules())


def test_trace_mode_restores_exact_preexisting_instance_forward():
    model = fresh_model()
    original = model.forward
    model.forward = original
    monitor = LiveTrainingMonitor(model, mode="trace")
    with monitor.observe(0):
        model(batches()[0][0]).sum().backward()
    assert model.__dict__["forward"] is original
    monitor.close()


def test_tuple_output_hook_attaches_first_tensor_not_last():
    model = fresh_model()
    observer = FXGradObserver(torch.fx.symbolic_trace(model))
    x = torch.tensor([1.0], requires_grad=True)
    y = torch.tensor([2.0], requires_grad=True)
    # First output is the only activation represented by the public node
    # record. Its gradient must not be overwritten by other tuple members.
    observer._attach_tensor_hook("tuple", x)
    observer._attach_tensor_hook("tuple", y)
    (x * 3 + y * 7).sum().backward()
    # This test exposes the undocumented overwrite; the proper tuple-aware
    # fix is to retain component-specific gradients and the first canonical one.
    assert torch.equal(observer.raw_activation_grads["tuple"], torch.tensor([3.0]))
