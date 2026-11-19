"""Regression oracles for diagnostic gradient isolation and failure cleanup."""
from __future__ import annotations

import copy

import pytest
import torch
from torch import nn

from graphyco.bridge.grad_bridge import DynamicExecutionBridge
from graphyco.observer.grad_observer import FXGradObserver, ModuleGradObserver


def _model():
    torch.manual_seed(1931)
    return nn.Sequential(nn.Linear(4, 7), nn.Tanh(), nn.Linear(7, 2))


@pytest.mark.parametrize("mode", ["trace", "module"])
@pytest.mark.parametrize("previous", [0.0, 0.25, -3.5])
def test_diagnostic_parameter_telemetry_is_independent_of_old_gradients(mode, previous):
    pristine = _model()
    with_old_grad = copy.deepcopy(pristine)

    old_grad_objects = {}
    for name, parameter in with_old_grad.named_parameters():
        parameter.grad = torch.full_like(parameter, previous)
        old_grad_objects[name] = parameter.grad

    reference = DynamicExecutionBridge(pristine, mode=mode, strict_trace=True)
    observed = DynamicExecutionBridge(with_old_grad, mode=mode, strict_trace=True)
    x = torch.tensor([[1.0, 2.0, 0.5, -0.25], [-1.0, 0.3, 0.6, 0.5]])
    try:
        state_ref = reference.step(x)
        state_obs = observed.step(x.clone())
        assert state_ref.node_records.keys() == state_obs.node_records.keys()
        assert any(
            n.parameter_gradient is not None for n in state_ref.node_records.values()
        )
        for name, left in state_ref.node_records.items():
            right = state_obs.node_records[name]
            assert (left.parameter_gradient is None) == (right.parameter_gradient is None)
            if left.parameter_gradient is not None:
                assert right.parameter_gradient.rms == pytest.approx(
                    left.parameter_gradient.rms, rel=1e-12, abs=1e-12
                )
                assert right.parameter_gradient.l2_norm == pytest.approx(
                    left.parameter_gradient.l2_norm, rel=1e-12, abs=1e-12
                )
        for name, parameter in with_old_grad.named_parameters():
            assert parameter.grad is old_grad_objects[name]
            torch.testing.assert_close(
                parameter.grad, torch.full_like(parameter, previous)
            )
        assert all(p.grad is None for p in pristine.parameters())
    finally:
        reference.close()
        observed.close()


@pytest.mark.parametrize("mode", ["trace", "module"])
def test_failed_diagnostic_backward_restores_same_original_grads(mode):
    model = _model()
    original = {}
    for name, parameter in model.named_parameters():
        parameter.grad = torch.ones_like(parameter) * 0.75
        original[name] = parameter.grad

    bridge = DynamicExecutionBridge(model, mode=mode, strict_trace=True)
    try:
        with pytest.raises(RuntimeError, match="does not require grad"):
            # A detached loss is invalid for backward and must not alter
            # pre-existing caller-owned parameter gradient objects.
            bridge.step(
                torch.ones(2, 4),
                loss_fn=lambda output: output.detach().sum(),
            )
        for name, parameter in model.named_parameters():
            assert parameter.grad is original[name]
            torch.testing.assert_close(
                parameter.grad, torch.full_like(parameter, 0.75)
            )
        assert bridge.trajectory == []
    finally:
        bridge.close()


@pytest.mark.parametrize("mode", ["trace", "module"])
def test_parameter_snapshot_never_retains_differentiable_gradient_graph(mode):
    model = _model()
    if mode == "trace":
        observer = FXGradObserver(torch.fx.symbolic_trace(model))
        observer.run(torch.ones(1, 4))
    else:
        observer = ModuleGradObserver(model)
    try:
        for param in model.parameters():
            # Deliberately differentiable gradient buffers occur with
            # higher-order optimization and custom autograd workflows.
            param.grad = torch.ones_like(param, requires_grad=True)
        observer.capture_parameter_gradients()
        assert observer.raw_param_grads
        for snapshot in observer.raw_param_grads.values():
            assert not snapshot.requires_grad
            assert snapshot.grad_fn is None
    finally:
        if isinstance(observer, ModuleGradObserver):
            observer.remove_hooks()
        model.zero_grad(set_to_none=True)
