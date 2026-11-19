"""Actual torchvision ResNet50 and deep Transformer monitor stress.

Run only in the dedicated CPU stress job after torchvision installation.
These are real PyTorch model implementations, NOT the earlier tiny residual
CNN/attention stand-ins. The test checks training gradients, optimizer state,
BatchNorm buffers, telemetry and bounded observer history.
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import os
import resource
import time
from pathlib import Path

import torch
from torch import nn

from graphyco.bridge.grad_bridge import LiveTrainingMonitor


class DeepTransformer(nn.Module):
    def __init__(self, layers: int = 8, width: int = 128):
        super().__init__()
        self.input_proj = nn.Linear(48, width)
        encoder = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=8,
            dim_feedforward=width * 4,
            dropout=0.0,
            batch_first=True,
        )
        self.stack = nn.TransformerEncoder(
            encoder, num_layers=layers,
            enable_nested_tensor=False,
        )
        self.head = nn.Linear(width, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.stack(self.input_proj(x)).mean(dim=1))


def fixture(family: str):
    torch.manual_seed(5712)
    gen = torch.Generator().manual_seed(119)
    if family == "resnet50":
        from torchvision.models import resnet50

        model = resnet50(weights=None, num_classes=10)
        x = torch.randn(2, 3, 64, 64, generator=gen)
        y = torch.randint(0, 10, (2,), generator=gen)
    elif family == "transformer8":
        model = DeepTransformer(layers=8)
        x = torch.randn(2, 24, 48, generator=gen)
        y = torch.randint(0, 10, (2,), generator=gen)
    else:
        raise ValueError("Unknown family {}".format(family))
    return model, x, y


def _rss_mib() -> float:
    with open("/proc/self/status", "r", encoding="utf-8") as source:
        for line in source:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024.0
    raise RuntimeError("Linux /proc/self/status unavailable")


def check_family(family: str, steps: int) -> dict:
    torch.set_num_threads(1)
    original, inputs, targets = fixture(family)
    monitored_model = copy.deepcopy(original)
    baseline_opt = torch.optim.SGD(original.parameters(), lr=0.001)
    observed_opt = torch.optim.SGD(monitored_model.parameters(), lr=0.001)
    monitor = LiveTrainingMonitor(
        monitored_model, mode="module", max_history_steps=3, log_interval=1,
    )
    rss_before = _rss_mib()
    plain_seconds = observed_seconds = 0.0
    observed_node_counts = []
    gradient_node_counts = []
    try:
        for step in range(steps):
            baseline_opt.zero_grad(set_to_none=True)
            began = time.perf_counter()
            a = original(inputs)
            loss_a = nn.functional.cross_entropy(a, targets)
            loss_a.backward()
            gradients_a = [p.grad.detach().clone() for p in original.parameters()]
            baseline_opt.step()
            plain_seconds += time.perf_counter() - began

            began = time.perf_counter()
            with monitor.observe(step):
                observed_opt.zero_grad(set_to_none=True)
                b = monitored_model(inputs)
                loss_b = nn.functional.cross_entropy(b, targets)
                loss_b.backward()
                gradients_b = [p.grad.detach().clone() for p in monitored_model.parameters()]
                observed_opt.step()
            observed_seconds += time.perf_counter() - began

            torch.testing.assert_close(a, b, atol=3e-5, rtol=3e-5)
            torch.testing.assert_close(loss_a, loss_b, atol=3e-5, rtol=3e-5)
            for left, right in zip(gradients_a, gradients_b):
                torch.testing.assert_close(left, right, atol=3e-5, rtol=3e-5)
            for left, right in zip(original.parameters(), monitored_model.parameters()):
                torch.testing.assert_close(left, right, atol=3e-5, rtol=3e-5)
            for left, right in zip(original.buffers(), monitored_model.buffers()):
                torch.testing.assert_close(left, right, atol=3e-5, rtol=3e-5)

            state = monitor.get_latest_state()
            assert state is not None
            assert state.node_records, "must actually capture model telemetry"
            observed_node_counts.append(len(state.node_records))
            gradient_node_counts.append(sum(
                rec.activation_gradient is not None
                for rec in state.node_records.values()
            ))
            assert len(monitor.trajectory) <= 3
            assert all(torch.isfinite(p).all() for p in monitored_model.parameters())

        gc.collect()
        return {
            "family": family,
            "steps": steps,
            "parameters": sum(p.numel() for p in original.parameters()),
            "input_shape": list(inputs.shape),
            "baseline_mean_step_seconds": plain_seconds / steps,
            "monitored_mean_step_seconds": observed_seconds / steps,
            "overhead_ratio": observed_seconds / max(plain_seconds, 1e-12),
            "telemetry_node_count": observed_node_counts,
            "gradient_observed_node_count": gradient_node_counts,
            "retained_trajectory": len(monitor.trajectory),
            "rss_start_mib": rss_before,
            "rss_end_mib": _rss_mib(),
            "peak_process_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
            "status": "passed",
        }
    finally:
        monitor.close()
        assert not any(m._forward_hooks for m in monitored_model.modules())
        assert not any(m._backward_hooks for m in monitored_model.modules())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=("resnet50", "transformer8"), required=True)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.steps < 2:
        parser.error("steps must be >=2")
    results = check_family(args.family, args.steps)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
