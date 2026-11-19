"""CPU architecture-family performance matrix with isolated plain/trace/module runs."""

from __future__ import annotations

import argparse
import json
import math
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path

import torch
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x)
        h = self.relu(self.bn1(h))
        return self.relu(x + self.conv2(h))


class TinyResidualCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Conv2d(3, 8, 3, padding=1)
        self.blocks = nn.Sequential(ResidualBlock(8), ResidualBlock(8))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(8, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.blocks(self.stem(x))
        return self.head(self.pool(h).flatten(1))


class TinyAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.input = nn.Linear(32, 32)
        self.attn = nn.MultiheadAttention(
            embed_dim=32, num_heads=4, dropout=0.0, batch_first=True,
        )
        self.norm = nn.LayerNorm(32)
        self.head = nn.Linear(32, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input(x)
        attended, _ = self.attn(h, h, h, need_weights=False)
        return self.head(self.norm(h + attended).mean(dim=1))


def fixture(family: str):
    torch.manual_seed(927)
    generator = torch.Generator().manual_seed(617)
    if family == "mlp":
        model = nn.Sequential(
            nn.Linear(32, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 10),
        )
        inputs = torch.randn(8, 32, generator=generator)
        targets = torch.randint(0, 10, (8,), generator=generator)
    elif family == "residual_cnn":
        model = TinyResidualCNN()
        inputs = torch.randn(4, 3, 24, 24, generator=generator)
        targets = torch.randint(0, 10, (4,), generator=generator)
    elif family == "attention":
        model = TinyAttention()
        inputs = torch.randn(4, 12, 32, generator=generator)
        targets = torch.randint(0, 10, (4,), generator=generator)
    else:
        raise ValueError("unsupported family %s" % family)
    return model, inputs, targets


def _measure(case: str, family: str, warmup: int, steps: int) -> dict:
    from graphyco.bridge.grad_bridge import LiveTrainingMonitor

    torch.set_num_threads(1)
    model, inputs, targets = fixture(family)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    monitor = (
        LiveTrainingMonitor(
            model, mode=case, max_history_steps=8,
            strict_trace=(case == "trace"),
        )
        if case != "plain" else None
    )
    start = 0.0
    try:
        for step in range(steps + warmup):
            if step == warmup:
                start = time.perf_counter()
            if monitor is None:
                optimizer.zero_grad(set_to_none=True)
                loss = nn.functional.cross_entropy(model(inputs), targets)
                loss.backward()
                optimizer.step()
            else:
                with monitor.observe(step):
                    optimizer.zero_grad(set_to_none=True)
                    loss = nn.functional.cross_entropy(model(inputs), targets)
                    loss.backward()
                    optimizer.step()
        elapsed = (time.perf_counter() - start) / steps
        assert torch.isfinite(loss)
        return {
            "family": family,
            "mode": case,
            "mean_step_ms": elapsed * 1000.0,
            "loss": float(loss.detach()),
            "peak_process_rss_mib": resource.getrusage(
                resource.RUSAGE_SELF
            ).ru_maxrss / 1024.0,
            "observed_steps": len(monitor.trajectory) if monitor else 0,
        }
    finally:
        if monitor is not None:
            monitor.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("plain", "trace", "module"))
    parser.add_argument("--family", choices=("mlp", "residual_cnn", "attention"))
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.steps < 1 or args.warmup < 1:
        parser.error("steps and warmup must be positive")
    if args.case is not None:
        if args.family is None:
            parser.error("--family is required for a single-case run")
        print(json.dumps(_measure(args.case, args.family, args.warmup, args.steps)))
        return

    rows = {}
    for family in ("mlp", "residual_cnn", "attention"):
        rows[family] = {}
        for case in ("plain", "trace", "module"):
            cmd = [
                sys.executable, str(Path(__file__).resolve()),
                "--case", case, "--family", family,
                "--steps", str(args.steps), "--warmup", str(args.warmup),
            ]
            run = subprocess.run(
                cmd, check=True, capture_output=True, text=True, timeout=180,
            )
            result = json.loads(run.stdout)
            rows[family][case] = result
        plain_ms = rows[family]["plain"]["mean_step_ms"]
        for case in ("trace", "module"):
            rows[family][case]["overhead_factor"] = (
                rows[family][case]["mean_step_ms"] / plain_ms
            )
            assert math.isfinite(rows[family][case]["overhead_factor"])
            assert rows[family][case]["overhead_factor"] < 100.0, (
                family, case, "gross performance regression"
            )

    result = {
        "method": "isolated CPU architecture-family wall time and peak process RSS",
        "torch": torch.__version__,
        "python": platform.python_version(),
        "steps": args.steps,
        "warmup": args.warmup,
        "models": rows,
        "notes": (
            "Small representative networks, not production-scale ResNet or LLM. "
            "No GPU or compiler results. Peak RSS includes interpreter startup."
        ),
    }
    print(json.dumps(result, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
