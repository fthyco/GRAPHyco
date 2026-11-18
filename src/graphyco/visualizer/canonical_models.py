"""Self-contained small PyTorch models for GRAPHyco smoke benchmarks.

These are *toy graph motifs*, not checkpoints or reference implementations
of full torchvision architectures. They require no external validation files.
"""
from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch import nn
from torch.nn import functional as F


class MLP(nn.Module):
    """Depth-N vector MLP with ReLU between linear layers."""

    def __init__(self, depth: int = 4, width: int = 64):
        super().__init__()
        if depth < 1 or width < 1:
            raise ValueError("depth and width must be positive")
        blocks = []
        for i in range(depth):
            blocks.append(nn.Linear(width, width))
            if i < depth - 1:
                blocks.append(nn.ReLU())
        self.layers = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class ResNetBlock(nn.Module):
    """Vector-space residual add; not a full convolutional ResNet."""

    def __init__(self, dim: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        return F.relu(self.fc2(F.relu(self.fc1(x))) + shortcut)


class DenseBlock(nn.Module):
    """Three dense connections with explicit concatenation."""

    def __init__(self, dim: int = 32):
        super().__init__()
        self.layer1 = nn.Linear(dim, dim)
        self.layer2 = nn.Linear(2 * dim, dim)
        self.layer3 = nn.Linear(3 * dim, dim)
        self.project = nn.Linear(4 * dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = F.relu(self.layer1(x))
        b = F.relu(self.layer2(torch.cat((x, a), dim=-1)))
        c = F.relu(self.layer3(torch.cat((x, a, b), dim=-1)))
        return self.project(torch.cat((x, a, b, c), dim=-1))


class UNetToy(nn.Module):
    """Vector down/up projection with a skip; not an image U-Net."""

    def __init__(self, dim: int = 32):
        super().__init__()
        self.encoder = nn.Linear(dim, dim)
        self.down = nn.Linear(dim, dim // 2)
        self.bottleneck = nn.Linear(dim // 2, dim // 2)
        self.up = nn.Linear(dim // 2, dim)
        self.decoder = nn.Linear(dim * 2, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip = F.relu(self.encoder(x))
        compressed = F.relu(self.down(skip))
        latent = F.relu(self.bottleneck(compressed))
        restored = F.relu(self.up(latent))
        return self.decoder(torch.cat((skip, restored), dim=-1))


class SelfAttentionToy(nn.Module):
    """Batch-first self-attention + residual LayerNorm."""

    def __init__(self, dim: int = 64, heads: int = 4):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attention, _ = self.attn(x, x, x, need_weights=False)
        return self.norm(x + attention)


def get_canonical_models() -> Dict[str, Tuple[nn.Module, torch.Tensor]]:
    """Return fresh toy models and compatible inputs without changing global RNG."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        return {
            "Sequential MLP": (MLP(depth=4, width=64), torch.randn(8, 64)),
            "ResNet Block": (ResNetBlock(dim=64), torch.randn(8, 64)),
            "DenseBlock": (DenseBlock(dim=32), torch.randn(8, 32)),
            "U-Net Toy": (UNetToy(dim=32), torch.randn(8, 32)),
            "Self-Attention Toy": (
                SelfAttentionToy(dim=64, heads=4), torch.randn(8, 4, 64),
            ),
        }
