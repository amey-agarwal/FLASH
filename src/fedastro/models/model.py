"""Small classifiers. Capacity is deliberately low (CLAUDE.md rule 6).

GroupNorm is used instead of BatchNorm: Opacus rejects BatchNorm (Phase 5) and batch
statistics also misbehave under tiny, label-skewed silo batches.
"""

from __future__ import annotations

import torch
from torch import nn

from fedastro.config import ModelConfig


class MLP(nn.Module):
    def __init__(self, in_features: int, hidden: tuple[int, ...], n_classes: int, dropout: float):
        super().__init__()
        layers: list[nn.Module] = []
        width = in_features
        for h in hidden:
            layers += [nn.Linear(width, h), nn.ReLU(), nn.Dropout(dropout)]
            width = h
        layers.append(nn.Linear(width, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CNN1D(nn.Module):
    """Conv blocks (Conv1d -> GroupNorm -> ReLU -> MaxPool), global average pool, linear head."""

    def __init__(self, in_channels: int, channels: tuple[int, ...], kernel_size: int,
                 n_classes: int, dropout: float):
        super().__init__()
        blocks: list[nn.Module] = []
        c_in = in_channels
        for c in channels:
            blocks += [
                nn.Conv1d(c_in, c, kernel_size, padding=kernel_size // 2),
                nn.GroupNorm(min(4, c), c),
                nn.ReLU(),
                nn.MaxPool1d(2),
            ]
            c_in = c
        self.features = nn.Sequential(*blocks)
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(c_in, n_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.features(x)
        return self.head(h.mean(dim=2, keepdim=True))


def build_model(cfg: ModelConfig, input_shape: tuple[int, ...], n_classes: int) -> nn.Module:
    """input_shape excludes the batch dimension: (n_features,) or (n_channels, length)."""
    if cfg.arch == "mlp":
        if len(input_shape) != 1:
            raise ValueError(f"mlp expects 1-D input, got shape {input_shape}")
        return MLP(input_shape[0], cfg.hidden, n_classes, cfg.dropout)
    if cfg.arch == "cnn1d":
        if len(input_shape) != 2:
            raise ValueError(f"cnn1d expects (channels, length) input, got shape {input_shape}")
        return CNN1D(input_shape[0], cfg.cnn_channels, cfg.kernel_size, n_classes, cfg.dropout)
    raise ValueError(f"unknown arch {cfg.arch!r}")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
