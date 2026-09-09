from __future__ import annotations

import torch
from torch import Tensor, nn


class TerminologyCNN(nn.Module):
    """Table 6 three-layer CNN for 49-dimensional terminology vectors."""

    def __init__(self, input_dim: int = 49, output_dim: int = 128,
                 channels: tuple[int, int, int] = (64, 96, 128)):
        super().__init__()
        c1, c2, c3 = channels
        self.net = nn.Sequential(
            nn.Conv1d(1, c1, 3, padding=1), nn.GELU(), nn.BatchNorm1d(c1),
            nn.Conv1d(c1, c2, 3, padding=1), nn.GELU(), nn.BatchNorm1d(c2),
            nn.Conv1d(c2, c3, 3, padding=1), nn.GELU(), nn.BatchNorm1d(c3),
            nn.AdaptiveMaxPool1d(1),
        )
        self.projection = nn.Linear(c3, output_dim)

    def forward(self, terminology: Tensor) -> Tensor:
        return self.projection(self.net(terminology.unsqueeze(1)).squeeze(-1))


class InteractionFeatureEncoder(nn.Module):
    def __init__(self, input_dim: int, output_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.LayerNorm(256), nn.GELU(),
            nn.Linear(256, output_dim), nn.LayerNorm(output_dim),
        )

    def forward(self, features: Tensor) -> Tensor:
        return self.net(features)


class HistoryBiLSTM(nn.Module):
    def __init__(self, input_dim: int, output_dim: int = 128):
        super().__init__()
        if output_dim % 2:
            raise ValueError("BiLSTM output dimension must be even")
        self.lstm = nn.LSTM(input_dim, output_dim // 2, num_layers=1,
                            batch_first=True, bidirectional=True)

    def forward(self, history: Tensor, mask: Tensor | None = None) -> Tensor:
        output, _ = self.lstm(history)
        if mask is None:
            return output.mean(1)
        weights = mask.float().unsqueeze(-1)
        return (output * weights).sum(1) / weights.sum(1).clamp_min(1.0)


class GatedFusion(nn.Module):
    """Paper equation g=sigmoid(Wg X+b), X=g*(Wp X+b), mapped to 128."""

    def __init__(self, input_dim: int = 384, output_dim: int = 128):
        super().__init__()
        self.gate = nn.Linear(input_dim, output_dim)
        self.projection = nn.Linear(input_dim, output_dim)

    def forward(self, *representations: Tensor) -> Tensor:
        concatenated = torch.cat(representations, -1)
        gate = torch.sigmoid(self.gate(concatenated))
        return gate * self.projection(concatenated)

