from __future__ import annotations

from collections.abc import Iterable
from itertools import pairwise

import torch
from torch import Tensor, nn


class MLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: Iterable[int],
                 activation: type[nn.Module] = nn.SiLU, layer_norm: bool = True):
        super().__init__()
        dims = [input_dim, *hidden_dims]
        layers: list[nn.Module] = []
        for left, right in pairwise(dims):
            layers.append(nn.Linear(left, right))
            if layer_norm:
                layers.append(nn.LayerNorm(right))
            layers.append(activation())
        layers.append(nn.Linear(dims[-1], output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.network(x)


@torch.no_grad()
def soft_update(source: nn.Module, target: nn.Module, tau: float) -> None:
    for source_parameter, target_parameter in zip(source.parameters(), target.parameters()):
        target_parameter.lerp_(source_parameter, tau)
