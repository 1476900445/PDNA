from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import torch
from torch import Tensor, nn
from torch.distributions import Normal
from torch.nn import functional as F

from pdna.nn import MLP, soft_update


class PolicyCandidates(NamedTuple):
    base: Tensor
    edited: Tensor
    selected: Tensor
    q_values: Tensor


class BasePolicy(nn.Module):
    def __init__(self, state_dim: int, encoding_dim: int, hidden_dims: list[int]):
        super().__init__()
        self.net = MLP(state_dim, encoding_dim, hidden_dims)

    def forward(self, state: Tensor) -> Tensor:
        return self.net(state)


class GaussianEditPolicy(nn.Module):
    def __init__(self, state_dim: int, encoding_dim: int, hidden_dims: list[int],
                 std_min: float = 0.01, std_max: float = 0.35):
        super().__init__()
        self.encoding_dim, self.std_min, self.std_max = encoding_dim, std_min, std_max
        self.net = MLP(state_dim + encoding_dim, 2 * encoding_dim, hidden_dims)

    def distribution(self, state: Tensor, base: Tensor) -> Normal:
        mean, raw_scale = self.net(torch.cat((state, base), -1)).chunk(2, -1)
        scale = self.std_min + (self.std_max - self.std_min) * torch.sigmoid(raw_scale)
        return Normal(mean, scale)

    def forward(self, state: Tensor, base: Tensor, deterministic: bool = False) -> tuple[Tensor, Tensor]:
        distribution = self.distribution(state, base)
        delta = distribution.mean if deterministic else distribution.rsample()
        return base + delta, distribution.log_prob(delta).sum(-1, keepdim=True)


class QCritic(nn.Module):
    def __init__(self, state_dim: int, encoding_dim: int, hidden_dims: list[int]):
        super().__init__()
        self.net = MLP(state_dim + encoding_dim, 1, hidden_dims)

    def forward(self, state: Tensor, action_encoding: Tensor) -> Tensor:
        return self.net(torch.cat((state, action_encoding), -1))


@dataclass
class EXPOLosses:
    behavior_cloning: Tensor
    critic: Tensor
    edit_policy: Tensor


class ExpressivePolicyLayer(nn.Module):
    """EXPO base/edit/Q/on-the-fly policy corresponding to Equation (11)."""

    def __init__(self, state_dim: int, classes: int, encoding_dim: int,
                 hidden_dims: list[int], gamma: float = 0.99, target_tau: float = 0.005,
                 edit_std_min: float = 0.01, edit_std_max: float = 0.35):
        super().__init__()
        if encoding_dim != classes + 1:
            raise ValueError("PDNA reserves one extra action dimension for the empty initial state")
        self.classes, self.encoding_dim = classes, encoding_dim
        self.gamma, self.target_tau = gamma, target_tau
        self.base_policy = BasePolicy(state_dim, encoding_dim, hidden_dims)
        self.edit_policy = GaussianEditPolicy(
            state_dim, encoding_dim, hidden_dims, edit_std_min, edit_std_max
        )
        self.critic = QCritic(state_dim, encoding_dim, hidden_dims)
        self.target_critic = QCritic(state_dim, encoding_dim, hidden_dims)
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.target_critic.requires_grad_(False)

    def candidates(self, state: Tensor, deterministic: bool = False) -> PolicyCandidates:
        z_base = self.base_policy(state)
        z_edit, _ = self.edit_policy(state, z_base, deterministic)
        stacked = torch.stack((z_base, z_edit), dim=1)
        expanded_state = state[:, None, :].expand(-1, 2, -1)
        q = self.critic(expanded_state.reshape(-1, state.shape[-1]),
                        stacked.reshape(-1, self.encoding_dim)).reshape(-1, 2)
        selected_index = q.argmax(-1)
        selected = stacked[torch.arange(state.shape[0], device=state.device), selected_index]
        return PolicyCandidates(z_base, z_edit, selected, q)

    def decode(self, encoding: Tensor) -> tuple[Tensor, Tensor]:
        # Empty-state dimension is excluded during inference, as specified by Table 4.
        probabilities = F.softmax(encoding[..., :self.classes], dim=-1)
        return probabilities, probabilities.argmax(-1)

    def forward(self, state: Tensor, deterministic: bool = False) -> tuple[Tensor, Tensor, PolicyCandidates]:
        candidates = self.candidates(state, deterministic)
        probabilities, action = self.decode(candidates.selected)
        return probabilities, action, candidates

    def losses(self, state: Tensor, action_class: Tensor, action_encoding: Tensor,
               reward: Tensor, next_state: Tensor, done: Tensor) -> EXPOLosses:
        base = self.base_policy(state)
        bc = F.cross_entropy(base[:, :self.classes], action_class.long())
        with torch.no_grad():
            next_z = self.candidates(next_state, deterministic=True).selected
            target = reward + self.gamma * (1.0 - done) * self.target_critic(next_state, next_z)
        critic_loss = F.mse_loss(self.critic(state, action_encoding), target)
        edited, log_probability = self.edit_policy(state, base.detach())
        # Q ascent with a small entropy term keeps the Gaussian edit neighborhood expressive.
        edit_loss = -self.critic(state, edited).mean() + 1e-3 * log_probability.mean()
        return EXPOLosses(bc, critic_loss, edit_loss)

    @torch.no_grad()
    def update_target(self) -> None:
        soft_update(self.critic, self.target_critic, self.target_tau)

