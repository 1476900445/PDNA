from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from pdna.data.replay import TransitionBatch
from pdna.nn import MLP, soft_update


class ConditionalFlowVelocity(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: list[int]):
        super().__init__()
        self.net = MLP(state_dim + action_dim + 1, action_dim, hidden_dims)

    def forward(self, state: Tensor, action_tau: Tensor, tau: Tensor) -> Tensor:
        return self.net(torch.cat((state, action_tau, tau), dim=-1))


class OneStepPolicy(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: list[int]):
        super().__init__()
        self.net = MLP(state_dim + action_dim, action_dim, hidden_dims)

    def forward(self, state: Tensor, noise: Tensor) -> Tensor:
        lac = torch.cat((state, noise), -1)
        return self.net(lac)


class QEnsemble(nn.Module):
    def __init__(self, count: int, state_dim: int, action_dim: int, hidden_dims: list[int]):
        super().__init__()
        self.networks = nn.ModuleList(
            [MLP(state_dim + action_dim, 1, hidden_dims) for _ in range(count)]
        )

    def forward(self, state: Tensor, action: Tensor) -> Tensor:
        x = torch.cat((state, action), -1)
        return torch.stack([network(x) for network in self.networks], dim=0)

    def conservative(self, state: Tensor, action: Tensor) -> Tensor:
        return self(state, action).min(dim=0).values


@dataclass
class FQLLosses:
    flow_matching: Tensor
    critic: Tensor
    distillation: Tensor
    q_guidance: Tensor
    one_step_total: Tensor


class FQLAgent(nn.Module):
    """Flow Q-Learning implementation of Equations (15)-(17)."""

    def __init__(self, state_dim: int = 7, action_dim: int = 1,
                 hidden_dims: list[int] | None = None, q_networks: int = 2,
                 gamma: float = 0.99, target_tau: float = 0.005, flow_steps: int = 10,
                 bc_coefficient: float = 1.0, q_guidance_coefficient: float = 1.0,
                 action_min: float = 0.0, action_max: float = 1.0):
        super().__init__()
        hidden_dims = hidden_dims or [512] * 4
        self.state_dim, self.action_dim = state_dim, action_dim
        self.gamma, self.target_tau, self.flow_steps = gamma, target_tau, flow_steps
        self.bc_coefficient = bc_coefficient
        self.q_guidance_coefficient = q_guidance_coefficient
        self.action_min, self.action_max = action_min, action_max
        self.flow = ConditionalFlowVelocity(state_dim, action_dim, hidden_dims)
        self.one_step = OneStepPolicy(state_dim, action_dim, hidden_dims)
        self.q = QEnsemble(q_networks, state_dim, action_dim, hidden_dims)
        self.target_q = copy.deepcopy(self.q).requires_grad_(False)

    def sample_flow(self, state: Tensor, noise: Tensor | None = None) -> Tensor:
        action = (torch.randn(state.shape[0], self.action_dim, device=state.device)
                  if noise is None else noise)
        dt = 1.0 / self.flow_steps
        for step in range(self.flow_steps):
            tau = torch.full((state.shape[0], 1), step * dt, device=state.device)
            action = action + dt * self.flow(state, action, tau)
        return action.clamp(self.action_min, self.action_max)

    def act(self, state: Tensor, noise: Tensor | None = None) -> Tensor:
        noise = (torch.randn(state.shape[0], self.action_dim, device=state.device)
                 if noise is None else noise)
        scale = self.action_max - self.action_min
        return self.one_step(state, noise).sigmoid().mul(scale).add(self.action_min)

    def losses(self, batch: TransitionBatch) -> FQLLosses:
        state, action = batch.states, batch.actions
        noise = torch.randn_like(action)
        tau = torch.rand(action.shape[0], 1, device=action.device)
        action_tau = (1.0 - tau) * noise + tau * action
        flow_loss = F.mse_loss(self.flow(state, action_tau, tau), action - noise)

        with torch.no_grad():
            next_action = self.act(batch.next_states)
            target = batch.rewards + self.gamma * (1.0 - batch.dones) * self.target_q.conservative(
                batch.next_states, next_action
            )
        q_predictions = self.q(state, action)
        critic_loss = sum(F.mse_loss(prediction, target) for prediction in q_predictions)

        shared_noise = torch.randn_like(action)
        with torch.no_grad():
            flow_action = self.sample_flow(state, shared_noise)
        one_step_action = self.act(state, shared_noise)
        distillation = F.mse_loss(one_step_action, flow_action)
        q_guidance = -self.q.conservative(state, one_step_action).mean()
        policy_loss = self.bc_coefficient * distillation + self.q_guidance_coefficient * q_guidance
        return FQLLosses(flow_loss, critic_loss, distillation, q_guidance, policy_loss)

    @torch.no_grad()
    def update_target(self) -> None:
        soft_update(self.q, self.target_q, self.target_tau)
