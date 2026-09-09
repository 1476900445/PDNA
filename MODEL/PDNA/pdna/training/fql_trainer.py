from __future__ import annotations

from dataclasses import asdict

import torch
from torch.optim import Adam

from pdna.data import MixedReplayBuffer
from pdna.fql import FQLAgent


class FQLTrainer:
    """Separate updates avoid unintended gradients across flow, critic and one-step policy."""

    def __init__(self, model: FQLAgent, replay: MixedReplayBuffer, learning_rate: float = 3e-4,
                 batch_size: int = 256, max_grad_norm: float = 10.0):
        self.model, self.replay, self.batch_size = model, replay, batch_size
        self.flow_optimizer = Adam(model.flow.parameters(), lr=learning_rate)
        self.q_optimizer = Adam(model.q.parameters(), lr=learning_rate)
        self.policy_optimizer = Adam(model.one_step.parameters(), lr=learning_rate)
        self.max_grad_norm = max_grad_norm

    @staticmethod
    def _step(loss, optimizer, parameters, max_grad_norm: float) -> None:
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, max_grad_norm)
        optimizer.step()

    def train_step(self, device: torch.device | str) -> dict[str, float]:
        batch = self.replay.sample(self.batch_size).to(device)

        losses = self.model.losses(batch)
        self._step(losses.flow_matching, self.flow_optimizer,
                   self.model.flow.parameters(), self.max_grad_norm)

        losses = self.model.losses(batch)
        self._step(losses.critic, self.q_optimizer, self.model.q.parameters(), self.max_grad_norm)

        for parameter in self.model.q.parameters():
            parameter.requires_grad_(False)
        losses = self.model.losses(batch)
        self._step(losses.one_step_total, self.policy_optimizer,
                   self.model.one_step.parameters(), self.max_grad_norm)
        for parameter in self.model.q.parameters():
            parameter.requires_grad_(True)
        self.model.update_target()
        return {key: float(value.detach()) for key, value in asdict(losses).items()}

