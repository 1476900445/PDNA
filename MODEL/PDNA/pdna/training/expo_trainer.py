from __future__ import annotations

import torch
from torch import Tensor
from torch.optim import Adam

from pdna.expo.policy import ExpressivePolicyLayer


class EXPOTrainer:
    def __init__(self, layer: ExpressivePolicyLayer, base_lr: float = 3e-4,
                 critic_lr: float = 3e-4, edit_lr: float = 1e-4):
        self.layer = layer
        self.base_optimizer = Adam(layer.base_policy.parameters(), lr=base_lr)
        self.critic_optimizer = Adam(layer.critic.parameters(), lr=critic_lr)
        self.edit_optimizer = Adam(layer.edit_policy.parameters(), lr=edit_lr)

    def pretrain_behavior(self, state: Tensor, action_class: Tensor) -> float:
        logits = self.layer.base_policy(state)[:, :self.layer.classes]
        loss = torch.nn.functional.cross_entropy(logits, action_class.long())
        self.base_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.base_optimizer.step()
        return float(loss.detach())

    def online_step(self, state: Tensor, action_class: Tensor, action_encoding: Tensor,
                    reward: Tensor, next_state: Tensor, done: Tensor) -> dict[str, float]:
        losses = self.layer.losses(
            state, action_class, action_encoding, reward, next_state, done
        )
        self.base_optimizer.zero_grad(set_to_none=True)
        losses.behavior_cloning.backward()
        self.base_optimizer.step()

        losses = self.layer.losses(
            state, action_class, action_encoding, reward, next_state, done
        )
        self.critic_optimizer.zero_grad(set_to_none=True)
        losses.critic.backward()
        self.critic_optimizer.step()

        for parameter in self.layer.critic.parameters():
            parameter.requires_grad_(False)
        losses = self.layer.losses(
            state, action_class, action_encoding, reward, next_state, done
        )
        self.edit_optimizer.zero_grad(set_to_none=True)
        losses.edit_policy.backward()
        self.edit_optimizer.step()
        for parameter in self.layer.critic.parameters():
            parameter.requires_grad_(True)
        self.layer.update_target()
        return {
            "behavior_cloning": float(losses.behavior_cloning.detach()),
            "critic": float(losses.critic.detach()),
            "edit_policy": float(losses.edit_policy.detach()),
        }
