from __future__ import annotations

from dataclasses import dataclass
from threading import RLock

import numpy as np
import torch
from torch import Tensor


@dataclass
class TransitionBatch:
    states: Tensor
    actions: Tensor
    rewards: Tensor
    next_states: Tensor
    dones: Tensor

    def to(self, device: torch.device | str) -> TransitionBatch:
        return TransitionBatch(*(value.to(device) for value in self.__dict__.values()))

    @classmethod
    def concat(cls, batches: list[TransitionBatch]) -> TransitionBatch:
        return cls(*(torch.cat([getattr(b, key) for b in batches]) for key in cls.__annotations__))


class ReplayBuffer:
    def __init__(self, capacity: int, state_dim: int, action_dim: int, seed: int = 42):
        self.capacity, self.state_dim, self.action_dim = capacity, state_dim, action_dim
        self.states = np.empty((capacity, state_dim), np.float32)
        self.actions = np.empty((capacity, action_dim), np.float32)
        self.rewards = np.empty((capacity, 1), np.float32)
        self.next_states = np.empty((capacity, state_dim), np.float32)
        self.dones = np.empty((capacity, 1), np.float32)
        self.size = self.cursor = 0
        self.rng, self.lock = np.random.default_rng(seed), RLock()

    def add(self, state, action, reward, next_state, done) -> None:
        with self.lock:
            i = self.cursor
            self.states[i], self.actions[i] = state, action
            self.rewards[i], self.next_states[i], self.dones[i] = reward, next_state, done
            self.cursor = (i + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)

    def sample(self, count: int) -> TransitionBatch:
        with self.lock:
            if count > self.size:
                raise ValueError(f"Requested {count} transitions from buffer of size {self.size}")
            index = self.rng.choice(self.size, count, replace=False)
            arrays = (self.states[index], self.actions[index], self.rewards[index],
                      self.next_states[index], self.dones[index])
            return TransitionBatch(*(torch.from_numpy(a.copy()) for a in arrays))

    def __len__(self) -> int:
        return self.size


class MixedReplayBuffer:
    """Exact paper-specified 8:2 offline/online sampler for FQL fine-tuning."""

    def __init__(self, offline: ReplayBuffer, online: ReplayBuffer,
                 offline_ratio: float = 0.8):
        self.offline, self.online, self.offline_ratio = offline, online, offline_ratio

    def sample(self, batch_size: int) -> TransitionBatch:
        online_count = round(batch_size * (1.0 - self.offline_ratio))
        offline_count = batch_size - online_count
        if len(self.online) < online_count:
            offline_count, online_count = batch_size, 0
        batches = [self.offline.sample(offline_count)]
        if online_count:
            batches.append(self.online.sample(online_count))
        batch = TransitionBatch.concat(batches)
        permutation = torch.randperm(batch_size)
        return TransitionBatch(*(v[permutation] for v in batch.__dict__.values()))
