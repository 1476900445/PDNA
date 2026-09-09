from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor, nn

from .policy import ExpressivePolicyLayer, PolicyCandidates


@dataclass
class EXPOOutput:
    intent_probabilities: Tensor
    intent: Tensor
    strategy_probabilities: Tensor
    strategy: Tensor
    intent_candidates: PolicyCandidates
    strategy_candidates: PolicyCandidates


class EXPOHierarchicalAgent(nn.Module):
    def __init__(self, intent_state_dim: int = 40, strategy_state_dim: int = 45,
                 hidden_dims: list[int] | None = None, gamma: float = 0.99,
                 target_tau: float = 0.005, edit_std_min: float = 0.01,
                 edit_std_max: float = 0.35):
        super().__init__()
        hidden_dims = hidden_dims or [512, 512, 256]
        self.intent_layer = ExpressivePolicyLayer(
            intent_state_dim, 23, 24, hidden_dims, gamma, target_tau,
            edit_std_min, edit_std_max
        )
        self.strategy_layer = ExpressivePolicyLayer(
            strategy_state_dim, 10, 11, hidden_dims, gamma, target_tau,
            edit_std_min, edit_std_max
        )

    def forward(self, intent_state: Tensor, strategy_state_factory,
                deterministic: bool = False) -> EXPOOutput:
        i_prob, intent, i_candidates = self.intent_layer(intent_state, deterministic)
        strategy_state = strategy_state_factory(i_prob)
        s_prob, strategy, s_candidates = self.strategy_layer(strategy_state, deterministic)
        return EXPOOutput(i_prob, intent, s_prob, strategy, i_candidates, s_candidates)
