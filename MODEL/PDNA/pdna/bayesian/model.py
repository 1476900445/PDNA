from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from pdna.nn import MLP

from .cpt import ConditionalProbabilityTables, DAGBayesInference


@dataclass
class OpponentFeatures:
    personality_probs: Tensor
    intent_probs: Tensor
    strategy_probs: Tensor
    addon_sensitivity: Tensor
    history: Tensor

    @property
    def personality(self) -> Tensor:
        return self.personality_probs.argmax(-1)

    @property
    def intent(self) -> Tensor:
        return self.intent_probs.argmax(-1)

    @property
    def strategy(self) -> Tensor:
        return self.strategy_probs.argmax(-1)


class BayesianFeatureRecognizer(nn.Module):
    """Equations (1)-(7): recurrent encoding, C->I->S inference and add-on sensitivity."""

    def __init__(self, utterance_dim: int, history_dim: int = 128, lexical_dim: int = 128,
                 hidden_dims: list[int] | None = None, lambda_intent: float = 0.5,
                 lambda_strategy: float = 0.5, eps: float = 1e-8):
        super().__init__()
        hidden_dims = hidden_dims or [512, 256]
        self.history_dim, self.lambda_intent, self.lambda_strategy = (
            history_dim, lambda_intent, lambda_strategy
        )
        self.history_encoder = MLP(utterance_dim + history_dim, history_dim, hidden_dims)
        self.lexical_encoder = MLP(utterance_dim, lexical_dim, hidden_dims)
        self.personality_likelihood = nn.Linear(lexical_dim + history_dim, 5)
        self.intent_likelihood = nn.Linear(lexical_dim + history_dim, 23)
        self.strategy_likelihood = nn.Linear(lexical_dim + history_dim, 10)

        # W_CI, W_II, W_IS, W_SS from Equations (4)-(5).
        self.w_ci = nn.Parameter(torch.empty(5, 23))
        self.w_ii = nn.Parameter(torch.eye(23))
        self.w_is = nn.Parameter(torch.empty(23, 10))
        self.w_ss = nn.Parameter(torch.eye(10))
        nn.init.xavier_uniform_(self.w_ci)
        nn.init.xavier_uniform_(self.w_is)

        self.term_strength = nn.Sequential(nn.Linear(lexical_dim, 1), nn.Sigmoid())
        self.intent_influence = nn.Sequential(nn.Linear(23, 1), nn.Sigmoid())
        self.personality_influence = nn.Sequential(nn.Linear(5, 1), nn.Sigmoid())
        self.strategy_influence = nn.Sequential(nn.Linear(10, 1), nn.Sigmoid())
        self.eps = eps

    def initial_history(self, batch_size: int, *, device: torch.device | str) -> Tensor:
        return torch.zeros(batch_size, self.history_dim, device=device)

    @staticmethod
    def _renormalize(probability: Tensor) -> Tensor:
        probability = probability.clamp_min(0.0)
        return probability / probability.sum(-1, keepdim=True).clamp_min(1e-8)

    def forward(self, utterance: Tensor, previous_history: Tensor,
                cpts: ConditionalProbabilityTables, term_weights: Tensor | None = None) -> OpponentFeatures:
        # Equation (1): X_t = MLP_enc(O_t, X_{t-1}).
        history = self.history_encoder(torch.cat([utterance, previous_history], dim=-1))
        lexical = self.lexical_encoder(utterance)
        observation = torch.cat([lexical, history], dim=-1)
        l_c = F.softmax(self.personality_likelihood(observation), dim=-1)
        l_i = F.softmax(self.intent_likelihood(observation), dim=-1)
        l_s = F.softmax(self.strategy_likelihood(observation), dim=-1)

        tables = cpts.to(utterance.device)
        base = DAGBayesInference(tables, self.eps).infer(l_c, l_i, l_s)
        prior_c = tables.personality_prior.expand_as(base.personality)
        delta_c = base.personality - prior_c
        prior_i = base.personality @ tables.intent_given_personality
        delta_i = base.intent - prior_i
        prior_s = base.intent @ tables.strategy_given_intent
        delta_s = base.strategy - prior_s

        # Directed correction, Equations (4)-(5). W matrices align parent/child spaces.
        corrected_i = base.intent + self.lambda_intent * (
            delta_c @ self.w_ci + delta_i @ self.w_ii
        )
        corrected_i = self._renormalize(corrected_i)
        corrected_s = base.strategy + self.lambda_strategy * (
            delta_i @ self.w_is + delta_s @ self.w_ss
        )
        corrected_s = self._renormalize(corrected_s)

        if term_weights is None:
            lexical_component = self.term_strength(lexical)
        else:
            # sum_k w_k * strength(o_addon_k), with normalized w_k.
            normalized = term_weights / term_weights.sum(-1, keepdim=True).clamp_min(self.eps)
            lexical_component = normalized.sum(-1, keepdim=True) * self.term_strength(lexical)
        sensitivity = (
            0.80 * lexical_component
            + 0.10 * self.intent_influence(corrected_i)
            + 0.05 * self.personality_influence(base.personality)
            + 0.05 * self.strategy_influence(corrected_s)
        ).clamp(0.0, 1.0)
        return OpponentFeatures(base.personality, corrected_i, corrected_s, sensitivity, history)
