from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


def _normalize_counts(counts: Tensor, alpha: float, dim: int = -1) -> Tensor:
    smoothed = counts.float() + alpha
    return smoothed / smoothed.sum(dim=dim, keepdim=True).clamp_min(1e-12)


@dataclass
class ConditionalProbabilityTables:
    """CPTs for the paper's C -> I -> S -> O DAG (Equation 2)."""

    personality_prior: Tensor                 # [C]
    intent_given_personality: Tensor          # [C, I]
    strategy_given_intent: Tensor             # [I, S]
    observation_given_strategy: Tensor | None # learned neural likelihood is used if absent

    @classmethod
    def fit(cls, personality: Tensor, intent: Tensor, strategy: Tensor,
            personality_count: int = 5, intent_count: int = 23,
            strategy_count: int = 10, alpha: float = 1.0) -> ConditionalProbabilityTables:
        if not (len(personality) == len(intent) == len(strategy)):
            raise ValueError("CPT labels must contain the same number of samples")
        c = torch.bincount(personality, minlength=personality_count)
        ci = torch.zeros(personality_count, intent_count)
        is_ = torch.zeros(intent_count, strategy_count)
        for c_idx, i_idx, s_idx in zip(personality.tolist(), intent.tolist(), strategy.tolist()):
            ci[c_idx, i_idx] += 1
            is_[i_idx, s_idx] += 1
        return cls(
            personality_prior=_normalize_counts(c, alpha),
            intent_given_personality=_normalize_counts(ci, alpha),
            strategy_given_intent=_normalize_counts(is_, alpha),
            observation_given_strategy=None,
        )

    @classmethod
    def uniform(cls, personality_count: int = 5, intent_count: int = 23,
                strategy_count: int = 10) -> ConditionalProbabilityTables:
        return cls(
            torch.full((personality_count,), 1.0 / personality_count),
            torch.full((personality_count, intent_count), 1.0 / intent_count),
            torch.full((intent_count, strategy_count), 1.0 / strategy_count),
            None,
        )

    def to(self, device: torch.device | str) -> ConditionalProbabilityTables:
        return ConditionalProbabilityTables(
            self.personality_prior.to(device), self.intent_given_personality.to(device),
            self.strategy_given_intent.to(device),
            None if self.observation_given_strategy is None else self.observation_given_strategy.to(device),
        )

    def state_dict(self) -> dict[str, Tensor | None]:
        return self.__dict__.copy()


@dataclass
class BayesianPosterior:
    personality: Tensor
    intent: Tensor
    strategy: Tensor


class DAGBayesInference:
    """Differentiable categorical inference constrained by empirical CPTs."""

    def __init__(self, tables: ConditionalProbabilityTables, eps: float = 1e-8):
        self.tables, self.eps = tables, eps

    def infer(self, lexical_c: Tensor, lexical_i: Tensor, lexical_s: Tensor) -> BayesianPosterior:
        # Lexical heads represent P(O_t | C), P(O_t | I), P(O_t | S).
        prior_c = self.tables.personality_prior.expand_as(lexical_c)
        post_c = prior_c * lexical_c.clamp_min(self.eps)
        post_c = post_c / post_c.sum(-1, keepdim=True).clamp_min(self.eps)

        prior_i = post_c @ self.tables.intent_given_personality
        post_i = prior_i * lexical_i.clamp_min(self.eps)
        post_i = post_i / post_i.sum(-1, keepdim=True).clamp_min(self.eps)

        prior_s = post_i @ self.tables.strategy_given_intent
        post_s = prior_s * lexical_s.clamp_min(self.eps)
        post_s = post_s / post_s.sum(-1, keepdim=True).clamp_min(self.eps)
        return BayesianPosterior(post_c, post_i, post_s)

