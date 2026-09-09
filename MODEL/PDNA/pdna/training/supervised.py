from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F

from pdna.bayesian import BayesianFeatureRecognizer, ConditionalProbabilityTables
from pdna.setformer import SeTformer, SeTformerBatch


def bayesian_multitask_loss(model: BayesianFeatureRecognizer, utterance: Tensor,
                            previous_history: Tensor, cpt: ConditionalProbabilityTables,
                            personality_label: Tensor, intent_label: Tensor,
                            strategy_label: Tensor, addon_label: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
    prediction = model(utterance, previous_history, cpt)
    terms = {
        "personality": F.nll_loss(prediction.personality_probs.clamp_min(1e-8).log(),
                                  personality_label.long()),
        "intent": F.nll_loss(prediction.intent_probs.clamp_min(1e-8).log(), intent_label.long()),
        "strategy": F.nll_loss(prediction.strategy_probs.clamp_min(1e-8).log(),
                               strategy_label.long()),
        "addon": F.l1_loss(prediction.addon_sensitivity.squeeze(-1), addon_label.float()),
    }
    return sum(terms.values()), terms


class SeTformerTrainer:
    def __init__(self, model: SeTformer, learning_rate: float = 3e-4):
        self.model = model
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    def train_step(self, batch: SeTformerBatch) -> dict[str, float]:
        components = self.model.losses(batch)
        self.optimizer.zero_grad(set_to_none=True)
        components.total.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        return {key: float(value.detach()) for key, value in components.__dict__.items()}
