from __future__ import annotations

from torch import Tensor
from torch.nn import functional as F


def deepseek_finetune_loss(intent_strategy_logits: Tensor, labels: Tensor,
                           generated_politeness: Tensor, target_politeness: Tensor,
                           terminology_loss: Tensor, alpha1: float = 0.4,
                           alpha2: float = 0.2, alpha3: float = 0.4) -> Tensor:
    """Equations (21)-(22) for the domain-fine-tuned DeepSeek-14B corrector."""
    intent_strategy = F.cross_entropy(intent_strategy_logits, labels)
    politeness = F.mse_loss(generated_politeness, target_politeness)
    return alpha1 * intent_strategy + alpha2 * politeness + alpha3 * terminology_loss

