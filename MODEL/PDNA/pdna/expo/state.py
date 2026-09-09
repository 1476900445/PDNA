from __future__ import annotations

import torch
from torch import Tensor, nn


class EXPOStateBuilder(nn.Module):
    """Builds Equation (8) and (10) states with paper-specified 40/45 dimensions.

    C is represented by a 5-way OCEAN posterior; I and S are categorical
    probability vectors. The raw concatenations (109 and 132 dimensions) are projected
    to the state dimensions reported in Table 4.
    """

    INTENT_RAW_DIM = 2 * 5 + 3 * 23 + 3 * 10
    STRATEGY_RAW_DIM = INTENT_RAW_DIM + 23

    def __init__(self, intent_state_dim: int = 40, strategy_state_dim: int = 45):
        super().__init__()
        self.intent_projection = nn.Sequential(
            nn.Linear(self.INTENT_RAW_DIM, 128), nn.LayerNorm(128), nn.SiLU(),
            nn.Linear(128, intent_state_dim),
        )
        self.strategy_projection = nn.Sequential(
            nn.Linear(self.STRATEGY_RAW_DIM, 128), nn.LayerNorm(128), nn.SiLU(),
            nn.Linear(128, strategy_state_dim),
        )

    def intent_state(self, c_opp_prev: Tensor, c_opp: Tensor, i_opp_prev: Tensor,
                     i_our_prev: Tensor, i_opp: Tensor, s_opp_prev: Tensor,
                     s_our_prev: Tensor, s_opp: Tensor) -> Tensor:
        raw = torch.cat((c_opp_prev, c_opp, i_opp_prev, i_our_prev, i_opp,
                         s_opp_prev, s_our_prev, s_opp), dim=-1)
        return self.intent_projection(raw)

    def strategy_state(self, intent_state_inputs: tuple[Tensor, ...], current_intent: Tensor) -> Tensor:
        raw = torch.cat((*intent_state_inputs, current_intent), dim=-1)
        return self.strategy_projection(raw)


def one_hot(index: Tensor, classes: int) -> Tensor:
    return torch.nn.functional.one_hot(index.long(), classes).float()
