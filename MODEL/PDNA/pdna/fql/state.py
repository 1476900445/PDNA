from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor


def build_fql_state(opponent_offer_utilities: Sequence[float], self_offer_utilities: Sequence[float],
                    current_turn: int, max_turns: int,
                    device: str | torch.device = "cpu") -> Tensor:
    """Equation (12): three opponent/self offer-utility pairs plus turn ratio."""
    if max_turns <= 0:
        raise ValueError("max_turns must be positive")
    opp = ([0.0] * 3 + list(opponent_offer_utilities))[-3:]
    own = ([0.0] * 3 + list(self_offer_utilities))[-3:]
    values = [opp[0], own[0], opp[1], own[1], opp[2], own[2], current_turn / max_turns]
    return torch.tensor(values, dtype=torch.float32, device=device)


def price_utility(offered_price: float, reserved_price: float, initial_price: float) -> float:
    """Equation (14), clipped to the normalized utility range."""
    denominator = initial_price - reserved_price
    if abs(denominator) < 1e-12:
        raise ValueError("InitialPrice and ReservedPrice must differ")
    return min(1.0, max(0.0, (offered_price - reserved_price) / denominator))


def negotiation_reward(agreement: bool, interrupted_or_timed_out: bool,
                       agreement_utility: float = 0.0) -> float:
    """Equation (9), shared by EXPO and FQL."""
    if agreement:
        return float(agreement_utility)
    if interrupted_or_timed_out:
        return -1.0
    return 0.0
