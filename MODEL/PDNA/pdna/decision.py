from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ConsistencyDecision:
    intent_index: int
    strategy_index: int
    utility: float
    adjusted: bool
    reason: str | None


class HighLowConsistencyResolver:
    """Maintains EXPO strategy/FQL utility direction consistency described in Section 3.3.

    Exact class-to-direction mappings are dataset semantics rather than a paper formula, so
    the mapping is injected. -1 means concession, 0 neutral, +1 firm/increasing self utility.
    """

    def __init__(self, strategy_directions: dict[int, int], tolerance: float = 0.01):
        self.strategy_directions = strategy_directions
        self.tolerance = tolerance

    def resolve(self, intent: int, strategy: int, previous_utility: float,
                proposed_utility: float) -> ConsistencyDecision:
        direction = self.strategy_directions.get(strategy, 0)
        delta = proposed_utility - previous_utility
        conflict = (direction < 0 and delta > self.tolerance) or (
            direction > 0 and delta < -self.tolerance
        )
        if not conflict:
            return ConsistencyDecision(intent, strategy, proposed_utility, False, None)
        # The paper says the high-level decision is adjusted. We conservatively assign
        # no_strategy (index 7 in Table 3) instead of silently overriding FQL utility.
        return ConsistencyDecision(
            intent, 7, proposed_utility, True,
            f"strategy direction {direction:+d} conflicts with utility delta {delta:+.4f}",
        )

