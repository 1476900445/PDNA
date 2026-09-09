from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Outcome:
    values: dict[str, Any]


class InverseUtilityMapper:
    """Equation (13): interval search with estimated opponent utility tie-breaking."""

    def __init__(self, outcome_source: Iterable[Outcome] | Callable[[], Iterable[Outcome]],
                 self_utility: Callable[[Outcome], float],
                 opponent_utility_estimator: Callable[[Outcome], float], tolerance: float = 0.01):
        self.outcome_source = outcome_source
        self.self_utility = self_utility
        self.opponent_utility_estimator = opponent_utility_estimator
        self.tolerance = tolerance

    def _outcomes(self) -> list[Outcome]:
        values = self.outcome_source() if callable(self.outcome_source) else self.outcome_source
        return list(values)

    def map(self, suggested_utility: float) -> Outcome:
        all_outcomes = self._outcomes()
        if not all_outcomes:
            raise LookupError("Outcome space is empty")
        upper = min(1.0, suggested_utility + self.tolerance)
        candidates = [o for o in all_outcomes if suggested_utility <= self.self_utility(o) <= upper]
        if not candidates:
            distance = min(abs(self.self_utility(o) - suggested_utility) for o in all_outcomes)
            candidates = [o for o in all_outcomes
                          if abs(abs(self.self_utility(o) - suggested_utility) - distance) < 1e-12]
        return max(candidates, key=self.opponent_utility_estimator)

