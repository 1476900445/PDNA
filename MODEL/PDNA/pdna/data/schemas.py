from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass(frozen=True)
class Offer:
    issues: dict[str, Any]
    self_utility: float
    opponent_utility_estimate: float | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.self_utility <= 1.0:
            raise ValueError("self_utility must lie in [0, 1]")


@dataclass(frozen=True)
class NegotiationTurn:
    turn_id: int
    speaker: Literal["self", "opponent"]
    utterance: str
    personality: list[float] | None = None
    intent: int | None = None
    strategy: int | None = None
    politeness: float | None = None
    addon_sensitivity: float | None = None
    offer: Offer | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Dialogue:
    dialogue_id: str
    scenario_id: str
    turns: list[NegotiationTurn]
    agreement: bool | None = None
    final_utility: float | None = None


@dataclass(frozen=True)
class Transition:
    state: list[float]
    action: list[float]
    reward: float
    next_state: list[float]
    done: bool

