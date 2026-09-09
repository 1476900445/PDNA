from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .knowledge import DomainKnowledgeBase


class QualityScorer(Protocol):
    def intent_match(self, text: str, target_intent: str) -> float: ...
    def strategy_match(self, text: str, target_strategy: str) -> float: ...
    def politeness(self, text: str) -> float: ...


class CorrectionBackend(Protocol):
    def revise(self, text: str, *, target_intent: str, target_strategy: str,
               target_politeness: float, evidence: list[str],
               deviations: list[str]) -> str: ...


@dataclass(frozen=True)
class InspectionResult:
    text: str
    passed: bool
    corrections: int
    intent_score: float
    strategy_score: float
    politeness_score: float
    unresolved: tuple[str, ...]


class InspectionCorrectionLoop:
    """Equations (21)-(22) inference loop with paper thresholds and at most 3 revisions."""

    def __init__(self, scorer: QualityScorer, backend: CorrectionBackend,
                 knowledge: DomainKnowledgeBase, intent_threshold: float = 0.85,
                 strategy_threshold: float = 0.80, politeness_tolerance: float = 0.05,
                 max_corrections: int = 3):
        self.scorer, self.backend, self.knowledge = scorer, backend, knowledge
        self.intent_threshold, self.strategy_threshold = intent_threshold, strategy_threshold
        self.politeness_tolerance, self.max_corrections = politeness_tolerance, max_corrections

    def inspect(self, text: str, target_intent: str, target_strategy: str,
                target_politeness: float) -> InspectionResult:
        corrections = 0
        while True:
            i_score = self.scorer.intent_match(text, target_intent)
            s_score = self.scorer.strategy_match(text, target_strategy)
            p_score = self.scorer.politeness(text)
            deviations = []
            if i_score < self.intent_threshold:
                deviations.append("intent")
            if s_score < self.strategy_threshold:
                deviations.append("strategy")
            if abs(p_score - target_politeness) > self.politeness_tolerance:
                deviations.append("tone")
            if not deviations or corrections >= self.max_corrections:
                return InspectionResult(
                    text, not deviations, corrections, i_score, s_score, p_score, tuple(deviations)
                )
            text = self.backend.revise(
                text, target_intent=target_intent, target_strategy=target_strategy,
                target_politeness=target_politeness, evidence=self.knowledge.evidence_for(text),
                deviations=deviations,
            )
            corrections += 1

