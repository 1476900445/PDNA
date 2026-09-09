from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class DomainKnowledgeBase:
    terminology: dict[str, str] = field(default_factory=dict)
    product_constraints: dict[str, Any] = field(default_factory=dict)
    intent_strategy_rules: dict[str, list[str]] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path) -> DomainKnowledgeBase:
        path = Path(path)
        if not path.exists():
            return cls()
        with path.open("r", encoding="utf-8") as stream:
            return cls(**json.load(stream))

    def evidence_for(self, text: str) -> list[str]:
        return [definition for term, definition in self.terminology.items() if term in text]

