from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import Dataset

from .schemas import Dialogue, NegotiationTurn, Offer, Transition


def _offer(raw: dict[str, Any] | None) -> Offer | None:
    return None if raw is None else Offer(**raw)


def _turn(raw: dict[str, Any]) -> NegotiationTurn:
    raw = dict(raw)
    raw["offer"] = _offer(raw.get("offer"))
    return NegotiationTurn(**raw)


class CloudDialogueDataset(Dataset[Dialogue]):
    """JSONL reader for Cloud-Data/IND/NegoChat annotations under ../LLM_DATA.

    No placeholder samples are generated: an empty or absent file yields an empty dataset
    (unless ``required=True``), allowing the engineering project to be initialized before
    the private data are copied into place.
    """

    def __init__(self, root: str | Path, split: str, required: bool = False):
        self.path = Path(root) / f"{split}.jsonl"
        if required and not self.path.exists():
            raise FileNotFoundError(self.path)
        self.records = list(self._read()) if self.path.exists() else []

    def _read(self) -> Iterator[Dialogue]:
        with self.path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    raw = json.loads(line)
                    yield Dialogue(
                        dialogue_id=str(raw["dialogue_id"]),
                        scenario_id=str(raw["scenario_id"]),
                        turns=[_turn(turn) for turn in raw["turns"]],
                        agreement=raw.get("agreement"),
                        final_utility=raw.get("final_utility"),
                    )
                except Exception as exc:
                    raise ValueError(f"Invalid dialogue at {self.path}:{line_number}") from exc

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dialogue:
        return self.records[index]


class NegotiationTransitionDataset(Dataset[Transition]):
    """Reads FQL/EXPO transitions from ../NEGMAS_DATA/{split}.jsonl."""

    def __init__(self, root: str | Path, split: str, state_dim: int, action_dim: int,
                 required: bool = False):
        self.path = Path(root) / f"{split}.jsonl"
        self.state_dim, self.action_dim = state_dim, action_dim
        if required and not self.path.exists():
            raise FileNotFoundError(self.path)
        self.records = list(self._read()) if self.path.exists() else []

    def _read(self) -> Iterator[Transition]:
        with self.path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                raw = json.loads(line)
                transition = Transition(**raw)
                if len(transition.state) != self.state_dim or len(transition.next_state) != self.state_dim:
                    raise ValueError(f"State dimension mismatch at {self.path}:{line_number}")
                if len(transition.action) != self.action_dim:
                    raise ValueError(f"Action dimension mismatch at {self.path}:{line_number}")
                yield transition

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Transition:
        return self.records[index]


def collate_dialogue(batch: list[dict[str, Any]]) -> dict[str, Tensor | list[str]]:
    """Pads already-vectorized examples while retaining source utterances for inspection."""
    if not batch:
        raise ValueError("Cannot collate an empty batch")
    result: dict[str, Tensor | list[str]] = {}
    for key in batch[0]:
        values = [item[key] for item in batch]
        result[key] = torch.stack(values) if isinstance(values[0], Tensor) else values
    return result

