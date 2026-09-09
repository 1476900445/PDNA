from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    pass


class AttrDict(dict):
    """Recursive mapping with attribute access and no silent missing keys."""

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc


def _convert(value: Any) -> Any:
    if isinstance(value, dict):
        return AttrDict({k: _convert(v) for k, v in value.items()})
    if isinstance(value, list):
        return [_convert(v) for v in value]
    return value


@dataclass(frozen=True)
class PDNAConfig:
    raw: AttrDict
    project_root: Path

    @property
    def paths(self) -> AttrDict:
        return self.raw.paths

    def resolve_path(self, name: str) -> Path:
        return (self.project_root / self.paths[name]).resolve()

    def validate(self) -> None:
        labels = self.raw.labels
        if len(labels.personalities) != 5:
            raise ConfigError("PDNA requires five OCEAN personality dimensions")
        if len(labels.intents) != 23:
            raise ConfigError("PDNA requires exactly 23 intent classes")
        if len(labels.strategies) != 10:
            raise ConfigError("PDNA requires exactly 10 strategy classes")
        if self.raw.expo.intent.state_dim != 40 or self.raw.expo.strategy.state_dim != 45:
            raise ConfigError("Paper-specified EXPO state dimensions are 40 and 45")
        if self.raw.fql.state_dim != 7 or self.raw.fql.action_dim != 1:
            raise ConfigError("Paper-specified FQL dimensions are state=7, action=1")
        ratio = self.raw.fql.offline_ratio + self.raw.fql.online_ratio
        if abs(ratio - 1.0) > 1e-9:
            raise ConfigError("FQL offline/online ratios must sum to one")


def load_config(path: str | Path = "configs/pdna.yaml") -> PDNAConfig:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as stream:
        raw: Mapping[str, Any] = yaml.safe_load(stream)
    config = PDNAConfig(_convert(dict(raw)), config_path.parent.parent)
    config.validate()
    return config

