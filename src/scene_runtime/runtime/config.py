"""YAML configuration loading with optional extends."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge override into base."""
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML file; resolve ``extends`` relative to the file directory."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data: dict[str, Any] = yaml.safe_load(f) or {}

    extends = data.pop("extends", None)
    if extends:
        parent_path = path.parent / extends
        parent = load_yaml(parent_path)
        data = _deep_merge(parent, data)
    return data


def load_config(config_path: Path, strategy_name: str | None = None) -> dict[str, Any]:
    """
    Load main config and optionally merge a strategy YAML.

    Parameters
    ----------
    config_path:
        Path to e.g. ``configs/raspberry_pi4.yaml``.
    strategy_name:
        Strategy key matching ``configs/strategies/<name>.yaml``.
    """
    config = load_yaml(config_path)
    if strategy_name:
        repo_root = _find_repo_root(config_path)
        strategy_path = repo_root / "configs" / "strategies" / f"{strategy_name}.yaml"
        if strategy_path.exists():
            strategy_cfg = load_yaml(strategy_path)
            config = _deep_merge(config, strategy_cfg)
        config.setdefault("project", {})["strategy"] = strategy_name
    return config


def _find_repo_root(start: Path) -> Path:
    """Walk up from start to find directory containing ``configs/``."""
    current = Path(start).resolve()
    if current.is_file():
        current = current.parent
    for parent in [current, *current.parents]:
        if (parent / "configs").is_dir():
            return parent
    return current
