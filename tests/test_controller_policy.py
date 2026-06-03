"""Tests for runtime decision controller policies."""

from __future__ import annotations

from scene_runtime.controller.policies import apply_fixed_policy
from scene_runtime.controller.runtime_controller import RuntimeDecisionController
from scene_runtime.runtime.config import load_config
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_fixed_low_power_policy() -> None:
    cfg = load_config(ROOT / "configs" / "default.yaml", "fixed_low_power")
    action = apply_fixed_policy(cfg)
    assert action is not None
    assert action.inference_interval == 4
    assert action.input_resolution == 320


def test_adaptive_strategy_returns_placeholder_action() -> None:
    """Backbone: adaptive path returns balanced defaults, not tuned policy yet."""
    cfg = load_config(ROOT / "configs" / "default.yaml", "scene_thermal_coadaptive")
    ctrl = RuntimeDecisionController(cfg)
    action = ctrl.decide(
        {"workload": "heavy"},
        {"thermal_state": "normal"},
    )
    assert action.mode == "balanced_placeholder"
    assert action.input_resolution == cfg["runtime"]["default_input_resolution"]


def test_unknown_thermal_balanced() -> None:
    cfg = load_config(ROOT / "configs" / "default.yaml", "default")
    ctrl = RuntimeDecisionController(cfg)
    action = ctrl.decide(
        {"workload": "medium"},
        {"thermal_state": "unknown"},
    )
    assert action.mode == "balanced_unknown"
