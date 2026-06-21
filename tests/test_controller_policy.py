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


def test_scene_thermal_normal_heavy_uses_high_workload_action() -> None:
    cfg = load_config(ROOT / "configs" / "default.yaml", "scene_thermal_coadaptive")
    ctrl = RuntimeDecisionController(cfg)
    action = ctrl.decide(
        {"workload": "heavy"},
        {"thermal_state": "normal"},
    )
    assert action.mode == "scene_heavy"
    assert action.input_resolution == cfg["runtime"]["default_input_resolution"]
    assert action.inference_interval == cfg["runtime"]["default_inference_interval"]
    assert action.cpu_threads == cfg["runtime"]["default_cpu_threads"]
    assert action.governor == "performance"
    assert action.query_budget == 300


def test_scene_thermal_hot_reduces_visible_workload() -> None:
    cfg = load_config(ROOT / "configs" / "default.yaml", "scene_thermal_coadaptive")
    ctrl = RuntimeDecisionController(cfg)
    action = ctrl.decide(
        {"workload": "heavy"},
        {"thermal_state": "hot"},
    )
    assert action.mode == "scene_heavy_thermal_hot"
    assert action.input_resolution == 320
    assert action.inference_interval == 4
    assert action.cpu_threads == 2
    assert action.governor == "powersave"
    assert action.decoder_layers == 3
    assert action.query_budget == 100


def test_scene_thermal_critical_uses_emergency_workload_reduction() -> None:
    cfg = load_config(ROOT / "configs" / "default.yaml", "scene_thermal_coadaptive")
    ctrl = RuntimeDecisionController(cfg)
    action = ctrl.decide(
        {"workload": "heavy"},
        {"thermal_state": "hot", "temp_c": 84.0},
    )
    assert action.mode == "scene_heavy_thermal_critical"
    assert action.input_resolution == 320
    assert action.inference_interval == 8
    assert action.cpu_threads == 1
    assert action.governor == "powersave"
    assert action.decoder_layers == 2
    assert action.query_budget == 60


def test_scene_thermal_critical_pressure_keeps_escalating() -> None:
    cfg = load_config(ROOT / "configs" / "default.yaml", "scene_thermal_coadaptive")
    ctrl = RuntimeDecisionController(cfg)

    critical = ctrl.decide(
        {"workload": "medium"},
        {"thermal_state": "critical", "temp_c": 82.0},
    )
    critical_plus = ctrl.decide(
        {"workload": "medium"},
        {"thermal_state": "critical", "temp_c": 85.0},
    )
    critical_max = ctrl.decide(
        {"workload": "medium"},
        {"thermal_state": "critical", "temp_c": 89.0},
    )

    assert critical.mode == "scene_medium_thermal_critical"
    assert critical.inference_interval == 8
    assert critical.query_budget == 60
    assert critical_plus.mode == "scene_medium_thermal_critical_plus"
    assert critical_plus.inference_interval == 12
    assert critical_plus.query_budget == 50
    assert critical_max.mode == "scene_medium_thermal_critical_max"
    assert critical_max.inference_interval == 16
    assert critical_max.query_budget == 40


def test_critical_pressure_level_uses_hold_before_downgrade() -> None:
    cfg = load_config(ROOT / "configs" / "default.yaml", "scene_thermal_coadaptive")
    cfg["thermal"]["pressure_hold_frames"] = 2
    ctrl = RuntimeDecisionController(cfg)

    plus = ctrl.decide(
        {"workload": "medium"},
        {"thermal_state": "critical", "temp_c": 85.0},
    )
    held_1 = ctrl.decide(
        {"workload": "medium"},
        {"thermal_state": "critical", "temp_c": 82.0},
    )
    held_2 = ctrl.decide(
        {"workload": "medium"},
        {"thermal_state": "critical", "temp_c": 82.0},
    )
    downgraded = ctrl.decide(
        {"workload": "medium"},
        {"thermal_state": "critical", "temp_c": 82.0},
    )

    assert plus.mode == "scene_medium_thermal_critical_plus"
    assert held_1.mode == "scene_medium_thermal_critical_plus"
    assert held_2.mode == "scene_medium_thermal_critical_plus"
    assert downgraded.mode == "scene_medium_thermal_critical"


def test_thermal_guard_holds_hot_state_before_recovery() -> None:
    cfg = load_config(ROOT / "configs" / "default.yaml", "scene_thermal_coadaptive")
    cfg["thermal"]["hot_hold_frames"] = 2
    cfg["thermal"]["warm_hold_frames"] = 0
    ctrl = RuntimeDecisionController(cfg)

    hot = ctrl.decide({"workload": "medium"}, {"thermal_state": "hot", "temp_c": 76.0})
    held_1 = ctrl.decide({"workload": "medium"}, {"thermal_state": "normal", "temp_c": 55.0})
    held_2 = ctrl.decide({"workload": "medium"}, {"thermal_state": "normal", "temp_c": 55.0})
    cooled = ctrl.decide({"workload": "medium"}, {"thermal_state": "normal", "temp_c": 55.0})

    assert hot.mode == "scene_medium_thermal_hot"
    assert held_1.mode == "scene_medium_thermal_hot"
    assert held_2.mode == "scene_medium_thermal_hot"
    assert cooled.mode == "scene_medium_thermal_warm"


def test_thermal_only_ignores_scene_but_reacts_to_hot_device() -> None:
    cfg = load_config(ROOT / "configs" / "default.yaml", "thermal_only")
    ctrl = RuntimeDecisionController(cfg)
    action = ctrl.decide(
        {"workload": "heavy"},
        {"thermal_state": "hot"},
    )
    assert action.mode == "scene_medium_thermal_hot"
    assert action.input_resolution == 320
    assert action.inference_interval == 4
    assert action.governor == "powersave"
    assert action.query_budget == 100


def test_scene_only_ignores_hot_thermal_state() -> None:
    cfg = load_config(ROOT / "configs" / "default.yaml", "scene_only")
    ctrl = RuntimeDecisionController(cfg)
    action = ctrl.decide(
        {"workload": "light"},
        {"thermal_state": "hot"},
    )
    assert action.mode == "scene_light"
    assert action.input_resolution == 480
    assert action.inference_interval == 2
    assert action.governor == "ondemand"


def test_classify_runtime_state() -> None:
    cfg = load_config(ROOT / "configs" / "default.yaml", "scene_thermal_coadaptive")
    ctrl = RuntimeDecisionController(cfg)
    state = ctrl.classify_runtime_state(
        {"workload": "heavy"},
        {"thermal_state": "warm", "temp_c": 70.0},
    )
    assert state["workload"] == "heavy"
    assert state["thermal_state"] == "warm"
    assert state["temp_c"] == 70.0


def test_unknown_thermal_balanced() -> None:
    cfg = load_config(ROOT / "configs" / "default.yaml", "default")
    ctrl = RuntimeDecisionController(cfg)
    action = ctrl.decide(
        {"workload": "medium"},
        {"thermal_state": "unknown"},
    )
    assert action.mode == "balanced_unknown"
