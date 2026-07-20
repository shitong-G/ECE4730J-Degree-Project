"""Tests for PWM fan control policy."""

from __future__ import annotations

from scene_runtime.device.fan import PwmFanController


def test_pwm_fan_duty_increases_with_temperature() -> None:
    fan = PwmFanController(
        {
            "project": {"strategy": "scene_thermal_interval_lk"},
            "thermal": {"warm_max_c": 68.0, "critical_c": 76.0},
            "fan": {
                "enabled": True,
                "enabled_strategies": ["scene_thermal_interval_lk"],
                "on_temp_c": 66.0,
                "off_temp_c": 60.0,
                "full_temp_c": 80.0,
                "min_duty_cycle": 0.25,
                "max_duty_cycle": 0.9,
            },
        }
    )

    low = fan.update({"thermal_state": "hot", "temp_c": 66.0}, "scene_medium_thermal_hot")
    high = fan.update(
        {"thermal_state": "critical", "temp_c": 80.0},
        "scene_medium_thermal_critical",
    )

    assert low.enabled is True
    assert low.duty_cycle == 0.25
    assert high.duty_cycle == 0.9


def test_pwm_fan_disabled_for_other_strategy() -> None:
    fan = PwmFanController(
        {
            "project": {"strategy": "native_rtdetr"},
            "fan": {
                "enabled": True,
                "enabled_strategies": ["scene_thermal_interval_lk"],
            },
        }
    )

    state = fan.update({"thermal_state": "critical", "temp_c": 90.0}, "fixed")

    assert state.enabled is False
    assert state.mode == "disabled"
