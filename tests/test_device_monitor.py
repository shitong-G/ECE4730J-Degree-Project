"""Tests for device state monitor (non-Pi safe)."""

from __future__ import annotations

from scene_runtime.device.state_monitor import DeviceStateMonitor


def test_snapshot_keys() -> None:
    mon = DeviceStateMonitor()
    snap = mon.snapshot()
    assert "temp_c" in snap
    assert "freq_mhz_avg" in snap
    assert "arm_clock_mhz" in snap
    assert "thermal_state" in snap
    assert snap["thermal_state"] in ("normal", "warm", "hot", "critical", "unknown")


def test_temperature_graceful() -> None:
    mon = DeviceStateMonitor()
    # On Windows dev machines this is typically None
    temp = mon.read_temperature_c()
    assert temp is None or isinstance(temp, float)


def test_snapshot_thermal_override() -> None:
    mon = DeviceStateMonitor()
    snap = mon.snapshot({"thermal": {"override_state": "hot", "override_temp_c": 82.5}})
    assert snap["thermal_state"] == "hot"
    assert snap["temp_c"] == 82.5


def test_snapshot_temperature_override_drives_thermal_state() -> None:
    mon = DeviceStateMonitor()
    snap = mon.snapshot(
        {
            "thermal": {
                "normal_max_c": 58.0,
                "warm_max_c": 66.0,
                "override_temp_c": 70.0,
            }
        }
    )
    assert snap["thermal_state"] == "hot"
    assert snap["temp_c"] == 70.0


def test_throttling_dict() -> None:
    mon = DeviceStateMonitor()
    th = mon.read_throttling_state()
    assert isinstance(th, dict)
