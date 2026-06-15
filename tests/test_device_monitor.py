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
    assert snap["thermal_state"] in ("normal", "warm", "hot", "unknown")


def test_temperature_graceful() -> None:
    mon = DeviceStateMonitor()
    # On Windows dev machines this is typically None
    temp = mon.read_temperature_c()
    assert temp is None or isinstance(temp, float)


def test_throttling_dict() -> None:
    mon = DeviceStateMonitor()
    th = mon.read_throttling_state()
    assert isinstance(th, dict)
