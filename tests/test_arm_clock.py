"""Tests for ARM clock reader."""

from __future__ import annotations

import subprocess
from unittest.mock import MagicMock

from scene_runtime.device.arm_clock import read_arm_clock_mhz


def test_read_arm_clock_mhz_parses_vcgencmd(monkeypatch) -> None:
    monkeypatch.setattr("scene_runtime.device.arm_clock.shutil.which", lambda _: "/usr/bin/vcgencmd")

    def fake_run(cmd, **kwargs):
        assert cmd == ["vcgencmd", "measure_clock", "arm"]
        return subprocess.CompletedProcess(cmd, 0, stdout="frequency(48)=1531406208\n", stderr="")

    monkeypatch.setattr("scene_runtime.device.arm_clock.subprocess.run", fake_run)
    assert read_arm_clock_mhz() == 1531.406208


def test_read_arm_clock_mhz_unavailable(monkeypatch) -> None:
    monkeypatch.setattr("scene_runtime.device.arm_clock.shutil.which", lambda _: None)
    assert read_arm_clock_mhz() is None
