"""Aggregated Raspberry Pi device state monitor with graceful degradation."""

from __future__ import annotations

from typing import Any

from scene_runtime.device.frequency import read_cpu_frequencies_mhz
from scene_runtime.device.power import read_power_w
from scene_runtime.device.temperature import read_temperature_c
from scene_runtime.device.throttling import read_throttling_state


class DeviceStateMonitor:
    """
    Records Raspberry Pi runtime state: temperature, frequency, throttling, power.

    All reads degrade to None or empty dict on non-Pi development machines.
    """

    def read_temperature_c(self) -> float | None:
        """Current CPU temperature in Celsius."""
        return read_temperature_c()

    def read_cpu_frequency_mhz(self) -> dict[str, float | int]:
        """Per-CPU and average frequency in MHz."""
        return read_cpu_frequencies_mhz()

    def read_throttling_state(self) -> dict[str, bool | str | None]:
        """Throttling flags from vcgencmd when available."""
        return read_throttling_state()

    def read_power_w(self) -> float | None:
        """Instantaneous power in watts, or None if unavailable."""
        return read_power_w()

    def thermal_state(self, config: dict[str, Any] | None = None) -> str:
        """
        Map temperature to ``normal``, ``warm``, ``hot``, or ``unknown``.

        Thresholds are configurable via YAML ``thermal`` section.
        """
        temp = self.read_temperature_c()
        if temp is None:
            return "unknown"
        cfg = (config or {}).get("thermal", {})
        normal_max = float(cfg.get("normal_max_c", 65.0))
        warm_max = float(cfg.get("warm_max_c", 75.0))
        if temp < normal_max:
            return "normal"
        if temp < warm_max:
            return "warm"
        return "hot"

    def snapshot(self, config: dict[str, Any] | None = None) -> dict[str, Any]:
        """
        Capture full device state snapshot for controller and logging.

        Returns
        -------
        dict
            Keys: temp_c, freq_mhz, freq_mhz_avg, power_w, throttling, thermal_state.
        """
        freq = self.read_cpu_frequency_mhz()
        avg = freq.get("avg_mhz")
        return {
            "temp_c": self.read_temperature_c(),
            "freq_mhz": {k: v for k, v in freq.items() if k != "avg_mhz"},
            "freq_mhz_avg": avg,
            "power_w": self.read_power_w(),
            "throttling": self.read_throttling_state(),
            "thermal_state": self.thermal_state(config),
        }
