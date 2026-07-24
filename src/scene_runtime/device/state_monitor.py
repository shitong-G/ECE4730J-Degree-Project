"""Aggregated Raspberry Pi device state monitor with graceful degradation."""

from __future__ import annotations

import threading
import time
from typing import Any

from scene_runtime.device.arm_clock import read_arm_clock_mhz
from scene_runtime.device.frequency import read_cpu_frequencies_mhz
from scene_runtime.device.power import read_power_w
from scene_runtime.device.temperature import read_temperature_c
from scene_runtime.device.throttling import read_throttling_state


class DeviceStateMonitor:
    """
    Records Raspberry Pi runtime state: temperature, frequency, throttling, power.

    All reads degrade to None or empty dict on non-Pi development machines.
    """

    def __init__(self) -> None:
        self._firmware_lock = threading.Lock()
        self._firmware_thread: threading.Thread | None = None
        self._firmware_stop = threading.Event()
        self._firmware_poll_interval_sec = 1.0
        self._firmware_cache_max_age_sec = 10.0
        self._cached_arm_clock_mhz: float | None = None
        self._cached_arm_clock_time = 0.0
        self._cached_throttling: dict[str, bool | str | None] | None = None
        self._cached_throttling_time = 0.0
        self._cached_firmware_poll_ms = 0.0

    def read_temperature_c(self) -> float | None:
        """Current CPU temperature in Celsius."""
        return read_temperature_c()

    def read_arm_clock_mhz(self) -> float | None:
        """Actual ARM core clock from firmware (MHz), or None if unavailable."""
        return read_arm_clock_mhz()

    def read_cpu_frequency_mhz(self) -> dict[str, float | int]:
        """Per-CPU and average frequency in MHz."""
        return read_cpu_frequencies_mhz()

    def read_throttling_state(self) -> dict[str, bool | str | None]:
        """Throttling flags from vcgencmd when available."""
        return read_throttling_state()

    def read_power_w(self) -> float | None:
        """Instantaneous power in watts, or None if unavailable."""
        return read_power_w()

    def thermal_state(
        self,
        config: dict[str, Any] | None = None,
        temp_c: float | None = None,
    ) -> str:
        """
        Map temperature to ``normal``, ``warm``, ``hot``, ``critical``, or ``unknown``.

        Thresholds are configurable via YAML ``thermal`` section.
        """
        cfg = (config or {}).get("thermal", {})
        override = cfg.get("override_state")
        if override in {"normal", "warm", "hot", "critical", "unknown"}:
            return override

        override_temp = cfg.get("override_temp_c")
        temp = (
            float(override_temp)
            if override_temp is not None
            else temp_c if temp_c is not None
            else self.read_temperature_c()
        )
        if temp is None:
            return "unknown"
        normal_max = float(cfg.get("normal_max_c", 65.0))
        warm_max = float(cfg.get("warm_max_c", 75.0))
        critical = float(cfg.get("critical_c", warm_max + 7.0))
        if temp < normal_max:
            return "normal"
        if temp < warm_max:
            return "warm"
        if temp < critical:
            return "hot"
        return "critical"

    def snapshot(self, config: dict[str, Any] | None = None) -> dict[str, Any]:
        """
        Capture full device state snapshot for controller and logging.

        Returns
        -------
        dict
            Keys: temp_c, freq_mhz, freq_mhz_avg, arm_clock_mhz, power_w,
            throttling, thermal_state.
        """
        cfg = (config or {}).get("thermal", {})
        firmware_poll_interval_sec = float(
            cfg.get("firmware_poll_interval_sec", 1.0)
        )
        firmware_cache_max_age_sec = float(
            cfg.get("firmware_cache_max_age_sec", 10.0)
        )
        override_temp = cfg.get("override_temp_c")
        temp_c = float(override_temp) if override_temp is not None else self.read_temperature_c()
        freq = self.read_cpu_frequency_mhz()
        avg = freq.get("avg_mhz")
        self._ensure_firmware_monitor(
            poll_interval_sec=firmware_poll_interval_sec,
            cache_max_age_sec=firmware_cache_max_age_sec,
        )
        firmware = self._cached_firmware_snapshot()
        return {
            "temp_c": temp_c,
            "freq_mhz": {k: v for k, v in freq.items() if k != "avg_mhz"},
            "freq_mhz_avg": avg,
            "arm_clock_mhz": firmware["arm_clock_mhz"],
            "arm_clock_stale": firmware["arm_clock_stale"],
            "power_w": self.read_power_w(),
            "throttling": firmware["throttling"],
            "throttling_stale": firmware["throttling_stale"],
            "firmware_poll_ms": firmware["firmware_poll_ms"],
            "thermal_state": self.thermal_state(config, temp_c=temp_c),
        }

    def _ensure_firmware_monitor(
        self,
        *,
        poll_interval_sec: float,
        cache_max_age_sec: float,
    ) -> None:
        self._firmware_poll_interval_sec = max(0.1, poll_interval_sec)
        self._firmware_cache_max_age_sec = max(0.0, cache_max_age_sec)
        if self._firmware_thread is not None and self._firmware_thread.is_alive():
            return
        self._firmware_stop.clear()
        self._firmware_thread = threading.Thread(
            target=self._firmware_monitor_loop,
            name="device-firmware-monitor",
            daemon=True,
        )
        self._firmware_thread.start()

    def close(self) -> None:
        """Stop background firmware monitor if it was started."""
        self._firmware_stop.set()
        thread = self._firmware_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=0.5)

    def _firmware_monitor_loop(self) -> None:
        while not self._firmware_stop.is_set():
            self._poll_firmware_once()
            self._firmware_stop.wait(self._firmware_poll_interval_sec)

    def _poll_firmware_once(self) -> None:
        t0 = time.perf_counter()
        arm_clock = self.read_arm_clock_mhz()
        throttling = self.read_throttling_state()
        poll_time = time.perf_counter()
        poll_ms = (poll_time - t0) * 1000.0
        with self._firmware_lock:
            self._cached_firmware_poll_ms = poll_ms
            if arm_clock is not None:
                self._cached_arm_clock_mhz = arm_clock
                self._cached_arm_clock_time = poll_time
            if throttling.get("available", True) or throttling.get("raw") is not None:
                self._cached_throttling = throttling
                self._cached_throttling_time = poll_time

    def _cached_firmware_snapshot(self) -> dict[str, Any]:
        now = time.perf_counter()
        with self._firmware_lock:
            cached_arm_clock = self._cached_arm_clock_mhz
            cached_arm_time = self._cached_arm_clock_time
            cached_throttling = (
                dict(self._cached_throttling)
                if self._cached_throttling is not None
                else None
            )
            cached_throttling_time = self._cached_throttling_time
            poll_ms = self._cached_firmware_poll_ms
            poll_interval_sec = self._firmware_poll_interval_sec
            cache_max_age_sec = self._firmware_cache_max_age_sec

        arm_age = now - cached_arm_time if cached_arm_time > 0.0 else float("inf")
        throttling_age = (
            now - cached_throttling_time
            if cached_throttling_time > 0.0
            else float("inf")
        )
        arm_clock_mhz = (
            cached_arm_clock
            if arm_age <= max(0.0, cache_max_age_sec)
            else None
        )
        throttling = (
            cached_throttling
            if cached_throttling is not None
            and throttling_age <= max(0.0, cache_max_age_sec)
            else {"available": False, "raw": None}
        )
        throttling["stale"] = bool(
            cached_throttling is not None and throttling_age > poll_interval_sec
        )
        return {
            "arm_clock_mhz": arm_clock_mhz,
            "arm_clock_stale": bool(
                cached_arm_clock is not None and arm_age > poll_interval_sec
            ),
            "throttling": throttling,
            "throttling_stale": bool(throttling.get("stale")),
            "firmware_poll_ms": poll_ms,
        }
