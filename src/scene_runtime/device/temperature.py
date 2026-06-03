"""CPU temperature reading via Linux thermal sysfs."""

from __future__ import annotations

from pathlib import Path

THERMAL_ZONE = Path("/sys/class/thermal/thermal_zone0/temp")


def read_temperature_c() -> float | None:
    """
    Read CPU temperature in degrees Celsius from thermal_zone0.

    Returns None if sysfs is unavailable (e.g. on Windows dev machines).
    """
    try:
        if not THERMAL_ZONE.exists():
            return None
        raw = THERMAL_ZONE.read_text(encoding="utf-8").strip()
        return float(raw) / 1000.0
    except (OSError, ValueError):
        return None
