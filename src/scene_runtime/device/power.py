"""Power measurement placeholder for future INA219 / USB meter integration."""

from __future__ import annotations


def read_power_w() -> float | None:
    """
    Read instantaneous power draw in watts.

    Returns None by default. TODO: integrate INA219 or external USB power meter.
    """
    return None
