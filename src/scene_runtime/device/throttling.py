"""Raspberry Pi throttling state via vcgencmd."""

from __future__ import annotations

import shutil
import subprocess


def read_throttling_state() -> dict[str, bool | str | None]:
    """
    Parse ``vcgencmd get_throttled`` when available.

    Returns flags for under-voltage, arm frequency capped, currently throttled, etc.
    Gracefully returns ``available: False`` on non-Pi systems.
    """
    if not shutil.which("vcgencmd"):
        return {"available": False, "raw": None}

    try:
        proc = subprocess.run(
            ["vcgencmd", "get_throttled"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        raw = proc.stdout.strip()
        # Format: throttled=0x50005
        hex_val = 0
        if "=" in raw:
            hex_str = raw.split("=", 1)[1]
            hex_val = int(hex_str, 16)
        return {
            "available": True,
            "raw": raw,
            "under_voltage": bool(hex_val & 0x1),
            "arm_freq_capped": bool(hex_val & 0x2),
            "currently_throttled": bool(hex_val & 0x4),
            "soft_temp_limit": bool(hex_val & 0x8),
        }
    except (subprocess.SubprocessError, ValueError):
        return {"available": False, "raw": None}
