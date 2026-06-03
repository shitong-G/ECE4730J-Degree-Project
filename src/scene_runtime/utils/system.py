"""Cross-platform system helpers."""

from __future__ import annotations

import platform
from pathlib import Path


def is_raspberry_pi() -> bool:
    """Heuristic check for Raspberry Pi Linux environment."""
    try:
        model = Path("/proc/device-tree/model").read_text(encoding="utf-8")
        return "Raspberry Pi" in model
    except OSError:
        return False


def platform_info() -> dict[str, str]:
    return {
        "system": platform.system(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "is_raspberry_pi": str(is_raspberry_pi()),
    }
