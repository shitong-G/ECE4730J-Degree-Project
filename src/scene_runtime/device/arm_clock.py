"""ARM core clock via Raspberry Pi firmware (vcgencmd)."""

from __future__ import annotations

import shutil
import subprocess


def read_arm_clock_mhz() -> float | None:
    """
    Read actual ARM clock from VideoCore firmware in MHz.

    On Raspberry Pi this reflects firmware thermal throttling more accurately
    than ``scaling_cur_freq`` sysfs, which may stay at the governor maximum.
    Returns ``None`` when ``vcgencmd`` is unavailable (e.g. dev workstation).
    """
    if not shutil.which("vcgencmd"):
        return None

    try:
        proc = subprocess.run(
            ["vcgencmd", "measure_clock", "arm"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        raw = proc.stdout.strip()
        if proc.returncode != 0 or "=" not in raw:
            return None
        hz = int(raw.split("=", 1)[1])
        return hz / 1_000_000.0
    except (subprocess.SubprocessError, ValueError, OSError):
        return None
