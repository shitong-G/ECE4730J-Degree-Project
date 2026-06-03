"""CPU frequency reading via cpufreq sysfs."""

from __future__ import annotations

from pathlib import Path


def read_cpu_frequencies_mhz() -> dict[str, float | int]:
    """
    Read per-CPU current scaling frequencies in MHz.

    Returns dict like ``{"cpu0": 1500.0, "avg_mhz": 1500.0}`` or empty on failure.
    """
    cpus: dict[str, float] = {}
    pattern = Path("/sys/devices/system/cpu")
    try:
        for cpu_dir in sorted(pattern.glob("cpu[0-9]*")):
            freq_path = (
                cpu_dir / "cpufreq" / "scaling_cur_freq"
            )
            if not freq_path.exists():
                continue
            khz = int(freq_path.read_text(encoding="utf-8").strip())
            cpus[cpu_dir.name] = khz / 1000.0
    except OSError:
        return {}

    if not cpus:
        return {}
    avg = sum(cpus.values()) / len(cpus)
    result: dict[str, float | int] = dict(cpus)
    result["avg_mhz"] = avg
    return result
