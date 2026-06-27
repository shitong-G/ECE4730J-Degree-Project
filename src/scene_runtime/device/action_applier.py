"""Best-effort OS-level application of runtime actions."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from scene_runtime.controller.actions import RuntimeAction


CPU_ROOT = Path("/sys/devices/system/cpu")


@dataclass
class AppliedRuntimeState:
    """Requested vs applied OS-level runtime state."""

    requested_governor: str | None
    applied_governor: str | None
    governor_applied: bool | None
    requested_cpu_affinity: str | None
    applied_cpu_affinity: str | None
    cpu_affinity_applied: bool | None


class RuntimeActionApplier:
    """Apply governor and CPU affinity when supported by the host OS."""

    def __init__(self, *, enabled: bool = False) -> None:
        self._enabled = enabled
        self._last_governor: str | None = None
        self._last_affinity: tuple[int, ...] | None = None

    def apply(self, action: RuntimeAction) -> AppliedRuntimeState:
        requested_affinity = _format_affinity(action.cpu_affinity)
        if not self._enabled:
            return AppliedRuntimeState(
                requested_governor=action.governor,
                applied_governor=self._read_governor(),
                governor_applied=None,
                requested_cpu_affinity=requested_affinity,
                applied_cpu_affinity=self._read_affinity(),
                cpu_affinity_applied=None,
            )

        governor_applied = self._apply_governor(action.governor)
        affinity_applied = self._apply_affinity(action.cpu_affinity)
        return AppliedRuntimeState(
            requested_governor=action.governor,
            applied_governor=self._read_governor(),
            governor_applied=governor_applied,
            requested_cpu_affinity=requested_affinity,
            applied_cpu_affinity=self._read_affinity(),
            cpu_affinity_applied=affinity_applied,
        )

    def _apply_governor(self, governor: str | None) -> bool | None:
        if not governor:
            return None
        if governor == self._last_governor:
            return True

        paths = sorted(CPU_ROOT.glob("cpu[0-9]*/cpufreq/scaling_governor"))
        if not paths:
            return False
        ok = True
        for path in paths:
            try:
                path.write_text(governor, encoding="utf-8")
            except OSError:
                ok = False
        if ok:
            self._last_governor = governor
        return ok

    def _apply_affinity(self, affinity: list[int] | None) -> bool | None:
        if affinity is None:
            return None
        requested = tuple(int(cpu) for cpu in affinity)
        if requested == self._last_affinity:
            return True
        if not hasattr(os, "sched_setaffinity"):
            return False
        try:
            os.sched_setaffinity(0, set(requested))
        except (OSError, ValueError):
            return False
        self._last_affinity = requested
        return True

    @staticmethod
    def _read_governor() -> str | None:
        path = CPU_ROOT / "cpu0" / "cpufreq" / "scaling_governor"
        try:
            if not path.exists():
                return None
            return path.read_text(encoding="utf-8").strip()
        except OSError:
            return None

    @staticmethod
    def _read_affinity() -> str | None:
        if not hasattr(os, "sched_getaffinity"):
            return None
        try:
            return _format_affinity(sorted(os.sched_getaffinity(0)))
        except OSError:
            return None


def _format_affinity(affinity: list[int] | tuple[int, ...] | None) -> str | None:
    if affinity is None:
        return None
    return ",".join(str(int(cpu)) for cpu in affinity)
