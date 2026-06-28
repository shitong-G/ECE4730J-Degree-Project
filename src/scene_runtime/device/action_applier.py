"""Best-effort OS-level application of runtime actions."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from scene_runtime.controller.actions import RuntimeAction


CPU_ROOT = Path("/sys/devices/system/cpu")


@dataclass
class AppliedRuntimeState:
    """Requested vs applied OS-level runtime state."""

    requested_governor: str | None
    applied_governor: str | None
    governor_applied: bool | None
    governor_apply_error: str | None
    requested_cpu_affinity: str | None
    applied_cpu_affinity: str | None
    cpu_affinity_applied: bool | None
    cpu_affinity_apply_error: str | None


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
                governor_apply_error="disabled",
                requested_cpu_affinity=requested_affinity,
                applied_cpu_affinity=self._read_affinity(),
                cpu_affinity_applied=None,
                cpu_affinity_apply_error="disabled",
            )

        governor_applied, governor_error = self._apply_governor(action.governor)
        affinity_applied, affinity_error = self._apply_affinity(action.cpu_affinity)
        return AppliedRuntimeState(
            requested_governor=action.governor,
            applied_governor=self._read_governor(),
            governor_applied=governor_applied,
            governor_apply_error=governor_error,
            requested_cpu_affinity=requested_affinity,
            applied_cpu_affinity=self._read_affinity(),
            cpu_affinity_applied=affinity_applied,
            cpu_affinity_apply_error=affinity_error,
        )

    def _apply_governor(self, governor: str | None) -> tuple[bool | None, str | None]:
        if not governor:
            return None, None

        paths = sorted(CPU_ROOT.glob("cpu[0-9]*/cpufreq/scaling_governor"))
        if not paths:
            return False, "scaling_governor_not_found"

        available = self._available_governors(paths)
        if available and governor not in available:
            return False, f"governor_unavailable:{','.join(available)}"

        errors: list[str] = []
        for path in paths:
            try:
                path.write_text(f"{governor}\n", encoding="utf-8")
            except OSError:
                writable = os.access(path, os.W_OK)
                errors.append(f"{path}:write_failed:writable={writable}")

        applied = self._read_governors(paths)
        mismatched = {path: value for path, value in applied.items() if value != governor}
        if errors or mismatched:
            details = []
            if errors:
                details.extend(errors)
            if mismatched:
                mismatch_text = ",".join(
                    f"{path.name}={value}" for path, value in mismatched.items()
                )
                details.append(f"readback_mismatch:{mismatch_text}")
            return False, ";".join(details)

        if applied:
            self._last_governor = governor
        return True, None

    def _apply_affinity(self, affinity: list[int] | None) -> tuple[bool | None, str | None]:
        if affinity is None:
            return None, None
        requested = tuple(int(cpu) for cpu in affinity)
        if requested == self._last_affinity:
            current = self._read_affinity()
            return current == _format_affinity(requested), None
        if not hasattr(os, "sched_setaffinity"):
            return False, "sched_setaffinity_unavailable"
        try:
            os.sched_setaffinity(0, set(requested))
        except (OSError, ValueError) as exc:
            return False, f"{type(exc).__name__}:{exc}"
        self._last_affinity = requested
        current = self._read_affinity()
        if current != _format_affinity(requested):
            return False, f"readback_mismatch:{current}"
        return True, None

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

    @staticmethod
    def _available_governors(paths: Iterable[Path]) -> list[str]:
        for path in paths:
            available_path = path.with_name("scaling_available_governors")
            try:
                text = available_path.read_text(encoding="utf-8").strip()
            except OSError:
                continue
            if text:
                return text.split()
        return []

    @staticmethod
    def _read_governors(paths: Iterable[Path]) -> dict[Path, str | None]:
        values: dict[Path, str | None] = {}
        for path in paths:
            try:
                values[path] = path.read_text(encoding="utf-8").strip()
            except OSError:
                values[path] = None
        return values


def _format_affinity(affinity: list[int] | tuple[int, ...] | None) -> str | None:
    if affinity is None:
        return None
    return ",".join(str(int(cpu)) for cpu in affinity)
