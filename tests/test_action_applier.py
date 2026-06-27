"""Tests for best-effort OS runtime action applier."""

from __future__ import annotations

from scene_runtime.controller.actions import RuntimeAction
from scene_runtime.device.action_applier import RuntimeActionApplier


def test_action_applier_disabled_reports_requested_state() -> None:
    action = RuntimeAction(
        mode="test",
        input_resolution=320,
        inference_interval=4,
        cpu_threads=2,
        cpu_affinity=[0, 1],
        governor="powersave",
    )
    applied = RuntimeActionApplier(enabled=False).apply(action)
    assert applied.requested_governor == "powersave"
    assert applied.requested_cpu_affinity == "0,1"
    assert applied.governor_applied is None
    assert applied.cpu_affinity_applied is None
