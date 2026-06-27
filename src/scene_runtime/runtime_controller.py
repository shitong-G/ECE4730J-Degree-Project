"""Backward-compatible import for the canonical runtime decision controller."""

from scene_runtime.controller.runtime_controller import (
    RuntimeDecisionController,
    THERMAL_LEVELS,
    THERMAL_NAMES,
)

__all__ = ["RuntimeDecisionController", "THERMAL_LEVELS", "THERMAL_NAMES"]
