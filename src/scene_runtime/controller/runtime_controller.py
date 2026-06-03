"""Rule-based runtime decision controller."""

from __future__ import annotations

from typing import Any

from scene_runtime.controller.actions import RuntimeAction
from scene_runtime.controller.policies import apply_fixed_policy


class RuntimeDecisionController:
    """
    Maps scene workload and device thermal state to a ``RuntimeAction``.

    BACKBONE: fixed strategies via YAML work; adaptive co-adaptive rules are not
    implemented yet (see README TODO — Member 3).
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self._config = config
        runtime = config.get("runtime", {})
        self._default_res = int(runtime.get("default_input_resolution", 640))
        self._default_interval = int(runtime.get("default_inference_interval", 1))
        self._default_threads = int(runtime.get("default_cpu_threads", 4))

    def decide(
        self,
        scene_state: dict[str, Any],
        device_state: dict[str, Any],
        recent_metrics: dict[str, Any] | None = None,
    ) -> RuntimeAction:
        """
        Produce runtime action from scene, device, and optional recent metrics.

        Parameters
        ----------
        scene_state:
            Output of ``SceneWorkloadEstimator.update()``.
        device_state:
            Output of ``DeviceStateMonitor.snapshot()``.
        recent_metrics:
            Optional rolling FPS / latency from runtime loop.
        """
        _ = recent_metrics  # reserved for future latency-aware rules

        fixed = apply_fixed_policy(self._config)
        if fixed is not None:
            return fixed

        policy = self._config.get("policy", {})
        use_scene = policy.get("use_scene", True)
        use_thermal = policy.get("use_thermal", True)

        workload = scene_state.get("workload", "medium") if use_scene else "medium"
        thermal = device_state.get("thermal_state", "unknown")
        if not use_thermal:
            thermal = "normal"

        return self._rule_based_action(workload, thermal)

    def _rule_based_action(self, workload: str, thermal: str) -> RuntimeAction:
        """
        BACKBONE: safe balanced defaults only.

        TODO(Member 3): scene × thermal co-adaptive rules (hot/warm/normal × light/medium/heavy).
        """
        _ = workload
        mode = "balanced_unknown" if thermal == "unknown" else "balanced_placeholder"
        return self._balanced(mode)

    def _balanced(self, mode: str) -> RuntimeAction:
        return RuntimeAction(
            mode=mode,
            input_resolution=self._default_res,
            inference_interval=self._default_interval,
            cpu_threads=self._default_threads,
            governor="ondemand",
            decoder_layers=None,
            query_budget=200,
        )
