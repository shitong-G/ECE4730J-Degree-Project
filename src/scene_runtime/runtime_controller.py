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

    def classify_runtime_state(
        self,
        scene_state: dict[str, Any],
        device_state: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Step 4 — fuse scene workload and SoC thermal into a runtime state (backbone).

        Maps to the figure's **Scene Complexity** + **SoC Temp Sensor** inputs before
        the **Layer Router & Schedule**. Feature policy uses this dict in ``decide()``.
        """
        policy = self._config.get("policy", {})
        workload = scene_state.get("workload", "medium")
        if not policy.get("use_scene", True):
            workload = "medium"
        thermal = device_state.get("thermal_state", "unknown")
        if not policy.get("use_thermal", True):
            thermal = "normal"
        return {
            "workload": workload,
            "thermal_state": thermal,
            "temp_c": device_state.get("temp_c"),
            # TODO(Member 3): layer_schedule_hint, query_budget_hint from scene × thermal
        }

    def decide(
        self,
        scene_state: dict[str, Any],
        device_state: dict[str, Any],
        recent_metrics: dict[str, Any] | None = None,
    ) -> RuntimeAction:
        """
        Step 5 — select ``RuntimeAction`` (Layer Router & Schedule + query/layer knobs).

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

        runtime_state = self.classify_runtime_state(scene_state, device_state)
        return self._rule_based_action(
            runtime_state["workload"],
            runtime_state["thermal_state"],
        )

    def _rule_based_action(self, workload: str, thermal: str) -> RuntimeAction:
        """
        BACKBONE: safe balanced defaults only.

        TODO(Member 3): scene × thermal co-adaptive rules (hot/warm/normal × light/medium/heavy).
        """
        if thermal == "unknown":
            return self._balanced("balanced_unknown")

        action = self._scene_action(workload)
        if thermal == "warm":
            return self._thermal_adjust(action, "warm")
        if thermal == "hot":
            return self._thermal_adjust(action, "hot")
        return action

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

    def _scene_action(self, workload: str) -> RuntimeAction:
        """Return the normal-temperature action selected by scene workload."""
        if workload == "light":
            return RuntimeAction(
                mode="scene_light",
                input_resolution=self._lower_resolution(self._default_res, steps=1),
                inference_interval=max(self._default_interval + 1, 2),
                cpu_threads=max(1, self._default_threads - 1),
                governor="ondemand",
                decoder_layers=4,
                query_budget=120,
            )
        if workload == "heavy":
            return RuntimeAction(
                mode="scene_heavy",
                input_resolution=self._default_res,
                inference_interval=max(1, self._default_interval),
                cpu_threads=self._default_threads,
                governor="performance",
                decoder_layers=6,
                query_budget=300,
            )
        return self._balanced("scene_medium")

    def _thermal_adjust(self, action: RuntimeAction, thermal: str) -> RuntimeAction:
        """Lower runtime workload for warm or hot thermal states."""
        if thermal == "warm":
            return RuntimeAction(
                mode=f"{action.mode}_thermal_warm",
                input_resolution=self._lower_resolution(action.input_resolution, steps=1),
                inference_interval=max(action.inference_interval + 1, 2),
                cpu_threads=max(1, min(action.cpu_threads, self._default_threads - 1)),
                governor="ondemand",
                decoder_layers=self._min_optional(action.decoder_layers, 4),
                query_budget=self._min_optional(action.query_budget, 160),
            )

        return RuntimeAction(
            mode=f"{action.mode}_thermal_hot",
            input_resolution=self._lower_resolution(action.input_resolution, steps=2),
            inference_interval=max(action.inference_interval + 2, 3),
            cpu_threads=max(1, min(action.cpu_threads, 2)),
            governor="powersave",
            decoder_layers=self._min_optional(action.decoder_layers, 3),
            query_budget=self._min_optional(action.query_budget, 100),
        )

    @staticmethod
    def _lower_resolution(resolution: int, *, steps: int) -> int:
        """Step down by 160 px increments while keeping a practical lower bound."""
        return max(320, int(resolution) - 160 * steps)

    @staticmethod
    def _min_optional(value: int | None, limit: int) -> int:
        """Cap an optional runtime knob, defaulting to the cap when unset."""
        return min(value if value is not None else limit, limit)
