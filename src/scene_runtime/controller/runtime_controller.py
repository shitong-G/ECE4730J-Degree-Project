"""Rule-based runtime decision controller."""

from __future__ import annotations

from typing import Any

from scene_runtime.controller.actions import RuntimeAction
from scene_runtime.controller.policies import apply_fixed_policy


THERMAL_LEVELS = {
    "normal": 0,
    "warm": 1,
    "hot": 2,
    "critical": 3,
}
THERMAL_NAMES = {level: name for name, level in THERMAL_LEVELS.items()}


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
        thermal = config.get("thermal", {})
        self._normal_max_c = float(thermal.get("normal_max_c", 65.0))
        self._warm_max_c = float(thermal.get("warm_max_c", 75.0))
        self._critical_c = float(thermal.get("critical_c", self._warm_max_c + 7.0))
        self._hysteresis_c = float(thermal.get("hysteresis_c", 4.0))
        self._thermal_hold_frames = {
            "normal": 0,
            "warm": int(thermal.get("warm_hold_frames", 120)),
            "hot": int(thermal.get("hot_hold_frames", 180)),
            "critical": int(thermal.get("critical_hold_frames", 240)),
        }
        self._critical_plus_delta_c = float(thermal.get("critical_plus_delta_c", 3.0))
        self._critical_max_delta_c = float(thermal.get("critical_max_delta_c", 7.0))
        self._pressure_hysteresis_c = float(
            thermal.get("pressure_hysteresis_c", max(1.0, self._hysteresis_c / 2.0))
        )
        self._pressure_hold_frames = int(thermal.get("pressure_hold_frames", 120))
        self._thermal_guard_state = "normal"
        self._thermal_hold_remaining = 0
        self._critical_pressure_level = 0
        self._pressure_hold_remaining = 0

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
        thermal_state = runtime_state["thermal_state"]
        if self._config.get("policy", {}).get("use_thermal", True):
            thermal_state = self._guarded_thermal_state(
                thermal_state,
                runtime_state.get("temp_c"),
            )
        return self._rule_based_action(
            runtime_state["workload"],
            thermal_state,
            runtime_state.get("temp_c"),
        )

    def _rule_based_action(
        self,
        workload: str,
        thermal: str,
        temp_c: Any = None,
    ) -> RuntimeAction:
        """
        BACKBONE: safe balanced defaults only.

        TODO(Member 3): scene × thermal co-adaptive rules (hot/warm/normal × light/medium/heavy).
        """
        if thermal == "unknown":
            return self._balanced("balanced_unknown")

        action = self._scene_action(workload)
        if thermal == "warm":
            return self._thermal_adjust(action, "warm", temp_c)
        if thermal == "hot":
            return self._thermal_adjust(action, "hot", temp_c)
        if thermal == "critical":
            return self._thermal_adjust(action, "critical", temp_c)
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

    def _thermal_adjust(
        self,
        action: RuntimeAction,
        thermal: str,
        temp_c: Any = None,
    ) -> RuntimeAction:
        """Lower runtime workload for warm or hot thermal states."""
        if thermal == "warm":
            near_hot = self._temp_at_least(temp_c, self._warm_max_c - 2.0)
            interval_boost = 2 if near_hot else 1
            return RuntimeAction(
                mode=f"{action.mode}_thermal_warm",
                input_resolution=self._lower_resolution(action.input_resolution, steps=1),
                inference_interval=max(action.inference_interval + interval_boost, 2),
                cpu_threads=max(1, min(action.cpu_threads, self._default_threads - 1)),
                governor="ondemand",
                decoder_layers=self._min_optional(action.decoder_layers, 4),
                query_budget=self._min_optional(action.query_budget, 140 if near_hot else 160),
            )

        if thermal == "hot":
            near_critical = self._temp_at_least(temp_c, self._critical_c - 2.0)
            interval_boost = 5 if near_critical else 3
            return RuntimeAction(
                mode=f"{action.mode}_thermal_hot_plus" if near_critical else f"{action.mode}_thermal_hot",
                input_resolution=self._lower_resolution(action.input_resolution, steps=2),
                inference_interval=max(action.inference_interval + interval_boost, 4),
                cpu_threads=max(1, min(action.cpu_threads, 2)),
                governor="powersave",
                decoder_layers=self._min_optional(action.decoder_layers, 2 if near_critical else 3),
                query_budget=self._min_optional(action.query_budget, 80 if near_critical else 100),
            )

        critical_level = self._guarded_critical_pressure_level(temp_c)
        if critical_level >= 2:
            mode = f"{action.mode}_thermal_critical_max"
            interval = max(action.inference_interval + 14, 16)
            query_budget = 40
        elif critical_level == 1:
            mode = f"{action.mode}_thermal_critical_plus"
            interval = max(action.inference_interval + 10, 12)
            query_budget = 50
        else:
            mode = f"{action.mode}_thermal_critical"
            interval = max(action.inference_interval + 6, 8)
            query_budget = 60

        return RuntimeAction(
            mode=mode,
            input_resolution=320,
            inference_interval=interval,
            cpu_threads=1,
            governor="powersave",
            decoder_layers=2,
            query_budget=query_budget,
        )

    def _guarded_thermal_state(self, raw_state: str, temp_c: Any) -> str:
        """Stateful thermal guard with early entry, hysteresis, and slow recovery."""
        desired = self._desired_thermal_state(raw_state, temp_c)
        if desired == "unknown":
            return "unknown"

        current_level = THERMAL_LEVELS[self._thermal_guard_state]
        desired_level = THERMAL_LEVELS[desired]

        if desired_level > current_level:
            self._set_thermal_guard(desired)
            return self._thermal_guard_state

        if desired_level == current_level:
            if self._thermal_guard_state != "normal":
                self._thermal_hold_remaining = max(
                    self._thermal_hold_remaining,
                    self._thermal_hold_frames[self._thermal_guard_state],
                )
            return self._thermal_guard_state

        if self._thermal_hold_remaining > 0:
            self._thermal_hold_remaining -= 1
            return self._thermal_guard_state

        if self._can_cool_down_one_level(temp_c):
            next_level = max(desired_level, current_level - 1)
            self._set_thermal_guard(THERMAL_NAMES[next_level])

        return self._thermal_guard_state

    def _desired_thermal_state(self, raw_state: str, temp_c: Any) -> str:
        try:
            temp = float(temp_c)
        except (TypeError, ValueError):
            return raw_state if raw_state in THERMAL_LEVELS else "unknown"

        if temp >= self._critical_c:
            return "critical"
        if temp >= self._warm_max_c:
            return "hot"
        if temp >= self._normal_max_c:
            return "warm"
        return "normal"

    def _can_cool_down_one_level(self, temp_c: Any) -> bool:
        try:
            temp = float(temp_c)
        except (TypeError, ValueError):
            return True

        if self._thermal_guard_state == "critical":
            return temp < (self._critical_c - self._hysteresis_c)
        if self._thermal_guard_state == "hot":
            return temp < (self._warm_max_c - self._hysteresis_c)
        if self._thermal_guard_state == "warm":
            return temp < (self._normal_max_c - self._hysteresis_c)
        return True

    def _set_thermal_guard(self, state: str) -> None:
        self._thermal_guard_state = state
        self._thermal_hold_remaining = self._thermal_hold_frames[state]
        if state != "critical":
            self._critical_pressure_level = 0
            self._pressure_hold_remaining = 0

    @staticmethod
    def _temp_at_least(temp_c: Any, threshold: float) -> bool:
        try:
            return float(temp_c) >= threshold
        except (TypeError, ValueError):
            return False

    def _raw_critical_pressure_level(self, temp_c: Any) -> int:
        try:
            temp = float(temp_c)
        except (TypeError, ValueError):
            return 0
        if temp >= self._critical_c + self._critical_max_delta_c:
            return 2
        if temp >= self._critical_c + self._critical_plus_delta_c:
            return 1
        return 0

    def _guarded_critical_pressure_level(self, temp_c: Any) -> int:
        raw_level = self._raw_critical_pressure_level(temp_c)
        if raw_level > self._critical_pressure_level:
            self._critical_pressure_level = raw_level
            self._pressure_hold_remaining = self._pressure_hold_frames
            return self._critical_pressure_level

        if raw_level == self._critical_pressure_level:
            if raw_level > 0:
                self._pressure_hold_remaining = max(
                    self._pressure_hold_remaining,
                    self._pressure_hold_frames,
                )
            return self._critical_pressure_level

        if self._pressure_hold_remaining > 0:
            self._pressure_hold_remaining -= 1
            return self._critical_pressure_level

        if self._can_reduce_pressure(temp_c):
            self._critical_pressure_level = max(raw_level, self._critical_pressure_level - 1)
            self._pressure_hold_remaining = (
                self._pressure_hold_frames if self._critical_pressure_level > 0 else 0
            )

        return self._critical_pressure_level

    def _can_reduce_pressure(self, temp_c: Any) -> bool:
        try:
            temp = float(temp_c)
        except (TypeError, ValueError):
            return True
        if self._critical_pressure_level >= 2:
            threshold = self._critical_c + self._critical_max_delta_c
            return temp < threshold - self._pressure_hysteresis_c
        if self._critical_pressure_level == 1:
            threshold = self._critical_c + self._critical_plus_delta_c
            return temp < threshold - self._pressure_hysteresis_c
        return True

    @staticmethod
    def _lower_resolution(resolution: int, *, steps: int) -> int:
        """Step down by 160 px increments while keeping a practical lower bound."""
        return max(320, int(resolution) - 160 * steps)

    @staticmethod
    def _min_optional(value: int | None, limit: int) -> int:
        """Cap an optional runtime knob, defaulting to the cap when unset."""
        return min(value if value is not None else limit, limit)
