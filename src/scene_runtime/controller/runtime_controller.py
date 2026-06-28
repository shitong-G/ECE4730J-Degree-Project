"""Rule-based runtime decision controller."""

from __future__ import annotations

import time
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
        self._thermal_hold_sec = {
            "normal": 0.0,
            "warm": float(thermal.get("warm_hold_sec", 30.0)),
            "hot": float(thermal.get("hot_hold_sec", 45.0)),
            "critical": float(thermal.get("critical_hold_sec", 60.0)),
        }
        self._critical_plus_delta_c = float(thermal.get("critical_plus_delta_c", 3.0))
        self._critical_max_delta_c = float(thermal.get("critical_max_delta_c", 7.0))
        self._pressure_hysteresis_c = float(
            thermal.get("pressure_hysteresis_c", max(1.0, self._hysteresis_c / 2.0))
        )
        self._pressure_hold_frames = int(thermal.get("pressure_hold_frames", 120))
        self._pressure_hold_sec = float(thermal.get("pressure_hold_sec", 45.0))
        self._target_c = float(thermal.get("target_c", self._warm_max_c + 8.0))
        self._recovery_slope_c_per_min = float(
            thermal.get("recovery_slope_c_per_min", -0.15)
        )
        self._preemptive_slope_c_per_min = float(
            thermal.get("preemptive_slope_c_per_min", 0.4)
        )
        self._balanced_interval_cap = int(thermal.get("balanced_interval_cap", 12))
        self._balanced_normal_governor = str(
            thermal.get("balanced_normal_governor", "ondemand")
        )
        self._balanced_warm_resolution_steps = int(
            thermal.get("balanced_warm_resolution_steps", 0)
        )
        self._balanced_warm_near_hot_resolution_extra_steps = int(
            thermal.get("balanced_warm_near_hot_resolution_extra_steps", 0)
        )
        self._balanced_warm_interval_boost = int(
            thermal.get("balanced_warm_interval_boost", 0)
        )
        self._balanced_warm_thread_cap = int(
            thermal.get("balanced_warm_thread_cap", self._default_threads)
        )
        self._balanced_warm_governor = str(
            thermal.get("balanced_warm_governor", self._balanced_normal_governor)
        )
        self._balanced_hot_resolution_steps = int(
            thermal.get("balanced_hot_resolution_steps", 1)
        )
        self._balanced_hot_interval_boost = int(
            thermal.get("balanced_hot_interval_boost", 1)
        )
        self._balanced_hot_interval_min = int(
            thermal.get("balanced_hot_interval_min", max(self._default_interval + 1, 3))
        )
        self._balanced_hot_thread_cap = int(
            thermal.get("balanced_hot_thread_cap", self._default_threads)
        )
        self._balanced_hot_query_budget = int(
            thermal.get("balanced_hot_query_budget", 140)
        )
        self._balanced_hot_plus_resolution_steps = int(
            thermal.get("balanced_hot_plus_resolution_steps", 2)
        )
        self._balanced_hot_plus_interval_boost = int(
            thermal.get("balanced_hot_plus_interval_boost", 2)
        )
        self._balanced_hot_plus_interval_min = int(
            thermal.get("balanced_hot_plus_interval_min", max(self._default_interval + 2, 4))
        )
        self._balanced_hot_plus_thread_cap = int(
            thermal.get("balanced_hot_plus_thread_cap", max(1, self._default_threads - 1))
        )
        self._balanced_hot_plus_query_budget = int(
            thermal.get("balanced_hot_plus_query_budget", 100)
        )
        self._balanced_hot_governor = str(thermal.get("balanced_hot_governor", "ondemand"))
        self._balanced_hot_plus_governor = str(
            thermal.get("balanced_hot_plus_governor", "powersave")
        )
        self._balanced_critical_resolution_steps = int(
            thermal.get("balanced_critical_resolution_steps", 2)
        )
        self._balanced_critical_interval_boost = int(
            thermal.get("balanced_critical_interval_boost", 3)
        )
        self._balanced_critical_interval_min = int(
            thermal.get("balanced_critical_interval_min", max(self._default_interval + 3, 4))
        )
        self._balanced_critical_thread_cap = int(
            thermal.get("balanced_critical_thread_cap", 2)
        )
        self._balanced_critical_query_budget = int(
            thermal.get("balanced_critical_query_budget", 100)
        )
        self._balanced_critical_governor = str(
            thermal.get("balanced_critical_governor", "powersave")
        )
        self._thermal_guard_state = "normal"
        self._thermal_hold_remaining = 0
        self._thermal_hold_until = 0.0
        self._critical_pressure_level = 0
        self._pressure_hold_remaining = 0
        self._pressure_hold_until = 0.0
        self._last_raw_thermal_state = "unknown"
        self._last_control_thermal_state = "unknown"
        self._last_decision_reason = "init"
        self._last_thermal_pressure_level = 0
        self._last_temp_slope_c_per_min = 0.0
        self._last_temp_sample: tuple[float, float] | None = None

    @property
    def last_raw_thermal_state(self) -> str:
        """Most recent device-reported thermal state before controller guarding."""
        return self._last_raw_thermal_state

    @property
    def last_control_thermal_state(self) -> str:
        """Most recent thermal state actually used to select the runtime action."""
        return self._last_control_thermal_state

    @property
    def last_decision_reason(self) -> str:
        """Short explanation for the most recent thermal/action decision."""
        return self._last_decision_reason

    @property
    def last_thermal_pressure_level(self) -> int:
        """Most recent critical pressure level used by the controller."""
        return self._last_thermal_pressure_level

    @property
    def last_temp_slope_c_per_min(self) -> float:
        """Most recent temperature slope estimate in degrees C per minute."""
        return self._last_temp_slope_c_per_min

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

        self._last_raw_thermal_state = str(device_state.get("thermal_state", "unknown"))
        self._update_temp_slope(device_state.get("temp_c"))
        runtime_state = self.classify_runtime_state(scene_state, device_state)
        thermal_state = runtime_state["thermal_state"]
        if self._config.get("policy", {}).get("use_thermal", True):
            thermal_state = self._guarded_thermal_state(
                thermal_state,
                runtime_state.get("temp_c"),
            )
        self._last_control_thermal_state = thermal_state

        fixed = apply_fixed_policy(self._config)
        if fixed is not None:
            return fixed

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
            governor=self._balanced_normal_governor,
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
            if self._is_balanced_thermal_policy():
                return self._balanced_warm_action(action, temp_c)

            near_hot = self._temp_at_least(temp_c, self._warm_max_c - 2.0)
            interval_boost = 2 if near_hot else 1
            self._last_decision_reason = "warm_near_hot" if near_hot else "warm_preemptive"
            return RuntimeAction(
                mode=f"{action.mode}_thermal_warm",
                input_resolution=self._lower_resolution(action.input_resolution, steps=1),
                inference_interval=self._cap_interval(
                    max(action.inference_interval + interval_boost, 2)
                ),
                cpu_threads=max(1, min(action.cpu_threads, self._default_threads - 1)),
                governor="ondemand",
                decoder_layers=self._min_optional(action.decoder_layers, 4),
                query_budget=self._min_optional(action.query_budget, 140 if near_hot else 160),
            )

        if thermal == "hot":
            if self._is_balanced_thermal_policy():
                return self._balanced_hot_action(action, temp_c)

            near_critical = self._temp_at_least(temp_c, self._critical_c - 2.0)
            interval_boost = 4 if near_critical else 2
            self._last_decision_reason = "hot_near_critical" if near_critical else "hot_cooldown"
            return RuntimeAction(
                mode=f"{action.mode}_thermal_hot_plus" if near_critical else f"{action.mode}_thermal_hot",
                input_resolution=self._lower_resolution(action.input_resolution, steps=2),
                inference_interval=self._cap_interval(
                    max(action.inference_interval + interval_boost, 4)
                ),
                cpu_threads=max(1, min(action.cpu_threads, 2)),
                governor="powersave",
                decoder_layers=self._min_optional(action.decoder_layers, 2 if near_critical else 3),
                query_budget=self._min_optional(action.query_budget, 80 if near_critical else 100),
            )

        critical_level = self._guarded_critical_pressure_level(temp_c)
        if self._is_balanced_thermal_policy():
            mode = f"{action.mode}_thermal_critical_balanced"
            interval = max(
                action.inference_interval + self._balanced_critical_interval_boost,
                self._balanced_critical_interval_min,
            )
            query_budget = self._balanced_critical_query_budget
            if critical_level >= 2:
                mode = f"{action.mode}_thermal_critical_max_balanced"
                interval += 2
                query_budget = min(query_budget, 70)
            elif critical_level == 1:
                mode = f"{action.mode}_thermal_critical_plus_balanced"
                interval += 1
                query_budget = min(query_budget, 85)
            if self._is_cooling_fast_enough(temp_c):
                interval = max(self._balanced_hot_plus_interval_min, interval - 1)
                mode = f"{mode}_recovery"
                self._last_decision_reason = "critical_recovery"
            else:
                self._last_decision_reason = f"critical_pressure_{critical_level}"
        elif critical_level >= 2:
            mode = f"{action.mode}_thermal_critical_max"
            interval = max(action.inference_interval + 14, 16)
            query_budget = 40
            self._last_decision_reason = "critical_max_pressure"
        elif critical_level == 1:
            mode = f"{action.mode}_thermal_critical_plus"
            interval = max(action.inference_interval + 10, 12)
            query_budget = 50
            self._last_decision_reason = "critical_plus_pressure"
        else:
            mode = f"{action.mode}_thermal_critical"
            interval = max(action.inference_interval + 6, 8)
            query_budget = 60
            self._last_decision_reason = "critical_cooldown"

        return RuntimeAction(
            mode=mode,
            input_resolution=(
                self._lower_resolution(
                    action.input_resolution,
                    steps=self._balanced_critical_resolution_steps,
                )
                if self._is_balanced_thermal_policy()
                else 320
            ),
            inference_interval=self._cap_interval(interval),
            cpu_threads=(
                max(1, min(action.cpu_threads, self._balanced_critical_thread_cap))
                if self._is_balanced_thermal_policy()
                else 1
            ),
            governor=(
                self._balanced_critical_governor
                if self._is_balanced_thermal_policy()
                else "powersave"
            ),
            decoder_layers=2,
            query_budget=query_budget,
        )

    def _balanced_warm_action(self, action: RuntimeAction, temp_c: Any) -> RuntimeAction:
        near_hot = self._temp_at_least(temp_c, self._warm_max_c - 2.0)
        steps = self._balanced_warm_resolution_steps + (
            self._balanced_warm_near_hot_resolution_extra_steps if near_hot else 0
        )
        interval_boost = self._balanced_warm_interval_boost + (1 if near_hot else 0)
        self._last_decision_reason = (
            "balanced_warm_near_hot" if near_hot else "balanced_warm_hold"
        )
        return RuntimeAction(
            mode=f"{action.mode}_thermal_warm_balanced",
            input_resolution=self._lower_resolution(action.input_resolution, steps=steps),
            inference_interval=self._cap_interval(
                max(action.inference_interval + interval_boost, self._default_interval)
            ),
            cpu_threads=max(1, min(action.cpu_threads, self._balanced_warm_thread_cap)),
            governor=self._balanced_warm_governor,
            decoder_layers=self._min_optional(action.decoder_layers, 5),
            query_budget=self._min_optional(action.query_budget, 180 if near_hot else 200),
        )

    def _balanced_hot_action(self, action: RuntimeAction, temp_c: Any) -> RuntimeAction:
        near_critical = self._temp_at_least(temp_c, self._critical_c - 2.0)
        if near_critical:
            self._last_decision_reason = "balanced_hot_near_critical"
            return RuntimeAction(
                mode=f"{action.mode}_thermal_hot_plus_balanced",
                input_resolution=self._lower_resolution(
                    action.input_resolution,
                    steps=self._balanced_hot_plus_resolution_steps,
                ),
                inference_interval=self._cap_interval(
                    max(
                        action.inference_interval + self._balanced_hot_plus_interval_boost,
                        self._balanced_hot_plus_interval_min,
                    )
                ),
                cpu_threads=max(1, min(action.cpu_threads, self._balanced_hot_plus_thread_cap)),
                governor=self._balanced_hot_plus_governor,
                decoder_layers=self._min_optional(action.decoder_layers, 3),
                query_budget=self._min_optional(
                    action.query_budget,
                    self._balanced_hot_plus_query_budget,
                ),
            )

        self._last_decision_reason = "balanced_hot_sustain"
        return RuntimeAction(
            mode=f"{action.mode}_thermal_hot_balanced",
            input_resolution=self._lower_resolution(
                action.input_resolution,
                steps=self._balanced_hot_resolution_steps,
            ),
            inference_interval=self._cap_interval(
                max(
                    action.inference_interval + self._balanced_hot_interval_boost,
                    self._balanced_hot_interval_min,
                )
            ),
            cpu_threads=max(1, min(action.cpu_threads, self._balanced_hot_thread_cap)),
            governor=self._balanced_hot_governor,
            decoder_layers=self._min_optional(action.decoder_layers, 4),
            query_budget=self._min_optional(action.query_budget, self._balanced_hot_query_budget),
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

        now = time.monotonic()
        if self._thermal_hold_remaining > 0 or now < self._thermal_hold_until:
            self._thermal_hold_remaining -= 1
            if self._is_cooling_fast_enough(temp_c) and self._can_cool_down_one_level(temp_c):
                next_level = max(desired_level, current_level - 1)
                self._set_thermal_guard(THERMAL_NAMES[next_level])
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
        if (
            temp >= self._normal_max_c - 2.0
            and self._last_temp_slope_c_per_min >= self._preemptive_slope_c_per_min
        ):
            self._last_decision_reason = "preemptive_warm_slope"
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
        self._thermal_hold_until = time.monotonic() + self._thermal_hold_sec[state]
        if state != "critical":
            self._critical_pressure_level = 0
            self._pressure_hold_remaining = 0
            self._pressure_hold_until = 0.0
            self._last_thermal_pressure_level = 0

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
            self._pressure_hold_until = time.monotonic() + self._pressure_hold_sec
            self._last_thermal_pressure_level = self._critical_pressure_level
            return self._critical_pressure_level

        if raw_level == self._critical_pressure_level:
            if raw_level > 0:
                self._pressure_hold_remaining = max(
                    self._pressure_hold_remaining,
                    self._pressure_hold_frames,
                )
                self._pressure_hold_until = max(
                    self._pressure_hold_until,
                    time.monotonic() + self._pressure_hold_sec,
                )
            self._last_thermal_pressure_level = self._critical_pressure_level
            return self._critical_pressure_level

        if self._pressure_hold_remaining > 0 or time.monotonic() < self._pressure_hold_until:
            self._pressure_hold_remaining -= 1
            if self._is_cooling_fast_enough(temp_c) and self._can_reduce_pressure(temp_c):
                self._critical_pressure_level = max(raw_level, self._critical_pressure_level - 1)
                self._pressure_hold_remaining = (
                    self._pressure_hold_frames if self._critical_pressure_level > 0 else 0
                )
                self._pressure_hold_until = (
                    time.monotonic() + self._pressure_hold_sec
                    if self._critical_pressure_level > 0
                    else 0.0
                )
            self._last_thermal_pressure_level = self._critical_pressure_level
            return self._critical_pressure_level

        if self._can_reduce_pressure(temp_c):
            self._critical_pressure_level = max(raw_level, self._critical_pressure_level - 1)
            self._pressure_hold_remaining = (
                self._pressure_hold_frames if self._critical_pressure_level > 0 else 0
            )
            self._pressure_hold_until = (
                time.monotonic() + self._pressure_hold_sec
                if self._critical_pressure_level > 0
                else 0.0
            )

        self._last_thermal_pressure_level = self._critical_pressure_level
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

    def _update_temp_slope(self, temp_c: Any) -> None:
        try:
            temp = float(temp_c)
        except (TypeError, ValueError):
            return
        now = time.monotonic()
        if self._last_temp_sample is not None:
            prev_temp, prev_time = self._last_temp_sample
            dt = now - prev_time
            if dt > 0:
                self._last_temp_slope_c_per_min = (temp - prev_temp) / dt * 60.0
        self._last_temp_sample = (temp, now)

    def _is_cooling_fast_enough(self, temp_c: Any) -> bool:
        try:
            temp = float(temp_c)
        except (TypeError, ValueError):
            return False
        if temp > self._target_c + self._hysteresis_c:
            return False
        return self._last_temp_slope_c_per_min <= self._recovery_slope_c_per_min

    def _is_balanced_thermal_policy(self) -> bool:
        strategy = self._config.get("project", {}).get("strategy")
        return strategy == "thermal_balanced" or bool(
            self._config.get("policy", {}).get("thermal_balanced", False)
        )

    def _cap_interval(self, interval: int) -> int:
        if self._is_balanced_thermal_policy():
            return min(int(interval), self._balanced_interval_cap)
        return int(interval)
