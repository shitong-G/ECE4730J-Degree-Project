"""Temperature-driven RT-DETR query-budget controller."""

from __future__ import annotations

from typing import Any


LEVELS = {"normal": 0, "warm": 1, "hot": 2, "critical": 3}
NAMES = {value: key for key, value in LEVELS.items()}


class ThermalQueryBudgetController:
    """Select a bounded query count with hysteretic thermal recovery."""

    def __init__(self, config: dict[str, Any]) -> None:
        cfg = config.get("query_budget_control", {})
        thermal = config.get("thermal", {})
        inference = config.get("inference", {})
        self.enabled = bool(cfg.get("enabled", False))
        self._max_budget = max(1, int(inference.get("max_query_budget", 300)))
        self._budgets = {
            "normal": int(cfg.get("normal_budget", 64)),
            "warm": int(cfg.get("warm_budget", 48)),
            "hot": int(cfg.get("hot_budget", 40)),
            "critical": int(cfg.get("critical_budget", 32)),
            "unknown": int(cfg.get("unknown_budget", 64)),
        }
        self._normal_max = float(thermal.get("normal_max_c", 65.0))
        self._warm_max = float(thermal.get("warm_max_c", 75.0))
        self._critical = float(
            thermal.get("critical_c", self._warm_max + 7.0)
        )
        self._hysteresis = max(
            0.0,
            float(cfg.get("hysteresis_c", thermal.get("hysteresis_c", 4.0))),
        )
        self._state = "normal"
        for name, budget in self._budgets.items():
            if budget < 1 or budget > self._max_budget:
                raise ValueError(
                    f"query_budget_control.{name}_budget must be within "
                    f"1..{self._max_budget}"
                )

    @property
    def state(self) -> str:
        return self._state

    def update(self, raw_state: str, temp_c: Any) -> tuple[int, str]:
        if not self.enabled:
            return self._budgets["normal"], "disabled"
        desired = raw_state if raw_state in LEVELS else "unknown"
        if desired == "unknown":
            return self._budgets["unknown"], "unknown"

        current_level = LEVELS[self._state]
        desired_level = LEVELS[desired]
        if desired_level > current_level:
            self._state = desired
        elif desired_level < current_level and self._may_recover(temp_c):
            self._state = NAMES[max(desired_level, current_level - 1)]
        return self._budgets[self._state], self._state

    def _may_recover(self, temp_c: Any) -> bool:
        try:
            temperature = float(temp_c)
        except (TypeError, ValueError):
            return False
        boundary = {
            "warm": self._normal_max,
            "hot": self._warm_max,
            "critical": self._critical,
        }.get(self._state)
        return boundary is not None and temperature < boundary - self._hysteresis
