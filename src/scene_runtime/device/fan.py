"""PWM fan control for Raspberry Pi thermal experiments."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
from pathlib import Path
import sys
from typing import Any


@dataclass(frozen=True)
class FanState:
    enabled: bool
    duty_cycle: float
    mode: str


class PwmFanController:
    """Temperature-aware PWM fan controller with GPIO fallback."""

    def __init__(self, config: dict[str, Any]) -> None:
        cfg = config.get("fan", {})
        self._enabled = bool(cfg.get("enabled", False))
        enabled_strategies = cfg.get("enabled_strategies", ["scene_thermal_coadaptive"])
        strategy = str(config.get("project", {}).get("strategy", "default"))
        if enabled_strategies and "*" not in enabled_strategies and strategy not in enabled_strategies:
            self._enabled = False

        thermal = config.get("thermal", {})
        warm_max = float(thermal.get("warm_max_c", 72.0))
        critical = float(thermal.get("critical_c", warm_max + 7.0))
        self._pin = int(cfg.get("pwm_pin", 18))
        self._gpio_mode = str(cfg.get("gpio_mode", "BCM")).upper()
        self._frequency_hz = float(cfg.get("pwm_frequency_hz", 25000.0))
        self._on_temp_c = float(cfg.get("on_temp_c", max(warm_max - 2.0, 0.0)))
        self._off_temp_c = float(cfg.get("off_temp_c", self._on_temp_c - 4.0))
        self._full_temp_c = float(cfg.get("full_temp_c", critical + 4.0))
        self._min_duty = float(cfg.get("min_duty_cycle", 0.35))
        self._max_duty = float(cfg.get("max_duty_cycle", 1.0))
        self._temperature_only = bool(cfg.get("temperature_only", False))
        self._hold_low_on_close = bool(cfg.get("hold_low_on_close", True))

        self._active = False
        self._backend = ""
        self._gpio = None
        self._pwm = None
        self._lgpio_handle = None
        self._gpio_ready = False
        self._setup_attempted = False
        self._last_error = ""
        self._last_state = FanState(False, 0.0, "disabled" if not self._enabled else "off")

    @property
    def backend(self) -> str:
        """Return the active GPIO backend name, or an empty string if unavailable."""
        return self._backend

    @property
    def last_error(self) -> str:
        """Return the most recent GPIO setup/apply error."""
        return self._last_error

    def update(self, device_state: dict[str, Any], action_mode: str | None = None) -> FanState:
        if not self._enabled:
            self._last_state = FanState(False, 0.0, "disabled")
            return self._last_state

        temp_c = self._as_float(device_state.get("temp_c"))
        thermal_state = str(device_state.get("thermal_state") or "unknown")
        should_run = self._should_run(temp_c, thermal_state, action_mode)
        self._active = should_run

        if not should_run:
            state = FanState(False, 0.0, "off")
        else:
            state = FanState(True, self._duty_cycle(temp_c), "pwm")

        applied = self._apply(state)
        if state.enabled and not applied:
            state = FanState(True, state.duty_cycle, "pwm_no_gpio")
        self._last_state = state
        return state

    def close(self) -> None:
        if self._gpio_ready:
            self._force_gpio_low()
        if self._pwm is not None:
            try:
                if self._backend == "RPi.GPIO":
                    self._pwm.stop()
            except Exception:
                pass
            self._pwm = None
        if self._backend == "lgpio" and self._gpio is not None and self._lgpio_handle is not None:
            if not self._hold_low_on_close:
                try:
                    self._gpio.gpiochip_close(self._lgpio_handle)
                except Exception:
                    pass
                self._lgpio_handle = None
        if (
            self._gpio_ready
            and self._gpio is not None
            and self._backend == "RPi.GPIO"
            and not self._hold_low_on_close
        ):
            try:
                self._gpio.cleanup()
            except Exception:
                pass
        self._last_state = FanState(False, 0.0, "off")
        if not self._hold_low_on_close:
            self._gpio_ready = False
            self._backend = ""

    def _should_run(
        self,
        temp_c: float | None,
        thermal_state: str,
        action_mode: str | None,
    ) -> bool:
        if not self._temperature_only:
            if thermal_state in {"hot", "critical"}:
                return True
            if action_mode and (
                "_thermal_hot" in action_mode or "_thermal_critical" in action_mode
            ):
                return True
        if temp_c is None:
            return self._active
        if self._active:
            return temp_c >= self._off_temp_c
        return temp_c >= self._on_temp_c

    def _duty_cycle(self, temp_c: float | None) -> float:
        if temp_c is None or temp_c <= self._on_temp_c:
            return self._min_duty
        span = max(self._full_temp_c - self._on_temp_c, 0.1)
        ratio = min(max((temp_c - self._on_temp_c) / span, 0.0), 1.0)
        return self._min_duty + ratio * (self._max_duty - self._min_duty)

    def _apply(self, state: FanState) -> bool:
        if state == self._last_state:
            return self._gpio_ready or not state.enabled
        if not self._ensure_gpio():
            return False
        duty_percent = state.duty_cycle * 100.0 if state.enabled else 0.0
        try:
            if self._backend == "RPi.GPIO":
                if self._pwm is None:
                    return False
                self._pwm.ChangeDutyCycle(duty_percent)
                return True
            if self._backend == "lgpio":
                if self._gpio is None or self._lgpio_handle is None:
                    return False
                if state.enabled and duty_percent > 0:
                    self._gpio.tx_pwm(
                        self._lgpio_handle,
                        self._pin,
                        self._frequency_hz,
                        duty_percent,
                    )
                else:
                    self._gpio.gpio_write(self._lgpio_handle, self._pin, 0)
                return True
        except Exception as exc:
            self._last_error = f"{self._backend} apply failed: {exc}"
            return False
        return False

    def _force_gpio_low(self) -> None:
        """Stop PWM and actively hold the fan control pin low on shutdown."""
        try:
            if self._backend == "RPi.GPIO" and self._gpio is not None:
                if self._pwm is not None:
                    self._pwm.ChangeDutyCycle(0.0)
                self._gpio.output(self._pin, self._gpio.LOW)
            elif (
                self._backend == "lgpio"
                and self._gpio is not None
                and self._lgpio_handle is not None
            ):
                try:
                    self._gpio.tx_pwm(
                        self._lgpio_handle,
                        self._pin,
                        self._frequency_hz,
                        0.0,
                    )
                except Exception:
                    pass
                self._gpio.gpio_write(self._lgpio_handle, self._pin, 0)
        except Exception as exc:
            self._last_error = f"{self._backend} force-off failed: {exc}"

    def _ensure_gpio(self) -> bool:
        if self._gpio_ready:
            return True
        if self._setup_attempted:
            return False
        self._setup_attempted = True
        if self._setup_rpi_gpio():
            return True
        if self._gpio_mode == "BOARD":
            self._last_error += "; lgpio fallback skipped because gpio_mode=BOARD"
            return False
        return self._setup_lgpio()

    def _setup_rpi_gpio(self) -> bool:
        try:
            GPIO = self._import_module("RPi.GPIO")
        except Exception as exc:
            self._last_error = f"RPi.GPIO import failed: {exc}"
            return False
        self._gpio = GPIO
        try:
            GPIO.setwarnings(False)
            GPIO.setmode(GPIO.BOARD if self._gpio_mode == "BOARD" else GPIO.BCM)
            GPIO.setup(self._pin, GPIO.OUT)
            self._pwm = GPIO.PWM(self._pin, self._frequency_hz)
            self._pwm.start(0.0)
            self._gpio_ready = True
            self._backend = "RPi.GPIO"
        except Exception as exc:
            self._last_error = f"RPi.GPIO setup failed: {exc}"
            self._gpio_ready = False
        return self._gpio_ready

    def _setup_lgpio(self) -> bool:
        try:
            lgpio = self._import_module("lgpio")
        except Exception as exc:
            self._last_error += f"; lgpio import failed: {exc}"
            return False
        try:
            handle = lgpio.gpiochip_open(0)
            lgpio.gpio_claim_output(handle, self._pin, 0)
            lgpio.gpio_write(handle, self._pin, 0)
            self._gpio = lgpio
            self._lgpio_handle = handle
            self._gpio_ready = True
            self._backend = "lgpio"
        except Exception as exc:
            self._last_error += f"; lgpio setup failed: {exc}"
            self._gpio_ready = False
        return self._gpio_ready

    @staticmethod
    def _import_module(name: str):
        """Import GPIO modules, including Ubuntu apt packages outside virtualenvs."""
        try:
            return importlib.import_module(name)
        except Exception as first_exc:
            for path in (
                "/usr/lib/python3/dist-packages",
                "/usr/local/lib/python3/dist-packages",
            ):
                if Path(path).exists() and path not in sys.path:
                    sys.path.append(path)
            try:
                return importlib.import_module(name)
            except Exception:
                raise first_exc

    @staticmethod
    def _as_float(value: Any) -> float | None:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
