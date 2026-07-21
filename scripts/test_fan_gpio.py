#!/usr/bin/env python3
"""Manually test Raspberry Pi GPIO PWM fan output."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scene_runtime.device.fan import PwmFanController
from scene_runtime.runtime.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "raspberry_pi4.yaml")
    parser.add_argument("--strategy", default="scene_thermal_interval_lk")
    parser.add_argument("--pin", type=int, default=None, help="Override fan.pwm_pin")
    parser.add_argument("--frequency-hz", type=float, default=None)
    parser.add_argument("--duty", type=float, default=0.5, help="Duty cycle from 0.0 to 1.0")
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Demonstrate PWM speed control by stepping through several duty cycles.",
    )
    parser.add_argument(
        "--duties",
        default="0.25,0.4,0.6,0.8,1.0",
        help="Comma-separated duty cycle list used by --sweep.",
    )
    parser.add_argument(
        "--step-seconds",
        type=float,
        default=4.0,
        help="Seconds to hold each duty cycle in --sweep mode.",
    )
    parser.add_argument("--on-temp-c", type=float, default=0.0)
    parser.add_argument("--full-temp-c", type=float, default=1.0)
    return parser.parse_args()


def _build_config(args: argparse.Namespace, duty: float) -> dict:
    duty = min(max(float(duty), 0.0), 1.0)
    config = load_config(args.config, args.strategy)
    fan_cfg = config.setdefault("fan", {})
    fan_cfg["enabled"] = True
    fan_cfg["enabled_strategies"] = [args.strategy]
    fan_cfg["on_temp_c"] = float(args.on_temp_c)
    fan_cfg["off_temp_c"] = -1.0
    fan_cfg["full_temp_c"] = float(args.full_temp_c)
    fan_cfg["min_duty_cycle"] = duty
    fan_cfg["max_duty_cycle"] = duty
    if args.pin is not None:
        fan_cfg["pwm_pin"] = int(args.pin)
    if args.frequency_hz is not None:
        fan_cfg["pwm_frequency_hz"] = float(args.frequency_hz)
    return config


def _run_one_step(args: argparse.Namespace, duty: float, seconds: float) -> None:
    config = _build_config(args, duty)

    controller = PwmFanController(config)
    try:
        state = controller.update({"temp_c": float(args.full_temp_c), "thermal_state": "hot"})
        print(
            f"fan state: enabled={state.enabled} duty={state.duty_cycle:.3f} "
            f"mode={state.mode} backend={controller.backend or '--'}"
        )
        if state.mode == "pwm_no_gpio":
            raise RuntimeError(
                "GPIO PWM was not applied. "
                f"Last GPIO error: {controller.last_error or 'unknown'}"
            )
        time.sleep(max(0.0, float(seconds)))
    finally:
        controller.close()
        print("fan off")


def _parse_duties(text: str) -> list[float]:
    duties: list[float] = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        duties.append(min(max(float(item), 0.0), 1.0))
    if not duties:
        raise ValueError("--duties must contain at least one numeric duty cycle")
    return duties


def main() -> None:
    args = parse_args()
    if not args.sweep:
        _run_one_step(args, min(max(float(args.duty), 0.0), 1.0), args.seconds)
        return

    duties = _parse_duties(args.duties)
    print(
        "PWM sweep: "
        + ", ".join(f"{duty:.2f}" for duty in duties)
        + f" at {args.frequency_hz or 'config default'} Hz"
    )
    for duty in duties:
        print(f"\nstep duty={duty:.2f} for {args.step_seconds:.1f}s")
        _run_one_step(args, duty, args.step_seconds)


if __name__ == "__main__":
    main()
