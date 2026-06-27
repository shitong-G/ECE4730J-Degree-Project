#!/usr/bin/env python3
"""Generate synthetic thermal-aware experiment logs for analysis testing.

This script does not run ONNX or OpenCV. It creates CSV files with the same
schema as RuntimeLogger so plotting and summary scripts can be tested before
running long Raspberry Pi experiments.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scene_runtime.controller.runtime_controller import RuntimeDecisionController


LOG_COLUMNS = [
    "timestamp",
    "frame_id",
    "strategy",
    "workload",
    "thermal_state",
    "raw_thermal_state",
    "control_thermal_state",
    "action_mode",
    "decision_reason",
    "thermal_pressure_level",
    "temp_slope_c_per_min",
    "temp_c",
    "freq_mhz_avg",
    "arm_clock_mhz",
    "power_w",
    "throttling_raw",
    "under_voltage",
    "arm_freq_capped",
    "currently_throttled",
    "soft_temp_limit",
    "did_infer",
    "latency_ms",
    "fps",
    "loop_fps",
    "effective_inference_fps",
    "actual_inference_fps",
    "input_resolution",
    "inference_interval",
    "cpu_threads",
    "governor",
    "requested_governor",
    "applied_governor",
    "governor_applied",
    "requested_cpu_affinity",
    "applied_cpu_affinity",
    "cpu_affinity_applied",
    "decoder_layers",
    "query_budget",
    "detection_count",
    "confidence_mean",
]

PROFILE_COLUMNS = [
    "timestamp",
    "frame_id",
    "strategy",
    "did_infer",
    "frame_total_ms",
    "scene_ms",
    "device_ms",
    "runtime_state_ms",
    "decision_ms",
    "infer_outer_ms",
    "preprocess_ms",
    "build_feed_ms",
    "onnx_run_ms",
    "postprocess_ms",
    "infer_total_ms",
    "summary_ms",
    "main_log_write_ms",
]

STRATEGIES = [
    "native_rtdetr",
    "fixed_frame_skip",
    "fixed_low_power",
    "thermal_only",
    "thermal_balanced",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic thermal experiment logs")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "experiments" / "logs" / "synthetic_thermal")
    parser.add_argument("--prefix", default="synthetic")
    parser.add_argument("--duration-sec", type=float, default=900.0)
    parser.add_argument("--frame-period-sec", type=float, default=0.2)
    parser.add_argument("--ambient-c", type=float, default=45.0)
    parser.add_argument("--start-temp-c", type=float, default=52.0)
    parser.add_argument("--strategies", default=",".join(STRATEGIES))
    return parser.parse_args()


def thermal_state(temp_c: float, cfg: dict[str, float]) -> str:
    if temp_c < cfg["normal_max_c"]:
        return "normal"
    if temp_c < cfg["warm_max_c"]:
        return "warm"
    if temp_c < cfg["critical_c"]:
        return "hot"
    return "critical"


def build_config(strategy: str) -> dict[str, object]:
    policy = {
        "use_scene": strategy not in {"native_rtdetr", "fixed_frame_skip", "fixed_low_power"},
        "use_thermal": strategy in {"thermal_only", "thermal_balanced"},
        "thermal_balanced": strategy == "thermal_balanced",
        "fixed_inference_interval": None,
        "fixed_input_resolution": None,
        "fixed_cpu_threads": None,
        "fixed_cpu_affinity": None,
        "fixed_governor": None,
    }
    if strategy == "native_rtdetr":
        policy.update(
            fixed_inference_interval=1,
            fixed_input_resolution=640,
            fixed_cpu_threads=4,
            fixed_governor="performance",
        )
    elif strategy == "fixed_frame_skip":
        policy.update(
            fixed_inference_interval=3,
            fixed_input_resolution=480,
            fixed_cpu_threads=4,
        )
    elif strategy == "fixed_low_power":
        policy.update(
            fixed_inference_interval=4,
            fixed_input_resolution=320,
            fixed_cpu_threads=2,
            fixed_cpu_affinity=[0, 1],
            fixed_governor="powersave",
        )

    return {
        "project": {"strategy": strategy},
        "runtime": {
            "default_input_resolution": 480,
            "default_inference_interval": 2,
            "default_cpu_threads": 3,
        },
        "thermal": {
            "normal_max_c": 58.0,
            "warm_max_c": 66.0,
            "critical_c": 76.0,
            "hysteresis_c": 5.0,
            "warm_hold_frames": 90,
            "hot_hold_frames": 150,
            "critical_hold_frames": 220,
            "critical_plus_delta_c": 2.0,
            "critical_max_delta_c": 5.0,
            "pressure_hysteresis_c": 2.0,
            "pressure_hold_frames": 90,
        },
        "policy": policy,
    }


def heat_rate_c_per_sec(action_interval: int, resolution: int, strategy: str) -> float:
    if strategy == "native_rtdetr":
        base = 0.070
    elif strategy == "fixed_frame_skip":
        base = 0.045
    elif strategy == "fixed_low_power":
        base = 0.024
    else:
        base = 0.030 + 0.035 / max(action_interval, 1)
    return base * (resolution / 480.0)


def latency_ms(strategy: str, resolution: int, temp_c: float, did_infer: bool) -> float:
    if not did_infer:
        return 0.0
    base = {
        "native_rtdetr": 4200.0,
        "fixed_frame_skip": 3400.0,
        "fixed_low_power": 2100.0,
        "thermal_only": 3000.0,
        "thermal_balanced": 2900.0,
    }.get(strategy, 3000.0)
    scale = (resolution / 480.0) ** 1.8
    throttle_penalty = 1.25 if temp_c >= 85.0 else 1.0
    return base * scale * throttle_penalty


def write_logs(strategy: str, args: argparse.Namespace) -> Path:
    cfg = build_config(strategy)
    thermal_cfg = cfg["thermal"]
    ctrl = RuntimeDecisionController(cfg)

    frame_count = max(1, int(args.duration_sec / args.frame_period_sec))
    log_path = args.output_dir / f"{args.prefix}_{strategy}.csv"
    profile_path = log_path.with_name(f"{log_path.stem}_profile.csv")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    temp_c = float(args.start_temp_c)
    inference_times: list[float] = []
    inference_counter = 0

    with log_path.open("w", encoding="utf-8", newline="") as log_handle, profile_path.open(
        "w", encoding="utf-8", newline=""
    ) as profile_handle:
        log_writer = csv.DictWriter(log_handle, fieldnames=LOG_COLUMNS)
        profile_writer = csv.DictWriter(profile_handle, fieldnames=PROFILE_COLUMNS)
        log_writer.writeheader()
        profile_writer.writeheader()

        for frame_id in range(frame_count):
            timestamp = frame_id * args.frame_period_sec
            raw_state = thermal_state(temp_c, thermal_cfg)
            action = ctrl.decide(
                {"workload": "medium"},
                {"thermal_state": raw_state, "temp_c": temp_c},
            )
            did_infer = (inference_counter % max(action.inference_interval, 1)) == 0
            inference_counter += 1

            if did_infer:
                inference_times.append(timestamp)
            if len(inference_times) >= 2:
                actual_fps = (len(inference_times) - 1) / max(
                    1e-9, inference_times[-1] - inference_times[0]
                )
            else:
                actual_fps = 0.0

            loop_fps = 1.0 / args.frame_period_sec
            infer_fps_est = loop_fps / max(action.inference_interval, 1)
            lat = latency_ms(strategy, action.input_resolution, temp_c, did_infer)
            currently_throttled = temp_c >= 85.0
            soft_temp_limit = temp_c >= 80.0
            arm_freq = 1500.0
            if currently_throttled:
                arm_freq = 900.0
            elif soft_temp_limit:
                arm_freq = 1200.0

            detection_count = max(0, int(3 + 2 * math.sin(frame_id / 37.0)))
            confidence = 0.76 if action.input_resolution >= 480 else 0.69

            log_writer.writerow(
                {
                    "timestamp": f"{timestamp:.3f}",
                    "frame_id": frame_id,
                    "strategy": strategy,
                    "workload": "medium",
                    "thermal_state": ctrl.last_control_thermal_state,
                    "raw_thermal_state": ctrl.last_raw_thermal_state,
                    "control_thermal_state": ctrl.last_control_thermal_state,
                    "action_mode": action.mode,
                    "decision_reason": ctrl.last_decision_reason,
                    "thermal_pressure_level": ctrl.last_thermal_pressure_level,
                    "temp_slope_c_per_min": f"{ctrl.last_temp_slope_c_per_min:.3f}",
                    "temp_c": f"{temp_c:.3f}",
                    "freq_mhz_avg": f"{arm_freq:.1f}",
                    "arm_clock_mhz": f"{arm_freq:.1f}",
                    "power_w": "",
                    "throttling_raw": "throttled=0x4" if currently_throttled else "throttled=0x0",
                    "under_voltage": False,
                    "arm_freq_capped": currently_throttled,
                    "currently_throttled": currently_throttled,
                    "soft_temp_limit": soft_temp_limit,
                    "did_infer": did_infer,
                    "latency_ms": f"{lat:.3f}",
                    "fps": f"{loop_fps:.3f}",
                    "loop_fps": f"{loop_fps:.3f}",
                    "effective_inference_fps": f"{infer_fps_est:.3f}",
                    "actual_inference_fps": f"{actual_fps:.3f}",
                    "input_resolution": action.input_resolution,
                    "inference_interval": action.inference_interval,
                    "cpu_threads": action.cpu_threads,
                    "governor": action.governor,
                    "requested_governor": action.governor,
                    "applied_governor": "",
                    "governor_applied": "",
                    "requested_cpu_affinity": "",
                    "applied_cpu_affinity": "",
                    "cpu_affinity_applied": "",
                    "decoder_layers": action.decoder_layers,
                    "query_budget": action.query_budget,
                    "detection_count": detection_count,
                    "confidence_mean": f"{confidence:.3f}",
                }
            )

            profile_writer.writerow(
                {
                    "timestamp": f"{timestamp:.3f}",
                    "frame_id": frame_id,
                    "strategy": strategy,
                    "did_infer": did_infer,
                    "frame_total_ms": f"{args.frame_period_sec * 1000.0:.3f}",
                    "scene_ms": 1.0,
                    "device_ms": 1.0,
                    "runtime_state_ms": 0.1,
                    "decision_ms": 0.1,
                    "infer_outer_ms": f"{lat:.3f}",
                    "preprocess_ms": f"{lat * 0.04:.3f}" if did_infer else 0.0,
                    "build_feed_ms": f"{lat * 0.01:.3f}" if did_infer else 0.0,
                    "onnx_run_ms": f"{lat * 0.90:.3f}" if did_infer else 0.0,
                    "postprocess_ms": f"{lat * 0.05:.3f}" if did_infer else 0.0,
                    "infer_total_ms": f"{lat:.3f}",
                    "summary_ms": 0.1,
                    "main_log_write_ms": 0.1,
                }
            )

            heat = heat_rate_c_per_sec(action.inference_interval, action.input_resolution, strategy)
            cooling = max(0.0, temp_c - args.ambient_c) * 0.0028
            temp_c += (heat - cooling) * args.frame_period_sec

    return log_path


def main() -> None:
    args = parse_args()
    strategies = [item.strip() for item in args.strategies.split(",") if item.strip()]
    for strategy in strategies:
        if strategy not in STRATEGIES:
            raise SystemExit(f"Unsupported synthetic strategy: {strategy}")
        log_path = write_logs(strategy, args)
        print(f"Generated {log_path}")
        print(f"Generated {log_path.with_name(log_path.stem + '_profile.csv')}")


if __name__ == "__main__":
    main()
