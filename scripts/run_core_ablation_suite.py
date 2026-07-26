#!/usr/bin/env python3
"""Run the focused thesis ablation: quantization, LK, ROI, and PWM fan.

The five cumulative conditions are:

1. Native FP32 RT-DETR on every frame.
2. Quantization-only INT8 RT-DETR on every frame.
3. INT8 + event-triggered LK, with ROI refresh disabled.
4. INT8 + event-triggered LK + 320 ROI refresh, with the fan disabled.
5. The same full detection pipeline as condition 4, with threshold/PWM fan control.

Before every condition the CPU is cooled to approximately the same temperature.
Only the first condition performs temperature-driven RT-DETR warmup, so the
suite does not begin from an unrealistically cold idle device.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from run_controlled_experiment_suite import (
    cpu_temp_c,
    first_logged_temperature,
    run_with_temperature_trace,
    sha256,
    start_cooldown_fan,
    system_snapshot,
    verify_fan_hardware,
    write_json,
)


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ExperimentSpec:
    key: str
    title: str
    strategy: str
    model_condition: str
    use_model_family: bool
    disable_roi_refresh: bool
    fan_control: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "raspberry_pi4.yaml",
    )
    parser.add_argument("--video", type=Path, default=ROOT / "data" / "sample.mp4")
    parser.add_argument(
        "--native-model",
        type=Path,
        default=ROOT / "models" / "rtdetr_r18_lite_pi4_640.onnx",
    )
    parser.add_argument(
        "--quantized-model-320",
        type=Path,
        default=ROOT / "models" / "rtdetr_r18_lite_pi4_320_int8.onnx",
    )
    parser.add_argument(
        "--quantized-model-640",
        type=Path,
        default=ROOT / "models" / "rtdetr_r18_lite_pi4_640_int8.onnx",
    )
    parser.add_argument("--duration-min", type=float, default=20.0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "experiments" / "core_ablation_suite",
    )
    parser.add_argument(
        "--run-id",
        default=datetime.now().strftime("core_ablation_%Y%m%d_%H%M%S"),
    )
    parser.add_argument("--cooldown-temp-c", type=float, default=50.0)
    parser.add_argument("--cooldown-tolerance-c", type=float, default=0.5)
    parser.add_argument("--max-wait-min", type=float, default=90.0)
    parser.add_argument("--poll-sec", type=float, default=5.0)
    parser.add_argument("--cooldown-fan-settle-sec", type=float, default=2.0)
    parser.add_argument("--temperature-trace-sec", type=float, default=1.0)
    parser.add_argument("--progress-sec", type=float, default=30.0)
    parser.add_argument("--first-warmup-temp-c", type=float, default=50.0)
    parser.add_argument("--first-warmup-max-sec", type=float, default=900.0)
    parser.add_argument("--loop-video", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--log-detections", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--evaluate-quality",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Compare all ablations with the FP32 baseline after the suite completes.",
    )
    parser.add_argument("--fan-preflight", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cooldown-fan", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fan-on-temp-c", type=float, default=68.0)
    parser.add_argument("--fan-off-temp-c", type=float, default=62.0)
    parser.add_argument("--fan-full-temp-c", type=float, default=82.0)
    parser.add_argument("--fan-min-duty-cycle", type=float, default=0.35)
    parser.add_argument("--fan-max-duty-cycle", type=float, default=1.0)
    parser.add_argument(
        "--apply-runtime-actions",
        action="store_true",
        help="Not recommended for the focused ablation; changes OS scheduling too.",
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Validate inputs and print the five commands without running them.",
    )
    return parser.parse_args()


def experiment_specs() -> list[ExperimentSpec]:
    return [
        ExperimentSpec(
            key="baseline_native_fp32",
            title="Native FP32 RT-DETR, every frame",
            strategy="native_rtdetr",
            model_condition="native_fp32_640",
            use_model_family=False,
            disable_roi_refresh=False,
            fan_control="disabled",
        ),
        ExperimentSpec(
            key="quantized_only",
            title="INT8 quantization only, every frame",
            strategy="native_rtdetr",
            model_condition="quantized_int8_640",
            use_model_family=False,
            disable_roi_refresh=False,
            fan_control="disabled",
        ),
        ExperimentSpec(
            key="quantized_lk_no_roi",
            title="INT8 + event-triggered LK, ROI disabled",
            strategy="scene_track_lk",
            model_condition="quantized_int8_640",
            use_model_family=False,
            disable_roi_refresh=True,
            fan_control="disabled",
        ),
        ExperimentSpec(
            key="quantized_lk_roi_fan_off",
            title="INT8 + event-triggered LK + 320 ROI, fan disabled",
            strategy="scene_track_lk",
            model_condition="quantized_int8_320_640",
            use_model_family=True,
            disable_roi_refresh=False,
            fan_control="disabled",
        ),
        ExperimentSpec(
            key="quantized_lk_roi_pwm_fan",
            title="INT8 + event-triggered LK + 320 ROI + threshold/PWM fan",
            strategy="scene_track_lk",
            model_condition="quantized_int8_320_640",
            use_model_family=True,
            disable_roi_refresh=False,
            fan_control="enabled",
        ),
    ]


def validate_args(args: argparse.Namespace) -> None:
    if args.duration_min <= 0:
        raise ValueError("--duration-min must be positive")
    if args.cooldown_temp_c <= 0 or args.cooldown_tolerance_c < 0:
        raise ValueError("Cooldown temperature must be positive and tolerance non-negative")
    if args.max_wait_min <= 0 or args.poll_sec <= 0:
        raise ValueError("Wait and polling durations must be positive")
    if args.first_warmup_temp_c < 0 or args.first_warmup_max_sec <= 0:
        raise ValueError("Warmup temperature must be non-negative and timeout positive")
    if not args.fan_off_temp_c <= args.fan_on_temp_c <= args.fan_full_temp_c:
        raise ValueError("Fan thresholds must satisfy off <= on <= full")
    if not 0.0 <= args.fan_min_duty_cycle <= args.fan_max_duty_cycle <= 1.0:
        raise ValueError("Fan duty cycles must satisfy 0 <= min <= max <= 1")
    paths = [
        args.config,
        args.video,
        args.native_model,
        args.quantized_model_320,
        args.quantized_model_640,
    ]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Required input(s) not found: " + ", ".join(missing))


def model_arguments(args: argparse.Namespace, spec: ExperimentSpec) -> list[str]:
    if spec.key == "baseline_native_fp32":
        return ["--model", str(args.native_model)]
    if spec.use_model_family:
        return [
            "--model-paths-by-resolution",
            ",".join(
                [
                    f"320={args.quantized_model_320}",
                    f"640={args.quantized_model_640}",
                ]
            ),
        ]
    return ["--model", str(args.quantized_model_640)]


def build_command(
    args: argparse.Namespace,
    spec: ExperimentSpec,
    output: Path,
    *,
    first_run: bool,
) -> list[str]:
    command = [
        sys.executable,
        str(ROOT / "scripts" / "run_experiment.py"),
        "--config",
        str(args.config),
        "--strategy",
        spec.strategy,
        "--video",
        str(args.video),
        "--duration-min",
        str(args.duration_min),
        "--output",
        str(output),
        "--fan-control",
        spec.fan_control,
    ]
    if args.loop_video:
        command.append("--loop-video")
    if args.log_detections:
        command.append("--log-detections")
    if args.apply_runtime_actions:
        command.append("--apply-runtime-actions")
    if spec.disable_roi_refresh:
        command.append("--disable-roi-refresh")
    if spec.fan_control == "enabled":
        command.extend(
            [
                "--fan-temperature-only",
                "--fan-on-temp-c",
                str(args.fan_on_temp_c),
                "--fan-off-temp-c",
                str(args.fan_off_temp_c),
                "--fan-full-temp-c",
                str(args.fan_full_temp_c),
                "--fan-min-duty-cycle",
                str(args.fan_min_duty_cycle),
                "--fan-max-duty-cycle",
                str(args.fan_max_duty_cycle),
            ]
        )
    if first_run and args.first_warmup_temp_c > 0:
        command.extend(
            [
                "--warmup-until-temp-c",
                str(args.first_warmup_temp_c),
                "--warmup-max-sec",
                str(args.first_warmup_max_sec),
            ]
        )
    command.extend(model_arguments(args, spec))
    return command


def cool_to_start_temperature(
    args: argparse.Namespace,
    trace_path: Path,
) -> tuple[list[dict[str, Any]], float]:
    """Cool to the upper edge of the target window, then release the GPIO fan."""
    upper = args.cooldown_temp_c + args.cooldown_tolerance_c
    deadline = time.monotonic() + args.max_wait_min * 60.0
    samples: list[dict[str, Any]] = []
    controller = None
    try:
        while True:
            temperature = cpu_temp_c()
            if temperature is None:
                raise RuntimeError("CPU temperature is unavailable during cooldown")
            samples.append(
                {
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "temp_c": temperature,
                    "target_c": args.cooldown_temp_c,
                    "upper_bound_c": upper,
                    "fan_active": controller is not None,
                }
            )
            if temperature <= upper:
                break
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"CPU did not cool to <= {upper:.2f}C within "
                    f"{args.max_wait_min:.1f} min; last={temperature:.2f}C"
                )
            if args.cooldown_fan and controller is None:
                controller = start_cooldown_fan(args)
                print("[core-suite] cooldown fan: 100% PWM", flush=True)
            mode = "fan=100%" if controller is not None else "passive"
            print(
                f"[core-suite] cooling: {temperature:.2f}C > {upper:.2f}C ({mode})",
                flush=True,
            )
            time.sleep(args.poll_sec)
    finally:
        if controller is not None:
            controller.close()
            print("[core-suite] cooldown fan stopped; GPIO released", flush=True)
            if args.cooldown_fan_settle_sec > 0:
                time.sleep(args.cooldown_fan_settle_sec)
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    with trace_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(samples[0].keys()))
        writer.writeheader()
        writer.writerows(samples)
    final_temperature = cpu_temp_c()
    if final_temperature is None:
        raise RuntimeError("CPU temperature became unavailable after cooldown")
    return samples, final_temperature


def print_plan(args: argparse.Namespace, specs: list[ExperimentSpec]) -> None:
    preview_root = args.output_dir / args.run_id
    for index, spec in enumerate(specs, start=1):
        output = preview_root / f"{index:02d}_{spec.key}" / "runtime.csv"
        command = build_command(args, spec, output, first_run=index == 1)
        print(f"\n{index}. {spec.title}")
        print("   " + subprocess.list2cmdline(command))


def run_quality_analysis(
    args: argparse.Namespace,
    suite_dir: Path,
    run_dirs: list[Path],
) -> dict[str, Any]:
    analysis_dir = suite_dir / "analysis"
    analysis_dir.mkdir()
    teacher_runtime = run_dirs[0] / "runtime.csv"
    teacher_detections = run_dirs[0] / "runtime_detections.jsonl"
    student_runtime = [run_dir / "runtime.csv" for run_dir in run_dirs[1:]]
    student_detections = [
        run_dir / "runtime_detections.jsonl" for run_dir in run_dirs[1:]
    ]
    command = [
        sys.executable,
        str(ROOT / "scripts" / "evaluate_strategy_detection_quality.py"),
        "--teacher",
        str(teacher_detections),
        "--students",
        *[str(path) for path in student_detections],
        "--teacher-csv",
        str(teacher_runtime),
        "--include-teacher-summary",
        "--student-csvs",
        *[str(path) for path in student_runtime],
        "--output",
        str(analysis_dir / "core_quality_summary.csv"),
        "--matches-output",
        str(analysis_dir / "core_quality_frames.csv"),
        "--plot-output",
        str(analysis_dir / "core_quality.png"),
        "--label-source",
        "student",
    ]
    print("[core-suite] evaluating pseudo-label quality", flush=True)
    completed = subprocess.run(command, cwd=ROOT)
    result = {
        "command": command,
        "returncode": completed.returncode,
        "summary": str(analysis_dir / "core_quality_summary.csv"),
        "frame_details": str(analysis_dir / "core_quality_frames.csv"),
        "plot": str(analysis_dir / "core_quality.png"),
    }
    if completed.returncode != 0:
        print(
            "[core-suite] WARNING: quality analysis failed, but all raw experiment "
            "logs are preserved.",
            flush=True,
        )
    return result


def main() -> None:
    args = parse_args()
    validate_args(args)
    specs = experiment_specs()
    if args.plan_only:
        print_plan(args, specs)
        return

    suite_dir = args.output_dir / args.run_id
    suite_dir.mkdir(parents=True, exist_ok=False)
    manifest: dict[str, Any] = {
        "protocol": "focused cumulative ablation with controlled cooldown",
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "ablation_order": [asdict(spec) for spec in specs],
        "input_sha256": sha256(args.video),
        "model_sha256": {
            "native_fp32_640": sha256(args.native_model),
            "quantized_int8_320": sha256(args.quantized_model_320),
            "quantized_int8_640": sha256(args.quantized_model_640),
        },
        "fan_protocol": {
            "formal_on_temp_c": args.fan_on_temp_c,
            "formal_off_temp_c": args.fan_off_temp_c,
            "formal_full_temp_c": args.fan_full_temp_c,
            "formal_min_duty_cycle": args.fan_min_duty_cycle,
            "formal_max_duty_cycle": args.fan_max_duty_cycle,
            "cooldown_duty_cycle": 1.0 if args.cooldown_fan else 0.0,
        },
        "system_before": system_snapshot(),
        "runs": [],
    }
    write_json(suite_dir / "manifest.json", manifest)
    verify_fan_hardware(args, suite_dir)

    run_dirs: list[Path] = []
    for index, spec in enumerate(specs, start=1):
        label = f"{index:02d}_{spec.key}"
        run_dir = suite_dir / label
        run_dir.mkdir()
        run_dirs.append(run_dir)
        _, pre_child_temperature = cool_to_start_temperature(
            args,
            run_dir / "cooldown_trace.csv",
        )
        output = run_dir / "runtime.csv"
        command = build_command(
            args,
            spec,
            output,
            first_run=index == 1,
        )
        print(
            f"\n[core-suite] starting {index}/{len(specs)}: {spec.title}\n"
            f"  model condition: {spec.model_condition}\n"
            f"  LK enabled: {'yes' if spec.strategy == 'scene_track_lk' else 'no'}\n"
            f"  ROI refresh: {'no' if spec.disable_roi_refresh or spec.strategy != 'scene_track_lk' else 'yes'}\n"
            f"  formal PWM fan: {'yes' if spec.fan_control == 'enabled' else 'no'}\n"
            f"  temperature before child startup: {pre_child_temperature:.2f}C\n"
            f"  output directory: {run_dir}",
            flush=True,
        )
        run_meta: dict[str, Any] = {
            **asdict(spec),
            "order": index,
            "command": command,
            "pre_child_temperature_c": pre_child_temperature,
            "first_run_temperature_warmup": bool(
                index == 1 and args.first_warmup_temp_c > 0
            ),
            "system_before": system_snapshot(),
            "started_utc": datetime.now(timezone.utc).isoformat(),
        }
        write_json(run_dir / "run_manifest.json", run_meta)
        returncode = run_with_temperature_trace(
            command,
            run_dir / "temperature_trace.csv",
            args.temperature_trace_sec,
            args.progress_sec,
        )
        run_meta.update(
            {
                "finished_utc": datetime.now(timezone.utc).isoformat(),
                "returncode": returncode,
                "formal_start_temperature_c": first_logged_temperature(output),
                "end_temperature_c": cpu_temp_c(),
                "system_after": system_snapshot(),
                "runtime_log": str(output),
                "profile_log": str(output.with_name("runtime_profile.csv")),
                "detections_log": str(
                    output.with_name("runtime_detections.jsonl")
                ),
            }
        )
        write_json(run_dir / "run_manifest.json", run_meta)
        manifest["runs"].append(run_meta)
        write_json(suite_dir / "manifest.json", manifest)
        print(
            f"[core-suite] finished {index}/{len(specs)}: {spec.key}; "
            f"returncode={returncode}; formal start="
            f"{run_meta['formal_start_temperature_c']}C; "
            f"end={run_meta['end_temperature_c']}C",
            flush=True,
        )
        if returncode != 0:
            raise RuntimeError(
                f"Experiment {spec.key} failed with exit code {returncode}; suite stopped"
            )

    manifest["system_after"] = system_snapshot()
    manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()
    if args.evaluate_quality and args.log_detections:
        manifest["quality_analysis"] = run_quality_analysis(
            args,
            suite_dir,
            run_dirs,
        )
    write_json(suite_dir / "manifest.json", manifest)
    print(f"Focused core ablation complete: {suite_dir}", flush=True)


if __name__ == "__main__":
    main()
