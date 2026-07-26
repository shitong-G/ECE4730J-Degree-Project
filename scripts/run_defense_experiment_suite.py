#!/usr/bin/env python3
"""Run the defense-oriented A-E RT-DETR experiment matrix.

Identical conditions shared by multiple thesis questions are executed once and
reused in the analysis. Every formal run starts after controlled cooldown; only
the first run performs an RT-DETR temperature warmup.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from run_controlled_experiment_suite import (
    cpu_temp_c,
    first_logged_temperature,
    run_with_temperature_trace,
    sha256,
    system_snapshot,
    verify_fan_hardware,
    write_json,
)
from run_core_ablation_suite import cool_to_start_temperature


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ExperimentSpec:
    key: str
    title: str
    groups: tuple[str, ...]
    strategy: str
    model_kind: str
    fixed_interval: int | None = None
    enable_lk: bool = False
    disable_roi: bool = False
    fan_control: str = "disabled"


def all_specs() -> list[ExperimentSpec]:
    """Return unique conditions; group membership expresses result reuse."""
    return [
        ExperimentSpec(
            "native_fp32",
            "Native FP32 RT-DETR, every frame, passive cooling",
            ("A", "B"),
            "native_rtdetr",
            "native",
        ),
        ExperimentSpec(
            "int8_every_frame",
            "INT8 RT-DETR, every frame, passive cooling",
            ("B", "C"),
            "native_rtdetr",
            "int8_640",
        ),
        ExperimentSpec(
            "int8_fixed_skip_2",
            "INT8 fixed skip, detector interval 2, no tracker",
            ("C",),
            "native_rtdetr",
            "int8_640",
            fixed_interval=2,
        ),
        ExperimentSpec(
            "int8_fixed_skip_5",
            "INT8 fixed skip, detector interval 5, no tracker",
            ("C",),
            "native_rtdetr",
            "int8_640",
            fixed_interval=5,
        ),
        ExperimentSpec(
            "int8_fixed_skip_10",
            "INT8 fixed skip, detector interval 10, no tracker",
            ("C",),
            "native_rtdetr",
            "int8_640",
            fixed_interval=10,
        ),
        ExperimentSpec(
            "int8_periodic_lk_5",
            "INT8 periodic detect-and-LK, detector interval 5",
            ("C",),
            "native_rtdetr",
            "int8_640",
            fixed_interval=5,
            enable_lk=True,
        ),
        ExperimentSpec(
            "int8_event_lk",
            "INT8 event-triggered LK, full-frame refresh only",
            ("C", "D"),
            "scene_track_lk",
            "int8_640",
            disable_roi=True,
        ),
        ExperimentSpec(
            "int8_event_lk_roi",
            "INT8 event-triggered LK with 320 ROI refresh",
            ("D",),
            "scene_track_lk",
            "int8_roi_family",
        ),
        ExperimentSpec(
            "proposed_software",
            "INT8 + event LK + ROI + scene/thermal software controller",
            ("D", "E"),
            "scene_thermal_interval_lk",
            "int8_adaptive_family",
        ),
        ExperimentSpec(
            "native_fp32_pwm_fan",
            "Native FP32 RT-DETR with threshold/PWM fan",
            ("E",),
            "native_rtdetr",
            "native",
            fan_control="enabled",
        ),
        ExperimentSpec(
            "proposed_software_pwm_fan",
            "Proposed software controller with threshold/PWM fan",
            ("E",),
            "scene_thermal_interval_lk",
            "int8_adaptive_family",
            fan_control="enabled",
        ),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "raspberry_pi4.yaml")
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
        "--quantized-model-480",
        type=Path,
        default=ROOT / "models" / "rtdetr_r18_lite_pi4_480_int8.onnx",
    )
    parser.add_argument(
        "--quantized-model-640",
        type=Path,
        default=ROOT / "models" / "rtdetr_r18_lite_pi4_640_int8.onnx",
    )
    parser.add_argument(
        "--groups",
        default="ABCDE",
        help="Subset of defense groups to run, e.g. ABC, D, or A,C,E.",
    )
    parser.add_argument("--duration-min", type=float, default=20.0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "experiments" / "defense_suite",
    )
    parser.add_argument(
        "--run-id",
        default=datetime.now().strftime("defense_%Y%m%d_%H%M%S"),
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
    parser.add_argument("--fan-on-temp-c", type=float, default=68.0)
    parser.add_argument("--fan-off-temp-c", type=float, default=62.0)
    parser.add_argument("--fan-full-temp-c", type=float, default=82.0)
    parser.add_argument("--fan-min-duty-cycle", type=float, default=0.35)
    parser.add_argument("--fan-max-duty-cycle", type=float, default=1.0)
    parser.add_argument("--loop-video", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--log-detections", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cooldown-fan", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fan-preflight", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--evaluate-quality", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--make-plots", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--apply-runtime-actions", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    return parser.parse_args()


def selected_groups(raw: str) -> set[str]:
    groups = {char for char in raw.upper() if char in "ABCDE"}
    invalid = {char for char in raw.upper() if char.isalpha() and char not in "ABCDE"}
    if invalid or not groups:
        raise ValueError("--groups must contain only A, B, C, D, and/or E")
    return groups


def select_specs(groups: set[str]) -> list[ExperimentSpec]:
    specs = all_specs()
    selected = [spec for spec in specs if groups.intersection(spec.groups)]
    native = specs[0]
    if native not in selected:
        selected.insert(0, native)
    return selected


def validate_args(args: argparse.Namespace, specs: list[ExperimentSpec]) -> None:
    if args.duration_min <= 0:
        raise ValueError("--duration-min must be positive")
    if args.cooldown_temp_c <= 0 or args.cooldown_tolerance_c < 0:
        raise ValueError("Cooldown temperature must be positive and tolerance non-negative")
    if args.max_wait_min <= 0 or args.poll_sec <= 0:
        raise ValueError("Wait and polling durations must be positive")
    if not args.fan_off_temp_c <= args.fan_on_temp_c <= args.fan_full_temp_c:
        raise ValueError("Fan thresholds must satisfy off <= on <= full")
    if not 0 <= args.fan_min_duty_cycle <= args.fan_max_duty_cycle <= 1:
        raise ValueError("Fan duty cycles must satisfy 0 <= min <= max <= 1")
    required = {args.config, args.video}
    kinds = {spec.model_kind for spec in specs}
    if "native" in kinds:
        required.add(args.native_model)
    if kinds.intersection({"int8_640", "int8_roi_family", "int8_adaptive_family"}):
        required.add(args.quantized_model_640)
    if kinds.intersection({"int8_roi_family", "int8_adaptive_family"}):
        required.add(args.quantized_model_320)
    if "int8_adaptive_family" in kinds:
        required.add(args.quantized_model_480)
    missing = [str(path) for path in sorted(required, key=str) if not path.exists()]
    if missing:
        raise FileNotFoundError("Required input(s) not found: " + ", ".join(missing))


def model_args(args: argparse.Namespace, spec: ExperimentSpec) -> list[str]:
    if spec.model_kind == "native":
        return ["--model", str(args.native_model)]
    if spec.model_kind == "int8_640":
        return ["--model", str(args.quantized_model_640)]
    mappings = [
        f"320={args.quantized_model_320}",
        f"640={args.quantized_model_640}",
    ]
    if spec.model_kind == "int8_adaptive_family":
        mappings.insert(1, f"480={args.quantized_model_480}")
    return ["--model-paths-by-resolution", ",".join(mappings)]


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
    if spec.fixed_interval is not None:
        command.extend(
            [
                "--fixed-inference-interval",
                str(spec.fixed_interval),
                "--fixed-input-resolution",
                "640",
            ]
        )
    if spec.enable_lk:
        command.append("--enable-lk-tracking")
    if spec.disable_roi:
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
    return command + model_args(args, spec)


def print_plan(args: argparse.Namespace, specs: list[ExperimentSpec]) -> None:
    root = args.output_dir / args.run_id
    print(f"Selected groups: {','.join(sorted(selected_groups(args.groups)))}")
    print(f"Unique formal runs: {len(specs)}")
    for index, spec in enumerate(specs, 1):
        output = root / f"{index:02d}_{spec.key}" / "runtime.csv"
        memberships = ",".join(spec.groups)
        print(f"\n{index:02d}. [{memberships}] {spec.title}")
        print("    " + subprocess.list2cmdline(build_command(args, spec, output, first_run=index == 1)))


def run_quality_analysis(suite_dir: Path, run_dirs: list[Path]) -> int:
    analysis = suite_dir / "analysis"
    analysis.mkdir(exist_ok=True)
    teacher = run_dirs[0]
    students = run_dirs[1:]
    command = [
        sys.executable,
        str(ROOT / "scripts" / "evaluate_strategy_detection_quality.py"),
        "--teacher",
        str(teacher / "runtime_detections.jsonl"),
        "--students",
        *[str(path / "runtime_detections.jsonl") for path in students],
        "--teacher-csv",
        str(teacher / "runtime.csv"),
        "--include-teacher-summary",
        "--student-csvs",
        *[str(path / "runtime.csv") for path in students],
        "--output",
        str(analysis / "quality_summary.csv"),
        "--matches-output",
        str(analysis / "quality_frames.csv"),
        "--plot-output",
        str(analysis / "quality_overview.png"),
        "--label-source",
        "student",
    ]
    return subprocess.run(command, cwd=ROOT).returncode


def run_plot_analysis(suite_dir: Path) -> int:
    command = [
        sys.executable,
        str(ROOT / "scripts" / "analyze_defense_experiment_suite.py"),
        "--suite-dir",
        str(suite_dir),
    ]
    return subprocess.run(command, cwd=ROOT).returncode


def main() -> None:
    args = parse_args()
    groups = selected_groups(args.groups)
    specs = select_specs(groups)
    validate_args(args, specs)
    if args.plan_only:
        print_plan(args, specs)
        return

    suite_dir = args.output_dir / args.run_id
    suite_dir.mkdir(parents=True, exist_ok=False)
    manifest: dict[str, Any] = {
        "protocol": "defense A-E matrix with unique-condition reuse and controlled cooldown",
        "selected_groups": sorted(groups),
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "experiment_order": [asdict(spec) for spec in specs],
        "input_sha256": sha256(args.video),
        "model_sha256": {
            "native_640": sha256(args.native_model) if args.native_model.exists() else None,
            "int8_320": sha256(args.quantized_model_320) if args.quantized_model_320.exists() else None,
            "int8_480": sha256(args.quantized_model_480) if args.quantized_model_480.exists() else None,
            "int8_640": sha256(args.quantized_model_640) if args.quantized_model_640.exists() else None,
        },
        "system_before": system_snapshot(),
        "runs": [],
    }
    write_json(suite_dir / "manifest.json", manifest)
    if any(spec.fan_control == "enabled" for spec in specs) or args.cooldown_fan:
        verify_fan_hardware(args, suite_dir)

    run_dirs: list[Path] = []
    for index, spec in enumerate(specs, 1):
        run_dir = suite_dir / f"{index:02d}_{spec.key}"
        run_dir.mkdir()
        run_dirs.append(run_dir)
        _, pre_temperature = cool_to_start_temperature(
            args, run_dir / "cooldown_trace.csv"
        )
        output = run_dir / "runtime.csv"
        command = build_command(args, spec, output, first_run=index == 1)
        print(
            f"\n[defense-suite] starting {index}/{len(specs)}: {spec.key}\n"
            f"  thesis groups: {','.join(spec.groups)}\n"
            f"  condition: {spec.title}\n"
            f"  model set: {spec.model_kind}\n"
            f"  detector interval: {spec.fixed_interval or 'controller/event'}\n"
            f"  LK: {'yes' if spec.enable_lk or 'lk' in spec.strategy else 'no'}\n"
            f"  formal PWM fan: {'yes' if spec.fan_control == 'enabled' else 'no'}\n"
            f"  pre-child temperature: {pre_temperature:.2f}C\n"
            f"  output: {run_dir}",
            flush=True,
        )
        run_meta: dict[str, Any] = {
            **asdict(spec),
            "order": index,
            "command": command,
            "pre_child_temperature_c": pre_temperature,
            "first_run_temperature_warmup": index == 1,
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "system_before": system_snapshot(),
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
                "detections_log": str(run_dir / "runtime_detections.jsonl"),
            }
        )
        write_json(run_dir / "run_manifest.json", run_meta)
        manifest["runs"].append(run_meta)
        write_json(suite_dir / "manifest.json", manifest)
        print(
            f"[defense-suite] finished {index}/{len(specs)}: {spec.key}; "
            f"returncode={returncode}; formal_start="
            f"{run_meta['formal_start_temperature_c']}C; end={run_meta['end_temperature_c']}C",
            flush=True,
        )
        if returncode:
            raise RuntimeError(f"{spec.key} failed with exit code {returncode}")

    if args.evaluate_quality and args.log_detections:
        manifest["quality_analysis_returncode"] = run_quality_analysis(
            suite_dir, run_dirs
        )
    if args.make_plots:
        manifest["plot_analysis_returncode"] = run_plot_analysis(suite_dir)
    manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["system_after"] = system_snapshot()
    write_json(suite_dir / "manifest.json", manifest)
    print(f"Defense experiment suite complete: {suite_dir}", flush=True)


if __name__ == "__main__":
    main()
