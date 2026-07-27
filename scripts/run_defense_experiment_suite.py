#!/usr/bin/env python3
"""Run the defense-oriented A-E RT-DETR experiment matrix.

Identical conditions shared by multiple thesis questions are executed once and
reused in the analysis. Every formal run starts after controlled cooldown; only
the first run performs an RT-DETR temperature warmup.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
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
    query_budget_override: int | None = None
    temperature_query_budget: bool = False


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
            "Event LK + ROI, full query budget baseline (Q=300)",
            ("D",),
            "scene_track_lk",
            "dynamic_query_roi_family",
            query_budget_override=300,
        ),
        ExperimentSpec(
            "int8_event_lk_roi_q64",
            "Event LK + ROI, fixed reduced query budget (Q=64)",
            ("D",),
            "scene_track_lk",
            "dynamic_query_roi_family",
            query_budget_override=64,
        ),
        ExperimentSpec(
            "int8_event_lk_roi_qthermal",
            "Event LK + ROI, query-only thermal adaptation (Q=64/48/40/32)",
            ("D",),
            "scene_track_lk",
            "dynamic_query_roi_family",
            temperature_query_budget=True,
        ),
        ExperimentSpec(
            "proposed_software",
            "Full controller: thermal resolution/interval plus query allocation",
            ("D", "E"),
            "scene_thermal_interval_lk",
            "dynamic_query_adaptive_family",
            temperature_query_budget=True,
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
            "dynamic_query_adaptive_family",
            fan_control="enabled",
            temperature_query_budget=True,
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
        "--dynamic-query-model-320",
        type=Path,
        default=ROOT / "models" / "rtdetr_r18_lite_pi4_320_int8_dynamic_q.onnx",
    )
    parser.add_argument(
        "--dynamic-query-model-480",
        type=Path,
        default=ROOT / "models" / "rtdetr_r18_lite_pi4_480_int8_dynamic_q.onnx",
    )
    parser.add_argument(
        "--dynamic-query-model-640",
        type=Path,
        default=ROOT / "models" / "rtdetr_r18_lite_pi4_640_int8_dynamic_q.onnx",
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
    parser.add_argument(
        "--resume-dir",
        type=Path,
        default=None,
        help=(
            "Resume an existing suite directory. Successful conditions are "
            "kept and only incomplete conditions are run."
        ),
    )
    parser.add_argument(
        "--resume-rerun-from-index",
        type=int,
        default=None,
        help=(
            "When resuming, invalidate and rerun this 1-based condition index "
            "and all later conditions. Old directories are preserved."
        ),
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
    parser.add_argument(
        "--query-budget-normal", type=int, default=64,
        help="Normal-temperature graph query budget for adaptive conditions.",
    )
    parser.add_argument(
        "--query-budget-warm", type=int, default=48,
        help="Warm-temperature graph query budget for adaptive conditions.",
    )
    parser.add_argument(
        "--query-budget-hot", type=int, default=40,
        help="Hot-temperature graph query budget for adaptive conditions.",
    )
    parser.add_argument(
        "--query-budget-critical", type=int, default=32,
        help="Critical-temperature graph query budget for adaptive conditions.",
    )
    parser.add_argument(
        "--query-budget-hysteresis-c", type=float, default=4.0,
        help="Temperature hysteresis used by the adaptive query controller.",
    )
    parser.add_argument(
        "--thermal-normal-max-c",
        type=float,
        default=58.0,
        help="Shared normal-to-warm boundary for every formal condition.",
    )
    parser.add_argument(
        "--thermal-warm-max-c",
        type=float,
        default=66.0,
        help="Shared warm-to-hot boundary for every formal condition.",
    )
    parser.add_argument(
        "--thermal-critical-c",
        type=float,
        default=76.0,
        help="Shared hot-to-critical boundary for every formal condition.",
    )
    parser.add_argument(
        "--thermal-hysteresis-c",
        type=float,
        default=4.0,
        help="Shared thermal-state hysteresis for every formal condition.",
    )
    parser.add_argument("--fan-on-temp-c", type=float, default=68.0)
    parser.add_argument("--fan-off-temp-c", type=float, default=62.0)
    parser.add_argument("--fan-full-temp-c", type=float, default=82.0)
    parser.add_argument("--fan-min-duty-cycle", type=float, default=0.35)
    parser.add_argument("--fan-max-duty-cycle", type=float, default=1.0)
    parser.add_argument("--loop-video", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--log-detections", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cooldown-fan", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fan-preflight", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--power-preflight",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fail on current or historical undervoltage before/after formal runs.",
    )
    parser.add_argument(
        "--require-controller-coverage",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Require the proposed software run to observe both a 480-pixel "
            "detector call and a thermal interval change."
        ),
    )
    parser.add_argument("--evaluate-quality", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--make-plots", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--apply-runtime-actions",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply the configured performance governor/affinity (default: enabled).",
    )
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
    if not (
        args.thermal_normal_max_c
        <= args.thermal_warm_max_c
        <= args.thermal_critical_c
    ):
        raise ValueError(
            "Shared thermal boundaries must satisfy normal <= warm <= critical"
        )
    if args.thermal_hysteresis_c < 0:
        raise ValueError("--thermal-hysteresis-c cannot be negative")
    budgets = (
        args.query_budget_normal,
        args.query_budget_warm,
        args.query_budget_hot,
        args.query_budget_critical,
    )
    if any(int(value) <= 0 for value in budgets):
        raise ValueError("All query budgets must be positive")
    if not args.query_budget_critical <= args.query_budget_hot <= args.query_budget_warm <= args.query_budget_normal:
        raise ValueError(
            "Query budgets must satisfy critical <= hot <= warm <= normal"
        )
    if args.query_budget_hysteresis_c < 0:
        raise ValueError("--query-budget-hysteresis-c cannot be negative")
    if args.resume_rerun_from_index is not None and args.resume_rerun_from_index < 1:
        raise ValueError("--resume-rerun-from-index must be positive")
    if args.resume_rerun_from_index is not None and args.resume_dir is None:
        raise ValueError("--resume-rerun-from-index requires --resume-dir")
    required = {args.config, args.video}
    kinds = {spec.model_kind for spec in specs}
    if "native" in kinds:
        required.add(args.native_model)
    if "int8_640" in kinds:
        required.add(args.quantized_model_640)
    if kinds.intersection({"dynamic_query_roi_family", "dynamic_query_adaptive_family"}):
        required.update(
            {args.dynamic_query_model_320, args.dynamic_query_model_640}
        )
    if "dynamic_query_adaptive_family" in kinds:
        required.add(args.dynamic_query_model_480)
    missing = [str(path) for path in sorted(required, key=str) if not path.exists()]
    if missing:
        raise FileNotFoundError("Required input(s) not found: " + ", ".join(missing))


def model_args(args: argparse.Namespace, spec: ExperimentSpec) -> list[str]:
    if spec.model_kind == "native":
        return ["--model", str(args.native_model)]
    if spec.model_kind == "int8_640":
        return ["--model", str(args.quantized_model_640)]
    mappings = [
        f"320={args.dynamic_query_model_320}",
        f"640={args.dynamic_query_model_640}",
    ]
    if spec.model_kind == "dynamic_query_adaptive_family":
        mappings.insert(1, f"480={args.dynamic_query_model_480}")
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
    command.extend(
        [
            "--thermal-normal-max-c",
            str(args.thermal_normal_max_c),
            "--thermal-warm-max-c",
            str(args.thermal_warm_max_c),
            "--thermal-critical-c",
            str(args.thermal_critical_c),
            "--thermal-hysteresis-c",
            str(args.thermal_hysteresis_c),
        ]
    )
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
    if spec.model_kind.startswith("dynamic_query_"):
        command.extend(["--query-budget-mode", "strict"])
    if spec.query_budget_override is not None:
        command.extend(
            ["--query-budget-override", str(spec.query_budget_override)]
        )
    if spec.temperature_query_budget:
        command.extend(
            [
                "--temperature-query-budget",
                "--query-budget-normal",
                str(args.query_budget_normal),
                "--query-budget-warm",
                str(args.query_budget_warm),
                "--query-budget-hot",
                str(args.query_budget_hot),
                "--query-budget-critical",
                str(args.query_budget_critical),
                "--query-budget-hysteresis-c",
                str(args.query_budget_hysteresis_c),
            ]
        )
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


def assert_no_undervoltage(context: str) -> None:
    completed = subprocess.run(
        ["vcgencmd", "get_throttled"],
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Cannot verify Raspberry Pi power state during {context}: "
            f"{completed.stderr.strip() or completed.stdout.strip()}"
        )
    match = re.search(r"0x([0-9a-fA-F]+)", completed.stdout)
    if match is None:
        raise RuntimeError(
            f"Unrecognised vcgencmd get_throttled output during {context}: "
            f"{completed.stdout!r}"
        )
    mask = int(match.group(1), 16)
    current_undervoltage = bool(mask & 0x1)
    historical_undervoltage = bool(mask & 0x10000)
    if current_undervoltage or historical_undervoltage:
        raise RuntimeError(
            f"Undervoltage invalidates the experiment ({context}): "
            f"throttled=0x{mask:x}. Fix the PSU/cable and reboot until "
            "`vcgencmd get_throttled` reports 0x0."
        )
    print(
        f"[defense-suite] power check passed ({context}): "
        f"throttled=0x{mask:x}",
        flush=True,
    )


def observed_runtime_coverage(path: Path) -> dict[str, Any]:
    """Summarize actual controller actions, not merely configured options."""
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    def counts(column: str, *, inferred_only: bool = False) -> dict[str, int]:
        selected = rows
        if inferred_only:
            selected = [
                row
                for row in rows
                if str(row.get("did_infer", "")).strip().lower() == "true"
            ]
        result: dict[str, int] = {}
        for row in selected:
            value = str(row.get(column, ""))
            result[value] = result.get(value, 0) + 1
        return result

    return {
        "frames": len(rows),
        "detector_call_resolutions": counts("detector_call_resolution", inferred_only=True),
        "inference_intervals": counts("inference_interval"),
        "action_modes": counts("action_mode"),
        "control_thermal_states": counts("control_thermal_state"),
        "query_budgets": counts("query_budget_applied", inferred_only=True),
    }


def validate_controller_coverage(
    spec: ExperimentSpec,
    coverage: dict[str, Any],
    *,
    required: bool,
) -> None:
    """Fail a proposed-controller run that never exercises its new controls."""
    if not required or spec.key != "proposed_software":
        return
    resolutions = coverage.get("detector_call_resolutions", {})
    intervals = coverage.get("inference_intervals", {})
    observed_480 = int(resolutions.get("480", 0))
    observed_non_unit_interval = 0
    for value, count in intervals.items():
        try:
            non_unit = float(value) != 1.0
        except (TypeError, ValueError):
            non_unit = bool(value)
        if non_unit:
            observed_non_unit_interval += count
    if observed_480 <= 0 or observed_non_unit_interval <= 0:
        raise RuntimeError(
            "Proposed controller coverage is incomplete: expected at least "
            "one 480-pixel detector call and one non-unit inference interval, "
            f"observed resolutions={resolutions}, intervals={intervals}. "
            "Use shared thermal thresholds or rerun with "
            "--no-require-controller-coverage only if intentionally testing "
            "a no-trigger case."
        )


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

    if args.resume_dir is not None:
        suite_dir = args.resume_dir.resolve()
        manifest_path = suite_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Cannot resume; missing manifest: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest_keys = [item.get("key") for item in manifest.get("experiment_order", [])]
        requested_keys = [spec.key for spec in specs]
        if manifest_keys != requested_keys:
            raise ValueError(
                "Resume arguments do not match the existing suite experiment order. "
                f"Existing={manifest_keys}; requested={requested_keys}"
            )
        if args.resume_rerun_from_index is not None:
            rerun_index = args.resume_rerun_from_index
            if rerun_index > len(specs):
                raise ValueError(
                    f"--resume-rerun-from-index must be <= {len(specs)}"
                )
            invalidated: list[dict[str, Any]] = []
            for item in list(manifest.get("runs", [])):
                order = int(item.get("order", 0))
                if order < rerun_index:
                    continue
                old_dir = suite_dir / f"{order:02d}_{item.get('key')}"
                if old_dir.exists():
                    backup = suite_dir / (
                        f"{old_dir.name}_invalidated_power_"
                        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    )
                    shutil.move(str(old_dir), str(backup))
                    invalidated.append(
                        {"key": item.get("key"), "order": order, "path": str(backup)}
                    )
            manifest["runs"] = [
                item
                for item in manifest.get("runs", [])
                if int(item.get("order", 0)) < rerun_index
            ]
            manifest.setdefault("invalidated_runs", []).extend(invalidated)
            write_json(suite_dir / "manifest.json", manifest)
            print(
                f"[defense-suite] invalidated {len(invalidated)} old run(s) "
                f"from index {rerun_index}; originals preserved",
                flush=True,
            )
        print(f"[defense-suite] resuming existing suite: {suite_dir}", flush=True)
    else:
        suite_dir = args.output_dir / args.run_id
        suite_dir.mkdir(parents=True, exist_ok=False)
        manifest = {
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
                "dynamic_query_int8_320": sha256(args.dynamic_query_model_320) if args.dynamic_query_model_320.exists() else None,
                "dynamic_query_int8_480": sha256(args.dynamic_query_model_480) if args.dynamic_query_model_480.exists() else None,
                "dynamic_query_int8_640": sha256(args.dynamic_query_model_640) if args.dynamic_query_model_640.exists() else None,
            },
            "system_before": system_snapshot(),
            "runs": [],
        }
        write_json(suite_dir / "manifest.json", manifest)
    if args.power_preflight:
        assert_no_undervoltage("suite start")
    if any(spec.fan_control == "enabled" for spec in specs) or args.cooldown_fan:
        verify_fan_hardware(args, suite_dir)
        if args.power_preflight:
            assert_no_undervoltage("after fan preflight")

    successful_keys = {
        item.get("key")
        for item in manifest.get("runs", [])
        if item.get("returncode") == 0
        and (suite_dir / f"{int(item.get('order', 0)):02d}_{item.get('key')}").exists()
    }
    run_dirs: list[Path] = []
    for index, spec in enumerate(specs, 1):
        run_dir = suite_dir / f"{index:02d}_{spec.key}"
        run_dirs.append(run_dir)
        if spec.key in successful_keys:
            print(
                f"[defense-suite] keeping completed {index}/{len(specs)}: {spec.key}",
                flush=True,
            )
            continue
        if run_dir.exists():
            raise RuntimeError(
                f"Cannot safely resume {spec.key}: {run_dir} exists but is not "
                "marked successful. Preserve it and choose a new suite or move "
                "the incomplete directory before retrying."
            )
        run_dir.mkdir()
        _, pre_temperature = cool_to_start_temperature(
            args, run_dir / "cooldown_trace.csv"
        )
        if args.power_preflight:
            assert_no_undervoltage(f"before {spec.key}, after cooldown")
        output = run_dir / "runtime.csv"
        command = build_command(args, spec, output, first_run=index == 1)
        print(
            f"\n[defense-suite] starting {index}/{len(specs)}: {spec.key}\n"
            f"  thesis groups: {','.join(spec.groups)}\n"
            f"  condition: {spec.title}\n"
            f"  model set: {spec.model_kind}\n"
            f"  detector interval: {spec.fixed_interval or 'controller/event'}\n"
            f"  query budget: "
            f"{'thermal-adaptive' if spec.temperature_query_budget else spec.query_budget_override or 'model/action default'}\n"
            f"  shared thermal bands: normal<{args.thermal_normal_max_c:.1f}, "
            f"warm<{args.thermal_warm_max_c:.1f}, critical>={args.thermal_critical_c:.1f}C\n"
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
        if output.exists():
            coverage = observed_runtime_coverage(output)
            run_meta["observed_coverage"] = coverage
        write_json(run_dir / "run_manifest.json", run_meta)
        manifest["runs"] = [
            item for item in manifest.get("runs", []) if item.get("key") != spec.key
        ]
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
        validate_controller_coverage(
            spec,
            run_meta.get("observed_coverage", {}),
            required=args.require_controller_coverage,
        )
        if run_meta.get("observed_coverage"):
            print(
                f"[defense-suite] observed coverage {spec.key}: "
                f"resolutions={run_meta['observed_coverage']['detector_call_resolutions']}; "
                f"intervals={run_meta['observed_coverage']['inference_intervals']}",
                flush=True,
            )
        if args.power_preflight:
            assert_no_undervoltage(f"after {spec.key}")

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
