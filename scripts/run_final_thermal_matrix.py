#!/usr/bin/env python3
"""Run the final, repeatable software-only thermal experiment matrix.

This is intentionally a focused confirmatory protocol, not a parameter sweep.
It separates the six mechanisms required to support the thesis claims while
keeping all formal runs fan-free and using the same dynamic-query INT8 model
family wherever possible:

1. Native FP32 RT-DETR, every frame (hardware-performance reference).
2. Dynamic-query INT8 RT-DETR, Q=300, every frame (model reference).
3. Dynamic-query INT8 + event LK + ROI, Q=300 (temporal/ROI reference).
4. Same pipeline, fixed Q=40 (mean-budget static control).
5. Same pipeline, thermal-adaptive query budget only.
6. Full proposed controller: thermal interval/resolution + adaptive query.

By default there are 6 conditions x 3 repetitions x 20 minutes = 6 hours of
formal runtime.  With at most 8 minutes of forced cooldown per run and one
minute of process allowance, the planned upper bound is 8.7 hours; the suite
refuses a plan that exceeds --max-total-hours (10 hours by default).

Run on the Raspberry Pi from the project root, preferably with sudo -E:

  sudo -E .venv/bin/python scripts/run_final_thermal_matrix.py \
    --config configs/raspberry_pi4.yaml --video data/sample.mp4
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
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
    system_snapshot,
    verify_fan_hardware,
    write_json,
)
from run_core_ablation_suite import cool_to_start_temperature


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Condition:
    key: str
    title: str
    strategy: str
    model_kind: str
    query_budget: int | None = None
    temperature_query_budget: bool = False


def conditions() -> list[Condition]:
    """Return the minimal confirmatory matrix in causal-ablation order."""
    return [
        Condition(
            "fp32_native",
            "Native FP32 RT-DETR, 640, every frame",
            "native_rtdetr",
            "fp32",
        ),
        Condition(
            "int8_dynamic_q300_native",
            "Dynamic-query INT8 RT-DETR, 640, every frame, Q=300",
            "native_rtdetr",
            "dynamic",
            query_budget=300,
        ),
        Condition(
            "int8_dynamic_q300_lk_roi",
            "Dynamic-query INT8 + event LK + ROI refresh, Q=300",
            "scene_track_lk",
            "dynamic",
            query_budget=300,
        ),
        Condition(
            "int8_dynamic_q40_lk_roi",
            "Dynamic-query INT8 + event LK + ROI refresh, fixed Q=40",
            "scene_track_lk",
            "dynamic",
            query_budget=40,
        ),
        Condition(
            "int8_dynamic_qthermal_lk_roi",
            "Dynamic-query INT8 + event LK + ROI refresh, thermal query only",
            "scene_track_lk",
            "dynamic",
            temperature_query_budget=True,
        ),
        Condition(
            "proposed_software",
            "Full software controller: thermal interval/resolution + adaptive query",
            "scene_thermal_interval_lk",
            "dynamic",
            temperature_query_budget=True,
        ),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "raspberry_pi4.yaml")
    parser.add_argument("--video", type=Path, default=ROOT / "data" / "sample.mp4")
    parser.add_argument("--native-model", type=Path, default=ROOT / "models" / "rtdetr_r18_lite_pi4_640.onnx")
    parser.add_argument("--dynamic-query-model-320", type=Path, default=ROOT / "models" / "rtdetr_r18_lite_pi4_320_int8_dynamic_q.onnx")
    parser.add_argument("--dynamic-query-model-480", type=Path, default=ROOT / "models" / "rtdetr_r18_lite_pi4_480_int8_dynamic_q.onnx")
    parser.add_argument("--dynamic-query-model-640", type=Path, default=ROOT / "models" / "rtdetr_r18_lite_pi4_640_int8_dynamic_q.onnx")
    parser.add_argument("--duration-min", type=float, default=20.0)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "experiments" / "final_thermal_matrix")
    parser.add_argument("--run-id", default=datetime.now().strftime("final_thermal_%Y%m%d_%H%M%S"))
    parser.add_argument("--resume-dir", type=Path, default=None)
    parser.add_argument(
        "--start-temp-min-c", type=float, default=45.0,
        help="Minimum permitted one-shot temperature reading before a formal run.",
    )
    parser.add_argument(
        "--start-temp-max-c", type=float, default=50.0,
        help="Maximum permitted one-shot temperature reading before a formal run.",
    )
    parser.add_argument(
        "--start-temp-reading-tolerance-c", type=float, default=0.5,
        help=(
            "One-shot sensor tolerance applied only below --start-temp-min-c; "
            "default accepts 44.5–50.0C for the requested 45–50C window."
        ),
    )
    parser.add_argument("--max-cooldown-min", type=float, default=8.0)
    parser.add_argument("--poll-sec", type=float, default=5.0)
    parser.add_argument("--cooldown-fan-settle-sec", type=float, default=2.0)
    parser.add_argument("--temperature-trace-sec", type=float, default=1.0)
    parser.add_argument("--progress-sec", type=float, default=30.0)
    parser.add_argument("--max-total-hours", type=float, default=10.0)
    parser.add_argument("--startup-allowance-min", type=float, default=1.0)
    parser.add_argument("--query-budget-normal", type=int, default=64)
    parser.add_argument("--query-budget-warm", type=int, default=48)
    parser.add_argument("--query-budget-hot", type=int, default=40)
    parser.add_argument("--query-budget-critical", type=int, default=32)
    parser.add_argument("--query-budget-hysteresis-c", type=float, default=4.0)
    parser.add_argument("--thermal-normal-max-c", type=float, default=58.0)
    parser.add_argument("--thermal-warm-max-c", type=float, default=66.0)
    parser.add_argument("--thermal-critical-c", type=float, default=76.0)
    parser.add_argument("--thermal-hysteresis-c", type=float, default=4.0)
    parser.add_argument("--loop-video", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--log-detections", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cooldown-fan", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fan-preflight", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--power-preflight", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--apply-runtime-actions", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--plan-only", action="store_true")
    return parser.parse_args()


def ordered_conditions(repeat_index: int, specs: list[Condition]) -> list[Condition]:
    """Use a deterministic balanced order to reduce time-of-day/order bias."""
    if repeat_index % 3 == 0:
        return specs
    if repeat_index % 3 == 1:
        return list(reversed(specs))
    return specs[2:] + specs[:2]


def expected_plan_minutes(args: argparse.Namespace, count: int) -> float:
    return count * (
        args.duration_min + args.max_cooldown_min + args.startup_allowance_min
    )


def validate_args(args: argparse.Namespace, specs: list[Condition]) -> None:
    if args.duration_min <= 0 or args.repeats < 1:
        raise ValueError("--duration-min must be positive and --repeats must be >= 1")
    if args.max_cooldown_min <= 0 or args.max_total_hours <= 0:
        raise ValueError("Cooldown and total-duration limits must be positive")
    if args.start_temp_min_c <= 0 or args.start_temp_min_c > args.start_temp_max_c:
        raise ValueError("Start window must satisfy 0 < --start-temp-min-c <= --start-temp-max-c")
    if args.start_temp_reading_tolerance_c < 0:
        raise ValueError("--start-temp-reading-tolerance-c must be non-negative")
    if not args.thermal_normal_max_c <= args.thermal_warm_max_c <= args.thermal_critical_c:
        raise ValueError("Thermal boundaries must satisfy normal <= warm <= critical")
    budgets = (args.query_budget_critical, args.query_budget_hot, args.query_budget_warm, args.query_budget_normal)
    if any(value <= 0 for value in budgets) or tuple(sorted(budgets)) != budgets:
        raise ValueError("Query budgets must satisfy critical <= hot <= warm <= normal")
    paths = [
        args.config, args.video, args.native_model, args.dynamic_query_model_320,
        args.dynamic_query_model_480, args.dynamic_query_model_640,
    ]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Required input(s) not found: " + ", ".join(missing))
    planned = expected_plan_minutes(args, len(specs) * args.repeats)
    if planned > args.max_total_hours * 60.0:
        raise ValueError(
            f"Planned upper bound is {planned / 60.0:.2f} h, exceeding "
            f"--max-total-hours={args.max_total_hours:.2f}. Reduce repeats, "
            "duration, cooldown budget, or startup allowance."
        )


def dynamic_model_args(args: argparse.Namespace) -> list[str]:
    return [
        "--model-paths-by-resolution",
        ",".join([
            f"320={args.dynamic_query_model_320}",
            f"480={args.dynamic_query_model_480}",
            f"640={args.dynamic_query_model_640}",
        ]),
    ]


def build_command(args: argparse.Namespace, condition: Condition, output: Path) -> list[str]:
    command = [
        sys.executable, str(ROOT / "scripts" / "run_experiment.py"),
        "--config", str(args.config), "--strategy", condition.strategy,
        "--video", str(args.video), "--duration-min", str(args.duration_min),
        "--output", str(output),
        # Fan is deliberately disabled during every formal run.  The fan may
        # only be used during cooldown to normalize the next start state.
        "--fan-control", "disabled",
        "--thermal-normal-max-c", str(args.thermal_normal_max_c),
        "--thermal-warm-max-c", str(args.thermal_warm_max_c),
        "--thermal-critical-c", str(args.thermal_critical_c),
        "--thermal-hysteresis-c", str(args.thermal_hysteresis_c),
    ]
    if args.loop_video:
        command.append("--loop-video")
    if args.log_detections:
        command.append("--log-detections")
    if args.apply_runtime_actions:
        command.append("--apply-runtime-actions")
    if condition.model_kind == "fp32":
        command.extend(["--model", str(args.native_model)])
    else:
        command.extend(["--query-budget-mode", "strict", *dynamic_model_args(args)])
        if condition.query_budget is not None:
            command.extend(["--query-budget-override", str(condition.query_budget)])
        if condition.temperature_query_budget:
            command.extend([
                "--temperature-query-budget",
                "--query-budget-normal", str(args.query_budget_normal),
                "--query-budget-warm", str(args.query_budget_warm),
                "--query-budget-hot", str(args.query_budget_hot),
                "--query-budget-critical", str(args.query_budget_critical),
                "--query-budget-hysteresis-c", str(args.query_budget_hysteresis_c),
            ])
    return command


def assert_no_undervoltage(context: str) -> None:
    completed = subprocess.run(["vcgencmd", "get_throttled"], text=True, capture_output=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"Cannot verify power state during {context}: {completed.stderr.strip()}")
    match = re.search(r"0x([0-9a-fA-F]+)", completed.stdout)
    if match is None:
        raise RuntimeError(f"Unrecognised vcgencmd output during {context}: {completed.stdout!r}")
    mask = int(match.group(1), 16)
    # bit 0=current under-voltage; bit 16=under-voltage occurred since boot
    if mask & ((1 << 0) | (1 << 16)):
        raise RuntimeError(f"Undervoltage detected during {context}: {completed.stdout.strip()}")


def observed_coverage(runtime_csv: Path) -> dict[str, dict[str, int]]:
    with runtime_csv.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    def count(column: str, inferred_only: bool = False) -> dict[str, int]:
        selected = rows if not inferred_only else [
            row for row in rows if row.get("did_infer", "").lower() == "true"
        ]
        values: dict[str, int] = {}
        for row in selected:
            value = row.get(column, "")
            values[value] = values.get(value, 0) + 1
        return values

    return {
        "detector_call_resolutions": count("detector_call_resolution", True),
        "inference_intervals": count("inference_interval"),
        "query_budgets": count("query_budget_applied", True),
        "control_thermal_states": count("control_thermal_state"),
    }


def cool_to_target(args: argparse.Namespace, trace_path: Path) -> tuple[list[dict[str, Any]], float]:
    """Cool to the top of the accepted window; no stability sampling is used."""
    cooldown_args = argparse.Namespace(
        **vars(args),
        cooldown_temp_c=args.start_temp_max_c,
        cooldown_tolerance_c=0.0,
        max_wait_min=args.max_cooldown_min,
    )
    return cool_to_start_temperature(cooldown_args, trace_path)


def validate_full_controller(coverage: dict[str, dict[str, int]]) -> None:
    intervals = coverage["inference_intervals"]
    budgets = coverage["query_budgets"]
    states = coverage["control_thermal_states"]
    non_unit = sum(count for interval, count in intervals.items() if interval not in {"", "1"})
    non_normal = sum(count for state, count in states.items() if state not in {"", "normal"})
    observed_budgets = {budget for budget in budgets if budget}
    if non_unit == 0 or non_normal == 0 or len(observed_budgets) < 2:
        raise RuntimeError(
            "Full-controller coverage incomplete: expected a non-normal thermal "
            "state, a non-unit inference interval, and at least two applied query "
            "budgets; observed "
            f"states={states}, intervals={intervals}, budgets={budgets}."
        )


def plan_entries(args: argparse.Namespace, specs: list[Condition]) -> list[tuple[int, int, Condition]]:
    return [
        (repeat_index + 1, order, condition)
        for repeat_index in range(args.repeats)
        for order, condition in enumerate(ordered_conditions(repeat_index, specs), 1)
    ]


def print_plan(args: argparse.Namespace, entries: list[tuple[int, int, Condition]]) -> None:
    print(f"Formal runs: {len(entries)}; planned upper bound: {expected_plan_minutes(args, len(entries)) / 60.0:.2f} h")
    for index, (repeat, order, condition) in enumerate(entries, 1):
        output = args.output_dir / args.run_id / f"r{repeat:02d}_{order:02d}_{condition.key}" / "runtime.csv"
        print(f"\n{index:02d}. repeat {repeat}, order {order}: {condition.title}")
        print("    " + subprocess.list2cmdline(build_command(args, condition, output)))


def main() -> None:
    args = parse_args()
    specs = conditions()
    entries = plan_entries(args, specs)
    validate_args(args, specs)
    if args.plan_only:
        print_plan(args, entries)
        return

    if args.resume_dir is not None:
        suite_dir = args.resume_dir.resolve()
        manifest_path = suite_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Cannot resume without {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        suite_dir = args.output_dir / args.run_id
        suite_dir.mkdir(parents=True, exist_ok=False)
        manifest = {
            "protocol": "final software-only thermal matrix; six causal conditions x repeated balanced order",
            "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
            "conditions": [asdict(condition) for condition in specs],
            "schedule": [{"repeat": repeat, "order": order, **asdict(condition)} for repeat, order, condition in entries],
            "input_sha256": sha256(args.video),
            "model_sha256": {
                "fp32_640": sha256(args.native_model),
                "dynamic_query_320": sha256(args.dynamic_query_model_320),
                "dynamic_query_480": sha256(args.dynamic_query_model_480),
                "dynamic_query_640": sha256(args.dynamic_query_model_640),
            },
            "system_before": system_snapshot(),
            "runs": [],
        }
        write_json(suite_dir / "manifest.json", manifest)

    if args.fan_preflight or args.cooldown_fan:
        verify_fan_hardware(args, suite_dir)
    if args.power_preflight:
        assert_no_undervoltage("suite start")

    completed = {item.get("run_key") for item in manifest.get("runs", []) if item.get("returncode") == 0}
    started = time.monotonic()
    for index, (repeat, order, condition) in enumerate(entries, 1):
        run_key = f"r{repeat:02d}_{order:02d}_{condition.key}"
        run_dir = suite_dir / run_key
        if run_key in completed:
            print(f"[final-matrix] keeping completed {run_key}", flush=True)
            continue
        if run_dir.exists():
            # Preserve partial traces (for example, a cooldown that reached its
            # target immediately before an interrupted run) and retry the
            # condition on resume instead of requiring a manual filesystem edit.
            backup = suite_dir / (
                f"{run_key}_incomplete_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            shutil.move(str(run_dir), str(backup))
            manifest.setdefault("preserved_incomplete_runs", []).append(
                {"run_key": run_key, "path": str(backup)}
            )
            write_json(suite_dir / "manifest.json", manifest)
            print(
                f"[final-matrix] preserved incomplete {run_key} at {backup.name}; retrying it",
                flush=True,
            )
        remaining_min = (args.max_total_hours * 3600.0 - (time.monotonic() - started)) / 60.0
        minimum_needed = args.duration_min + args.startup_allowance_min
        if remaining_min < minimum_needed:
            raise TimeoutError(f"Total suite budget exhausted before {run_key}; remaining={remaining_min:.1f} min")

        run_dir.mkdir()
        _, pre_temperature = cool_to_target(args, run_dir / "cooldown_trace.csv")
        accepted_min = args.start_temp_min_c - args.start_temp_reading_tolerance_c
        if not accepted_min <= pre_temperature <= args.start_temp_max_c:
            raise RuntimeError(
                f"{run_key} pre-run temperature {pre_temperature:.2f}C is outside "
                f"the permitted {accepted_min:.1f}–{args.start_temp_max_c:.1f}C "
                "one-shot sensor window."
            )
        if args.power_preflight:
            assert_no_undervoltage(f"before {run_key}")
        output = run_dir / "runtime.csv"
        command = build_command(args, condition, output)
        run_meta: dict[str, Any] = {
            "run_key": run_key, "repeat": repeat, "order_within_repeat": order,
            "condition": asdict(condition), "command": command,
            "pre_child_temperature_c": pre_temperature,
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "system_before": system_snapshot(),
        }
        write_json(run_dir / "run_manifest.json", run_meta)
        print(f"\n[final-matrix] {index}/{len(entries)} {run_key}: {condition.title}", flush=True)
        returncode = run_with_temperature_trace(command, run_dir / "temperature_trace.csv", args.temperature_trace_sec, args.progress_sec)
        formal_start = first_logged_temperature(output)
        run_meta.update({
            "finished_utc": datetime.now(timezone.utc).isoformat(),
            "returncode": returncode,
            "formal_start_temperature_c": formal_start,
            "end_temperature_c": cpu_temp_c(),
            "runtime_log": str(output),
            "detections_log": str(run_dir / "runtime_detections.jsonl"),
            "system_after": system_snapshot(),
        })
        if output.exists():
            run_meta["observed_coverage"] = observed_coverage(output)
        write_json(run_dir / "run_manifest.json", run_meta)
        manifest["runs"] = [item for item in manifest.get("runs", []) if item.get("run_key") != run_key]
        manifest["runs"].append(run_meta)
        write_json(suite_dir / "manifest.json", manifest)
        if returncode:
            raise RuntimeError(f"{run_key} failed with exit code {returncode}")
        if condition.key == "proposed_software":
            validate_full_controller(run_meta["observed_coverage"])
        if args.power_preflight:
            assert_no_undervoltage(f"after {run_key}")

    manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["system_after"] = system_snapshot()
    write_json(suite_dir / "manifest.json", manifest)
    print(f"Final thermal matrix complete: {suite_dir}", flush=True)


if __name__ == "__main__":
    main()
