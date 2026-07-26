"""Run the bounded RT-DETR query-budget sensitivity ablation.

This is intentionally separate from the main defense matrix.  It holds the
scene/LK/ROI policy and model resolution family fixed, and varies only the
graph query budget: Q=32, 48, 64, 100, and 300.  Each run is cooled to the
same temperature window and the resulting logs are compared with the native
FP32 teacher after the sweep.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from run_controlled_experiment_suite import (
    cpu_temp_c,
    first_logged_temperature,
    run_with_temperature_trace,
    sha256,
    system_snapshot,
    write_json,
)
from run_core_ablation_suite import cool_to_start_temperature


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BUDGETS = (32, 48, 64, 100, 300)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "raspberry_pi4.yaml")
    parser.add_argument("--video", type=Path, default=ROOT / "data" / "sample.mp4")
    parser.add_argument("--native-model", type=Path, default=ROOT / "models" / "rtdetr_r18_lite_pi4_640.onnx")
    parser.add_argument("--dynamic-query-model-320", type=Path, default=ROOT / "models" / "rtdetr_r18_lite_pi4_320_int8_dynamic_q.onnx")
    parser.add_argument("--dynamic-query-model-640", type=Path, default=ROOT / "models" / "rtdetr_r18_lite_pi4_640_int8_dynamic_q.onnx")
    parser.add_argument("--duration-min", type=float, default=20.0)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "experiments" / "query_budget_sweep")
    parser.add_argument("--run-id", default=datetime.now().strftime("query_sweep_%Y%m%d_%H%M%S"))
    parser.add_argument("--cooldown-temp-c", type=float, default=50.0)
    parser.add_argument("--cooldown-tolerance-c", type=float, default=0.5)
    parser.add_argument("--max-wait-min", type=float, default=90.0)
    parser.add_argument("--poll-sec", type=float, default=5.0)
    parser.add_argument("--cooldown-fan-settle-sec", type=float, default=2.0)
    parser.add_argument("--temperature-trace-sec", type=float, default=1.0)
    parser.add_argument("--progress-sec", type=float, default=30.0)
    parser.add_argument("--first-warmup-temp-c", type=float, default=50.0)
    parser.add_argument("--first-warmup-max-sec", type=float, default=900.0)
    parser.add_argument("--teacher-cycle-frames", type=int, default=0)
    parser.add_argument("--loop-video", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--log-detections", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cooldown-fan", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--apply-runtime-actions", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--plan-only", action="store_true")
    return parser.parse_args()


def validate(args: argparse.Namespace) -> None:
    if not args.log_detections:
        raise ValueError("query-budget sweep requires --log-detections")
    required = [
        args.config,
        args.video,
        args.native_model,
        args.dynamic_query_model_320,
        args.dynamic_query_model_640,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Required input(s) not found: " + ", ".join(missing))
    if args.duration_min <= 0 or args.max_wait_min <= 0 or args.poll_sec <= 0:
        raise ValueError("duration, max wait, and polling intervals must be positive")
    if args.cooldown_temp_c <= 0 or args.cooldown_tolerance_c < 0:
        raise ValueError("cooldown temperature must be positive and tolerance non-negative")


def command(args: argparse.Namespace, budget: int, output: Path, *, first_run: bool) -> list[str]:
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "run_experiment.py"),
        "--config", str(args.config),
        "--strategy", "scene_track_lk",
        "--video", str(args.video),
        "--duration-min", str(args.duration_min),
        "--output", str(output),
        "--fan-control", "disabled",
        "--query-budget-mode", "strict",
        "--query-budget-override", str(budget),
        "--model-paths-by-resolution",
        f"320={args.dynamic_query_model_320},640={args.dynamic_query_model_640}",
    ]
    if args.loop_video:
        cmd.append("--loop-video")
    if args.log_detections:
        cmd.append("--log-detections")
    if args.apply_runtime_actions:
        cmd.append("--apply-runtime-actions")
    if first_run:
        cmd.extend([
            "--warmup-until-temp-c", str(args.first_warmup_temp_c),
            "--warmup-max-sec", str(args.first_warmup_max_sec),
        ])
    return cmd


def print_plan(args: argparse.Namespace, suite_dir: Path) -> None:
    for index, budget in enumerate(DEFAULT_BUDGETS, 1):
        output = suite_dir / f"{index:02d}_q{budget}" / "runtime.csv"
        print(f"{index}. Q={budget}")
        print("   " + subprocess.list2cmdline(command(args, budget, output, first_run=index == 1)))


def run_quality(args: argparse.Namespace, suite_dir: Path) -> int:
    analysis = suite_dir / "analysis"
    analysis.mkdir(exist_ok=True)
    runs = [suite_dir / f"{index:02d}_q{budget}" for index, budget in enumerate(DEFAULT_BUDGETS, 1)]
    teacher = suite_dir / "teacher_native_fp32"
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "evaluate_strategy_detection_quality.py"),
        "--teacher", str(teacher / "runtime_detections.jsonl"),
        "--students", *[str(run / "runtime_detections.jsonl") for run in runs],
        "--teacher-csv", str(teacher / "runtime.csv"),
        "--include-teacher-summary",
        "--student-csvs", *[str(run / "runtime.csv") for run in runs],
        "--output", str(analysis / "quality_summary.csv"),
        "--matches-output", str(analysis / "quality_frames.csv"),
        "--plot-output", str(analysis / "quality_overview.png"),
        "--label-source", "student",
    ]
    if args.teacher_cycle_frames > 0:
        cmd.extend(["--teacher-cycle-frames", str(args.teacher_cycle_frames)])
    return subprocess.run(cmd, cwd=ROOT).returncode


def main() -> None:
    args = parse_args()
    validate(args)
    suite_dir = (args.output_dir / args.run_id).resolve()
    if args.plan_only:
        print_plan(args, suite_dir)
        return
    suite_dir.mkdir(parents=True, exist_ok=False)
    manifest = {
        "protocol": "query-budget sensitivity sweep",
        "budgets": list(DEFAULT_BUDGETS),
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "input_sha256": sha256(args.video),
        "model_sha256": {"native": sha256(args.native_model), "dynamic_320": sha256(args.dynamic_query_model_320), "dynamic_640": sha256(args.dynamic_query_model_640)},
        "system_before": system_snapshot(),
        "runs": [],
    }
    write_json(suite_dir / "manifest.json", manifest)

    # The teacher is recorded once at the beginning so all Q values use the
    # same native FP32 reference sequence.
    teacher_dir = suite_dir / "teacher_native_fp32"
    teacher_dir.mkdir()
    _, teacher_temperature = cool_to_start_temperature(
        args, teacher_dir / "cooldown_trace.csv"
    )
    teacher_output = teacher_dir / "runtime.csv"
    teacher_cmd = [
        sys.executable, str(ROOT / "scripts" / "run_experiment.py"),
        "--config", str(args.config), "--strategy", "native_rtdetr",
        "--video", str(args.video), "--duration-min", str(args.duration_min),
        "--output", str(teacher_output), "--fan-control", "disabled",
        "--model", str(args.native_model), "--log-detections",
    ]
    if args.loop_video:
        teacher_cmd.append("--loop-video")
    if args.apply_runtime_actions:
        teacher_cmd.append("--apply-runtime-actions")
    teacher_cmd.extend(["--warmup-until-temp-c", str(args.first_warmup_temp_c), "--warmup-max-sec", str(args.first_warmup_max_sec)])
    print(
        f"[query-sweep] starting native FP32 teacher at {teacher_temperature:.2f}C",
        flush=True,
    )
    returncode = run_with_temperature_trace(teacher_cmd, teacher_dir / "temperature_trace.csv", args.temperature_trace_sec, args.progress_sec)
    if returncode != 0:
        raise SystemExit(returncode)

    for index, budget in enumerate(DEFAULT_BUDGETS, 1):
        run_dir = suite_dir / f"{index:02d}_q{budget}"
        run_dir.mkdir()
        _, temperature = cool_to_start_temperature(args, run_dir / "cooldown_trace.csv")
        output = run_dir / "runtime.csv"
        cmd = command(args, budget, output, first_run=False)
        print(f"\n[query-sweep] starting {index}/{len(DEFAULT_BUDGETS)}: Q={budget}", flush=True)
        print(f"  pre-child temperature: {temperature:.2f}C", flush=True)
        run_meta = {
            "key": f"q{budget}",
            "title": f"Fixed graph query budget Q={budget}",
            "groups": ["query_sweep"],
            "strategy": "scene_track_lk",
            "model_kind": "dynamic_query_640",
            "query_budget_override": budget,
            "temperature_query_budget": False,
            "order": index,
            "command": cmd,
            "pre_child_temperature_c": temperature,
        }
        write_json(run_dir / "run_manifest.json", run_meta)
        returncode = run_with_temperature_trace(cmd, run_dir / "temperature_trace.csv", args.temperature_trace_sec, args.progress_sec)
        run_meta.update({"returncode": returncode, "formal_start_temperature_c": first_logged_temperature(output), "end_temperature_c": cpu_temp_c(), "finished_utc": datetime.now(timezone.utc).isoformat()})
        write_json(run_dir / "run_manifest.json", run_meta)
        manifest["runs"].append(run_meta)
        write_json(suite_dir / "manifest.json", manifest)
        if returncode != 0:
            raise SystemExit(returncode)

    quality_returncode = run_quality(args, suite_dir)
    if quality_returncode != 0:
        raise SystemExit(quality_returncode)
    analyzer = [sys.executable, str(ROOT / "scripts" / "analyze_defense_experiment_suite.py"), "--suite-dir", str(suite_dir)]
    if subprocess.run(analyzer, cwd=ROOT).returncode != 0:
        raise SystemExit(1)
    print(f"[query-sweep] complete: {suite_dir}", flush=True)


if __name__ == "__main__":
    main()
