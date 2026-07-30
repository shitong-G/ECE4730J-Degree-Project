#!/usr/bin/env python3
"""Run a repeatable Raspberry Pi thermal/latency matrix for SOTA baselines.

This mirrors the formal structure of run_final_thermal_matrix.py, but it only
runs external detector baselines. It is intended for comparing non-accuracy
metrics such as CPU temperature, wall throughput, and inference latency against
the already-collected proposed-method runs.
"""

from __future__ import annotations

import argparse
import csv
import json
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
    run_with_temperature_trace,
    sha256,
    system_snapshot,
    verify_fan_hardware,
    write_json,
)
from run_final_thermal_matrix import (
    assert_no_undervoltage,
    condition_start_temperature,
)


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class SotaCondition:
    key: str
    title: str
    detector: str
    model: Path
    imgsz: int
    nanodet_input_size: int | None = None


def conditions(args: argparse.Namespace) -> list[SotaCondition]:
    selected = {item.strip() for item in args.models.split(",") if item.strip()}
    specs = [
        SotaCondition(
            "yolov8n_640",
            "YOLOv8n, input 640",
            "yolov8n",
            args.yolov8n_model,
            640,
        ),
        SotaCondition(
            "nanodet_plus_m_input640",
            "NanoDet-Plus-m checkpoint, input 640",
            "nanodet_plus_m_input640",
            args.nanodet_checkpoint,
            640,
            nanodet_input_size=640,
        ),
        SotaCondition(
            "picodet_l_640",
            "PicoDet-L, input 640",
            "pp_picodet_l_640",
            args.picodet_l_640_dir,
            640,
        ),
    ]
    if selected == {"all"}:
        return specs
    unknown = selected - {spec.key for spec in specs}
    if unknown:
        raise ValueError(f"Unknown --models entries: {sorted(unknown)}")
    return [spec for spec in specs if spec.key in selected]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, default=ROOT / "data" / "sample3.mp4")
    parser.add_argument("--models", default="all", help="Comma-separated keys or all")
    parser.add_argument("--max-frames", type=int, default=600)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "experiments" / "sota_thermal_matrix")
    parser.add_argument("--visualization-dir", type=Path, default=ROOT / "experiments" / "visualizations" / "sota_thermal_matrix")
    parser.add_argument("--run-id", default=datetime.now().strftime("sota_thermal_%Y%m%d_%H%M%S"))
    parser.add_argument("--resume-dir", type=Path, default=None)
    parser.add_argument("--yolov8n-model", type=Path, default=ROOT / "models" / "baselines" / "yolov8n_640.onnx")
    parser.add_argument("--nanodet-checkpoint", type=Path, default=ROOT / "models" / "baselines" / "nanodet-plus-m_320.ckpt")
    parser.add_argument("--picodet-l-640-dir", type=Path, default=ROOT / "models" / "baselines" / "picodet_l_640_coco_lcnet_portable")
    parser.add_argument("--start-temp-min-c", type=float, default=45.0)
    parser.add_argument("--start-temp-max-c", type=float, default=50.0)
    parser.add_argument("--start-temp-reading-tolerance-c", type=float, default=0.5)
    parser.add_argument("--max-cooldown-min", type=float, default=8.0)
    parser.add_argument("--poll-sec", type=float, default=5.0)
    parser.add_argument("--preheat-workers", type=int, default=4)
    parser.add_argument("--cooldown-fan-settle-sec", type=float, default=2.0)
    parser.add_argument("--temperature-trace-sec", type=float, default=1.0)
    parser.add_argument("--progress-sec", type=float, default=30.0)
    parser.add_argument("--max-total-hours", type=float, default=6.0)
    parser.add_argument("--startup-allowance-min", type=float, default=1.0)
    parser.add_argument("--cooldown-fan", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fan-preflight", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--power-preflight", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    return parser.parse_args()


def ordered_conditions(repeat_index: int, specs: list[SotaCondition]) -> list[SotaCondition]:
    if repeat_index % 2 == 0:
        return specs
    return list(reversed(specs))


def plan_entries(specs: list[SotaCondition], repeats: int) -> list[tuple[int, int, SotaCondition]]:
    return [
        (repeat_index + 1, order, condition)
        for repeat_index in range(repeats)
        for order, condition in enumerate(ordered_conditions(repeat_index, specs), 1)
    ]


def expected_plan_minutes(args: argparse.Namespace, count: int) -> float:
    # External baselines are frame-count limited, so use a conservative process
    # allowance rather than a duration-min argument.
    return count * (args.max_cooldown_min + args.startup_allowance_min + 10.0)


def validate_args(args: argparse.Namespace, specs: list[SotaCondition]) -> None:
    if args.repeats < 1 or args.max_frames < 1:
        raise ValueError("--repeats and --max-frames must be positive")
    if args.start_temp_min_c <= 0 or args.start_temp_min_c > args.start_temp_max_c:
        raise ValueError("Start window must satisfy 0 < --start-temp-min-c <= --start-temp-max-c")
    missing = [str(path) for path in [args.video, *(spec.model for spec in specs)] if not path.exists()]
    if missing:
        raise FileNotFoundError("Required input(s) not found: " + ", ".join(missing))
    planned = expected_plan_minutes(args, len(specs) * args.repeats)
    if planned > args.max_total_hours * 60.0:
        raise ValueError(
            f"Planned upper bound is {planned / 60.0:.2f} h, exceeding "
            f"--max-total-hours={args.max_total_hours:.2f}."
        )


def build_command(args: argparse.Namespace, condition: SotaCondition, output: Path, run_dir: Path) -> list[str]:
    command = [
        sys.executable,
        str(ROOT / "scripts" / "run_sota_external_detector.py"),
        "--detector",
        condition.detector,
        "--video",
        str(args.video),
        "--model",
        str(condition.model),
        "--output-csv",
        str(output),
        "--visualization-dir",
        str(args.visualization_dir / args.run_id / run_dir.name),
        "--imgsz",
        str(condition.imgsz),
        "--device",
        "cpu",
        "--threads",
        str(args.threads),
        "--max-frames",
        str(args.max_frames),
    ]
    if condition.nanodet_input_size is not None:
        command.extend(["--nanodet-input-size", str(condition.nanodet_input_size)])
    if args.save_video:
        command.append("--save-video")
    return command


def read_summary_csv(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return rows[0] if rows else {}


def temperature_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    values: list[float] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                values.append(float(row["temp_c"]))
            except Exception:
                pass
    if not values:
        return {}
    return {
        "temperature_samples": len(values),
        "temperature_start_c": values[0],
        "temperature_end_c": values[-1],
        "temperature_min_c": min(values),
        "temperature_max_c": max(values),
        "temperature_mean_c": sum(values) / len(values),
    }


def write_suite_summary(suite_dir: Path, manifest: dict[str, Any]) -> None:
    rows: list[dict[str, Any]] = []
    for run in manifest.get("runs", []):
        summary = dict(run.get("summary", {}))
        temp = dict(run.get("temperature_summary", {}))
        rows.append({
            "run_key": run.get("run_key"),
            "condition": run.get("condition", {}).get("key"),
            "title": run.get("condition", {}).get("title"),
            "returncode": run.get("returncode"),
            "elapsed_sec": run.get("elapsed_sec"),
            "pre_child_temperature_c": run.get("pre_child_temperature_c"),
            "end_temperature_c": run.get("end_temperature_c"),
            "frames": summary.get("frames", ""),
            "wall_sec": summary.get("wall_sec", ""),
            "throughput_fps": summary.get("throughput_fps", ""),
            "inference_ms_mean": summary.get("inference_ms_mean", ""),
            "preprocess_ms_mean": summary.get("preprocess_ms_mean", ""),
            "postprocess_ms_mean": summary.get("postprocess_ms_mean", ""),
            "detection_count_mean": summary.get("detection_count_mean", ""),
            **temp,
        })
    if not rows:
        return
    with (suite_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def print_plan(args: argparse.Namespace, entries: list[tuple[int, int, SotaCondition]]) -> None:
    print(f"SOTA runs: {len(entries)}; planned upper bound: {expected_plan_minutes(args, len(entries)) / 60.0:.2f} h")
    for index, (repeat, order, condition) in enumerate(entries, 1):
        run_dir = args.output_dir / args.run_id / f"r{repeat:02d}_{order:02d}_{condition.key}"
        print(f"\n{index:02d}. repeat {repeat}, order {order}: {condition.title}")
        print("    " + subprocess.list2cmdline(build_command(args, condition, run_dir / "runtime.csv", run_dir)))


def main() -> None:
    args = parse_args()
    specs = conditions(args)
    entries = plan_entries(specs, args.repeats)
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
            "protocol": "SOTA thermal/latency matrix; external detectors only; final-matrix style start conditioning",
            "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
            "conditions": [asdict(condition) for condition in specs],
            "schedule": [{"repeat": repeat, "order": order, **asdict(condition)} for repeat, order, condition in entries],
            "input_sha256": sha256(args.video),
            "model_sha256": {condition.key: sha256(condition.model) if condition.model.is_file() else "directory" for condition in specs},
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
            print(f"[sota-matrix] keeping completed {run_key}", flush=True)
            continue
        if run_dir.exists():
            backup = suite_dir / f"{run_key}_incomplete_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            shutil.move(str(run_dir), str(backup))
            manifest.setdefault("preserved_incomplete_runs", []).append({"run_key": run_key, "path": str(backup)})
            write_json(suite_dir / "manifest.json", manifest)
        remaining_min = (args.max_total_hours * 3600.0 - (time.monotonic() - started)) / 60.0
        if remaining_min < args.startup_allowance_min:
            raise TimeoutError(f"Total suite budget exhausted before {run_key}; remaining={remaining_min:.1f} min")

        run_dir.mkdir()
        pre_temperature = condition_start_temperature(args, run_dir)
        if args.power_preflight:
            assert_no_undervoltage(f"before {run_key}")
        output = run_dir / "runtime.csv"
        command = build_command(args, condition, output, run_dir)
        run_meta: dict[str, Any] = {
            "run_key": run_key,
            "repeat": repeat,
            "order_within_repeat": order,
            "condition": asdict(condition),
            "command": command,
            "pre_child_temperature_c": pre_temperature,
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "system_before": system_snapshot(),
        }
        write_json(run_dir / "run_manifest.json", run_meta)
        print(f"\n[sota-matrix] {index}/{len(entries)} {run_key}: {condition.title}", flush=True)
        run_started = time.time()
        returncode = run_with_temperature_trace(
            command,
            run_dir / "temperature_trace.csv",
            args.temperature_trace_sec,
            args.progress_sec,
        )
        elapsed = time.time() - run_started
        run_meta.update({
            "finished_utc": datetime.now(timezone.utc).isoformat(),
            "returncode": returncode,
            "elapsed_sec": round(elapsed, 3),
            "end_temperature_c": cpu_temp_c(),
            "runtime_log": str(output),
            "summary": read_summary_csv(output),
            "temperature_summary": temperature_summary(run_dir / "temperature_trace.csv"),
            "system_after": system_snapshot(),
        })
        write_json(run_dir / "run_manifest.json", run_meta)
        manifest["runs"] = [item for item in manifest.get("runs", []) if item.get("run_key") != run_key]
        manifest["runs"].append(run_meta)
        write_json(suite_dir / "manifest.json", manifest)
        write_suite_summary(suite_dir, manifest)
        if returncode:
            raise RuntimeError(f"{run_key} failed with exit code {returncode}")
        if args.power_preflight:
            assert_no_undervoltage(f"after {run_key}")

    manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["system_after"] = system_snapshot()
    write_json(suite_dir / "manifest.json", manifest)
    write_suite_summary(suite_dir, manifest)
    print(f"SOTA thermal matrix complete: {suite_dir}", flush=True)


if __name__ == "__main__":
    main()
