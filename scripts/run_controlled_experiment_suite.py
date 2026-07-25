#!/usr/bin/env python3
"""Run one thermally-normalised, full-metric experiment for every strategy.

Each strategy is run exactly once.  Before *every* run the suite waits until
the CPU temperature is inside the same target window and has remained stable
for a configurable number of samples.  A failed temperature normalisation is
an error by default: silently continuing would invalidate a thermal comparison.

For academically comparable results use a fixed, looped video.  A live camera
is supported only when the experiment is explicitly intended to measure an
end-to-end live pipeline; it does not replay the same visual scene per run.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]

# Explicit ordering keeps the baseline first and makes a one-pass experiment
# reproducible.  Every name maps to configs/strategies/<name>.yaml.
DEFAULT_STRATEGIES = [
    "native_rtdetr",
    "default",
    "static_affinity",
    "fixed_low_power",
    "fixed_frame_skip",
    "thermal_only",
    "thermal_balanced",
    "thermal_interval_first",
    "scene_only",
    "scene_track_lk",
    "scene_thermal_coadaptive",
    "scene_thermal_interval_first",
    "scene_thermal_interval_lk",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "raspberry_pi4.yaml")
    parser.add_argument("--video", type=Path, default=ROOT / "data" / "sample.mp4")
    parser.add_argument(
        "--camera",
        choices=["csi"],
        default=None,
        help="Use a live CSI camera. Prefer the default fixed video for fair strategy comparison.",
    )
    parser.add_argument("--model", type=Path, default=None, help="Optional single ONNX model override.")
    parser.add_argument(
        "--quantized-model-320", "--optimized-model-320",
        dest="optimized_model_320",
        type=Path,
        default=None,
        help="Quantization-only ONNX model used at 320 for every non-baseline strategy.",
    )
    parser.add_argument(
        "--quantized-model-480", "--optimized-model-480",
        dest="optimized_model_480",
        type=Path,
        default=None,
        help="Quantization-only ONNX model used at 480 for every non-baseline strategy.",
    )
    parser.add_argument(
        "--quantized-model-640", "--optimized-model-640",
        dest="optimized_model_640",
        type=Path,
        default=None,
        help="Quantization-only ONNX model used at 640 for every non-baseline strategy.",
    )
    parser.add_argument("--duration-min", type=float, default=20.0)
    parser.add_argument("--strategies", default=",".join(DEFAULT_STRATEGIES))
    parser.add_argument("--run-id", default=datetime.now().strftime("controlled_%Y%m%d_%H%M%S"))
    parser.add_argument("--output-dir", type=Path, default=ROOT / "experiments" / "controlled_suite")
    parser.add_argument("--start-temp-c", type=float, default=None)
    parser.add_argument("--temp-tolerance-c", type=float, default=0.5)
    parser.add_argument("--stable-samples", type=int, default=5)
    parser.add_argument("--poll-sec", type=float, default=10.0)
    parser.add_argument("--max-wait-min", type=float, default=90.0)
    parser.add_argument("--temperature-trace-sec", type=float, default=1.0)
    parser.add_argument(
        "--detr-warmup-temp-c",
        type=float,
        default=50.0,
        help=(
            "Run real RT-DETR inference in the child process until this temperature, "
            "then begin formal logging. Set to none is not supported by CLI; use 0 to disable."
        ),
    )
    parser.add_argument("--detr-warmup-max-sec", type=float, default=900.0)
    parser.add_argument(
        "--preheat-workers",
        type=int,
        default=0,
        help=(
            "CPU worker processes used only when the device starts below the lower "
            "temperature bound.  They stop at the upper bound, then the suite waits "
            "for passive cooling into the stable target window.  0 disables preheating."
        ),
    )
    parser.add_argument(
        "--allow-temperature-timeout",
        action="store_true",
        help="Continue after a normalisation timeout (not recommended for a comparative experiment).",
    )
    parser.add_argument("--loop-video", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--log-detections", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--apply-runtime-actions", action="store_true")
    parser.add_argument("--enable-thread-sessions", action="store_true")
    parser.add_argument("--thread-session-counts", default=None)
    parser.add_argument(
        "--fan-preflight",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require a real GPIO PWM fan test before the suite starts (default: enabled).",
    )
    parser.add_argument(
        "--cooldown-fan",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Run the PWM fan at full duty only while cooling between experiments, "
            "then stop and release GPIO before RT-DETR warmup (default: enabled)."
        ),
    )
    parser.add_argument(
        "--cooldown-fan-settle-sec",
        type=float,
        default=2.0,
        help="Fan-off settling time before RT-DETR warmup begins.",
    )
    parser.add_argument(
        "--progress-sec",
        type=float,
        default=30.0,
        help="Print a live child-process heartbeat at this interval.",
    )
    return parser.parse_args()


def cpu_temp_c() -> float | None:
    for path in sorted(Path("/sys/class/thermal").glob("thermal_zone*/temp")):
        try:
            raw = float(path.read_text(encoding="utf-8").strip())
            return raw / 1000.0 if raw > 1000 else raw
        except (OSError, ValueError):
            continue
    return None


def sha256(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_governors() -> dict[str, str | None]:
    values: dict[str, str | None] = {}
    for path in sorted(Path("/sys/devices/system/cpu").glob("cpu[0-9]*/cpufreq/scaling_governor")):
        try:
            values[str(path)] = path.read_text(encoding="utf-8").strip()
        except OSError:
            values[str(path)] = None
    return values


def system_snapshot() -> dict[str, Any]:
    affinity: list[int] | None = None
    if hasattr(os, "sched_getaffinity"):
        try:
            affinity = sorted(os.sched_getaffinity(0))
        except OSError:
            pass
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python": sys.version,
        "cpu_count": os.cpu_count(),
        "process_affinity": affinity,
        "governors": read_governors(),
        "kernel": platform.release(),
    }


def _start_preheat_workers(count: int) -> list[subprocess.Popen[Any]]:
    """Start bounded local CPU load processes; caller must always terminate them."""
    worker_code = "while True: x = 1234567 * 7654321"
    return [
        subprocess.Popen([sys.executable, "-c", worker_code])
        for _ in range(max(0, int(count)))
    ]


def _stop_processes(processes: list[subprocess.Popen[Any]]) -> None:
    for process in processes:
        if process.poll() is None:
            process.terminate()
    for process in processes:
        try:
            process.wait(timeout=3.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def wait_for_normalised_temperature(args: argparse.Namespace) -> tuple[list[dict[str, Any]], float | None]:
    """Require N consecutive readings inside the target temperature window."""
    samples: list[dict[str, Any]] = []
    stable = 0
    deadline = time.monotonic() + args.max_wait_min * 60.0
    preheat_processes: list[subprocess.Popen[Any]] = []
    preheat_was_used = False
    lower = args.start_temp_c - args.temp_tolerance_c
    upper = args.start_temp_c + args.temp_tolerance_c
    try:
        while True:
            temp = cpu_temp_c()
            sample = {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "temp_c": temp,
                "phase": "preheat" if preheat_processes else "stabilise",
            }
            samples.append(sample)
            if temp is None:
                message = "CPU temperature is unavailable; cannot normalise experimental start state."
                if not args.allow_temperature_timeout:
                    raise RuntimeError(message)
                print(f"[warn] {message}", flush=True)
                return samples, None

            # A precondition cycle heats at most once.  Reheating after an
            # overshoot creates a large thermal oscillation on fan-cooled
            # devices and makes an equal-temperature start less likely.
            if (
                temp < lower
                and args.preheat_workers > 0
                and not preheat_processes
                and not preheat_was_used
            ):
                preheat_processes = _start_preheat_workers(args.preheat_workers)
                preheat_was_used = True
                stable = 0
                print(
                    f"Preheating with {args.preheat_workers} worker(s): {temp:.2f}C < {lower:.2f}C",
                    flush=True,
                )

            # Stop at the upper boundary so the timed run begins during passive
            # cooling, not while an artificial workload is still active.
            if preheat_processes and temp >= upper:
                _stop_processes(preheat_processes)
                preheat_processes = []
                stable = 0
                print(
                    f"Preheat complete at {temp:.2f}C; waiting for passive stabilisation.",
                    flush=True,
                )

            within = lower <= temp <= upper and not preheat_processes
            stable = stable + 1 if within else 0
            print(
                f"Temperature conditioning: {temp:.2f}C; target "
                f"{args.start_temp_c:.2f}±{args.temp_tolerance_c:.2f}C; "
                f"stable samples {stable}/{args.stable_samples}",
                flush=True,
            )
            if stable >= args.stable_samples:
                if preheat_was_used:
                    samples.append({
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                        "temp_c": temp,
                        "phase": "preheat_then_stable",
                    })
                return samples, temp
            if time.monotonic() >= deadline:
                message = "Temperature normalisation timed out."
                if not args.allow_temperature_timeout:
                    raise RuntimeError(message + " Refusing to start a non-comparable run.")
                print(f"[warn] {message} Continuing by explicit override.", flush=True)
                return samples, temp
            # Thermal inertia after stopping a CPU preheat is much faster than
            # the normal 10 s polling cadence.  Sample at 1 Hz while passively
            # cooling so the target window is not skipped between samples.
            poll_interval = 1.0 if preheat_was_used else max(1.0, args.poll_sec)
            time.sleep(poll_interval)
    finally:
        _stop_processes(preheat_processes)


def build_command(
    args: argparse.Namespace,
    strategy: str,
    output: Path,
    *,
    use_optimized_models: bool,
) -> list[str]:
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "run_experiment.py"),
        "--config", str(args.config),
        "--strategy", strategy,
        "--duration-min", str(args.duration_min),
        "--output", str(output),
    ]
    if args.camera is not None:
        cmd.extend(["--camera", args.camera])
    else:
        cmd.extend(["--video", str(args.video)])
        if args.loop_video:
            cmd.append("--loop-video")
    if args.log_detections:
        cmd.append("--log-detections")
    if args.apply_runtime_actions:
        cmd.append("--apply-runtime-actions")
    if args.enable_thread_sessions:
        cmd.append("--enable-thread-sessions")
    if args.thread_session_counts:
        cmd.extend(["--thread-session-counts", args.thread_session_counts])
    if args.detr_warmup_temp_c > 0:
        cmd.extend([
            "--warmup-until-temp-c", str(args.detr_warmup_temp_c),
            "--warmup-max-sec", str(args.detr_warmup_max_sec),
        ])
    if use_optimized_models:
        cmd.extend([
            "--model-paths-by-resolution",
            ",".join([
                f"320={args.optimized_model_320}",
                f"480={args.optimized_model_480}",
                f"640={args.optimized_model_640}",
            ]),
        ])
    elif args.model is not None:
        cmd.extend(["--model", str(args.model)])
    return cmd


def run_with_temperature_trace(
    cmd: list[str],
    trace_path: Path,
    interval_sec: float,
    progress_sec: float,
) -> int:
    """Run one child process and write an independent 1 Hz thermal trace."""
    print("+ " + " ".join(cmd), flush=True)
    with trace_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["timestamp_utc", "elapsed_sec", "temp_c"])
        writer.writeheader()
        started = time.monotonic()
        next_progress = max(1.0, progress_sec)
        process = subprocess.Popen(cmd, cwd=ROOT)
        print(
            f"[suite] child started: pid={process.pid}; thermal trace={trace_path}",
            flush=True,
        )
        while process.poll() is None:
            elapsed = time.monotonic() - started
            temp = cpu_temp_c()
            writer.writerow({
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "elapsed_sec": round(elapsed, 3),
                "temp_c": temp,
            })
            handle.flush()
            if elapsed >= next_progress:
                temp_text = f"{temp:.2f}C" if temp is not None else "unavailable"
                print(
                    f"[suite] running: pid={process.pid}; elapsed={elapsed:.1f}s; "
                    f"temperature={temp_text}",
                    flush=True,
                )
                next_progress += max(1.0, progress_sec)
            time.sleep(max(0.1, interval_sec))
        writer.writerow({
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "elapsed_sec": round(time.monotonic() - started, 3),
            "temp_c": cpu_temp_c(),
        })
        handle.flush()
        return int(process.returncode or 0)


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def first_logged_temperature(path: Path) -> float | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8", newline="") as handle:
        row = next(csv.DictReader(handle), None)
    if not row:
        return None
    try:
        return float(row["temp_c"])
    except (KeyError, TypeError, ValueError):
        return None


def verify_fan_hardware(args: argparse.Namespace, suite_dir: Path) -> None:
    """Fail before the suite if fan control would degrade to ``pwm_no_gpio``."""
    if not args.fan_preflight:
        return
    command = [
        sys.executable,
        str(ROOT / "scripts" / "test_fan_gpio.py"),
        "--config", str(args.config),
        "--strategy", "scene_thermal_interval_lk",
        "--duty", "1.0",
        "--seconds", "1.0",
    ]
    completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    result = {
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }
    write_json(suite_dir / "fan_preflight.json", result)
    if completed.returncode != 0 or "mode=pwm " not in completed.stdout:
        raise RuntimeError(
            "Fan preflight failed: real GPIO PWM was not applied. Fix RPi.GPIO/lgpio "
            "permissions, daemon, wiring, or pin configuration before this thermal suite. "
            "See fan_preflight.json for the exact error."
        )


def start_cooldown_fan(args: argparse.Namespace):
    """Start full-speed fan cooling using a controller that releases GPIO on close."""
    sys.path.insert(0, str(ROOT / "src"))
    from scene_runtime.device.fan import PwmFanController
    from scene_runtime.runtime.config import load_config

    config = load_config(args.config, "scene_thermal_interval_lk")
    config.setdefault("project", {})["strategy"] = "suite_cooldown"
    fan_cfg = config.setdefault("fan", {})
    fan_cfg.update({
        "enabled": True,
        "enabled_strategies": ["suite_cooldown"],
        "on_temp_c": 0.0,
        "off_temp_c": -1.0,
        "full_temp_c": 1.0,
        "min_duty_cycle": 1.0,
        "max_duty_cycle": 1.0,
        "hold_low_on_close": False,
    })
    controller = PwmFanController(config)
    state = controller.update({"temp_c": 100.0, "thermal_state": "critical"})
    if state.mode != "pwm":
        error = controller.last_error or "unknown GPIO error"
        controller.close()
        raise RuntimeError(f"Cooldown fan could not apply real PWM: {error}")
    return controller


def wait_until_cool_enough(
    args: argparse.Namespace,
    *,
    allow_fan: bool,
) -> tuple[list[dict[str, Any]], float | None]:
    """Wait only for cooling; the child then heats with real RT-DETR inference."""
    target = float(args.detr_warmup_temp_c)
    deadline = time.monotonic() + args.max_wait_min * 60.0
    samples: list[dict[str, Any]] = []
    controller = None
    try:
        while True:
            temp = cpu_temp_c()
            samples.append({
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "temp_c": temp,
                "phase": (
                    "fan_cool_before_detr_warmup"
                    if controller is not None
                    else "cool_before_detr_warmup"
                ),
            })
            if temp is not None and temp <= target:
                return samples, temp
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    f"CPU did not cool to <= {target:.2f}C before RT-DETR warmup."
                )
            if allow_fan and args.cooldown_fan and controller is None:
                controller = start_cooldown_fan(args)
                print(
                    "[suite] cooldown fan started at 100% PWM; "
                    "it will stop before RT-DETR warmup.",
                    flush=True,
                )
            current = f"{temp:.2f}C" if temp is not None else "unavailable"
            cooling = "fan=100%" if controller is not None else "passive cooling"
            print(
                f"Waiting to cool before RT-DETR warmup: "
                f"{current} > {target:.2f}C ({cooling})",
                flush=True,
            )
            time.sleep(max(1.0, args.poll_sec))
    finally:
        if controller is not None:
            controller.close()
            print("[suite] cooldown fan stopped; GPIO released.", flush=True)
            if args.cooldown_fan_settle_sec > 0:
                time.sleep(args.cooldown_fan_settle_sec)


def main() -> None:
    args = parse_args()
    if args.duration_min <= 0 or args.stable_samples < 1 or args.temp_tolerance_c < 0:
        raise ValueError("duration, stable-samples, and temp-tolerance must be positive")
    if args.preheat_workers < 0:
        raise ValueError("--preheat-workers cannot be negative")
    if args.detr_warmup_temp_c < 0 or args.detr_warmup_max_sec <= 0:
        raise ValueError("RT-DETR warmup temperature must be non-negative and max time positive")
    if args.cooldown_fan_settle_sec < 0:
        raise ValueError("--cooldown-fan-settle-sec cannot be negative")
    if args.progress_sec <= 0:
        raise ValueError("--progress-sec must be positive")
    if args.detr_warmup_temp_c > 0 and args.preheat_workers:
        raise ValueError("Use either --detr-warmup-temp-c or --preheat-workers, not both")
    if args.detr_warmup_temp_c <= 0 and args.start_temp_c is None:
        raise ValueError("--start-temp-c is required when RT-DETR warmup is disabled")
    optimized_models = (
        args.optimized_model_320,
        args.optimized_model_480,
        args.optimized_model_640,
    )
    if any(model is not None for model in optimized_models):
        if args.model is not None:
            raise ValueError("--model cannot be combined with --optimized-model-* options")
        missing_models = [
            name for name, model in zip(("320", "480", "640"), optimized_models) if model is None
        ]
        if missing_models:
            raise ValueError(
                "Provide all three optimized models (missing: " + ", ".join(missing_models) + ")"
            )
        absent_models = [str(model) for model in optimized_models if model is not None and not model.exists()]
        if absent_models:
            raise FileNotFoundError("Optimized model(s) not found: " + ", ".join(absent_models))
    if args.camera is None and not args.video.exists():
        raise FileNotFoundError(f"Fixed input video does not exist: {args.video}")
    strategies = [value.strip() for value in args.strategies.split(",") if value.strip()]
    if not strategies:
        raise ValueError("No strategies selected")
    if any(model is not None for model in optimized_models) and strategies[0] != "native_rtdetr":
        raise ValueError(
            "When using optimized models, the first strategy must be native_rtdetr "
            "so it remains the original FP32 baseline."
        )
    missing = [name for name in strategies if not (ROOT / "configs" / "strategies" / f"{name}.yaml").exists()]
    if missing:
        raise FileNotFoundError(f"Unknown strategy configuration(s): {', '.join(missing)}")

    suite_dir = args.output_dir / args.run_id
    suite_dir.mkdir(parents=True, exist_ok=False)
    manifest: dict[str, Any] = {
        "protocol": "single-run thermally-normalised controlled strategy suite",
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "strategies": strategies,
        "input_sha256": sha256(args.video) if args.camera is None else None,
        "config_sha256": sha256(args.config),
        "model_sha256": sha256(args.model) if args.model else None,
        "optimized_model_sha256": {
            str(resolution): sha256(model) if model else None
            for resolution, model in zip((320, 480, 640), optimized_models)
        },
        "system_before": system_snapshot(),
        "runs": [],
    }
    write_json(suite_dir / "manifest.json", manifest)
    verify_fan_hardware(args, suite_dir)

    for index, strategy in enumerate(strategies, start=1):
        label = f"{index:02d}_{strategy}"
        run_dir = suite_dir / label
        run_dir.mkdir()
        output = run_dir / "runtime.csv"
        conditioning, start_temp = (
            wait_until_cool_enough(args, allow_fan=index > 1)
            if args.detr_warmup_temp_c > 0
            else wait_for_normalised_temperature(args)
        )
        write_json(run_dir / "precondition_temperature.json", conditioning)
        # The first native RT-DETR run is the FP32 baseline.  Every following
        # strategy uses precisely the same 320/480/640 optimized model family.
        use_optimized_models = bool(optimized_models[0]) and index > 1
        command = build_command(
            args,
            strategy,
            output,
            use_optimized_models=use_optimized_models,
        )
        model_condition = (
            "quantization_only_int8" if use_optimized_models
            else "native_fp32_baseline" if bool(optimized_models[0])
            else "single_model_override" if args.model is not None
            else "config_default"
        )
        fan_expected = strategy in {"scene_thermal_coadaptive", "scene_thermal_interval_lk"}
        start_text = f"{start_temp:.2f}C" if start_temp is not None else "unavailable"
        print(
            f"\n[suite] starting {index}/{len(strategies)}: {strategy}\n"
            f"  model condition: {model_condition}\n"
            f"  pre-warmup temperature: {start_text}\n"
            f"  PWM fan expected: {'yes' if fan_expected else 'no'}\n"
            f"  output directory: {run_dir}",
            flush=True,
        )
        run_meta: dict[str, Any] = {
            "strategy": strategy,
            "model_condition": model_condition,
            "command": command,
            "pre_warmup_temperature_c": start_temp,
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
        run_meta.update({
            "finished_utc": datetime.now(timezone.utc).isoformat(),
            "returncode": returncode,
            "formal_start_temperature_c": first_logged_temperature(output),
            "end_temperature_c": cpu_temp_c(),
            "system_after": system_snapshot(),
            "runtime_log": str(output),
            "profile_log": str(output.with_name("runtime_profile.csv")),
            "detections_log": str(output.with_name("runtime_detections.jsonl")),
        })
        write_json(run_dir / "run_manifest.json", run_meta)
        print(
            f"[suite] finished {index}/{len(strategies)}: {strategy}; "
            f"returncode={returncode}; formal start temperature="
            f"{run_meta['formal_start_temperature_c']}; "
            f"end temperature={run_meta['end_temperature_c']}",
            flush=True,
        )
        manifest["runs"].append(run_meta)
        write_json(suite_dir / "manifest.json", manifest)
        if returncode != 0:
            raise RuntimeError(f"Strategy {strategy} failed with exit code {returncode}; suite stopped.")

    manifest["system_after"] = system_snapshot()
    manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(suite_dir / "manifest.json", manifest)
    print(f"Controlled experiment suite complete: {suite_dir}")


if __name__ == "__main__":
    main()
