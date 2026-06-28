#!/usr/bin/env python3
"""Micro-benchmark ONNX runtime knobs on a short video prefix."""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scene_runtime.controller.actions import RuntimeAction
from scene_runtime.device.action_applier import RuntimeActionApplier
from scene_runtime.inference.onnx_engine import ONNXRTDETREngine
from scene_runtime.utils.video import FrameSource


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark RT-DETR runtime knobs")
    parser.add_argument("--model", type=Path, default=ROOT / "models" / "rtdetr_r18_lite_pi4.onnx")
    parser.add_argument("--video", type=Path, default=ROOT / "data" / "sample.mp4")
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--resolutions", default="320,480,640")
    parser.add_argument("--threads", default="1,2,3,4")
    parser.add_argument("--governors", default="performance,ondemand,powersave")
    parser.add_argument("--apply-runtime-actions", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "experiments" / "logs" / "runtime_knob_benchmark.csv")
    return parser.parse_args()


def _parse_ints(text: str) -> list[int]:
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def _parse_items(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def _load_frames(video: Path, count: int) -> list:
    source = FrameSource(video, synthetic=False, max_frames=count, loop=True)
    try:
        return [frame for frame in source]
    finally:
        source.release()


def _resolved_resolution(engine: ONNXRTDETREngine, requested: int) -> int:
    """Resolve actual ONNX input size across old and new engine versions."""
    value = getattr(engine, "last_resolved_input_resolution", None)
    if value is not None:
        return int(value)
    resolver = getattr(engine, "_resolve_input_resolution", None)
    if callable(resolver):
        return int(resolver(int(requested)))
    fixed = getattr(engine, "_fixed_input_size", None)
    return int(fixed or requested)


def main() -> None:
    args = parse_args()
    resolutions = _parse_ints(args.resolutions)
    thread_counts = _parse_ints(args.threads)
    governors = _parse_items(args.governors)
    frames = _load_frames(args.video, max(1, args.frames + args.warmup))

    engine = ONNXRTDETREngine(
        model_path=str(args.model),
        dry_run=False,
        enable_thread_sessions=True,
        thread_session_counts=thread_counts,
    )
    engine.load()
    applier = RuntimeActionApplier(enabled=args.apply_runtime_actions)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    for governor in governors:
        for threads in thread_counts:
            for resolution in resolutions:
                action = RuntimeAction(
                    mode="benchmark",
                    input_resolution=resolution,
                    inference_interval=1,
                    cpu_threads=threads,
                    governor=governor,
                )
                applied = applier.apply(action)
                latencies: list[float] = []
                onnx_latencies: list[float] = []
                for index, frame in enumerate(frames):
                    t0 = time.perf_counter()
                    engine.infer(frame, action)
                    elapsed_ms = (time.perf_counter() - t0) * 1000.0
                    if index >= args.warmup:
                        latencies.append(elapsed_ms)
                        onnx_latencies.append(float(engine.last_profile.get("onnx_run_ms", 0.0)))
                row = {
                    "governor": governor,
                    "threads": threads,
                    "requested_resolution": resolution,
                    "resolved_resolution": _resolved_resolution(engine, resolution),
                    "governor_applied": applied.governor_applied,
                    "governor_apply_error": applied.governor_apply_error,
                    "mean_total_ms": sum(latencies) / len(latencies),
                    "mean_onnx_ms": sum(onnx_latencies) / len(onnx_latencies),
                    "frames": len(latencies),
                }
                rows.append(row)
                print(row, flush=True)

    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved benchmark: {args.output}")


if __name__ == "__main__":
    main()
