#!/usr/bin/env python3
"""Summarize experiment CSV logs into baseline metrics."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, median


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize scene-runtime baseline CSV logs")
    parser.add_argument("--input", type=Path, required=True, help="Experiment CSV log")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/results/baseline_native"),
        help="Directory for JSON/CSV summary outputs",
    )
    parser.add_argument("--label", default=None, help="Optional run label")
    return parser.parse_args()


def _to_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        out = float(value)
    except ValueError:
        return None
    return out if math.isfinite(out) else None


def _series(rows: list[dict[str, str]], column: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = _to_float(row.get(column))
        if value is not None:
            values.append(value)
    return values


def _positive_series(rows: list[dict[str, str]], column: str) -> list[float]:
    return [value for value in _series(rows, column) if value > 0.0]


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    lower = math.floor(pos)
    upper = math.ceil(pos)
    if lower == upper:
        return ordered[lower]
    weight = pos - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _stat(values: list[float], op: str) -> float | None:
    if not values:
        return None
    if op == "mean":
        return mean(values)
    if op == "median":
        return median(values)
    if op == "min":
        return min(values)
    if op == "max":
        return max(values)
    raise ValueError(f"Unknown stat op: {op}")


def summarize(rows: list[dict[str, str]], label: str) -> dict[str, float | int | str | None]:
    timestamps = _series(rows, "timestamp")
    latencies = _positive_series(rows, "latency_ms")
    fps = _series(rows, "fps")
    loop_fps = _series(rows, "loop_fps") or fps
    effective_inference_fps = _series(rows, "effective_inference_fps")
    temps = _series(rows, "temp_c")
    freqs = _series(rows, "freq_mhz_avg")
    arm_clocks = _series(rows, "arm_clock_mhz")
    detections = _series(rows, "detection_count")
    confidences = _series(rows, "confidence_mean")

    return {
        "label": label,
        "strategy": rows[0].get("strategy") if rows else None,
        "wall_time_sec": (timestamps[-1] - timestamps[0]) if len(timestamps) >= 2 else 0.0,
        "total_frames": len(rows),
        "inference_frames": len(latencies),
        "skipped_frames": len(rows) - len(latencies),
        "latency_ms_mean": _stat(latencies, "mean"),
        "latency_ms_median": _stat(latencies, "median"),
        "latency_ms_p95": _percentile(latencies, 0.95),
        "latency_ms_p99": _percentile(latencies, 0.99),
        "latency_ms_max": _stat(latencies, "max"),
        "fps_mean": _stat(fps, "mean"),
        "fps_median": _stat(fps, "median"),
        "fps_min": _stat(fps, "min"),
        "fps_max": _stat(fps, "max"),
        "loop_fps_mean": _stat(loop_fps, "mean"),
        "effective_inference_fps_mean": _stat(effective_inference_fps, "mean"),
        "effective_inference_fps_median": _stat(effective_inference_fps, "median"),
        "effective_inference_fps_min": _stat(effective_inference_fps, "min"),
        "effective_inference_fps_max": _stat(effective_inference_fps, "max"),
        "temp_c_start": temps[0] if temps else None,
        "temp_c_end": temps[-1] if temps else None,
        "temp_c_mean": _stat(temps, "mean"),
        "temp_c_max": _stat(temps, "max"),
        "freq_mhz_avg_mean": _stat(freqs, "mean"),
        "arm_clock_mhz_mean": _stat(arm_clocks, "mean"),
        "arm_clock_mhz_min": _stat(arm_clocks, "min"),
        "arm_clock_mhz_max": _stat(arm_clocks, "max"),
        "detection_count_mean": _stat(detections, "mean"),
        "confidence_mean": _stat(confidences, "mean"),
    }


def main() -> None:
    args = parse_args()
    with args.input.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    label = args.label or args.input.stem
    summary = summarize(rows, label)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / f"{label}_summary.json"
    csv_path = args.output_dir / f"{label}_summary.csv"

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)

    print(f"Saved JSON summary: {json_path}")
    print(f"Saved CSV summary:  {csv_path}")


if __name__ == "__main__":
    main()
