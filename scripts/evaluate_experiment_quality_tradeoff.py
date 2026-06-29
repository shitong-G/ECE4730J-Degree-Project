#!/usr/bin/env python3
"""Compute quality-adjusted performance scores for experiment CSV logs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from statistics import mean


DEFAULT_QUALITY_WEIGHTS = {
    640: 1.0,
    480: 0.500945179584121,
    320: 0.21928166351606806,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate quality/performance/thermal trade-off from logs")
    parser.add_argument("logs", nargs="+", type=Path, help="Experiment CSV logs")
    parser.add_argument(
        "--quality-summary",
        type=Path,
        default=Path("experiments/results/quality/resolution_quality_summary.csv"),
        help="Optional resolution quality CSV from evaluate_resolution_quality.py",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/results/quality_tradeoff_summary.csv"),
    )
    parser.add_argument("--temp-threshold-c", type=float, default=80.0)
    return parser.parse_args()


def _to_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        out = float(value)
    except ValueError:
        return None
    return out if math.isfinite(out) else None


def _to_bool(value: str | None) -> bool | None:
    if value is None or value == "":
        return None
    lowered = value.strip().lower()
    if lowered in {"true", "1", "yes"}:
        return True
    if lowered in {"false", "0", "no"}:
        return False
    return None


def _mean(values: list[float]) -> float | None:
    return mean(values) if values else None


def _load_quality_weights(path: Path) -> dict[int, float]:
    if not path.exists():
        return dict(DEFAULT_QUALITY_WEIGHTS)
    weights: dict[int, float] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            resolution = _to_float(row.get("resolution"))
            recall = _to_float(row.get("pseudo_recall"))
            if resolution is not None and recall is not None:
                weights[int(resolution)] = max(0.0, min(1.0, recall))
    return weights or dict(DEFAULT_QUALITY_WEIGHTS)


def _duration_sec(rows: list[dict[str, str]]) -> float:
    timestamps = [_to_float(row.get("timestamp")) for row in rows]
    timestamps = [value for value in timestamps if value is not None]
    if len(timestamps) < 2:
        return 0.0
    return max(0.0, timestamps[-1] - timestamps[0])


def _infer_dt(rows: list[dict[str, str]], index: int, duration: float) -> float:
    current = _to_float(rows[index].get("timestamp"))
    if current is not None and index + 1 < len(rows):
        nxt = _to_float(rows[index + 1].get("timestamp"))
        if nxt is not None:
            return max(0.0, nxt - current)
    return duration / max(1, len(rows))


def _quality_weight(row: dict[str, str], weights: dict[int, float]) -> float:
    resolution = _to_float(row.get("resolved_input_resolution")) or _to_float(row.get("input_resolution"))
    if resolution is None:
        return 1.0
    resolution_int = int(resolution)
    if resolution_int in weights:
        return weights[resolution_int]
    return weights[min(weights, key=lambda item: abs(item - resolution_int))]


def summarize_log(path: Path, weights: dict[int, float], temp_threshold_c: float) -> dict[str, object]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    duration = _duration_sec(rows)
    if not rows:
        return {"log": str(path), "label": path.stem, "error": "empty_log"}

    raw_inference_count = 0
    quality_inference_count = 0.0
    temp_above_sec = 0.0
    soft_count = throttled_count = capped_count = 0
    bool_count = 0
    weighted_resolution_sec: Counter[str] = Counter()

    for index, row in enumerate(rows):
        dt = _infer_dt(rows, index, duration)
        temp = _to_float(row.get("temp_c"))
        if temp is not None and temp >= temp_threshold_c:
            temp_above_sec += dt
        for col, name in [
            ("soft_temp_limit", "soft"),
            ("currently_throttled", "throttled"),
            ("arm_freq_capped", "capped"),
        ]:
            value = _to_bool(row.get(col))
            if value is not None:
                if col == "soft_temp_limit":
                    soft_count += int(value)
                elif col == "currently_throttled":
                    throttled_count += int(value)
                else:
                    capped_count += int(value)
        if any(_to_bool(row.get(col)) is not None for col in ["soft_temp_limit", "currently_throttled", "arm_freq_capped"]):
            bool_count += 1

        res = row.get("resolved_input_resolution") or row.get("input_resolution") or "unknown"
        weighted_resolution_sec[str(res)] += dt
        if _to_bool(row.get("did_infer")):
            raw_inference_count += 1
            quality_inference_count += _quality_weight(row, weights)

    latencies = [
        value
        for value in (_to_float(row.get("latency_ms")) for row in rows)
        if value is not None and value > 0
    ]
    actual_fps = [
        value
        for value in (_to_float(row.get("actual_inference_fps")) for row in rows)
        if value is not None
    ]
    temps = [
        value for value in (_to_float(row.get("temp_c")) for row in rows) if value is not None
    ]
    powers = [
        value for value in (_to_float(row.get("power_w")) for row in rows) if value is not None
    ]
    qfps = quality_inference_count / duration if duration > 0 else None
    raw_fps = raw_inference_count / duration if duration > 0 else None
    thermal_penalty = 0.0
    if duration > 0:
        thermal_penalty += temp_above_sec / duration
    if bool_count > 0:
        thermal_penalty += throttled_count / bool_count + capped_count / bool_count
    utility = qfps / (1.0 + thermal_penalty) if qfps is not None else None

    return {
        "log": str(path),
        "label": path.stem,
        "strategy": rows[0].get("strategy"),
        "duration_sec": duration,
        "frames": len(rows),
        "raw_inference_count": raw_inference_count,
        "quality_adjusted_inference_count": quality_inference_count,
        "raw_detector_fps": raw_fps,
        "quality_adjusted_fps": qfps,
        "sustained_utility": utility,
        "latency_ms_mean": _mean(latencies),
        "latency_ms_p95": _percentile(latencies, 0.95),
        "actual_inference_fps_mean": _mean(actual_fps),
        "temp_c_mean": _mean(temps),
        "temp_c_max": max(temps) if temps else None,
        f"time_above_{int(temp_threshold_c)}c_sec": temp_above_sec,
        f"time_above_{int(temp_threshold_c)}c_ratio": temp_above_sec / duration if duration > 0 else None,
        "soft_temp_limit_ratio": soft_count / bool_count if bool_count else None,
        "currently_throttled_ratio": throttled_count / bool_count if bool_count else None,
        "arm_freq_capped_ratio": capped_count / bool_count if bool_count else None,
        "power_w_mean": _mean(powers),
        "resolution_duration_sec": json.dumps(dict(sorted(weighted_resolution_sec.items())), sort_keys=True),
    }


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    weight = pos - lo
    return ordered[lo] * (1.0 - weight) + ordered[hi] * weight


def main() -> None:
    args = parse_args()
    weights = _load_quality_weights(args.quality_summary)
    rows = [summarize_log(path, weights, args.temp_threshold_c) for path in args.logs]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Quality weights: {weights}")
    print(f"Saved trade-off summary: {args.output}")
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
