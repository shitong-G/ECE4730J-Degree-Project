#!/usr/bin/env python3
"""Print overall latency and serial stage-latency statistics from experiment CSVs."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from statistics import mean, median


MAIN_COLUMNS = [
    ("latency_ms", "RT-DETR latency from main log"),
    ("tracking_ms", "LK tracking"),
    ("loop_fps", "Loop FPS"),
    ("actual_inference_fps", "Actual inference FPS"),
    ("effective_inference_fps", "Effective inference FPS"),
]

PROFILE_COLUMNS = [
    ("serial_total_ms", "Serial total"),
    ("source_total_ms", "Frame source total"),
    ("source_wait_ms", "Source wait"),
    ("capture_ms", "Camera/video capture"),
    ("isp_ms", "Lite ISP"),
    ("source_resize_ms", "Source resize"),
    ("source_save_ms", "Source save latest"),
    ("source_runtime_resize_ms", "Runtime input resize"),
    ("source_consumer_wait_ms", "Consumer wait"),
    ("source_frame_age_ms", "Frame age at consume"),
    ("source_dropped_frames", "Dropped source frames"),
    ("source_error_count", "Source recoverable errors"),
    ("frame_total_ms", "Runtime after frame received"),
    ("scene_ms", "Scene features"),
    ("device_ms", "Device monitor"),
    ("runtime_state_ms", "Runtime state classify"),
    ("decision_ms", "Controller decision"),
    ("infer_outer_ms", "Inference outer call"),
    ("preprocess_ms", "RT-DETR preprocess"),
    ("build_feed_ms", "ONNX feed build"),
    ("session_select_ms", "ONNX session select"),
    ("onnx_run_ms", "ONNX run"),
    ("postprocess_ms", "RT-DETR postprocess"),
    ("infer_total_ms", "RT-DETR total"),
    ("summary_ms", "Detection summary"),
    ("main_log_write_ms", "Main log write"),
]

INFERENCE_REASON_GROUPS = [
    ("roi_refresh_", "ROI refresh"),
    ("forced_refresh_", "Full-frame forced refresh"),
    ("detector_frame", "Detector reset/init"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Main experiment CSV")
    parser.add_argument(
        "--profile",
        type=Path,
        default=None,
        help="Profile CSV. Defaults to <input_stem>_profile.csv next to --input.",
    )
    parser.add_argument(
        "--include-zero",
        action="store_true",
        help="Include zero values in latency statistics. Default excludes zeros.",
    )
    parser.add_argument(
        "--inference-only",
        action="store_true",
        help="For profile timing, only summarize rows where did_infer is true.",
    )
    parser.add_argument(
        "--top-slow-inferences",
        type=int,
        default=10,
        help="Print this many slowest inference rows. Use 0 to disable.",
    )
    parser.add_argument(
        "--top-slow-device",
        type=int,
        default=10,
        help="Print this many slowest device-monitor rows. Use 0 to disable.",
    )
    return parser.parse_args()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _to_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def _to_bool(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"true", "1", "yes"}


def _series(
    rows: list[dict[str, str]],
    column: str,
    *,
    include_zero: bool = False,
) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = _to_float(row.get(column))
        if value is None:
            continue
        if value == 0.0 and not include_zero:
            continue
        values.append(value)
    return values


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


def _fmt(value: float | int | None, digits: int = 3) -> str:
    if value is None:
        return "--"
    return f"{float(value):.{digits}f}"


def _print_stats(title: str, rows: list[dict[str, str]], columns: list[tuple[str, str]], include_zero: bool) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    print(
        f"{'stage':28s} {'n':>7s} {'mean':>11s} {'median':>11s} "
        f"{'p95':>11s} {'max':>11s}"
    )
    for column, label in columns:
        values = _series(rows, column, include_zero=include_zero)
        if not values:
            continue
        print(
            f"{label[:28]:28s} {len(values):7d} "
            f"{_fmt(mean(values)):>11s} {_fmt(median(values)):>11s} "
            f"{_fmt(_percentile(values, 0.95)):>11s} {_fmt(max(values)):>11s}"
        )


def _print_series_stats(title: str, groups: list[tuple[str, list[float]]]) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    print(
        f"{'group':36s} {'n':>7s} {'mean':>11s} {'median':>11s} "
        f"{'p95':>11s} {'max':>11s}"
    )
    for label, values in groups:
        if not values:
            continue
        print(
            f"{label[:36]:36s} {len(values):7d} "
            f"{_fmt(mean(values)):>11s} {_fmt(median(values)):>11s} "
            f"{_fmt(_percentile(values, 0.95)):>11s} {_fmt(max(values)):>11s}"
        )


def _print_inference_breakdown(rows: list[dict[str, str]], include_zero: bool) -> None:
    inference_rows = [row for row in rows if _to_bool(row.get("did_infer"))]
    if not inference_rows:
        return

    reason_values: dict[str, list[float]] = {}
    group_values: dict[str, list[float]] = {}
    for row in inference_rows:
        latency = _to_float(row.get("latency_ms"))
        if latency is None:
            continue
        if latency == 0.0 and not include_zero:
            continue
        reason = row.get("tracking_reason") or "unknown"
        reason_values.setdefault(reason, []).append(latency)
        group_label = "Other inference"
        for prefix, label in INFERENCE_REASON_GROUPS:
            if reason == prefix or reason.startswith(prefix):
                group_label = label
                break
        group_values.setdefault(group_label, []).append(latency)

    ordered_reasons = sorted(
        reason_values.items(),
        key=lambda item: (-len(item[1]), item[0]),
    )
    ordered_groups = sorted(
        group_values.items(),
        key=lambda item: (-len(item[1]), item[0]),
    )
    _print_series_stats("Inference Latency By Tracking Reason", ordered_reasons)
    _print_series_stats("Inference Latency By Refresh Type", ordered_groups)


def _print_lk_quality_confirmation(rows: list[dict[str, str]]) -> None:
    if not rows or "lk_quality_confirm_deferred" not in rows[0]:
        return

    deferred_rows = [
        row for row in rows if _to_bool(row.get("lk_quality_confirm_deferred"))
    ]
    confirming_rows = [
        row
        for row in rows
        if (row.get("tracking_reason") or "") == "lk_quality_degraded_confirming"
    ]
    forced_rows = [
        row
        for row in rows
        if (row.get("tracking_reason") or "")
        == "forced_refresh_lk_tracking_quality_degraded"
    ]
    raw_quality_rows = [
        row
        for row in rows
        if (row.get("tracking_reason") or "") == "lk_tracking_quality_degraded"
    ]
    total_deferred = max(
        [
            int(value)
            for value in (
                _to_float(row.get("lk_quality_confirm_total_deferred"))
                for row in rows
            )
            if value is not None
        ]
        or [0]
    )

    if not (deferred_rows or confirming_rows or forced_rows or raw_quality_rows):
        return

    print("\nLK Quality Refresh Confirmation")
    print("-------------------------------")
    print(f"deferred soft LK-quality refresh rows: {len(deferred_rows)}")
    print(f"total deferred since run start:        {total_deferred}")
    print(f"full-frame LK-quality refresh rows:   {len(forced_rows)}")
    if confirming_rows:
        ratios = _series(confirming_rows, "tracking_failure_ratio", include_zero=True)
        qualities = _series(confirming_rows, "tracking_mean_quality", include_zero=True)
        failed_boxes = _series(confirming_rows, "tracking_failed_box_count", include_zero=True)
        print(
            "deferred row context: "
            f"failure_ratio median={_fmt(median(ratios) if ratios else None)}, "
            f"quality median={_fmt(median(qualities) if qualities else None)}, "
            f"failed_boxes median={_fmt(median(failed_boxes) if failed_boxes else None, 0)}"
        )
    if forced_rows:
        ratios = _series(forced_rows, "tracking_failure_ratio", include_zero=True)
        qualities = _series(forced_rows, "tracking_mean_quality", include_zero=True)
        failed_boxes = _series(forced_rows, "tracking_failed_box_count", include_zero=True)
        counts = _series(forced_rows, "lk_quality_confirm_count", include_zero=True)
        print(
            "forced row context:  "
            f"confirm_count median={_fmt(median(counts) if counts else None, 0)}, "
            f"failure_ratio median={_fmt(median(ratios) if ratios else None)}, "
            f"quality median={_fmt(median(qualities) if qualities else None)}, "
            f"failed_boxes median={_fmt(median(failed_boxes) if failed_boxes else None, 0)}"
        )


def _print_roi_refresh_details(rows: list[dict[str, str]], include_zero: bool) -> None:
    roi_rows = [
        row
        for row in rows
        if _to_bool(row.get("roi_refresh_candidate"))
        or str(row.get("tracking_reason") or "").startswith("roi_refresh_")
    ]
    if not roi_rows:
        return

    groups: dict[str, list[dict[str, str]]] = {}
    for row in roi_rows:
        label = row.get("roi_refresh_reason") or row.get("tracking_reason") or "unknown"
        reject = row.get("roi_refresh_reject_reason") or ""
        if reject:
            label = f"{label} / rejected:{reject}"
        elif _to_bool(row.get("roi_refresh_applied")):
            label = f"{label} / applied"
        groups.setdefault(label, []).append(row)

    print("\nROI Refresh Candidate Details")
    print("-----------------------------")
    print(
        f"{'group':42s} {'n':>5s} {'lat_med':>9s} {'lat_p95':>9s} "
        f"{'lat_max':>9s} {'area_mean':>10s} {'area_max':>9s} "
        f"{'w_mean':>9s} {'h_mean':>9s}"
    )
    for label, group_rows in sorted(
        groups.items(),
        key=lambda item: (-len(item[1]), item[0]),
    ):
        latencies = _series(group_rows, "latency_ms", include_zero=include_zero)
        areas = _series(group_rows, "roi_refresh_area_ratio", include_zero=True)
        widths = _series(group_rows, "roi_refresh_width_px", include_zero=True)
        heights = _series(group_rows, "roi_refresh_height_px", include_zero=True)
        print(
            f"{label[:42]:42s} {len(group_rows):5d} "
            f"{_fmt(median(latencies) if latencies else None):>9s} "
            f"{_fmt(_percentile(latencies, 0.95) if latencies else None):>9s} "
            f"{_fmt(max(latencies) if latencies else None):>9s} "
            f"{_fmt(mean(areas) if areas else None, 4):>10s} "
            f"{_fmt(max(areas) if areas else None, 4):>9s} "
            f"{_fmt(mean(widths) if widths else None):>9s} "
            f"{_fmt(mean(heights) if heights else None):>9s}"
        )


def _print_inference_diagnostic_context(
    profile_rows: list[dict[str, str]],
    include_zero: bool,
) -> None:
    inference_rows = [row for row in profile_rows if _to_bool(row.get("did_infer"))]
    if not inference_rows:
        return
    if "diag_infer_start_bg_active" not in inference_rows[0]:
        return

    def onnx_values(rows: list[dict[str, str]]) -> list[float]:
        return _series(rows, "onnx_run_ms", include_zero=include_zero)

    groups = [
        (
            "bg active at inference start",
            onnx_values(
                [
                    row
                    for row in inference_rows
                    if (_to_float(row.get("diag_infer_start_bg_active")) or 0.0) >= 0.5
                ]
            ),
        ),
        (
            "bg inactive at inference start",
            onnx_values(
                [
                    row
                    for row in inference_rows
                    if (_to_float(row.get("diag_infer_start_bg_active")) or 0.0) < 0.5
                ]
            ),
        ),
        (
            "bg capture completed during inference",
            onnx_values(
                [
                    row
                    for row in inference_rows
                    if (_to_float(row.get("diag_infer_bg_captures_delta")) or 0.0) > 0.0
                ]
            ),
        ),
        (
            "no bg capture completed during inference",
            onnx_values(
                [
                    row
                    for row in inference_rows
                    if (_to_float(row.get("diag_infer_bg_captures_delta")) or 0.0) <= 0.0
                ]
            ),
        ),
        (
            "bg trigger skipped during inference",
            onnx_values(
                [
                    row
                    for row in inference_rows
                    if (_to_float(row.get("diag_infer_bg_skipped_delta")) or 0.0) > 0.0
                ]
            ),
        ),
        (
            "bg error during inference",
            onnx_values(
                [
                    row
                    for row in inference_rows
                    if (_to_float(row.get("diag_infer_bg_errors_delta")) or 0.0) > 0.0
                ]
            ),
        ),
    ]
    if not any(values for _, values in groups):
        return
    _print_series_stats("Inference ONNX Run By Diagnostic Context", groups)

    load_values = _series(inference_rows, "diag_infer_start_load1", include_zero=True)
    mem_values = _series(
        inference_rows,
        "diag_infer_start_mem_available_mb",
        include_zero=True,
    )
    thread_values = _series(
        inference_rows,
        "diag_infer_start_process_threads",
        include_zero=True,
    )
    if load_values or mem_values or thread_values:
        print("\nInference Host Diagnostic Ranges")
        print("--------------------------------")
        if load_values:
            print(
                "load1 start: "
                f"median={_fmt(median(load_values))}, "
                f"p95={_fmt(_percentile(load_values, 0.95))}, "
                f"max={_fmt(max(load_values))}"
            )
        if mem_values:
            print(
                "mem available MB start: "
                f"median={_fmt(median(mem_values))}, "
                f"p05={_fmt(_percentile(mem_values, 0.05))}, "
                f"min={_fmt(min(mem_values))}"
            )
        if thread_values:
            print(
                "process threads start: "
                f"median={_fmt(median(thread_values))}, "
                f"p95={_fmt(_percentile(thread_values, 0.95))}, "
                f"max={_fmt(max(thread_values))}"
            )


def _profile_by_frame_id(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {
        str(row.get("frame_id") or ""): row
        for row in rows
        if str(row.get("frame_id") or "") != ""
    }


def _print_slowest_inference_rows(
    main_rows: list[dict[str, str]],
    profile_rows: list[dict[str, str]],
    *,
    limit: int,
    include_zero: bool,
) -> None:
    if limit <= 0:
        return
    inference_rows: list[dict[str, str]] = []
    for row in main_rows:
        if not _to_bool(row.get("did_infer")):
            continue
        latency = _to_float(row.get("latency_ms"))
        if latency is None:
            continue
        if latency == 0.0 and not include_zero:
            continue
        inference_rows.append(row)
    if not inference_rows:
        return

    profile_lookup = _profile_by_frame_id(profile_rows)
    slowest = sorted(
        inference_rows,
        key=lambda row: _to_float(row.get("latency_ms")) or -1.0,
        reverse=True,
    )[:limit]

    print(f"\nTop {len(slowest)} Slowest Inference Rows")
    print("--------------------------------")
    print(
        f"{'frame':>7s} {'lat_ms':>9s} {'onnx_ms':>9s} {'scene_ms':>9s} "
        f"{'dev_ms':>8s} {'temp':>7s} {'freq':>7s} {'arm':>7s} "
        f"{'uv':>3s} {'uvH':>3s} {'throt':>6s} {'thrH':>4s} {'soft':>5s} {'roi_area':>8s} "
        f"{'bgS':>4s} {'bgE':>4s} {'bgCap':>5s} {'bgSkip':>6s} "
        f"{'load1':>7s} {'memMB':>7s} {'reason':36s}"
    )
    for row in slowest:
        frame_id = str(row.get("frame_id") or "")
        profile = profile_lookup.get(frame_id, {})
        print(
            f"{frame_id[:7]:>7s} "
            f"{_fmt(_to_float(row.get('latency_ms'))):>9s} "
            f"{_fmt(_to_float(profile.get('onnx_run_ms'))):>9s} "
            f"{_fmt(_to_float(profile.get('scene_ms'))):>9s} "
            f"{_fmt(_to_float(profile.get('device_ms'))):>8s} "
            f"{_fmt(_to_float(row.get('temp_c'))):>7s} "
            f"{_fmt(_to_float(row.get('freq_mhz_avg'))):>7s} "
            f"{_fmt(_to_float(row.get('arm_clock_mhz'))):>7s} "
            f"{_bool_label(row.get('under_voltage')):>3s} "
            f"{_bool_label(row.get('under_voltage_occurred')):>3s} "
            f"{_bool_label(row.get('currently_throttled')):>6s} "
            f"{_bool_label(row.get('throttled_occurred')):>4s} "
            f"{_bool_label(row.get('soft_temp_limit')):>5s} "
            f"{_fmt(_to_float(row.get('roi_refresh_area_ratio')), 4):>8s} "
            f"{_fmt(_to_float(profile.get('diag_infer_start_bg_active')), 0):>4s} "
            f"{_fmt(_to_float(profile.get('diag_infer_end_bg_active')), 0):>4s} "
            f"{_fmt(_to_float(profile.get('diag_infer_bg_captures_delta')), 0):>5s} "
            f"{_fmt(_to_float(profile.get('diag_infer_bg_skipped_delta')), 0):>6s} "
            f"{_fmt(_to_float(profile.get('diag_infer_start_load1'))):>7s} "
            f"{_fmt(_to_float(profile.get('diag_infer_start_mem_available_mb'))):>7s} "
            f"{str(row.get('tracking_reason') or 'unknown')[:36]:36s}"
        )


def _main_by_frame_id(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {
        str(row.get("frame_id") or ""): row
        for row in rows
        if str(row.get("frame_id") or "") != ""
    }


def _bool_label(value: str | None) -> str:
    text = str(value or "").strip()
    if not text:
        return "--"
    lowered = text.lower()
    if lowered in {"true", "1", "yes"}:
        return "Y"
    if lowered in {"false", "0", "no"}:
        return "N"
    return text[:3]


def _print_slowest_device_rows(
    main_rows: list[dict[str, str]],
    profile_rows: list[dict[str, str]],
    *,
    limit: int,
    include_zero: bool,
) -> None:
    if limit <= 0 or not profile_rows:
        return
    device_rows: list[dict[str, str]] = []
    for row in profile_rows:
        device_ms = _to_float(row.get("device_ms"))
        if device_ms is None:
            continue
        if device_ms == 0.0 and not include_zero:
            continue
        device_rows.append(row)
    if not device_rows:
        return

    main_lookup = _main_by_frame_id(main_rows)
    slowest = sorted(
        device_rows,
        key=lambda row: _to_float(row.get("device_ms")) or -1.0,
        reverse=True,
    )[:limit]

    print(f"\nTop {len(slowest)} Slowest Device Monitor Rows")
    print("-----------------------------------")
    print(
        f"{'frame':>7s} {'dev_ms':>9s} {'fw_ms':>8s} {'scene_ms':>9s} "
        f"{'infer':>5s} {'track_ms':>9s} {'temp':>7s} {'freq':>7s} "
        f"{'arm':>7s} {'armSt':>5s} {'thrSt':>5s} {'uv':>3s} {'uvH':>3s} "
        f"{'throt':>6s} {'thrH':>4s} {'soft':>5s} {'reason':36s}"
    )
    for profile in slowest:
        frame_id = str(profile.get("frame_id") or "")
        row = main_lookup.get(frame_id, {})
        print(
            f"{frame_id[:7]:>7s} "
            f"{_fmt(_to_float(profile.get('device_ms'))):>9s} "
            f"{_fmt(_to_float(row.get('firmware_poll_ms'))):>8s} "
            f"{_fmt(_to_float(profile.get('scene_ms'))):>9s} "
            f"{_bool_label(row.get('did_infer')):>5s} "
            f"{_fmt(_to_float(row.get('tracking_ms'))):>9s} "
            f"{_fmt(_to_float(row.get('temp_c'))):>7s} "
            f"{_fmt(_to_float(row.get('freq_mhz_avg'))):>7s} "
            f"{_fmt(_to_float(row.get('arm_clock_mhz'))):>7s} "
            f"{_bool_label(row.get('arm_clock_stale')):>5s} "
            f"{_bool_label(row.get('throttling_stale')):>5s} "
            f"{_bool_label(row.get('under_voltage')):>3s} "
            f"{_bool_label(row.get('under_voltage_occurred')):>3s} "
            f"{_bool_label(row.get('currently_throttled')):>6s} "
            f"{_bool_label(row.get('throttled_occurred')):>4s} "
            f"{_bool_label(row.get('soft_temp_limit')):>5s} "
            f"{str(row.get('tracking_reason') or 'unknown')[:36]:36s}"
        )


def _wall_time(rows: list[dict[str, str]]) -> float | None:
    timestamps = [_to_float(row.get("timestamp")) for row in rows]
    timestamps = [value for value in timestamps if value is not None]
    if len(timestamps) < 2:
        return None
    return max(timestamps) - min(timestamps)


def _default_profile_path(input_path: Path) -> Path:
    return input_path.with_name(input_path.stem + "_profile.csv")


def print_stage_latency_summary(
    input_path: Path,
    *,
    profile_path: Path | None = None,
    include_zero: bool = False,
    inference_only: bool = False,
    top_slow_inferences: int = 10,
    top_slow_device: int = 10,
) -> None:
    """Read logs and print overall/stage latency statistics."""
    main_rows = _read_csv(input_path)
    resolved_profile_path = profile_path or _default_profile_path(input_path)
    profile_rows = _read_csv(resolved_profile_path) if resolved_profile_path.exists() else []

    if inference_only and profile_rows:
        profile_rows = [row for row in profile_rows if _to_bool(row.get("did_infer"))]

    inference_frames = sum(1 for row in main_rows if _to_bool(row.get("did_infer")))
    print(f"Input log:    {input_path}")
    print(f"Profile log:  {resolved_profile_path if resolved_profile_path.exists() else 'not found'}")
    print(f"Total frames: {len(main_rows)}")
    print(f"Inferences:   {inference_frames}")
    print(f"Wall time s:  {_fmt(_wall_time(main_rows))}")

    _print_stats("Overall From Main CSV", main_rows, MAIN_COLUMNS, include_zero)
    _print_inference_breakdown(main_rows, include_zero)
    _print_lk_quality_confirmation(main_rows)
    _print_roi_refresh_details(main_rows, include_zero)
    if profile_rows:
        _print_inference_diagnostic_context(profile_rows, include_zero)
    _print_slowest_inference_rows(
        main_rows,
        profile_rows,
        limit=top_slow_inferences,
        include_zero=include_zero,
    )
    _print_slowest_device_rows(
        main_rows,
        profile_rows,
        limit=top_slow_device,
        include_zero=include_zero,
    )
    if profile_rows:
        title = "Stage Latency From Profile CSV"
        if inference_only:
            title += " (Inference Rows Only)"
        _print_stats(title, profile_rows, PROFILE_COLUMNS, include_zero)
    else:
        print("\nStage Latency From Profile CSV")
        print("------------------------------")
        print("Profile CSV not found. Run a newer experiment or pass --profile explicitly.")


def main() -> None:
    args = parse_args()
    print_stage_latency_summary(
        args.input,
        profile_path=args.profile,
        include_zero=args.include_zero,
        inference_only=args.inference_only,
        top_slow_inferences=args.top_slow_inferences,
        top_slow_device=args.top_slow_device,
    )


if __name__ == "__main__":
    main()
