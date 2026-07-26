"""Smoke tests for the defense experiment analysis."""

from __future__ import annotations

import csv
from pathlib import Path

from scripts.analyze_defense_experiment_suite import make_plots, summarize_run


def test_defense_summary_and_query_plots(tmp_path: Path) -> None:
    run_dir = tmp_path / "01_native_fp32"
    run_dir.mkdir()
    rows = [
        {
            "timestamp": str(1000 + frame_id),
            "did_infer": "True",
            "detector_invocation_count": str(frame_id + 1),
            "full_detector_invocation_count": str(frame_id + 1),
            "roi_detector_invocation_count": "0",
            "roi_refresh_applied": "False",
            "latency_ms": str(100 + frame_id),
            "loop_fps": "8.0",
            "actual_inference_fps": "8.0",
            "temp_c": str(50 + frame_id),
            "arm_clock_mhz": "1500",
            "tracking_mode": "detect_reset",
            "fan_duty_cycle": "0",
            "currently_throttled": "False",
            "soft_temp_limit": "False",
        }
        for frame_id in range(5)
    ]
    with (run_dir / "runtime.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    run_meta = {
        "key": "native_fp32",
        "title": "Native FP32",
        "groups": ["A", "B"],
        "strategy": "native_rtdetr",
        "model_kind": "native",
        "fixed_interval": None,
        "fan_control": "disabled",
    }
    quality = {
        "pseudo_recall": "1",
        "precision_proxy": "1",
        "mean_matched_iou": "1",
        "mean_center_error_norm": "0",
        "detection_count_ratio": "1",
        "infer_frame_pseudo_recall": "1",
        "noninfer_frame_pseudo_recall": "1",
        "noninfer_frame_precision_proxy": "1",
        "lost_object_frame_ratio": "0",
    }
    summary = summarize_run(run_dir, run_meta, quality)
    assert summary["detector_invocation_count"] == 5
    assert summary["detector_invocation_ratio"] == 1.0
    assert summary["pseudo_recall"] == 1.0

    output_dir = tmp_path / "analysis"
    output_dir.mkdir()
    make_plots([(summary, rows)], [summary], output_dir)
    assert len(list(output_dir.glob("*.png"))) == 9
