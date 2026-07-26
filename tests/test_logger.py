"""Tests for runtime logger schema."""

from __future__ import annotations

import csv
from pathlib import Path

from scene_runtime.runtime.logger import LOG_COLUMNS, LogRecord, RuntimeLogger


def test_log_record_validate() -> None:
    rec = LogRecord(
        timestamp=1.0,
        frame_id=0,
        strategy="test",
        workload="medium",
        thermal_state="normal",
        raw_thermal_state="normal",
        control_thermal_state="normal",
        action_mode="scene_medium",
        decision_reason="normal",
        thermal_pressure_level=0,
        temp_slope_c_per_min=0.0,
        temp_c=None,
        freq_mhz_avg=None,
        arm_clock_mhz=None,
        arm_clock_stale=None,
        firmware_poll_ms=0.0,
        power_w=None,
        throttling_raw=None,
        throttling_stale=None,
        under_voltage=None,
        arm_freq_capped=None,
        currently_throttled=None,
        soft_temp_limit=None,
        under_voltage_occurred=None,
        arm_freq_capped_occurred=None,
        throttled_occurred=None,
        soft_temp_limit_occurred=None,
        did_infer=True,
        detector_invocation_count=1,
        detector_invocation_ratio=1.0,
        full_detector_invocation_count=1,
        full_detector_invocation_ratio=1.0,
        roi_detector_invocation_count=0,
        roi_detector_invocation_ratio=0.0,
        detector_call_type="full",
        detector_call_resolution=640,
        tracking_mode="detect_reset",
        tracking_reason="detector_frame",
        tracking_ms=0.0,
        tracking_track_count_before=0,
        tracking_track_count_after=0,
        tracking_failed_box_count=0,
        tracking_failure_ratio=0.0,
        tracking_mean_quality=1.0,
        tracking_should_refresh=False,
        lk_quality_confirm_count=0,
        lk_quality_confirm_deferred=False,
        lk_quality_confirm_total_deferred=0,
        roi_refresh_candidate=False,
        roi_refresh_applied=False,
        roi_refresh_reason=None,
        roi_refresh_reject_reason=None,
        roi_refresh_area_ratio=0.0,
        roi_refresh_width_px=0.0,
        roi_refresh_height_px=0.0,
        roi_refresh_max_area_ratio=0.0,
        latency_ms=10.0,
        fps=15.0,
        loop_fps=15.0,
        effective_inference_fps=15.0,
        actual_inference_fps=15.0,
        input_resolution=640,
        resolved_input_resolution=640,
        inference_interval=1,
        cpu_threads=4,
        governor="ondemand",
        requested_governor="ondemand",
        applied_governor=None,
        governor_applied=None,
        governor_apply_error=None,
        requested_cpu_affinity=None,
        applied_cpu_affinity=None,
        cpu_affinity_applied=None,
        cpu_affinity_apply_error=None,
        decoder_layers=None,
        query_budget=200,
        query_budget_requested=200,
        query_budget_applied=200,
        query_budget_mode="graph_input",
        query_budget_supported=True,
        query_budget_source="temperature",
        query_budget_temperature_state="warm",
        query_output_count=200,
        fan_enabled=False,
        fan_duty_cycle=0.0,
        fan_mode="disabled",
        detection_count=2,
        confidence_mean=0.7,
    )
    rec.validate()
    d = rec.to_dict()
    assert set(d.keys()) == set(LOG_COLUMNS)


def test_logger_writes_csv(tmp_path: Path) -> None:
    path = tmp_path / "test.csv"
    logger = RuntimeLogger(path, fmt="csv")
    logger.open()
    rec = LogRecord(
        timestamp=1.0,
        frame_id=0,
        strategy="dry",
        workload="light",
        thermal_state="normal",
        raw_thermal_state="normal",
        control_thermal_state="normal",
        action_mode="scene_light",
        decision_reason="normal",
        thermal_pressure_level=0,
        temp_slope_c_per_min=0.0,
        temp_c=50.0,
        freq_mhz_avg=1500.0,
        arm_clock_mhz=1531.4,
        arm_clock_stale=False,
        firmware_poll_ms=12.5,
        power_w=None,
        throttling_raw="throttled=0x0",
        throttling_stale=False,
        under_voltage=False,
        arm_freq_capped=False,
        currently_throttled=False,
        soft_temp_limit=False,
        under_voltage_occurred=True,
        arm_freq_capped_occurred=False,
        throttled_occurred=True,
        soft_temp_limit_occurred=False,
        did_infer=True,
        detector_invocation_count=1,
        detector_invocation_ratio=1.0,
        full_detector_invocation_count=0,
        full_detector_invocation_ratio=0.0,
        roi_detector_invocation_count=1,
        roi_detector_invocation_ratio=1.0,
        detector_call_type="roi",
        detector_call_resolution=320,
        tracking_mode="detect_reset",
        tracking_reason="detector_frame",
        tracking_ms=0.0,
        tracking_track_count_before=0,
        tracking_track_count_after=0,
        tracking_failed_box_count=0,
        tracking_failure_ratio=0.0,
        tracking_mean_quality=1.0,
        tracking_should_refresh=False,
        lk_quality_confirm_count=0,
        lk_quality_confirm_deferred=False,
        lk_quality_confirm_total_deferred=0,
        roi_refresh_candidate=True,
        roi_refresh_applied=False,
        roi_refresh_reason="unexplained_motion_outside_tracks",
        roi_refresh_reject_reason="area_too_large",
        roi_refresh_area_ratio=0.30,
        roi_refresh_width_px=320.0,
        roi_refresh_height_px=288.0,
        roi_refresh_max_area_ratio=0.25,
        latency_ms=5.0,
        fps=20.0,
        loop_fps=20.0,
        effective_inference_fps=10.0,
        actual_inference_fps=9.5,
        input_resolution=480,
        resolved_input_resolution=480,
        inference_interval=2,
        cpu_threads=2,
        governor=None,
        requested_governor=None,
        applied_governor=None,
        governor_applied=None,
        governor_apply_error=None,
        requested_cpu_affinity="0,1",
        applied_cpu_affinity="0,1",
        cpu_affinity_applied=True,
        cpu_affinity_apply_error=None,
        decoder_layers=None,
        query_budget=None,
        query_budget_requested=300,
        query_budget_applied=None,
        query_budget_mode="unsupported_inactive",
        query_budget_supported=False,
        query_budget_source="action",
        query_budget_temperature_state=None,
        query_output_count=None,
        fan_enabled=True,
        fan_duty_cycle=0.5,
        fan_mode="pwm_no_gpio",
        detection_count=0,
        confidence_mean=0.0,
    )
    logger.write(rec)
    logger.close()

    with path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        row = next(reader)
    assert row["workload"] == "light"
    assert row["strategy"] == "dry"
    assert row["did_infer"] == "True"
    assert row["detector_invocation_count"] == "1"
    assert row["roi_detector_invocation_ratio"] == "1.0"
    assert row["detector_call_type"] == "roi"
    assert row["detector_call_resolution"] == "320"
    assert row["query_budget_mode"] == "unsupported_inactive"
    assert row["effective_inference_fps"] == "10.0"
    assert row["actual_inference_fps"] == "9.5"
    assert row["currently_throttled"] == "False"
    assert row["under_voltage_occurred"] == "True"
    assert row["throttled_occurred"] == "True"
    assert row["arm_clock_stale"] == "False"
    assert row["throttling_stale"] == "False"
    assert row["firmware_poll_ms"] == "12.5"
    assert row["cpu_affinity_applied"] == "True"
    assert row["fan_mode"] == "pwm_no_gpio"
    assert row["tracking_track_count_after"] == "0"
    assert row["tracking_failed_box_count"] == "0"
    assert row["lk_quality_confirm_deferred"] == "False"
    assert row["roi_refresh_candidate"] == "True"
    assert row["roi_refresh_reject_reason"] == "area_too_large"
    assert row["roi_refresh_area_ratio"] == "0.3"
