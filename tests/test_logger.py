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
        action_mode="scene_medium",
        temp_c=None,
        freq_mhz_avg=None,
        arm_clock_mhz=None,
        power_w=None,
        did_infer=True,
        latency_ms=10.0,
        fps=15.0,
        loop_fps=15.0,
        effective_inference_fps=15.0,
        input_resolution=640,
        inference_interval=1,
        cpu_threads=4,
        governor="ondemand",
        decoder_layers=None,
        query_budget=200,
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
        action_mode="scene_light",
        temp_c=50.0,
        freq_mhz_avg=1500.0,
        arm_clock_mhz=1531.4,
        power_w=None,
        did_infer=True,
        latency_ms=5.0,
        fps=20.0,
        loop_fps=20.0,
        effective_inference_fps=10.0,
        input_resolution=480,
        inference_interval=2,
        cpu_threads=2,
        governor=None,
        decoder_layers=None,
        query_budget=None,
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
    assert row["effective_inference_fps"] == "10.0"
