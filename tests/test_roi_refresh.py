"""Tests for ROI detector refresh helpers."""

from __future__ import annotations

import numpy as np

from scene_runtime.controller.actions import RuntimeAction
from scene_runtime.inference.postprocess import Detection
from scene_runtime.runtime.loop import RuntimeLoop
from scene_runtime.tracking.lk_tracker import LKTrackingReport


def test_roi_mapping_to_full_detection_resolution() -> None:
    detections = [
        Detection(class_id=1, score=0.9, bbox=(80.0, 80.0, 240.0, 240.0)),
    ]

    mapped = RuntimeLoop._map_roi_detections_to_full_resolution(
        detections,
        roi=(100.0, 50.0, 300.0, 250.0),
        roi_resolution=320,
        frame_shape=(480, 640, 3),
        output_resolution=480,
    )

    assert len(mapped) == 1
    x1, y1, x2, y2 = mapped[0].bbox
    assert round(x1, 3) == 112.5
    assert round(y1, 3) == 100.0
    assert round(x2, 3) == 187.5
    assert round(y2, 3) == 200.0


def test_merge_detections_keeps_best_overlapping_box() -> None:
    base = [
        Detection(class_id=1, score=0.7, bbox=(10.0, 10.0, 100.0, 100.0)),
    ]
    updates = [
        Detection(class_id=1, score=0.9, bbox=(12.0, 12.0, 98.0, 98.0)),
        Detection(class_id=2, score=0.8, bbox=(200.0, 200.0, 250.0, 250.0)),
    ]

    merged = RuntimeLoop._merge_detections(base, updates)

    assert len(merged) == 2
    assert merged[0].score == 0.9
    assert {d.class_id for d in merged} == {1, 2}


def test_roi_build_expands_failed_box() -> None:
    loop = RuntimeLoop.__new__(RuntimeLoop)
    loop._roi_refresh_expand_ratio = 2.0
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    report = LKTrackingReport(
        reason="lk_tracking_quality_degraded",
        failed_boxes_frame=[np.asarray([100.0, 120.0, 150.0, 160.0], dtype=np.float32)],
    )

    roi = RuntimeLoop._build_roi(loop, frame, report.failed_boxes_frame)

    assert roi is not None
    x1, y1, x2, y2 = roi
    assert x1 < 100.0
    assert y1 < 120.0
    assert x2 > 150.0
    assert y2 > 160.0


def test_roi_refresh_empty_detection_does_not_request_full_fallback() -> None:
    class EmptyEngine:
        last_profile = {"infer_total_ms": 12.0}
        last_resolved_input_resolution = 320

        def infer(self, _frame, _action):
            return []

    loop = RuntimeLoop.__new__(RuntimeLoop)
    loop._roi_refresh_enabled = True
    loop._last_detection_resolution = 480
    loop._roi_refresh_resolution = 320
    loop._roi_refresh_min_survivors = 1
    loop._roi_refresh_lk_quality_enabled = False
    loop._roi_refresh_max_failure_ratio = 0.60
    loop._roi_refresh_max_area_ratio = 0.45
    loop._roi_refresh_lk_max_failed_boxes = 1
    loop._roi_refresh_lk_max_failure_ratio = 0.40
    loop._roi_refresh_lk_max_area_ratio = 0.18
    loop._roi_refresh_expand_ratio = 1.8
    loop._engine = EmptyEngine()

    action = RuntimeAction(
        mode="test",
        input_resolution=480,
        inference_interval=1,
        cpu_threads=1,
    )
    report = LKTrackingReport(
        reason="unexplained_motion_outside_tracks",
        failure_ratio=0.25,
        track_count_after=1,
        refresh_boxes_frame=[
            np.asarray([100.0, 100.0, 150.0, 150.0], dtype=np.float32)
        ],
    )
    tracked = [Detection(class_id=1, score=0.7, bbox=(20.0, 20.0, 80.0, 80.0))]

    result = RuntimeLoop._try_roi_refresh(
        loop,
        np.zeros((480, 640, 3), dtype=np.uint8),
        action,
        report,
        tracked,
    )

    assert result is not None
    detections, _outer_ms, profile = result
    assert detections == tracked
    assert profile["infer_total_ms"] == 12.0


def test_lk_quality_roi_disabled_by_default() -> None:
    class CountingEngine:
        last_profile = {"infer_total_ms": 12.0}
        last_resolved_input_resolution = 320

        def __init__(self) -> None:
            self.calls = 0

        def infer(self, _frame, _action):
            self.calls += 1
            return []

    engine = CountingEngine()
    loop = RuntimeLoop.__new__(RuntimeLoop)
    loop._roi_refresh_enabled = True
    loop._last_detection_resolution = 480
    loop._roi_refresh_resolution = 320
    loop._roi_refresh_lk_quality_enabled = False
    loop._roi_refresh_expand_ratio = 1.8
    loop._engine = engine

    action = RuntimeAction(
        mode="test",
        input_resolution=480,
        inference_interval=1,
        cpu_threads=1,
    )
    report = LKTrackingReport(
        reason="lk_tracking_quality_degraded",
        failure_ratio=0.10,
        track_count_after=2,
        failed_boxes_frame=[np.asarray([100.0, 100.0, 150.0, 150.0], dtype=np.float32)],
    )

    result = RuntimeLoop._try_roi_refresh(
        loop,
        np.zeros((480, 640, 3), dtype=np.uint8),
        action,
        report,
        [],
    )

    assert result is None
    assert engine.calls == 0


def test_lk_quality_roi_rejects_nonlocal_failures() -> None:
    loop = RuntimeLoop.__new__(RuntimeLoop)
    loop._roi_refresh_min_survivors = 1
    loop._roi_refresh_lk_max_failed_boxes = 1
    loop._roi_refresh_lk_max_failure_ratio = 0.40

    too_many_boxes = LKTrackingReport(
        reason="lk_tracking_quality_degraded",
        failure_ratio=0.35,
        track_count_after=2,
        failed_boxes_frame=[
            np.asarray([10.0, 10.0, 20.0, 20.0], dtype=np.float32),
            np.asarray([40.0, 40.0, 50.0, 50.0], dtype=np.float32),
        ],
    )
    too_high_ratio = LKTrackingReport(
        reason="lk_tracking_quality_degraded",
        failure_ratio=0.50,
        track_count_after=1,
        failed_boxes_frame=[np.asarray([10.0, 10.0, 20.0, 20.0], dtype=np.float32)],
    )
    local_failure = LKTrackingReport(
        reason="lk_tracking_quality_degraded",
        failure_ratio=0.35,
        track_count_after=2,
        failed_boxes_frame=[np.asarray([10.0, 10.0, 20.0, 20.0], dtype=np.float32)],
    )

    assert not RuntimeLoop._lk_quality_roi_allowed(loop, too_many_boxes)
    assert not RuntimeLoop._lk_quality_roi_allowed(loop, too_high_ratio)
    assert RuntimeLoop._lk_quality_roi_allowed(loop, local_failure)


def test_lk_quality_confirmation_defers_one_soft_failure() -> None:
    loop = RuntimeLoop.__new__(RuntimeLoop)
    loop._lk_quality_confirm_enabled = True
    loop._lk_quality_confirm_frames = 1
    loop._lk_quality_confirm_max_failure_ratio = 0.60
    loop._lk_quality_confirm_min_survivors = 1
    loop._lk_quality_confirm_count = 0
    loop._lk_quality_confirm_total_deferred = 0

    first = LKTrackingReport(
        reason="lk_tracking_quality_degraded",
        should_refresh=True,
        failure_ratio=0.35,
        track_count_after=2,
    )
    second = LKTrackingReport(
        reason="lk_tracking_quality_degraded",
        should_refresh=True,
        failure_ratio=0.35,
        track_count_after=2,
    )

    RuntimeLoop._apply_lk_quality_confirmation(loop, first)
    RuntimeLoop._apply_lk_quality_confirmation(loop, second)

    assert first.reason == "lk_quality_degraded_confirming"
    assert first.should_refresh is False
    assert first.lk_quality_confirm_deferred is True
    assert first.lk_quality_confirm_count == 1
    assert second.reason == "lk_tracking_quality_degraded"
    assert second.should_refresh is True
    assert second.lk_quality_confirm_count == 2
    assert second.lk_quality_confirm_total_deferred == 1


def test_lk_quality_confirmation_does_not_defer_collapse() -> None:
    loop = RuntimeLoop.__new__(RuntimeLoop)
    loop._lk_quality_confirm_enabled = True
    loop._lk_quality_confirm_frames = 1
    loop._lk_quality_confirm_max_failure_ratio = 0.60
    loop._lk_quality_confirm_min_survivors = 1
    loop._lk_quality_confirm_count = 0
    loop._lk_quality_confirm_total_deferred = 0
    report = LKTrackingReport(
        reason="lk_tracking_quality_degraded",
        should_refresh=True,
        failure_ratio=1.0,
        track_count_after=0,
    )

    RuntimeLoop._apply_lk_quality_confirmation(loop, report)

    assert report.reason == "lk_tracking_quality_degraded"
    assert report.should_refresh is True
    assert report.lk_quality_confirm_deferred is False


def test_lk_quality_roi_rejects_large_roi_before_inference() -> None:
    class CountingEngine:
        last_profile = {"infer_total_ms": 12.0}
        last_resolved_input_resolution = 320

        def __init__(self) -> None:
            self.calls = 0

        def infer(self, _frame, _action):
            self.calls += 1
            return []

    engine = CountingEngine()
    loop = RuntimeLoop.__new__(RuntimeLoop)
    loop._roi_refresh_enabled = True
    loop._last_detection_resolution = 480
    loop._roi_refresh_resolution = 320
    loop._roi_refresh_min_survivors = 1
    loop._roi_refresh_lk_quality_enabled = True
    loop._roi_refresh_lk_max_failed_boxes = 1
    loop._roi_refresh_lk_max_failure_ratio = 0.40
    loop._roi_refresh_lk_max_area_ratio = 0.05
    loop._roi_refresh_expand_ratio = 1.8
    loop._engine = engine

    action = RuntimeAction(
        mode="test",
        input_resolution=480,
        inference_interval=1,
        cpu_threads=1,
    )
    report = LKTrackingReport(
        reason="lk_tracking_quality_degraded",
        failure_ratio=0.35,
        track_count_after=2,
        failed_boxes_frame=[np.asarray([100.0, 100.0, 300.0, 300.0], dtype=np.float32)],
    )

    result = RuntimeLoop._try_roi_refresh(
        loop,
        np.zeros((480, 640, 3), dtype=np.uint8),
        action,
        report,
        [],
    )

    assert result is None
    assert engine.calls == 0


def test_motion_roi_rejects_large_area_and_records_metadata() -> None:
    class CountingEngine:
        last_profile = {"infer_total_ms": 12.0}
        last_resolved_input_resolution = 320

        def __init__(self) -> None:
            self.calls = 0

        def infer(self, _frame, _action):
            self.calls += 1
            return []

    engine = CountingEngine()
    loop = RuntimeLoop.__new__(RuntimeLoop)
    loop._roi_refresh_enabled = True
    loop._last_detection_resolution = 480
    loop._roi_refresh_resolution = 320
    loop._roi_refresh_max_area_ratio = 0.25
    loop._roi_refresh_lk_max_area_ratio = 0.18
    loop._roi_refresh_expand_ratio = 1.8
    loop._engine = engine

    action = RuntimeAction(
        mode="test",
        input_resolution=480,
        inference_interval=1,
        cpu_threads=1,
    )
    report = LKTrackingReport(
        reason="unexplained_motion_outside_tracks",
        refresh_boxes_frame=[
            np.asarray([120.0, 80.0, 500.0, 420.0], dtype=np.float32)
        ],
    )

    result = RuntimeLoop._try_roi_refresh(
        loop,
        np.zeros((480, 640, 3), dtype=np.uint8),
        action,
        report,
        [],
    )

    assert result is None
    assert engine.calls == 0
    assert report.roi_refresh_candidate
    assert not report.roi_refresh_applied
    assert report.roi_refresh_reason == "unexplained_motion_outside_tracks"
    assert report.roi_refresh_reject_reason == "area_too_large"
    assert report.roi_refresh_area_ratio > report.roi_refresh_max_area_ratio


def test_healthy_tracking_defers_safety_refresh_until_hard_limit() -> None:
    loop = RuntimeLoop.__new__(RuntimeLoop)
    loop._motion_gate = None
    loop._safety_refresh_frames = 300
    loop._safety_refresh_defer_when_healthy = True
    loop._safety_refresh_hard_limit_frames = 900
    loop._safety_refresh_healthy_max_failure_ratio = 0.05
    loop._safety_refresh_healthy_min_quality = 0.75
    loop._frame_id = 300
    loop._last_full_detector_frame = 0

    action = RuntimeAction(
        mode="test",
        input_resolution=480,
        inference_interval=1,
        cpu_threads=1,
    )
    report = LKTrackingReport(
        mode="track",
        reason="lk_track",
        failure_ratio=0.0,
        mean_quality=0.95,
        track_count_after=2,
        should_refresh=False,
    )

    RuntimeLoop._apply_event_refresh_gate(
        loop,
        np.zeros((480, 640, 3), dtype=np.uint8),
        action,
        report,
        [],
    )

    assert not report.should_refresh
    assert report.reason == "track_healthy_safety_refresh_deferred"


def test_safety_refresh_hard_limit_overrides_healthy_deferral() -> None:
    loop = RuntimeLoop.__new__(RuntimeLoop)
    loop._motion_gate = None
    loop._safety_refresh_frames = 300
    loop._safety_refresh_defer_when_healthy = True
    loop._safety_refresh_hard_limit_frames = 900
    loop._safety_refresh_healthy_max_failure_ratio = 0.05
    loop._safety_refresh_healthy_min_quality = 0.75
    loop._frame_id = 900
    loop._last_full_detector_frame = 0

    action = RuntimeAction(
        mode="test",
        input_resolution=480,
        inference_interval=1,
        cpu_threads=1,
    )
    report = LKTrackingReport(
        mode="track",
        reason="lk_track",
        failure_ratio=0.0,
        mean_quality=0.95,
        track_count_after=2,
        should_refresh=False,
    )

    RuntimeLoop._apply_event_refresh_gate(
        loop,
        np.zeros((480, 640, 3), dtype=np.uint8),
        action,
        report,
        [],
    )

    assert report.should_refresh
    assert report.reason == "long_interval_safety_refresh_hard_limit"


def test_roi_slow_fuse_requires_consecutive_slow_refreshes() -> None:
    loop = RuntimeLoop.__new__(RuntimeLoop)
    loop._roi_slow_fuse_enabled = True
    loop._roi_slow_fuse_threshold_ms = 8000.0
    loop._roi_slow_fuse_consecutive_limit = 2
    loop._roi_slow_fuse_cooldown_frames = 45
    loop._roi_slow_fuse_until_frame = -1
    loop._roi_slow_fuse_consecutive_count = 0
    loop._frame_id = 100

    RuntimeLoop._update_roi_slow_fuse(loop, {"onnx_run_ms": 8100.0})

    assert loop._roi_slow_fuse_consecutive_count == 1
    assert loop._roi_slow_fuse_until_frame == -1

    loop._frame_id = 110
    RuntimeLoop._update_roi_slow_fuse(loop, {"onnx_run_ms": 8200.0})

    assert loop._roi_slow_fuse_consecutive_count == 0
    assert loop._roi_slow_fuse_until_frame == 155


def test_fast_roi_resets_slow_fuse_count() -> None:
    loop = RuntimeLoop.__new__(RuntimeLoop)
    loop._roi_slow_fuse_enabled = True
    loop._roi_slow_fuse_threshold_ms = 8000.0
    loop._roi_slow_fuse_consecutive_limit = 2
    loop._roi_slow_fuse_cooldown_frames = 45
    loop._roi_slow_fuse_until_frame = -1
    loop._roi_slow_fuse_consecutive_count = 0
    loop._frame_id = 100

    RuntimeLoop._update_roi_slow_fuse(loop, {"onnx_run_ms": 8100.0})
    RuntimeLoop._update_roi_slow_fuse(loop, {"onnx_run_ms": 900.0})
    loop._frame_id = 110
    RuntimeLoop._update_roi_slow_fuse(loop, {"onnx_run_ms": 8200.0})

    assert loop._roi_slow_fuse_consecutive_count == 1
    assert loop._roi_slow_fuse_until_frame == -1
