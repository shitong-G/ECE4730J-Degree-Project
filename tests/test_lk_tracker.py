"""Tests for SparseLKBoxTracker."""

from __future__ import annotations

import cv2
import numpy as np

from scene_runtime.inference.postprocess import Detection
from scene_runtime.tracking.lk_tracker import SparseLKBoxTracker


def _synthetic_frame() -> np.ndarray:
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    cv2.rectangle(frame, (20, 25), (55, 65), (220, 220, 220), thickness=-1)
    cv2.circle(frame, (33, 38), 5, (40, 40, 40), thickness=-1)
    cv2.line(frame, (23, 60), (52, 30), (30, 30, 30), thickness=2)
    cv2.rectangle(frame, (90, 45), (130, 88), (180, 180, 180), thickness=-1)
    cv2.circle(frame, (112, 66), 7, (20, 20, 20), thickness=-1)
    cv2.line(frame, (92, 84), (128, 49), (50, 50, 50), thickness=2)
    return frame


def test_sparse_lk_tracker_batches_multiple_tracks() -> None:
    frame0 = _synthetic_frame()
    matrix = np.float32([[1, 0, 4], [0, 1, 3]])
    frame1 = cv2.warpAffine(frame0, matrix, (frame0.shape[1], frame0.shape[0]))
    detections = [
        Detection(class_id=1, score=0.91, bbox=(20.0, 25.0, 55.0, 65.0)),
        Detection(class_id=2, score=0.82, bbox=(90.0, 45.0, 130.0, 88.0)),
    ]
    tracker = SparseLKBoxTracker(
        max_corners=16,
        min_valid_points=4,
        redetect_interval=10,
        redetect_min_points=6,
        win_size=15,
        max_level=2,
        max_iterations=15,
    )

    reset_report = tracker.reset(frame0, detections, input_resolution=160)
    tracked, report = tracker.update(frame1)

    assert reset_report.track_count_after == 2
    assert report.reason == "lk_track"
    assert report.track_count_before == 2
    assert report.track_count_after == 2
    assert len(tracked) == 2
    for original, updated in zip(detections, tracked):
        x1, y1, x2, y2 = updated.bbox
        ox1, oy1, ox2, oy2 = original.bbox
        assert abs((x1 - ox1) - 4.0) < 1.5
        assert abs((y1 - oy1) - 3.0) < 1.5
        assert abs((x2 - ox2) - 4.0) < 1.5
        assert abs((y2 - oy2) - 3.0) < 1.5
