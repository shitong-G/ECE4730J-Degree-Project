"""Tests for scene visual feature extraction."""

from __future__ import annotations

import numpy as np

from scene_runtime.scene.visual_features import (
    edge_density,
    frame_difference,
    image_entropy,
    motion_intensity,
)
from scene_runtime.scene.workload_estimator import SceneWorkloadEstimator


def test_frame_difference_zero_without_prev() -> None:
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    assert frame_difference(frame, None) == 0.0


def test_frame_difference_nonzero_with_motion() -> None:
    a = np.zeros((64, 64, 3), dtype=np.uint8)
    b = a.copy()
    b[10:20, 10:20] = 255
    assert frame_difference(b, a) > 0.0


def test_edge_density_range() -> None:
    frame = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    d = edge_density(frame)
    assert 0.0 <= d <= 1.0


def test_entropy_range() -> None:
    frame = np.full((32, 32, 3), 128, dtype=np.uint8)
    e = image_entropy(frame)
    assert 0.0 <= e <= 1.0


def test_motion_intensity() -> None:
    a = np.zeros((48, 48, 3), dtype=np.uint8)
    b = np.zeros((48, 48, 3), dtype=np.uint8)
    b[0:24, :] = 255
    m = motion_intensity(b, a)
    assert m > 0.0


def test_workload_estimator_update() -> None:
    est = SceneWorkloadEstimator()
    frame = np.random.randint(0, 255, (80, 80, 3), dtype=np.uint8)
    out = est.update(frame)
    # Backbone: classifier stub always returns medium until Member 1 implements rules
    assert out["workload"] == "medium"
    assert "edge_density" in out
    assert "entropy" in out
