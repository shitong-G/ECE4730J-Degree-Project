"""Tests for RT-DETR ONNX post-processing."""

import numpy as np

from scene_runtime.inference.postprocess import postprocess_rtdetr_outputs


def test_postprocess_rtdetr_outputs_filters_by_score() -> None:
    labels = np.array([[1, 2, 3]], dtype=np.int64)
    boxes = np.array([[[10, 20, 30, 40], [0, 0, 5, 5], [50, 50, 60, 60]]], dtype=np.float32)
    scores = np.array([[0.9, 0.2, 0.7]], dtype=np.float32)

    detections = postprocess_rtdetr_outputs(
        [labels, boxes, scores],
        score_threshold=0.5,
    )

    assert len(detections) == 2
    assert detections[0].class_id == 1
    assert detections[0].score == 0.9
    assert detections[0].bbox == (10.0, 20.0, 30.0, 40.0)
    assert detections[1].class_id == 3
