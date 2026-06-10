"""Detection post-processing for RT-DETR ONNX outputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class Detection:
    """Single object detection result."""

    class_id: int
    score: float
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2


def postprocess_rtdetr_outputs(
    raw_outputs: list[np.ndarray],
    score_threshold: float = 0.5,
) -> list[Detection]:
    """
    Convert RT-DETR ONNX raw tensors to ``Detection`` list.

    Official export order: ``labels``, ``boxes``, ``scores`` with batch dimension.
    """
    if len(raw_outputs) < 3:
        return []

    labels = np.asarray(raw_outputs[0])[0]
    boxes = np.asarray(raw_outputs[1])[0]
    scores = np.asarray(raw_outputs[2])[0]

    detections: list[Detection] = []
    for label, box, score in zip(labels, boxes, scores):
        if float(score) < score_threshold:
            continue
        x1, y1, x2, y2 = (float(v) for v in box[:4])
        detections.append(
            Detection(
                class_id=int(label),
                score=float(score),
                bbox=(x1, y1, x2, y2),
            )
        )
    return detections


def detections_summary(detections: list[Detection]) -> dict[str, Any]:
    """Compute count and confidence statistics for logging."""
    if not detections:
        return {"detection_count": 0, "confidence_mean": 0.0, "confidence_std": 0.0}
    scores = [d.score for d in detections]
    arr = np.array(scores, dtype=np.float32)
    return {
        "detection_count": len(detections),
        "confidence_mean": float(np.mean(arr)),
        "confidence_std": float(np.std(arr)),
    }
