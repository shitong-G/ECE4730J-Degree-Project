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

    TODO: align with actual exported ONNX output names and shapes from RT-DETR export.
    Expected placeholder layout: [boxes (N,4), scores (N,), labels (N,)].
    """
    if not raw_outputs:
        return []
    # TODO: implement real RT-DETR decode when model is available
    return []


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
