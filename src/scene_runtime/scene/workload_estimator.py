"""Scene workload estimation from visual and detection-history signals."""

from __future__ import annotations

from typing import Any

import numpy as np

from scene_runtime.scene.detection_history import DetectionHistory
from scene_runtime.scene.visual_features import (
    edge_density,
    frame_difference,
    image_entropy,
    motion_intensity,
)


class SceneWorkloadEstimator:
    """
    Estimates whether the current scene workload is light, medium, or heavy.

    Uses OpenCV/NumPy visual features and optional detection history.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        cfg = (config or {}).get("scene", {})
        self._light_edge_max = float(cfg.get("light_edge_density_max", 0.08))
        self._light_motion_max = float(cfg.get("light_motion_max", 0.05))
        self._heavy_edge_min = float(cfg.get("heavy_edge_density_min", 0.18))
        self._heavy_motion_min = float(cfg.get("heavy_motion_min", 0.15))
        self._heavy_det_min = int(cfg.get("heavy_detection_count_min", 8))

    def extract_features(
        self,
        frame: np.ndarray,
        prev_frame: np.ndarray | None = None,
        detection_history: DetectionHistory | None = None,
    ) -> dict[str, Any]:
        """
        Extract visual and history-based features for workload classification.

        Returns
        -------
        dict
            Feature keys used by ``classify_workload`` and ``update``.
        """
        hist = detection_history.summary() if detection_history else {}
        features: dict[str, Any] = {
            "frame_diff": frame_difference(frame, prev_frame),
            "motion_intensity": motion_intensity(frame, prev_frame),
            "edge_density": edge_density(frame),
            "entropy": image_entropy(frame),
            "prev_detection_count": int(hist.get("prev_detection_count", 0)),
            "confidence_mean": float(hist.get("confidence_mean", 0.0)),
            "confidence_std": float(hist.get("confidence_std", 0.0)),
            "prev_inference_latency_ms": float(
                hist.get("prev_inference_latency_ms", 0.0)
            ),
        }
        return features

    def classify_workload(self, features: dict[str, Any]) -> str:
        """Classify scene workload as ``light``, ``medium``, or ``heavy``."""
        edge = float(features.get("edge_density", 0.0))
        motion = float(features.get("motion_intensity", 0.0))
        detections = int(features.get("prev_detection_count", 0))

        if (
            edge >= self._heavy_edge_min
            or motion >= self._heavy_motion_min
            or detections >= self._heavy_det_min
        ):
            return "heavy"
        if edge <= self._light_edge_max and motion <= self._light_motion_max:
            return "light"
        return "medium"

    def update(
        self,
        frame: np.ndarray,
        prev_frame: np.ndarray | None = None,
        detection_history: DetectionHistory | None = None,
    ) -> dict[str, Any]:
        """
        Extract features, classify workload, and return full state dict.

        Returns
        -------
        dict
            Includes ``workload`` plus all feature fields.
        """
        features = self.extract_features(frame, prev_frame, detection_history)
        workload = self.classify_workload(features)
        return {
            "workload": workload,
            "frame_diff": features["frame_diff"],
            "motion_intensity": features["motion_intensity"],
            "edge_density": features["edge_density"],
            "entropy": features["entropy"],
            "prev_detection_count": features["prev_detection_count"],
            "confidence_mean": features["confidence_mean"],
            "confidence_std": features["confidence_std"],
        }
