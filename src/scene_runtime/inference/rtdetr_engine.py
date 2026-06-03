"""Abstract inference engine interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

from scene_runtime.inference.postprocess import Detection

if TYPE_CHECKING:
    from scene_runtime.controller.actions import RuntimeAction


class BaseInferenceEngine(ABC):
    """Abstract object detection inference engine."""

    @abstractmethod
    def load(self) -> None:
        """Load model weights / ONNX session."""

    @abstractmethod
    def preprocess(self, frame: np.ndarray, input_resolution: int) -> np.ndarray:
        """Resize and normalize frame for model input."""

    @abstractmethod
    def postprocess(self, raw_outputs: list[np.ndarray]) -> list[Detection]:
        """Decode raw model outputs to detections."""

    @abstractmethod
    def infer(self, frame: np.ndarray, config: RuntimeAction) -> list[Detection]:
        """Run full inference pipeline under runtime configuration."""
