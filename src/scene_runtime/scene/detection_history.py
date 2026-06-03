"""Rolling statistics from recent detection results."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class DetectionHistory:
    """Maintains recent detection counts, confidence stats, and latency."""

    max_entries: int = 30
    counts: list[int] = field(default_factory=list)
    confidences: list[float] = field(default_factory=list)
    latencies_ms: list[float] = field(default_factory=list)

    def push(
        self,
        detection_count: int,
        confidences: list[float],
        latency_ms: float,
    ) -> None:
        """Append one inference cycle to history."""
        self.counts.append(detection_count)
        self.confidences.extend(confidences)
        self.latencies_ms.append(latency_ms)
        if len(self.counts) > self.max_entries:
            self.counts.pop(0)
        if len(self.latencies_ms) > self.max_entries:
            self.latencies_ms.pop(0)
        # Trim confidence list coarsely
        max_conf = self.max_entries * 20
        if len(self.confidences) > max_conf:
            self.confidences = self.confidences[-max_conf:]

    def summary(self) -> dict[str, Any]:
        """Return aggregate stats for scene workload estimation."""
        import numpy as np

        prev_count = self.counts[-1] if self.counts else 0
        if self.confidences:
            conf_arr = np.array(self.confidences, dtype=np.float32)
            conf_mean = float(np.mean(conf_arr))
            conf_std = float(np.std(conf_arr))
        else:
            conf_mean = 0.0
            conf_std = 0.0
        prev_latency = self.latencies_ms[-1] if self.latencies_ms else 0.0
        return {
            "prev_detection_count": prev_count,
            "confidence_mean": conf_mean,
            "confidence_std": conf_std,
            "prev_inference_latency_ms": prev_latency,
        }
