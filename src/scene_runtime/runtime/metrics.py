"""Rolling FPS and latency metrics."""

from __future__ import annotations

import time
from collections import deque


class MetricsTracker:
    """Track frame timing, inference latency, and FPS."""

    def __init__(self, window: int = 30) -> None:
        self._frame_times: deque[float] = deque(maxlen=window)
        self._latencies_ms: deque[float] = deque(maxlen=window)
        self._inference_times: deque[float] = deque(maxlen=window)
        self._last_frame_time: float | None = None

    def mark_frame(self) -> None:
        now = time.perf_counter()
        if self._last_frame_time is not None:
            self._frame_times.append(now - self._last_frame_time)
        self._last_frame_time = now

    def record_latency(self, latency_ms: float) -> None:
        self._latencies_ms.append(latency_ms)

    def record_inference(self) -> None:
        """Record that one real inference was executed at the current time."""
        self._inference_times.append(time.perf_counter())

    @property
    def fps(self) -> float:
        if not self._frame_times:
            return 0.0
        avg_dt = sum(self._frame_times) / len(self._frame_times)
        return 1.0 / avg_dt if avg_dt > 0 else 0.0

    @property
    def avg_latency_ms(self) -> float:
        if not self._latencies_ms:
            return 0.0
        return sum(self._latencies_ms) / len(self._latencies_ms)

    @property
    def inference_fps(self) -> float:
        """Rolling actual inference FPS based on executed inference timestamps."""
        if len(self._inference_times) < 2:
            return 0.0
        elapsed = self._inference_times[-1] - self._inference_times[0]
        return (len(self._inference_times) - 1) / elapsed if elapsed > 0 else 0.0

    def snapshot(self) -> dict[str, float]:
        return {
            "fps": self.fps,
            "latency_ms": self.avg_latency_ms,
            "inference_fps": self.inference_fps,
        }
