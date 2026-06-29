"""JSONL logger for per-frame detection boxes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from scene_runtime.inference.postprocess import Detection


class DetectionLogger:
    """Append one JSON object per processed frame with current detections."""

    def __init__(self, path: Path | None) -> None:
        self._path = Path(path) if path is not None else None
        self._file = None

    @property
    def path(self) -> Path | None:
        return self._path

    def open(self) -> None:
        if self._path is None:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self._path.open("w", encoding="utf-8")

    def write(
        self,
        *,
        timestamp: float,
        frame_id: int,
        strategy: str,
        did_infer: bool,
        tracking_mode: str | None,
        tracking_reason: str | None,
        input_resolution: int,
        resolved_input_resolution: int | None,
        detections: list[Detection],
    ) -> None:
        if self._file is None:
            return
        row: dict[str, Any] = {
            "timestamp": timestamp,
            "frame_id": frame_id,
            "strategy": strategy,
            "did_infer": did_infer,
            "tracking_mode": tracking_mode,
            "tracking_reason": tracking_reason,
            "input_resolution": input_resolution,
            "resolved_input_resolution": resolved_input_resolution,
            "detections": [
                {
                    "class_id": det.class_id,
                    "score": det.score,
                    "bbox": list(det.bbox),
                }
                for det in detections
            ],
        }
        self._file.write(json.dumps(row, separators=(",", ":")) + "\n")
        self._file.flush()

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None
