"""Video and camera frame sources."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


class FrameSource:
    """
    Iterable frame source from video file, camera index, or synthetic dry-run.

    Parameters
    ----------
    video:
        Path to video file, camera index as int string, or None for synthetic.
    synthetic:
        Generate blank frames when no video (for dry-run without sample video).
    max_frames:
        Optional cap on frames for short tests.
    """

    def __init__(
        self,
        video: str | Path | int | None = None,
        *,
        synthetic: bool = False,
        synthetic_size: tuple[int, int] = (640, 480),
        max_frames: int | None = None,
    ) -> None:
        self._video = video
        self._synthetic = synthetic
        self._synthetic_size = synthetic_size
        self._max_frames = max_frames
        self._cap: cv2.VideoCapture | None = None
        self._count = 0

    def _open(self) -> None:
        if self._synthetic or self._video is None:
            return
        if isinstance(self._video, int):
            self._cap = cv2.VideoCapture(self._video)
        else:
            path = Path(self._video)
            if path.exists():
                self._cap = cv2.VideoCapture(str(path))
            else:
                # Fall back to synthetic if file missing in dry-run dev
                self._synthetic = True

    def __iter__(self) -> Iterator[np.ndarray]:
        self._open()
        while True:
            if self._max_frames is not None and self._count >= self._max_frames:
                break
            if self._synthetic or self._cap is None:
                frame = self._synthetic_frame()
            else:
                ok, frame = self._cap.read()
                if not ok:
                    break
            yield frame
            self._count += 1

    def _synthetic_frame(self) -> np.ndarray:
        w, h = self._synthetic_size
        # Slight variation so visual features are non-trivial
        base = (self._count * 3) % 255
        frame = np.full((h, w, 3), base, dtype=np.uint8)
        cv2.rectangle(
            frame,
            (50 + (self._count % 100), 50),
            (200, 200),
            (255 - base, 128, 64),
            2,
        )
        return frame

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
