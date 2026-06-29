"""Sparse Lucas-Kanade box tracker for skipped detector frames."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from scene_runtime.inference.postprocess import Detection


@dataclass
class LKTrackingReport:
    """Quality report for one tracker update."""

    mode: str = "disabled"
    reason: str = "disabled"
    tracking_ms: float = 0.0
    failure_ratio: float = 0.0
    mean_quality: float = 1.0
    track_count_before: int = 0
    track_count_after: int = 0
    should_refresh: bool = False


@dataclass
class _Track:
    class_id: int
    score: float
    bbox_frame: np.ndarray
    points: np.ndarray | None


class SparseLKBoxTracker:
    """Track RT-DETR boxes in original frame coordinates between detector calls."""

    def __init__(
        self,
        *,
        max_corners: int = 40,
        min_valid_points: int = 5,
        min_survival_ratio: float = 0.35,
        max_forward_backward_error: float = 1.5,
        max_failure_ratio: float = 0.30,
        min_box_side: int = 5,
    ) -> None:
        self.max_corners = int(max_corners)
        self.min_valid_points = int(min_valid_points)
        self.min_survival_ratio = float(min_survival_ratio)
        self.max_forward_backward_error = float(max_forward_backward_error)
        self.max_failure_ratio = float(max_failure_ratio)
        self.min_box_side = int(min_box_side)
        self._previous_gray: np.ndarray | None = None
        self._tracks: list[_Track] = []
        self._last_input_resolution: int | None = None
        self._last_frame_shape: tuple[int, int] | None = None

    def reset(
        self,
        frame: np.ndarray,
        detections: list[Detection],
        input_resolution: int | None,
    ) -> LKTrackingReport:
        """Initialize tracks from fresh detector outputs."""
        gray = self._gray(frame)
        height, width = gray.shape
        resolution = int(input_resolution or max(height, width))
        self._last_input_resolution = resolution
        self._last_frame_shape = (height, width)

        tracks: list[_Track] = []
        for detection in detections:
            bbox = self._detection_to_frame_bbox(detection, width, height, resolution)
            if bbox is None:
                continue
            tracks.append(
                _Track(
                    class_id=detection.class_id,
                    score=detection.score,
                    bbox_frame=bbox,
                    points=self._points_for_box(gray, bbox),
                )
            )
        self._previous_gray = gray
        self._tracks = tracks
        return LKTrackingReport(
            mode="detect_reset",
            reason="detector_frame",
            track_count_before=len(detections),
            track_count_after=len(tracks),
        )

    def update(self, frame: np.ndarray) -> tuple[list[Detection], LKTrackingReport]:
        """Update tracks on a skipped detector frame."""
        current_gray = self._gray(frame)
        self._last_frame_shape = current_gray.shape
        before = len(self._tracks)
        if self._previous_gray is None:
            self._previous_gray = current_gray
            return [], LKTrackingReport(
                mode="track",
                reason="no_previous_frame",
                failure_ratio=1.0,
                mean_quality=0.0,
                track_count_before=before,
                track_count_after=0,
                should_refresh=True,
            )
        if before == 0:
            self._previous_gray = current_gray
            return [], LKTrackingReport(
                mode="track",
                reason="no_tracks",
                track_count_before=0,
                track_count_after=0,
                should_refresh=True,
            )

        survivors: list[_Track] = []
        qualities: list[float] = []
        for track in self._tracks:
            updated, quality = self._update_one(track, current_gray)
            qualities.append(quality)
            if updated is not None:
                survivors.append(updated)

        self._previous_gray = current_gray
        self._tracks = survivors
        failed = before - len(survivors)
        failure_ratio = failed / max(1, before)
        mean_quality = float(np.mean(qualities)) if qualities else 0.0
        should_refresh = failure_ratio > self.max_failure_ratio
        reason = "lk_quality_degraded" if should_refresh else "lk_track"
        detections = self._tracks_to_detections()
        return detections, LKTrackingReport(
            mode="track",
            reason=reason,
            failure_ratio=failure_ratio,
            mean_quality=mean_quality,
            track_count_before=before,
            track_count_after=len(survivors),
            should_refresh=should_refresh,
        )

    @property
    def last_input_resolution(self) -> int | None:
        return self._last_input_resolution

    @staticmethod
    def _gray(frame: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def _points_for_box(self, gray: np.ndarray, bbox: np.ndarray) -> np.ndarray | None:
        height, width = gray.shape
        clipped = _clip_bbox(bbox, width, height)
        if clipped is None:
            return None
        x1, y1, x2, y2 = clipped.astype(int)
        if x2 - x1 < self.min_box_side or y2 - y1 < self.min_box_side:
            return None
        mask = np.zeros_like(gray, dtype=np.uint8)
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, thickness=-1)
        return cv2.goodFeaturesToTrack(
            gray,
            maxCorners=self.max_corners,
            qualityLevel=0.01,
            minDistance=5,
            mask=mask,
            blockSize=7,
            useHarrisDetector=False,
        )

    def _update_one(
        self,
        track: _Track,
        current_gray: np.ndarray,
    ) -> tuple[_Track | None, float]:
        if self._previous_gray is None or track.points is None or len(track.points) == 0:
            return None, 0.0

        next_points, status_forward, _ = cv2.calcOpticalFlowPyrLK(
            self._previous_gray,
            current_gray,
            track.points,
            None,
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
        if next_points is None or status_forward is None:
            return None, 0.0

        backward_points, status_backward, _ = cv2.calcOpticalFlowPyrLK(
            current_gray,
            self._previous_gray,
            next_points,
            None,
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
        if backward_points is None or status_backward is None:
            return None, 0.0

        old_xy = track.points.reshape(-1, 2)
        new_xy = next_points.reshape(-1, 2)
        backward_xy = backward_points.reshape(-1, 2)
        height, width = current_gray.shape
        valid = (
            status_forward.reshape(-1).astype(bool)
            & status_backward.reshape(-1).astype(bool)
            & (np.linalg.norm(old_xy - backward_xy, axis=1) <= self.max_forward_backward_error)
            & (new_xy[:, 0] >= 0)
            & (new_xy[:, 0] < width)
            & (new_xy[:, 1] >= 0)
            & (new_xy[:, 1] < height)
        )
        valid_count = int(valid.sum())
        quality = valid_count / max(1, len(track.points))
        if valid_count < self.min_valid_points or quality < self.min_survival_ratio:
            return None, quality

        shift = np.median(new_xy[valid] - old_xy[valid], axis=0).astype(np.float32)
        bbox = track.bbox_frame + np.asarray(
            [shift[0], shift[1], shift[0], shift[1]],
            dtype=np.float32,
        )
        bbox = _clip_bbox(bbox, width, height)
        if bbox is None:
            return None, quality
        if bbox[2] - bbox[0] < self.min_box_side or bbox[3] - bbox[1] < self.min_box_side:
            return None, quality

        points = self._points_for_box(current_gray, bbox)
        if points is None or len(points) < self.min_valid_points:
            points = new_xy[valid].reshape(-1, 1, 2).astype(np.float32)
        return _Track(track.class_id, track.score, bbox, points), quality

    def _tracks_to_detections(self) -> list[Detection]:
        if self._last_input_resolution is None or self._last_frame_shape is None:
            return []
        height, width = self._last_frame_shape
        resolution = self._last_input_resolution
        detections: list[Detection] = []
        for track in self._tracks:
            x1, y1, x2, y2 = track.bbox_frame
            detections.append(
                Detection(
                    class_id=track.class_id,
                    score=track.score,
                    bbox=(
                        float(x1 / width * resolution),
                        float(y1 / height * resolution),
                        float(x2 / width * resolution),
                        float(y2 / height * resolution),
                    ),
                )
            )
        return detections

    @staticmethod
    def _detection_to_frame_bbox(
        detection: Detection,
        width: int,
        height: int,
        input_resolution: int,
    ) -> np.ndarray | None:
        x1, y1, x2, y2 = detection.bbox
        bbox = np.asarray(
            [
                x1 / input_resolution * width,
                y1 / input_resolution * height,
                x2 / input_resolution * width,
                y2 / input_resolution * height,
            ],
            dtype=np.float32,
        )
        return _clip_bbox(bbox, width, height)


def _clip_bbox(bbox: np.ndarray, width: int, height: int) -> np.ndarray | None:
    x1, y1, x2, y2 = [float(value) for value in bbox]
    x1 = float(np.clip(x1, 0, width - 1))
    y1 = float(np.clip(y1, 0, height - 1))
    x2 = float(np.clip(x2, 0, width - 1))
    y2 = float(np.clip(y2, 0, height - 1))
    if x2 <= x1 or y2 <= y1:
        return None
    return np.asarray([x1, y1, x2, y2], dtype=np.float32)
