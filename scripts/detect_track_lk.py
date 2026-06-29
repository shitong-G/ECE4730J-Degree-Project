#!/usr/bin/env python3
"""Event-triggered RT-DETR + LK optical-flow tracking.

Put this file under <project-root>/tools/detect_track_lk.py and run it from the
project root. RT-DETR is treated as a black-box ONNX keyframe detector:

  first frame / event trigger -> RT-DETR detect -> (re)initialize LK tracks
  otherwise                   -> LK tracks existing boxes only

A non-learning residual-motion gate triggers a new RT-DETR call when a large
motion region appears outside every known tracked box. All annotated frames are
saved as JPGs; no MP4 is written.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence
import math

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scene_runtime.controller.actions import RuntimeAction
from scene_runtime.inference.onnx_engine import ONNXRTDETREngine


COCO80 = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]


@dataclass
class Detection:
    """bbox uses original video coordinates: [x1, y1, x2, y2]."""
    bbox: np.ndarray
    class_id: int
    score: float
    track_id: int = -1


@dataclass
class Track:
    track_id: int
    bbox: np.ndarray
    class_id: int
    score: float
    points: Optional[np.ndarray]
    age: int = 0
    quality: float = 1.0

    def to_detection(self) -> Detection:
        return Detection(self.bbox.copy(), self.class_id, self.score, self.track_id)


@dataclass
class TrackerReport:
    failure_ratio: float
    mean_quality: float
    failed_ids: list[int]
    before_count: int
    after_count: int


@dataclass
class GateReport:
    global_motion_ok: bool
    global_inliers: int
    residual_ratio: float
    outside_ratio: float
    largest_outside_component: int


def label_of(class_id: int) -> str:
    return COCO80[class_id] if 0 <= class_id < len(COCO80) else str(class_id)


def color_of(class_id: int) -> tuple[int, int, int]:
    palette = [
        (50, 220, 120), (80, 170, 255), (240, 180, 70), (220, 90, 90),
        (180, 120, 255), (70, 220, 220), (230, 120, 200), (120, 220, 80),
    ]
    return palette[class_id % len(palette)]


def clip_bbox(bbox: np.ndarray, width: int, height: int) -> Optional[np.ndarray]:
    x1, y1, x2, y2 = [float(value) for value in bbox]
    x1 = np.clip(x1, 0, width - 1)
    y1 = np.clip(y1, 0, height - 1)
    x2 = np.clip(x2, 0, width - 1)
    y2 = np.clip(y2, 0, height - 1)
    if x2 <= x1 or y2 <= y1:
        return None
    return np.asarray([x1, y1, x2, y2], dtype=np.float32)


def bbox_iou(a: np.ndarray, b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return inter / max(1e-6, area_a + area_b - inter)


def expand_bbox(bbox: np.ndarray, ratio: float, width: int, height: int) -> np.ndarray:
    x1, y1, x2, y2 = bbox.astype(np.float32)
    bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
    expanded = np.asarray(
        [x1 - ratio * bw, y1 - ratio * bh, x2 + ratio * bw, y2 + ratio * bh],
        dtype=np.float32,
    )
    clipped = clip_bbox(expanded, width, height)
    return clipped if clipped is not None else np.zeros(4, dtype=np.float32)


def transform_bbox(bbox: np.ndarray, affine: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = bbox.astype(np.float32)
    corners = np.asarray(
        [[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32
    ).reshape(-1, 1, 2)
    transformed = cv2.transform(corners, affine).reshape(-1, 2)
    low, high = transformed.min(axis=0), transformed.max(axis=0)
    return np.asarray([low[0], low[1], high[0], high[1]], dtype=np.float32)


def box_mask(shape: tuple[int, int], boxes: Iterable[np.ndarray], expand_ratio: float) -> np.ndarray:
    height, width = shape
    mask = np.zeros((height, width), dtype=np.uint8)
    for bbox in boxes:
        x1, y1, x2, y2 = expand_bbox(bbox, expand_ratio, width, height).astype(int)
        if x2 > x1 and y2 > y1:
            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, thickness=-1)
    return mask


class LKMultiObjectTracker:
    """
    Improved sparse Lucas-Kanade tracker.

    Default behavior:
    - Select Shi-Tomasi feature points uniformly in a 3x3 grid inside each bbox.
    - Use forward-backward LK validation.
    - Reject motion outliers with median + MAD.
    - Move the whole bbox by one robust translation vector.

    This avoids the original affine-warp problem where one side of a person box
    moves while the other side remains nearly fixed.
    """

    def __init__(
            self,
            *,
            max_corners: int = 54,
            min_valid_points: int = 6,
            min_survival_ratio: float = 0.35,
            max_fb_error: float = 1.5,
            max_failure_ratio: float = 0.30,
            refresh_iou: float = 0.30,
            min_box_side: int = 5,
            grid_size: int = 3,
    ) -> None:
        self.max_corners = max_corners
        self.min_valid_points = min_valid_points
        self.min_survival_ratio = min_survival_ratio
        self.max_fb_error = max_fb_error
        self.max_failure_ratio = max_failure_ratio
        self.refresh_iou = refresh_iou
        self.min_box_side = min_box_side
        self.grid_size = grid_size

        self.previous_gray: Optional[np.ndarray] = None
        self.tracks: list[Track] = []
        self.next_track_id = 0

    @staticmethod
    def _gray(frame: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def _points_for_box(
            self,
            gray: np.ndarray,
            bbox: np.ndarray,
    ) -> Optional[np.ndarray]:
        """
        Pick corners in a grid rather than from the entire bbox at once.

        This prevents all anchors from clustering on one textured area such as
        a face, shirt logo, arm, or one side of a walking person.
        """
        height, width = gray.shape
        clipped = clip_bbox(bbox, width, height)
        if clipped is None:
            return None

        x1, y1, x2, y2 = clipped.astype(int)
        if x2 - x1 < 3 or y2 - y1 < 3:
            return None

        per_cell = max(
            1,
            math.ceil(self.max_corners / (self.grid_size * self.grid_size)),
        )

        x_edges = np.linspace(
            x1, x2 + 1, self.grid_size + 1, dtype=int
        )
        y_edges = np.linspace(
            y1, y2 + 1, self.grid_size + 1, dtype=int
        )

        collected: list[np.ndarray] = []

        for row in range(self.grid_size):
            for col in range(self.grid_size):
                cell_x1 = int(x_edges[col])
                cell_x2 = int(x_edges[col + 1])
                cell_y1 = int(y_edges[row])
                cell_y2 = int(y_edges[row + 1])

                if cell_x2 - cell_x1 < 3 or cell_y2 - cell_y1 < 3:
                    continue

                mask = np.zeros_like(gray, dtype=np.uint8)
                cv2.rectangle(
                    mask,
                    (cell_x1, cell_y1),
                    (cell_x2 - 1, cell_y2 - 1),
                    255,
                    thickness=-1,
                )

                points = cv2.goodFeaturesToTrack(
                    gray,
                    maxCorners=per_cell,
                    qualityLevel=0.01,
                    minDistance=4,
                    mask=mask,
                    blockSize=7,
                    useHarrisDetector=False,
                )

                if points is not None:
                    collected.append(points)

        if collected:
            points = np.concatenate(collected, axis=0).astype(np.float32)

            unique_xy = np.unique(
                np.round(points.reshape(-1, 2), decimals=2),
                axis=0,
            )
            points = unique_xy.reshape(-1, 1, 2).astype(np.float32)

            if len(points) >= self.min_valid_points:
                return points[: self.max_corners]

        # Fallback for texture-poor objects.
        full_mask = np.zeros_like(gray, dtype=np.uint8)
        cv2.rectangle(full_mask, (x1, y1), (x2, y2), 255, thickness=-1)

        return cv2.goodFeaturesToTrack(
            gray,
            maxCorners=self.max_corners,
            qualityLevel=0.01,
            minDistance=5,
            mask=full_mask,
            blockSize=7,
            useHarrisDetector=False,
        )

    def _preserved_ids(
            self,
            detections: Sequence[Detection],
            old_tracks: Sequence[Track],
    ) -> dict[int, int]:
        candidates: list[tuple[float, int, int]] = []

        for detection_index, detection in enumerate(detections):
            for track_index, track in enumerate(old_tracks):
                if detection.class_id != track.class_id:
                    continue

                overlap = bbox_iou(detection.bbox, track.bbox)

                if overlap >= self.refresh_iou:
                    candidates.append(
                        (overlap, detection_index, track_index)
                    )

        candidates.sort(reverse=True)

        used_detections: set[int] = set()
        used_tracks: set[int] = set()
        matches: dict[int, int] = {}

        for _, detection_index, track_index in candidates:
            if detection_index in used_detections:
                continue
            if track_index in used_tracks:
                continue

            matches[detection_index] = old_tracks[track_index].track_id
            used_detections.add(detection_index)
            used_tracks.add(track_index)

        return matches

    def reset(
            self,
            frame: np.ndarray,
            detections: Sequence[Detection],
            preserve_ids_from: Sequence[Track] = (),
    ) -> list[Detection]:
        """Reinitialize tracks from a fresh RT-DETR detection frame."""
        gray = self._gray(frame)

        matched_ids = self._preserved_ids(
            detections,
            preserve_ids_from,
        )

        fresh_tracks: list[Track] = []
        output: list[Detection] = []

        for index, detection in enumerate(detections):
            bbox = clip_bbox(
                detection.bbox,
                gray.shape[1],
                gray.shape[0],
            )

            if bbox is None:
                continue

            track_id = matched_ids.get(index)

            if track_id is None:
                track_id = self.next_track_id
                self.next_track_id += 1

            track = Track(
                track_id=track_id,
                bbox=bbox,
                class_id=detection.class_id,
                score=detection.score,
                points=self._points_for_box(gray, bbox),
            )

            fresh_tracks.append(track)
            output.append(track.to_detection())

        self.previous_gray = gray
        self.tracks = fresh_tracks

        return output

    def _robust_translation(
            self,
            old_xy: np.ndarray,
            new_xy: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Estimate one robust translation for the whole bbox.

        Median displacement handles a minority of unreliable optical-flow points.
        MAD removes points moving inconsistently with the majority.
        """
        displacement = new_xy - old_xy

        median_shift = np.median(displacement, axis=0)

        point_residuals = np.linalg.norm(
            displacement - median_shift[None, :],
            axis=1,
            )

        mad = float(np.median(point_residuals))

        # Avoid threshold collapsing to zero for almost-static frames.
        threshold = max(1.0, 2.5 * 1.4826 * mad)

        inliers = point_residuals <= threshold

        if int(inliers.sum()) >= self.min_valid_points:
            median_shift = np.median(
                displacement[inliers],
                axis=0,
            )

        return median_shift.astype(np.float32), inliers

    def _update_track(
            self,
            track: Track,
            current_gray: np.ndarray,
    ) -> tuple[Optional[Track], float]:
        if (
                self.previous_gray is None
                or track.points is None
                or len(track.points) == 0
        ):
            return None, 0.0

        next_points, status_forward, _ = cv2.calcOpticalFlowPyrLK(
            self.previous_gray,
            current_gray,
            track.points,
            None,
            winSize=(21, 21),
            maxLevel=3,
            criteria=(
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                30,
                0.01,
            ),
        )

        if next_points is None or status_forward is None:
            return None, 0.0

        backward_points, status_backward, _ = cv2.calcOpticalFlowPyrLK(
            current_gray,
            self.previous_gray,
            next_points,
            None,
            winSize=(21, 21),
            maxLevel=3,
            criteria=(
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                30,
                0.01,
            ),
        )

        if backward_points is None or status_backward is None:
            return None, 0.0

        old_xy = track.points.reshape(-1, 2)
        new_xy = next_points.reshape(-1, 2)
        backward_xy = backward_points.reshape(-1, 2)

        height, width = current_gray.shape

        basic_valid = (
                status_forward.reshape(-1).astype(bool)
                & status_backward.reshape(-1).astype(bool)
                & (
                        np.linalg.norm(old_xy - backward_xy, axis=1)
                        <= self.max_fb_error
                )
                & (new_xy[:, 0] >= 0)
                & (new_xy[:, 0] < width)
                & (new_xy[:, 1] >= 0)
                & (new_xy[:, 1] < height)
        )

        basic_count = int(basic_valid.sum())
        basic_quality = basic_count / max(1, len(track.points))

        if (
                basic_count < self.min_valid_points
                or basic_quality < self.min_survival_ratio
        ):
            return None, basic_quality

        old_valid = old_xy[basic_valid]
        new_valid = new_xy[basic_valid]

        shift, robust_inliers = self._robust_translation(
            old_valid,
            new_valid,
        )

        robust_count = int(robust_inliers.sum())
        quality = robust_count / max(1, len(track.points))

        if (
                robust_count < self.min_valid_points
                or quality < self.min_survival_ratio
        ):
            return None, quality

        old_inliers = old_valid[robust_inliers]
        new_inliers = new_valid[robust_inliers]

        # Important: move all four bbox edges by exactly the same shift.
        # Do NOT use affine to directly warp bbox corners.
        updated_bbox = track.bbox + np.asarray(
            [shift[0], shift[1], shift[0], shift[1]],
            dtype=np.float32,
        )

        updated_bbox = clip_bbox(updated_bbox, width, height)

        if updated_bbox is None:
            return None, quality

        if updated_bbox[2] - updated_bbox[0] < self.min_box_side:
            return None, quality

        if updated_bbox[3] - updated_bbox[1] < self.min_box_side:
            return None, quality

        refreshed_points = self._points_for_box(
            current_gray,
            updated_bbox,
        )

        if (
                refreshed_points is None
                or len(refreshed_points) < self.min_valid_points
        ):
            refreshed_points = new_inliers.reshape(-1, 1, 2).astype(
                np.float32
            )

        return Track(
            track_id=track.track_id,
            bbox=updated_bbox,
            class_id=track.class_id,
            score=track.score,
            points=refreshed_points,
            age=track.age + 1,
            quality=quality,
        ), quality

    def update(self, frame: np.ndarray) -> TrackerReport:
        current_gray = self._gray(frame)
        before = len(self.tracks)

        if self.previous_gray is None:
            self.previous_gray = current_gray
            return TrackerReport(1.0, 0.0, [], before, before)

        if before == 0:
            self.previous_gray = current_gray
            return TrackerReport(0.0, 1.0, [], 0, 0)

        survivors: list[Track] = []
        failed_ids: list[int] = []
        qualities: list[float] = []

        for track in self.tracks:
            updated, quality = self._update_track(track, current_gray)

            qualities.append(quality)

            if updated is None:
                failed_ids.append(track.track_id)
            else:
                survivors.append(updated)

        self.previous_gray = current_gray
        self.tracks = survivors

        return TrackerReport(
            failure_ratio=len(failed_ids) / max(1, before),
            mean_quality=float(np.mean(qualities)) if qualities else 0.0,
            failed_ids=failed_ids,
            before_count=before,
            after_count=len(survivors),
        )

    def needs_refresh(self, report: TrackerReport) -> bool:
        return report.failure_ratio > self.max_failure_ratio

    def detections(self) -> list[Detection]:
        return [track.to_detection() for track in self.tracks]


class ResidualMotionGate:
    """Whole-frame, non-learning new-object/scene-change gate."""

    def __init__(
        self,
        *,
        gate_width: int = 320,
        pixel_threshold: int = 24,
        outside_ratio_threshold: float = 0.010,
        min_component_area: int = 120,
        scene_change_ratio_threshold: float = 0.35,
        mask_expand_ratio: float = 0.28,
        enable_camera_compensation: bool = True,
    ) -> None:
        self.gate_width = gate_width
        self.pixel_threshold = pixel_threshold
        self.outside_ratio_threshold = outside_ratio_threshold
        self.min_component_area = min_component_area
        self.scene_change_ratio_threshold = scene_change_ratio_threshold
        self.mask_expand_ratio = mask_expand_ratio
        self.enable_camera_compensation = enable_camera_compensation

    @staticmethod
    def _scale_bbox(bbox: np.ndarray, sx: float, sy: float) -> np.ndarray:
        return np.asarray([bbox[0] * sx, bbox[1] * sy, bbox[2] * sx, bbox[3] * sy], dtype=np.float32)

    def _global_affine(
        self,
        previous_gray: np.ndarray,
        current_gray: np.ndarray,
        previous_boxes: Sequence[np.ndarray],
    ) -> tuple[np.ndarray, bool, int]:
        height, width = previous_gray.shape
        identity = np.asarray([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
        if not self.enable_camera_compensation:
            return identity, True, 0

        background_mask = np.full((height, width), 255, dtype=np.uint8)
        covered = box_mask((height, width), previous_boxes, self.mask_expand_ratio)
        background_mask[covered > 0] = 0
        border = max(4, min(height, width) // 80)
        background_mask[:border] = 0
        background_mask[-border:] = 0
        background_mask[:, :border] = 0
        background_mask[:, -border:] = 0

        points = cv2.goodFeaturesToTrack(
            previous_gray,
            maxCorners=350,
            qualityLevel=0.01,
            minDistance=7,
            mask=background_mask,
            blockSize=7,
        )
        if points is None or len(points) < 12:
            return identity, False, 0

        next_points, forward_status, _ = cv2.calcOpticalFlowPyrLK(
            previous_gray, current_gray, points, None, winSize=(21, 21), maxLevel=3
        )
        if next_points is None or forward_status is None:
            return identity, False, 0
        backward_points, backward_status, _ = cv2.calcOpticalFlowPyrLK(
            current_gray, previous_gray, next_points, None, winSize=(21, 21), maxLevel=3
        )
        if backward_points is None or backward_status is None:
            return identity, False, 0

        src = points.reshape(-1, 2)
        dst = next_points.reshape(-1, 2)
        back = backward_points.reshape(-1, 2)
        valid = (
            forward_status.reshape(-1).astype(bool)
            & backward_status.reshape(-1).astype(bool)
            & (np.linalg.norm(src - back, axis=1) <= 2.0)
        )
        if int(valid.sum()) < 12:
            return identity, False, int(valid.sum())

        affine, inliers = cv2.estimateAffinePartial2D(
            src[valid], dst[valid], method=cv2.RANSAC, ransacReprojThreshold=3.0
        )
        if affine is None or inliers is None:
            return identity, False, 0
        inlier_count = int(inliers.reshape(-1).sum())
        return affine.astype(np.float32), inlier_count >= 12, inlier_count

    def analyze(
        self,
        previous_frame: np.ndarray,
        current_frame: np.ndarray,
        previous_boxes: Sequence[np.ndarray],
        current_boxes: Sequence[np.ndarray],
    ) -> GateReport:
        height, width = current_frame.shape[:2]
        small_height = max(1, round(height * self.gate_width / width))
        previous_small = cv2.resize(previous_frame, (self.gate_width, small_height))
        current_small = cv2.resize(current_frame, (self.gate_width, small_height))
        previous_gray = cv2.cvtColor(previous_small, cv2.COLOR_BGR2GRAY)
        current_gray = cv2.cvtColor(current_small, cv2.COLOR_BGR2GRAY)

        sx, sy = self.gate_width / width, small_height / height
        old_small_boxes = [self._scale_bbox(box, sx, sy) for box in previous_boxes]
        new_small_boxes = [self._scale_bbox(box, sx, sy) for box in current_boxes]
        affine, motion_ok, inliers = self._global_affine(previous_gray, current_gray, old_small_boxes)

        aligned_previous = cv2.warpAffine(
            previous_gray,
            affine,
            (self.gate_width, small_height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        difference = cv2.absdiff(aligned_previous, current_gray)
        difference = cv2.GaussianBlur(difference, (5, 5), 0)
        _, residual = cv2.threshold(difference, self.pixel_threshold, 255, cv2.THRESH_BINARY)
        residual = cv2.morphologyEx(residual, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8))
        residual = cv2.dilate(residual, np.ones((3, 3), dtype=np.uint8), iterations=1)

        transformed_old_boxes = [transform_bbox(box, affine) for box in old_small_boxes]
        explained = box_mask(
            (small_height, self.gate_width),
            list(transformed_old_boxes) + list(new_small_boxes),
            self.mask_expand_ratio,
        )
        outside = cv2.bitwise_and(residual, cv2.bitwise_not(explained))

        residual_ratio = np.count_nonzero(residual) / max(1, residual.size)
        outside_ratio = np.count_nonzero(outside) / max(1, outside.size)
        num_components, _, stats, _ = cv2.connectedComponentsWithStats(outside)
        largest = int(stats[1:, cv2.CC_STAT_AREA].max()) if num_components > 1 else 0

        return GateReport(
            global_motion_ok=motion_ok,
            global_inliers=inliers,
            residual_ratio=float(residual_ratio),
            outside_ratio=float(outside_ratio),
            largest_outside_component=largest,
        )

    def needs_refresh(self, report: GateReport) -> tuple[bool, str]:
        if not report.global_motion_ok and report.residual_ratio >= self.scene_change_ratio_threshold:
            return True, "scene_change_or_camera_motion"
        if (
            report.outside_ratio >= self.outside_ratio_threshold
            and report.largest_outside_component >= self.min_component_area
        ):
            return True, "unexplained_motion_outside_tracks"
        return False, "no_gate_trigger"


class ONNXKeyframeDetector:
    """Adapter around the existing project ONNXRTDETREngine."""

    def __init__(
        self,
        *,
        model_path: Path,
        model_template: Optional[str],
        resolution: int,
        threads: int,
        score_threshold: float,
        allowed_classes: Optional[set[int]],
    ) -> None:
        model_paths = None
        if model_template:
            model_paths = {resolution: model_template.format(resolution=resolution)}
        self.engine = ONNXRTDETREngine(
            model_path=str(model_path),
            model_paths_by_resolution=model_paths,
            enable_thread_sessions=True,
            thread_session_counts=[threads],
        )
        self.engine.load()
        self.action = RuntimeAction(
            mode="visualize",
            input_resolution=resolution,
            inference_interval=1,
            cpu_threads=threads,
            governor=None,
        )
        self.resolution = resolution
        self.score_threshold = score_threshold
        self.allowed_classes = allowed_classes

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """This is the only function in the script that invokes RT-DETR."""
        raw_detections = self.engine.infer(frame, self.action)
        resolved = self.engine.last_resolved_input_resolution or self.resolution
        height, width = frame.shape[:2]
        sx, sy = width / float(resolved), height / float(resolved)
        detections: list[Detection] = []
        for raw in raw_detections:
            if raw.score < self.score_threshold:
                continue
            if self.allowed_classes is not None and raw.class_id not in self.allowed_classes:
                continue
            x1, y1, x2, y2 = raw.bbox
            bbox = clip_bbox(
                np.asarray([x1 * sx, y1 * sy, x2 * sx, y2 * sy], dtype=np.float32),
                width,
                height,
            )
            if bbox is not None:
                detections.append(Detection(bbox, int(raw.class_id), float(raw.score)))
        return detections


class DetectTrackController:
    """DETECT -> TRACK -> DETECT state controller."""

    def __init__(
        self,
        detector: ONNXKeyframeDetector,
        tracker: LKMultiObjectTracker,
        gate: ResidualMotionGate,
        safety_refresh_frames: int,
    ) -> None:
        self.detector = detector
        self.tracker = tracker
        self.gate = gate
        self.safety_refresh_frames = safety_refresh_frames
        self.previous_frame: Optional[np.ndarray] = None
        self.previous_boxes: list[np.ndarray] = []
        self.last_detect_frame = -10**9
        self.detector_calls = 0

    def _detect_and_reset(self, frame: np.ndarray, frame_id: int, old_tracks: Sequence[Track]) -> tuple[list[Detection], float]:
        start = time.perf_counter()
        detections = self.detector.detect(frame)
        detector_latency_ms = (time.perf_counter() - start) * 1000.0
        output = self.tracker.reset(frame, detections, preserve_ids_from=old_tracks)
        self.previous_boxes = [track.bbox.copy() for track in self.tracker.tracks]
        self.last_detect_frame = frame_id
        self.detector_calls += 1
        return output, detector_latency_ms

    def process(self, frame: np.ndarray, frame_id: int) -> tuple[list[Detection], str, str, TrackerReport, GateReport, float]:
        empty_tracker = TrackerReport(0.0, 1.0, [], 0, 0)
        empty_gate = GateReport(True, 0, 0.0, 0.0, 0)

        if self.previous_frame is None:
            output, det_ms = self._detect_and_reset(frame, frame_id, [])
            self.previous_frame = frame.copy()
            return output, "DETECT", "first_frame", empty_tracker, empty_gate, det_ms

        old_tracks = list(self.tracker.tracks)
        tracker_report = self.tracker.update(frame)
        current_boxes = [track.bbox.copy() for track in self.tracker.tracks]
        gate_report = self.gate.analyze(self.previous_frame, frame, self.previous_boxes, current_boxes)

        refresh = False
        reason = "track_healthy"
        if self.tracker.needs_refresh(tracker_report):
            refresh, reason = True, "lk_tracking_quality_degraded"
        else:
            gate_refresh, gate_reason = self.gate.needs_refresh(gate_report)
            if gate_refresh:
                refresh, reason = True, gate_reason
            elif self.safety_refresh_frames > 0 and frame_id - self.last_detect_frame >= self.safety_refresh_frames:
                refresh, reason = True, "long_interval_safety_refresh"

        detector_latency_ms = 0.0
        if refresh:
            output, detector_latency_ms = self._detect_and_reset(frame, frame_id, old_tracks)
            mode = "DETECT"
        else:
            output = self.tracker.detections()
            self.previous_boxes = current_boxes
            mode = "TRACK"

        self.previous_frame = frame.copy()
        return output, mode, reason, tracker_report, gate_report, detector_latency_ms


def draw_visualization(
    frame: np.ndarray,
    detections: Sequence[Detection],
    *,
    frame_id: int,
    mode: str,
    reason: str,
    detector_calls: int,
    total_latency_ms: float,
    detector_latency_ms: float,
    tracker_report: TrackerReport,
    gate_report: GateReport,
) -> np.ndarray:
    output = frame.copy()
    for detection in detections:
        x1, y1, x2, y2 = detection.bbox.astype(int)
        color = color_of(detection.class_id)
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        source = "D" if mode == "DETECT" else "T"
        text = f"{source} id={detection.track_id} {label_of(detection.class_id)} {detection.score:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
        label_y = max(0, y1 - th - 7)
        cv2.rectangle(output, (x1, label_y), (x1 + tw + 6, label_y + th + 6), color, -1)
        cv2.putText(output, text, (x1 + 3, label_y + th + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (10, 10, 10), 1, cv2.LINE_AA)

    lines = [
        f"frame={frame_id} mode={mode} boxes={len(detections)} detector_calls={detector_calls}",
        f"reason={reason}",
        f"total={total_latency_ms:.1f} ms  RT-DETR={detector_latency_ms:.1f} ms  LKq={tracker_report.mean_quality:.2f} fail={tracker_report.failure_ratio:.2f}",
        f"outside={gate_report.outside_ratio:.3f} residual={gate_report.residual_ratio:.3f} component={gate_report.largest_outside_component} inliers={gate_report.global_inliers}",
    ]
    for row, line in enumerate(lines):
        y = 25 + row * 22
        cv2.putText(output, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.53, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(output, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.53, (20, 20, 20), 1, cv2.LINE_AA)
    return output


def save_contact_sheet(frames: Sequence[tuple[int, np.ndarray]], path: Path, columns: int = 4, thumb_width: int = 360) -> None:
    if not frames:
        return
    thumbnails: list[np.ndarray] = []
    for frame_id, image in frames:
        height, width = image.shape[:2]
        thumb_height = max(1, round(height * thumb_width / width))
        thumb = cv2.resize(image, (thumb_width, thumb_height))
        cv2.putText(thumb, f"frame {frame_id}", (10, thumb_height - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(thumb, f"frame {frame_id}", (10, thumb_height - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (255, 255, 0), 1, cv2.LINE_AA)
        thumbnails.append(thumb)

    cell_height = max(image.shape[0] for image in thumbnails)
    rows = (len(thumbnails) + columns - 1) // columns
    sheet = np.full((rows * cell_height, columns * thumb_width, 3), 230, dtype=np.uint8)
    for index, thumb in enumerate(thumbnails):
        row, column = divmod(index, columns)
        y, x = row * cell_height, column * thumb_width
        sheet[y:y + thumb.shape[0], x:x + thumb.shape[1]] = thumb
    cv2.imwrite(str(path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])


def parse_classes(text: str) -> Optional[set[int]]:
    if not text.strip():
        return None
    mapping = {name: index for index, name in enumerate(COCO80)}
    selected: set[int] = set()
    for token in text.split(","):
        token = token.strip().lower()
        if not token:
            continue
        if token.isdigit():
            class_id = int(token)
        elif token in mapping:
            class_id = mapping[token]
        else:
            raise argparse.ArgumentTypeError(f"Unknown COCO class: {token}")
        if not 0 <= class_id < len(COCO80):
            raise argparse.ArgumentTypeError(f"Invalid COCO class ID: {class_id}")
        selected.add(class_id)
    return selected or None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Event-triggered RT-DETR + LK tracking; outputs JPG frames.")
    parser.add_argument(
        "--mode",
        choices=["detect_track", "detect_only"],
        default="detect_track",
        help="detect_track uses event-triggered RT-DETR + LK; detect_only runs RT-DETR on every processed frame.",
    )
    parser.add_argument("--video", type=Path, default=ROOT / "data" / "sample.mp4")
    parser.add_argument("--model", type=Path, default=ROOT / "models" / "rtdetr_r18_lite_pi4_640.onnx")
    parser.add_argument("--model-template", default=None, help="Optional model template containing {resolution}.")
    parser.add_argument("--resolution", type=int, default=640)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--score-threshold", type=float, default=0.5)
    parser.add_argument("--classes", type=parse_classes, default=None, help="Optional: person,car,bus or 0,2,5")

    parser.add_argument("--output-dir", type=Path, default=ROOT / "experiments" / "visualizations" / "detect_track")
    parser.add_argument("--jpg-quality", type=int, default=92)
    parser.add_argument("--max-frames", type=int, default=300)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--safety-refresh-frames", type=int, default=300, help="0 disables fallback refresh.")

    parser.add_argument("--max-track-failure-ratio", type=float, default=0.30)
    parser.add_argument("--min-valid-points", type=int, default=6)
    parser.add_argument("--refresh-iou", type=float, default=0.30)

    parser.add_argument("--gate-width", type=int, default=320)
    parser.add_argument("--motion-threshold", type=int, default=24)
    parser.add_argument("--outside-ratio-threshold", type=float, default=0.010)
    parser.add_argument("--min-component-area", type=int, default=120)
    parser.add_argument("--scene-change-ratio-threshold", type=float, default=0.35)
    parser.add_argument("--mask-expand-ratio", type=float, default=0.28)
    parser.add_argument("--disable-camera-compensation", action="store_true")

    parser.add_argument("--contact-sheet-stride", type=int, default=30, help="0 disables contact sheet.")
    parser.add_argument("--contact-sheet-max", type=int, default=80)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.frame_stride < 1:
        raise ValueError("--frame-stride must be >= 1")
    if args.jpg_quality < 1 or args.jpg_quality > 100:
        raise ValueError("--jpg-quality must be between 1 and 100")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = args.output_dir / "frames"
    frames_dir.mkdir(exist_ok=True)

    detector = ONNXKeyframeDetector(
        model_path=args.model,
        model_template=args.model_template,
        resolution=args.resolution,
        threads=args.threads,
        score_threshold=args.score_threshold,
        allowed_classes=args.classes,
    )
    controller: DetectTrackController | None = None
    if args.mode == "detect_track":
        tracker = LKMultiObjectTracker(
            min_valid_points=args.min_valid_points,
            max_failure_ratio=args.max_track_failure_ratio,
            refresh_iou=args.refresh_iou,
        )
        gate = ResidualMotionGate(
            gate_width=args.gate_width,
            pixel_threshold=args.motion_threshold,
            outside_ratio_threshold=args.outside_ratio_threshold,
            min_component_area=args.min_component_area,
            scene_change_ratio_threshold=args.scene_change_ratio_threshold,
            mask_expand_ratio=args.mask_expand_ratio,
            enable_camera_compensation=not args.disable_camera_compensation,
        )
        controller = DetectTrackController(detector, tracker, gate, args.safety_refresh_frames)

    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open video: {args.video}")

    events_path = args.output_dir / "events.csv"
    contact_frames: list[tuple[int, np.ndarray]] = []
    source_frame_id = 0
    processed = 0
    detector_calls = 0
    total_latency_ms = 0.0
    total_detector_latency_ms = 0.0

    with events_path.open("w", newline="", encoding="utf-8") as event_file:
        writer = csv.DictWriter(event_file, fieldnames=[
            "source_frame", "saved_frame", "mode", "reason", "objects", "detector_calls",
            "total_latency_ms", "detector_latency_ms", "tracker_failure_ratio",
            "tracker_mean_quality", "global_motion_ok", "global_inliers", "residual_ratio",
            "outside_ratio", "largest_outside_component",
        ])
        writer.writeheader()

        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                if source_frame_id % args.frame_stride != 0:
                    source_frame_id += 1
                    continue

                if args.mode == "detect_only":
                    start = time.perf_counter()
                    detector_start = time.perf_counter()
                    detections = detector.detect(frame)
                    detector_ms = (time.perf_counter() - detector_start) * 1000.0
                    total_ms = (time.perf_counter() - start) * 1000.0
                    mode = "DETECT"
                    reason = "detect_only"
                    tracker_report = TrackerReport(0.0, 1.0, [], 0, 0)
                    gate_report = GateReport(True, 0, 0.0, 0.0, 0)
                    detector_call_count = detector_calls + 1
                else:
                    if controller is None:
                        raise RuntimeError("detect_track controller was not initialized")
                    start = time.perf_counter()
                    detections, mode, reason, tracker_report, gate_report, detector_ms = controller.process(frame, source_frame_id)
                    total_ms = (time.perf_counter() - start) * 1000.0
                    detector_call_count = controller.detector_calls
                total_latency_ms += total_ms
                if mode == "DETECT":
                    detector_calls += 1
                    total_detector_latency_ms += detector_ms
                    print(f"[frame {source_frame_id:06d}] DETECT reason={reason}; objects={len(detections)}; RT-DETR={detector_ms:.1f} ms")

                visualized = draw_visualization(
                    frame, detections,
                    frame_id=source_frame_id,
                    mode=mode,
                    reason=reason,
                    detector_calls=detector_call_count,
                    total_latency_ms=total_ms,
                    detector_latency_ms=detector_ms,
                    tracker_report=tracker_report,
                    gate_report=gate_report,
                )
                frame_path = frames_dir / f"frame_{source_frame_id:06d}.jpg"
                if not cv2.imwrite(str(frame_path), visualized, [cv2.IMWRITE_JPEG_QUALITY, args.jpg_quality]):
                    raise RuntimeError(f"Could not write {frame_path}")

                writer.writerow({
                    "source_frame": source_frame_id,
                    "saved_frame": frame_path.name,
                    "mode": mode,
                    "reason": reason,
                    "objects": len(detections),
                    "detector_calls": detector_call_count,
                    "total_latency_ms": f"{total_ms:.3f}",
                    "detector_latency_ms": f"{detector_ms:.3f}",
                    "tracker_failure_ratio": f"{tracker_report.failure_ratio:.6f}",
                    "tracker_mean_quality": f"{tracker_report.mean_quality:.6f}",
                    "global_motion_ok": int(gate_report.global_motion_ok),
                    "global_inliers": gate_report.global_inliers,
                    "residual_ratio": f"{gate_report.residual_ratio:.6f}",
                    "outside_ratio": f"{gate_report.outside_ratio:.6f}",
                    "largest_outside_component": gate_report.largest_outside_component,
                })

                if (
                    args.contact_sheet_stride > 0
                    and processed % args.contact_sheet_stride == 0
                    and len(contact_frames) < args.contact_sheet_max
                ):
                    contact_frames.append((source_frame_id, visualized.copy()))

                processed += 1
                source_frame_id += 1
                if args.max_frames > 0 and processed >= args.max_frames:
                    break
        finally:
            capture.release()

    if args.contact_sheet_stride > 0:
        save_contact_sheet(contact_frames, args.output_dir / "contact_sheet.jpg")

    average_total = total_latency_ms / max(1, processed)
    average_detect = total_detector_latency_ms / max(1, detector_calls)
    summary_path = args.output_dir / "summary.txt"
    summary_path.write_text(
        "\n".join([
            f"video: {args.video}",
            f"model: {args.model}",
            f"mode: {args.mode}",
            f"processed_visualized_frames: {processed}",
            f"detector_calls: {detector_calls}",
            f"detector_invocation_rate: {detector_calls / max(1, processed):.6f}",
            f"average_total_latency_ms: {average_total:.3f}",
            f"average_detect_frame_latency_ms: {average_detect:.3f}",
            f"frames_dir: {frames_dir}",
            f"events_csv: {events_path}",
        ]) + "\n",
        encoding="utf-8",
    )

    print("\nFinished.")
    print(f"Visualized JPG frames: {frames_dir}")
    print(f"Contact sheet:          {args.output_dir / 'contact_sheet.jpg'}")
    print(f"Events CSV:             {events_path}")
    print(f"Summary:                {summary_path}")
    print(f"RT-DETR invocation:     {detector_calls}/{max(1, processed)} ({detector_calls / max(1, processed):.2%})")
    print(f"Average latency:        {average_total:.2f} ms / processed frame")


if __name__ == "__main__":
    main()
