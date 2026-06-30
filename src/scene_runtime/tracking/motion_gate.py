"""Residual-motion gate for event-triggered detector refresh."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np


@dataclass
class MotionGateReport:
    """Scene-change/new-motion report used to decide detector refresh."""

    global_motion_ok: bool = True
    global_inliers: int = 0
    residual_ratio: float = 0.0
    outside_ratio: float = 0.0
    largest_outside_component: int = 0
    should_refresh: bool = False
    reason: str = "no_gate_trigger"


class ResidualMotionGate:
    """Detect unexplained motion outside tracked boxes."""

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
        self.gate_width = int(gate_width)
        self.pixel_threshold = int(pixel_threshold)
        self.outside_ratio_threshold = float(outside_ratio_threshold)
        self.min_component_area = int(min_component_area)
        self.scene_change_ratio_threshold = float(scene_change_ratio_threshold)
        self.mask_expand_ratio = float(mask_expand_ratio)
        self.enable_camera_compensation = bool(enable_camera_compensation)

    def analyze(
        self,
        previous_frame: np.ndarray | None,
        current_frame: np.ndarray,
        previous_boxes: Sequence[np.ndarray],
        current_boxes: Sequence[np.ndarray],
    ) -> MotionGateReport:
        if previous_frame is None:
            return MotionGateReport(reason="no_previous_frame")

        height, width = current_frame.shape[:2]
        if width <= 0 or height <= 0:
            return MotionGateReport(reason="invalid_frame")

        small_height = max(1, round(height * self.gate_width / width))
        previous_small = cv2.resize(previous_frame, (self.gate_width, small_height))
        current_small = cv2.resize(current_frame, (self.gate_width, small_height))
        previous_gray = cv2.cvtColor(previous_small, cv2.COLOR_BGR2GRAY)
        current_gray = cv2.cvtColor(current_small, cv2.COLOR_BGR2GRAY)

        sx, sy = self.gate_width / width, small_height / height
        old_small_boxes = [_scale_bbox(box, sx, sy) for box in previous_boxes]
        new_small_boxes = [_scale_bbox(box, sx, sy) for box in current_boxes]
        affine, motion_ok, inliers = self._global_affine(
            previous_gray,
            current_gray,
            old_small_boxes,
        )

        aligned_previous = cv2.warpAffine(
            previous_gray,
            affine,
            (self.gate_width, small_height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        difference = cv2.absdiff(aligned_previous, current_gray)
        difference = cv2.GaussianBlur(difference, (5, 5), 0)
        _, residual = cv2.threshold(
            difference,
            self.pixel_threshold,
            255,
            cv2.THRESH_BINARY,
        )
        residual = cv2.morphologyEx(
            residual,
            cv2.MORPH_OPEN,
            np.ones((3, 3), dtype=np.uint8),
        )
        residual = cv2.dilate(residual, np.ones((3, 3), dtype=np.uint8), iterations=1)

        transformed_old = [_transform_bbox(box, affine) for box in old_small_boxes]
        explained = _box_mask(
            (small_height, self.gate_width),
            [*transformed_old, *new_small_boxes],
            self.mask_expand_ratio,
        )
        outside = cv2.bitwise_and(residual, cv2.bitwise_not(explained))

        residual_ratio = float(np.count_nonzero(residual) / max(1, residual.size))
        outside_ratio = float(np.count_nonzero(outside) / max(1, outside.size))
        num_components, _, stats, _ = cv2.connectedComponentsWithStats(outside)
        largest = int(stats[1:, cv2.CC_STAT_AREA].max()) if num_components > 1 else 0

        should_refresh = False
        reason = "no_gate_trigger"
        if not motion_ok and residual_ratio >= self.scene_change_ratio_threshold:
            should_refresh = True
            reason = "scene_change_or_camera_motion"
        elif (
            outside_ratio >= self.outside_ratio_threshold
            and largest >= self.min_component_area
        ):
            should_refresh = True
            reason = "unexplained_motion_outside_tracks"

        return MotionGateReport(
            global_motion_ok=motion_ok,
            global_inliers=inliers,
            residual_ratio=residual_ratio,
            outside_ratio=outside_ratio,
            largest_outside_component=largest,
            should_refresh=should_refresh,
            reason=reason,
        )

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
        covered = _box_mask((height, width), previous_boxes, self.mask_expand_ratio)
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
            previous_gray,
            current_gray,
            points,
            None,
            winSize=(21, 21),
            maxLevel=3,
        )
        if next_points is None or forward_status is None:
            return identity, False, 0

        backward_points, backward_status, _ = cv2.calcOpticalFlowPyrLK(
            current_gray,
            previous_gray,
            next_points,
            None,
            winSize=(21, 21),
            maxLevel=3,
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
            src[valid],
            dst[valid],
            method=cv2.RANSAC,
            ransacReprojThreshold=3.0,
        )
        if affine is None or inliers is None:
            return identity, False, 0
        inlier_count = int(inliers.reshape(-1).sum())
        return affine.astype(np.float32), inlier_count >= 12, inlier_count


def _scale_bbox(bbox: np.ndarray, sx: float, sy: float) -> np.ndarray:
    return np.asarray(
        [bbox[0] * sx, bbox[1] * sy, bbox[2] * sx, bbox[3] * sy],
        dtype=np.float32,
    )


def _clip_bbox(bbox: np.ndarray, width: int, height: int) -> np.ndarray | None:
    x1, y1, x2, y2 = [float(value) for value in bbox]
    x1 = float(np.clip(x1, 0, width - 1))
    y1 = float(np.clip(y1, 0, height - 1))
    x2 = float(np.clip(x2, 0, width - 1))
    y2 = float(np.clip(y2, 0, height - 1))
    if x2 <= x1 or y2 <= y1:
        return None
    return np.asarray([x1, y1, x2, y2], dtype=np.float32)


def _box_mask(
    shape: tuple[int, int],
    boxes: Sequence[np.ndarray],
    expand_ratio: float,
) -> np.ndarray:
    height, width = shape
    mask = np.zeros((height, width), dtype=np.uint8)
    for bbox in boxes:
        clipped = _clip_bbox(_expand_bbox(bbox, expand_ratio), width, height)
        if clipped is None:
            continue
        x1, y1, x2, y2 = clipped.astype(int)
        if x2 > x1 and y2 > y1:
            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, thickness=-1)
    return mask


def _expand_bbox(bbox: np.ndarray, ratio: float) -> np.ndarray:
    x1, y1, x2, y2 = bbox.astype(np.float32)
    width = max(1.0, x2 - x1)
    height = max(1.0, y2 - y1)
    return np.asarray(
        [x1 - ratio * width, y1 - ratio * height, x2 + ratio * width, y2 + ratio * height],
        dtype=np.float32,
    )


def _transform_bbox(bbox: np.ndarray, affine: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = bbox.astype(np.float32)
    corners = np.asarray(
        [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
        dtype=np.float32,
    ).reshape(-1, 1, 2)
    transformed = cv2.transform(corners, affine).reshape(-1, 2)
    low = transformed.min(axis=0)
    high = transformed.max(axis=0)
    return np.asarray([low[0], low[1], high[0], high[1]], dtype=np.float32)
