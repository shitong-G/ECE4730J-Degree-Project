"""Low-cost visual feature extraction using OpenCV and NumPy."""

from __future__ import annotations

import cv2
import numpy as np


def frame_difference(frame: np.ndarray, prev_frame: np.ndarray | None) -> float:
    """
    Mean absolute difference between consecutive grayscale frames.

    Returns 0.0 when ``prev_frame`` is None.
    """
    if prev_frame is None:
        return 0.0
    gray_a = _to_gray(frame)
    gray_b = _to_gray(prev_frame)
    if gray_a.shape != gray_b.shape:
        gray_b = cv2.resize(gray_b, (gray_a.shape[1], gray_a.shape[0]))
    diff = cv2.absdiff(gray_a, gray_b)
    return float(np.mean(diff) / 255.0)


def motion_intensity(frame: np.ndarray, prev_frame: np.ndarray | None) -> float:
    """
    Motion intensity from frame differencing thresholded at 25.

    Normalized to [0, 1] as fraction of pixels with motion.
    """
    if prev_frame is None:
        return 0.0
    gray_a = _to_gray(frame)
    gray_b = _to_gray(prev_frame)
    if gray_a.shape != gray_b.shape:
        gray_b = cv2.resize(gray_b, (gray_a.shape[1], gray_a.shape[0]))
    diff = cv2.absdiff(gray_a, gray_b)
    _, mask = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
    return float(np.count_nonzero(mask) / mask.size)


def edge_density(frame: np.ndarray) -> float:
    """Canny edge pixel density normalized to [0, 1]."""
    gray = _to_gray(frame)
    edges = cv2.Canny(gray, 50, 150)
    return float(np.count_nonzero(edges) / edges.size)


def image_entropy(frame: np.ndarray) -> float:
    """Shannon entropy of grayscale histogram, normalized to [0, 1]."""
    gray = _to_gray(frame)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    hist = hist / (hist.sum() + 1e-8)
    nonzero = hist[hist > 0]
    entropy = -float(np.sum(nonzero * np.log2(nonzero)))
    # Max entropy for 256 bins is log2(256) = 8
    return entropy / 8.0


def _to_gray(frame: np.ndarray) -> np.ndarray:
    if len(frame.shape) == 2:
        return frame
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
