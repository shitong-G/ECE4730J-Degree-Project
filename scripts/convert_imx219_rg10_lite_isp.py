#!/usr/bin/env python3
"""Convert IMX219 RG10 raw Bayer frames to displayable BGR/JPEG images."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def read_rg10_raw(
    path: str | Path,
    width: int,
    height: int,
    stride_pixels: int | None = None,
    verbose: bool = True,
) -> np.ndarray:
    """Read a uint16 RG10 raw file and return visible 10-bit pixels as float32."""
    raw_flat = np.fromfile(path, dtype=np.uint16)
    if raw_flat.size == 0:
        raise ValueError(f"Raw file is empty: {path}")

    if stride_pixels is None:
        if raw_flat.size % height != 0:
            raise ValueError(
                f"Cannot infer stride: raw values={raw_flat.size}, height={height}"
            )
        stride_pixels = raw_flat.size // height

    if stride_pixels < width:
        raise ValueError(f"stride_pixels={stride_pixels} < width={width}")

    expected_values = stride_pixels * height
    if raw_flat.size != expected_values:
        raise ValueError(
            f"Size mismatch: got {raw_flat.size} uint16 values, "
            f"expected {expected_values} = {stride_pixels}*{height}"
        )

    raw16 = raw_flat.reshape(height, stride_pixels)[:, :width]
    if verbose:
        print(f"width={width}, height={height}, stride_pixels={stride_pixels}")
        print(f"raw16 min={raw16.min()}, max={raw16.max()}")

    if raw16.max() > 4095:
        if verbose:
            print("Detected left-aligned 10-bit data, using raw16 >> 6")
        raw10 = raw16 >> 6
    else:
        if verbose:
            print("Detected low-bit 10-bit data, using raw16 directly")
        raw10 = raw16

    return np.clip(raw10, 0, 1023).astype(np.float32)


def apply_gamma(rgb01: np.ndarray, gamma: float) -> np.ndarray:
    rgb01 = np.clip(rgb01, 0.0, 1.0)
    return np.power(rgb01, 1.0 / gamma)


def boost_saturation(bgr8: np.ndarray, saturation: float) -> np.ndarray:
    hsv = cv2.cvtColor(bgr8, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] *= saturation
    hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def sharpen(bgr8: np.ndarray, amount: float) -> np.ndarray:
    if amount <= 0:
        return bgr8
    blur = cv2.GaussianBlur(bgr8, (0, 0), 1.0)
    return cv2.addWeighted(bgr8, 1.0 + amount, blur, -amount, 0)


def rg10_to_gray_bgr(
    input_path: str | Path,
    width: int,
    height: int,
    stride_pixels: int | None = None,
    black_level: float = 64.0,
    white_level: float = 1023.0,
    gamma: float = 1.0,
    verbose: bool = True,
) -> np.ndarray:
    """Convert RG10 raw to fast grayscale BGR for runtime inference/tracking."""
    raw10 = read_rg10_raw(input_path, width, height, stride_pixels, verbose=verbose)
    denom = max(1.0, float(white_level) - float(black_level))
    gray01 = np.clip((raw10 - float(black_level)) / denom, 0.0, 1.0)
    if gamma > 0 and gamma != 1.0:
        gray01 = apply_gamma(gray01, gamma)
    gray8 = (gray01 * 255.0).astype(np.uint8)
    return cv2.cvtColor(gray8, cv2.COLOR_GRAY2BGR)


def rg10_to_bgr(
    input_path: str | Path,
    width: int,
    height: int,
    stride_pixels: int | None = None,
    black_level: float = 64.0,
    r_gain: float = 1.2,
    g_gain: float = 0.8,
    b_gain: float = 1.2,
    low_percentile: float = 0.5,
    high_percentile: float = 99.5,
    gamma: float = 2.2,
    saturation: float = 1.8,
    sharpen_amount: float = 0.2,
    resize_original: bool = True,
    verbose: bool = True,
) -> np.ndarray:
    """Run a lightweight ISP for IMX219 SRGGB10_1X10 raw frames."""
    raw10 = read_rg10_raw(input_path, width, height, stride_pixels, verbose=verbose)
    raw10 = np.clip(raw10 - black_level, 0, 1023)

    # SRGGB10_1X10: row0 R G R G, row1 G B G B.
    r = raw10[0::2, 0::2] * r_gain
    g1 = raw10[0::2, 1::2]
    g2 = raw10[1::2, 0::2]
    b = raw10[1::2, 1::2] * b_gain
    g = 0.5 * (g1 + g2) * g_gain

    if verbose:
        print(f"B mean={b.mean():.2f}, G mean={g.mean():.2f}, R mean={r.mean():.2f}")

    bgr = np.stack([b, g, r], axis=-1)
    lo = np.percentile(bgr, low_percentile)
    hi = np.percentile(bgr, high_percentile)
    if verbose:
        print(f"normalize lo={lo:.2f}, hi={hi:.2f}")

    bgr01 = np.clip((bgr - lo) / (hi - lo + 1e-6), 0.0, 1.0)
    bgr01 = apply_gamma(bgr01, gamma)
    bgr8 = (bgr01 * 255.0).astype(np.uint8)
    bgr8 = boost_saturation(bgr8, saturation)
    bgr8 = sharpen(bgr8, sharpen_amount)

    if resize_original:
        bgr8 = cv2.resize(bgr8, (width, height), interpolation=cv2.INTER_CUBIC)

    return bgr8


def convert_file(
    input_path: str | Path,
    output_path: str | Path,
    width: int,
    height: int,
    stride_pixels: int | None = None,
    mode: str = "lite-isp",
    verbose: bool = True,
    **kwargs,
) -> np.ndarray:
    """Convert one RG10 raw file and save the JPEG output."""
    if mode == "gray":
        bgr = rg10_to_gray_bgr(
            input_path=input_path,
            width=width,
            height=height,
            stride_pixels=stride_pixels,
            black_level=float(kwargs.get("black_level", 64.0)),
            gamma=1.0,
            verbose=verbose,
        )
    else:
        bgr = rg10_to_bgr(
            input_path=input_path,
            width=width,
            height=height,
            stride_pixels=stride_pixels,
            verbose=verbose,
            **kwargs,
        )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), bgr):
        raise RuntimeError(f"Failed to save image: {output_path}")
    if verbose:
        print(f"Saved {output_path}")
    return bgr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument(
        "--mode",
        choices=["lite-isp", "gray"],
        default="lite-isp",
        help="lite-isp runs color tuning; gray skips color tuning for fast runtime input.",
    )
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--stride-pixels", type=int, default=None)
    parser.add_argument("--output", default="preview_lite_isp.jpg")
    parser.add_argument("--black-level", type=float, default=64.0)
    parser.add_argument("--r-gain", type=float, default=1.2)
    parser.add_argument("--g-gain", type=float, default=0.8)
    parser.add_argument("--b-gain", type=float, default=1.2)
    parser.add_argument("--low-percentile", type=float, default=0.5)
    parser.add_argument("--high-percentile", type=float, default=99.5)
    parser.add_argument("--gamma", type=float, default=2.2)
    parser.add_argument("--saturation", type=float, default=1.8)
    parser.add_argument("--sharpen", type=float, default=0.2)
    parser.add_argument(
        "--resize-original",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resize lite ISP output back to the original raw resolution.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    convert_file(
        input_path=args.input,
        output_path=args.output,
        width=args.width,
        height=args.height,
        stride_pixels=args.stride_pixels,
        mode=args.mode,
        black_level=args.black_level,
        r_gain=args.r_gain,
        g_gain=args.g_gain,
        b_gain=args.b_gain,
        low_percentile=args.low_percentile,
        high_percentile=args.high_percentile,
        gamma=args.gamma,
        saturation=args.saturation,
        sharpen_amount=args.sharpen,
        resize_original=args.resize_original,
    )


if __name__ == "__main__":
    main()
