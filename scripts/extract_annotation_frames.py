#!/usr/bin/env python3
"""Extract selected video frames in exactly the runtime detector geometry.

The ONNX runtime directly resizes each decoded frame to a square input using
OpenCV's default interpolation (INTER_LINEAR), without crop or letterbox.
This utility produces annotation images under that same convention.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2


def parse_frame_ids(value: str) -> list[int]:
    frame_ids = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not frame_ids:
        raise argparse.ArgumentTypeError("at least one frame ID is required")
    if min(frame_ids) < 0:
        raise argparse.ArgumentTypeError("frame IDs must be non-negative")
    if len(set(frame_ids)) != len(frame_ids):
        raise argparse.ArgumentTypeError("frame IDs must be unique")
    return sorted(frame_ids)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--frame-ids", type=parse_frame_ids, required=True)
    parser.add_argument("--size", type=int, default=640)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    if args.size <= 0:
        parser.error("--size must be positive")
    if not args.video.is_file():
        parser.error(f"video does not exist: {args.video}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    wanted = set(args.frame_ids)
    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {args.video}")

    saved: list[int] = []
    frame_id = 0
    max_frame_id = max(wanted)
    try:
        while frame_id <= max_frame_id:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_id in wanted:
                # Matches OnnxRtdetrEngine._preprocess: direct square resize,
                # default OpenCV interpolation (INTER_LINEAR), no letterboxing.
                resized = cv2.resize(frame, (args.size, args.size))
                output_path = args.output_dir / f"frame_{frame_id:06d}.png"
                if not cv2.imwrite(str(output_path), resized):
                    raise RuntimeError(f"failed to write {output_path}")
                saved.append(frame_id)
            frame_id += 1
    finally:
        capture.release()

    missing = sorted(wanted - set(saved))
    if missing:
        raise RuntimeError(f"video ended before requested frame IDs: {missing}")
    print(f"saved {len(saved)} images ({args.size}x{args.size}) to {args.output_dir}")


if __name__ == "__main__":
    main()
