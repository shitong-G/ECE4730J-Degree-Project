#!/usr/bin/env python3
"""Densify sparse LabelImg VOC annotations by endpoint-based linear interpolation.

For each two consecutive manually annotated frames, same-class objects are
matched globally.  On intervening frames the box centre follows constant
velocity, while width and height are fixed to the arithmetic mean of the two
endpoint boxes.  Original XML files are never modified.
"""

from __future__ import annotations

import argparse
import csv
import math
import shutil
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


LABEL_ALIASES = {"people": "person"}


@dataclass(frozen=True)
class Box:
    label: str
    xyxy: tuple[float, float, float, float]

    @property
    def centre(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.xyxy
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @property
    def width(self) -> float:
        return self.xyxy[2] - self.xyxy[0]

    @property
    def height(self) -> float:
        return self.xyxy[3] - self.xyxy[1]

    @property
    def diagonal(self) -> float:
        return math.hypot(self.width, self.height)


def canonical_label(label: str) -> str:
    return LABEL_ALIASES.get(label.strip().lower(), label.strip().lower())


def read_voc(path: Path) -> list[Box]:
    root = ET.parse(path).getroot()
    boxes: list[Box] = []
    for obj in root.findall("object"):
        node = obj.find("bndbox")
        if node is None:
            continue
        boxes.append(Box(
            canonical_label(obj.findtext("name", default="")),
            tuple(float(node.findtext(key)) for key in ("xmin", "ymin", "xmax", "ymax")),
        ))
    return boxes


def pair_cost(a: Box, b: Box) -> float | None:
    """Return a conservative same-object association cost, or None if implausible."""
    if a.label != b.label:
        return None
    ax, ay = a.centre
    bx, by = b.centre
    distance = math.hypot(ax - bx, ay - by)
    # A generous but finite geometric gate prevents matching two different
    # distant objects of the same class while allowing motion across <=5 frames.
    gate = max(55.0, 1.6 * max(a.diagonal, b.diagonal))
    if distance > gate:
        return None
    area_a = max(a.width * a.height, 1.0)
    area_b = max(b.width * b.height, 1.0)
    return distance / gate + 0.25 * abs(math.log(area_a / area_b))


def match_boxes(start: list[Box], end: list[Box]) -> list[tuple[int, int, float]]:
    """Maximum-cardinality, minimum-cost class-aware matching by enumeration."""
    candidates = {(i, j): pair_cost(a, b) for i, a in enumerate(start) for j, b in enumerate(end)}
    candidates = {key: value for key, value in candidates.items() if value is not None}
    best: tuple[int, float, list[tuple[int, int, float]]] = (-1, float("inf"), [])

    def search(index: int, used_end: set[int], chosen: list[tuple[int, int, float]], cost: float) -> None:
        nonlocal best
        if index == len(start):
            candidate = (len(chosen), cost, chosen.copy())
            if candidate[0] > best[0] or (candidate[0] == best[0] and candidate[1] < best[1]):
                best = candidate
            return
        search(index + 1, used_end, chosen, cost)  # object may leave/enter view
        for end_index in range(len(end)):
            value = candidates.get((index, end_index))
            if value is None or end_index in used_end:
                continue
            used_end.add(end_index)
            chosen.append((index, end_index, value))
            search(index + 1, used_end, chosen, cost + value)
            chosen.pop()
            used_end.remove(end_index)

    search(0, set(), [], 0.0)
    return best[2]


def interpolate(a: Box, b: Box, ratio: float, size: int) -> Box:
    ax, ay = a.centre
    bx, by = b.centre
    centre_x = ax + (bx - ax) * ratio
    centre_y = ay + (by - ay) * ratio
    width = (a.width + b.width) / 2.0
    height = (a.height + b.height) / 2.0
    x1 = max(0.0, min(float(size), centre_x - width / 2.0))
    y1 = max(0.0, min(float(size), centre_y - height / 2.0))
    x2 = max(0.0, min(float(size), centre_x + width / 2.0))
    y2 = max(0.0, min(float(size), centre_y + height / 2.0))
    return Box(a.label, (x1, y1, x2, y2))


def write_voc(path: Path, boxes: list[Box], size: int, origin: str) -> None:
    root = ET.Element("annotation")
    ET.SubElement(root, "folder").text = path.parent.name
    ET.SubElement(root, "filename").text = path.with_suffix(".png").name
    source = ET.SubElement(root, "source")
    ET.SubElement(source, "database").text = origin
    node_size = ET.SubElement(root, "size")
    ET.SubElement(node_size, "width").text = str(size)
    ET.SubElement(node_size, "height").text = str(size)
    ET.SubElement(node_size, "depth").text = "3"
    ET.SubElement(root, "segmented").text = "0"
    for box in boxes:
        obj = ET.SubElement(root, "object")
        ET.SubElement(obj, "name").text = box.label
        ET.SubElement(obj, "pose").text = "Unspecified"
        ET.SubElement(obj, "truncated").text = "0"
        ET.SubElement(obj, "difficult").text = "0"
        bndbox = ET.SubElement(obj, "bndbox")
        for key, value in zip(("xmin", "ymin", "xmax", "ymax"), box.xyxy):
            ET.SubElement(bndbox, key).text = str(int(round(value)))
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def draw_preview(image: np.ndarray, boxes: list[Box], frame_id: int, origin: str) -> np.ndarray:
    canvas = image.copy()
    color = (44, 160, 44) if origin == "manual" else (38, 119, 197)
    for box in boxes:
        x1, y1, x2, y2 = (int(round(value)) for value in box.xyxy)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        cv2.putText(canvas, box.label, (x1, max(16, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    cv2.putText(canvas, f"frame {frame_id:03d} | {origin}", (12, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)
    return canvas


def write_preview(output_dir: Path, labels: dict[int, list[Box]], manual_ids: set[int]) -> None:
    ids = [0, 2, 4, 39, 41, 43, 83, 85, 87, 130, 132, 135, 178, 180, 183, 209, 211, 213]
    cells: list[np.ndarray] = []
    for frame_id in ids:
        image = cv2.imread(str(output_dir / f"frame_{frame_id:06d}.png"))
        if image is None:
            continue
        origin = "manual" if frame_id in manual_ids else "interpolated"
        image = draw_preview(image, labels[frame_id], frame_id, origin)
        cells.append(cv2.resize(image, (320, 320)))
    rows = [np.hstack(cells[index:index + 6]) for index in range(0, len(cells), 6)]
    cv2.imwrite(str(output_dir / "interpolation_preview.png"), np.vstack(rows))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations-dir", type=Path, default=Path("data/annotations/sample3_50frames_640"))
    parser.add_argument("--video", type=Path, default=Path("data/sample3.mp4"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/annotations/sample3_214frames_interpolated_640"))
    parser.add_argument("--size", type=int, default=640)
    args = parser.parse_args()
    xml_paths = sorted(args.annotations_dir.glob("frame_*.xml"))
    if len(xml_paths) < 2:
        raise RuntimeError("at least two manually annotated XML files are required")
    manual = {int(path.stem.rsplit("_", 1)[1]): read_voc(path) for path in xml_paths}
    manual_ids = sorted(manual)
    if not args.video.is_file():
        raise RuntimeError(f"video does not exist: {args.video}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    labels: dict[int, list[Box]] = {frame_id: [] for frame_id in range(manual_ids[0], manual_ids[-1] + 1)}
    labels.update(manual)
    audit: list[dict[str, object]] = []
    for start_id, end_id in zip(manual_ids, manual_ids[1:]):
        start, end = manual[start_id], manual[end_id]
        matches = match_boxes(start, end)
        for start_index, end_index, cost in matches:
            a, b = start[start_index], end[end_index]
            audit.append({"start_frame": start_id, "end_frame": end_id, "label": a.label, "start_index": start_index, "end_index": end_index, "association_cost": f"{cost:.5f}"})
            for frame_id in range(start_id + 1, end_id):
                labels.setdefault(frame_id, []).append(interpolate(a, b, (frame_id - start_id) / (end_id - start_id), args.size))
        audit.append({"start_frame": start_id, "end_frame": end_id, "label": "__summary__", "start_index": len(start), "end_index": len(end), "association_cost": f"matched={len(matches)}"})

    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {args.video}")
    try:
        for frame_id in range(manual_ids[0], manual_ids[-1] + 1):
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"video ended before frame {frame_id}")
            output_image = args.output_dir / f"frame_{frame_id:06d}.png"
            origin = "manual" if frame_id in manual else "interpolated"
            source_image = args.annotations_dir / output_image.name
            if origin == "manual" and source_image.is_file():
                shutil.copy2(source_image, output_image)
            else:
                image = cv2.resize(frame, (args.size, args.size))
                if not cv2.imwrite(str(output_image), image):
                    raise RuntimeError(f"could not write {output_image}")
            write_voc(args.output_dir / f"frame_{frame_id:06d}.xml", labels.get(frame_id, []), args.size, origin)
    finally:
        capture.release()

    with (args.output_dir / "interpolation_audit.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["start_frame", "end_frame", "label", "start_index", "end_index", "association_cost"])
        writer.writeheader(); writer.writerows(audit)
    with (args.output_dir / "README.txt").open("w", encoding="utf-8") as handle:
        handle.write("Manual labels were copied geometrically into this dense set and canonicalised (people -> person).\n")
        handle.write("Interpolated boxes: centre moves linearly; width and height are the arithmetic endpoint means.\n")
        handle.write("Only class-aware endpoint matches are interpolated; unmatched endpoint objects are omitted between labels.\n")
    write_preview(args.output_dir, labels, set(manual))
    print(f"manual_frames={len(manual)} interpolated_frames={len(labels) - len(manual)} total_frames={len(labels)}")
    print(f"matched_object_pairs={sum(1 for row in audit if row['label'] != '__summary__')}")
    print(f"output={args.output_dir}")


if __name__ == "__main__":
    main()
