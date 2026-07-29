#!/usr/bin/env python3
"""Compare detector quality on manually annotated video frames.

The script evaluates the same 50 LabelImg Pascal-VOC frames against:

* existing project runtime detections from the final experiment suite;
* YOLOv8n predictions generated directly on the annotated PNG frames;
* NanoDet-Plus predictions generated directly on the annotated PNG frames;
* a PicoDet local backend probe, reported in status.csv when it is unavailable.

All predictions are normalized to the runtime_detections.jsonl schema used by
the project, so the resulting CSVs can be reused by plotting/report scripts.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import subprocess
import sys
import time
import types
from collections import defaultdict
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from plot_manual_gt_accuracy import (  # noqa: E402
    COCO80,
    Box,
    metric_row,
    read_predictions,
    read_voc_annotations,
    score_frame,
)


PROJECT_METHODS = [
    (
        "rtdetr_fp32_native",
        "RT-DETR FP32 native",
        Path("r01_01_fp32_native/runtime_detections.jsonl"),
    ),
    (
        "lk_tracking",
        "LK tracking",
        Path("r01_03_int8_dynamic_q300_lk_roi/runtime_detections.jsonl"),
    ),
    (
        "proposed_software",
        "Proposed software",
        Path("r01_06_proposed_software/runtime_detections.jsonl"),
    ),
]


def frame_id_from_path(path: Path) -> int:
    match = re.search(r"frame_(\d+)", path.stem)
    if not match:
        raise ValueError(f"cannot parse frame id from {path.name}")
    return int(match.group(1))


def annotated_images(annotations_dir: Path) -> list[Path]:
    images = sorted(annotations_dir.glob("frame_*.png"), key=frame_id_from_path)
    if not images:
        raise RuntimeError(f"no annotated frame PNG files found in {annotations_dir}")
    return images


def write_jsonl(path: Path, rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")


def predict_yolov8n(images: list[Path], model_path: Path, output_jsonl: Path, conf: float) -> float:
    from ultralytics import YOLO

    model = YOLO(str(model_path))
    rows: list[dict[str, object]] = []
    start = time.perf_counter()
    for image_path in images:
        result = model.predict(
            source=str(image_path),
            imgsz=640,
            conf=conf,
            device="cpu",
            verbose=False,
        )[0]
        detections = []
        for box in result.boxes:
            xyxy = [float(value) for value in box.xyxy[0].tolist()]
            detections.append(
                {
                    "class_id": int(box.cls[0].item()),
                    "score": float(box.conf[0].item()),
                    "bbox": xyxy,
                }
            )
        rows.append({"frame_id": frame_id_from_path(image_path), "detections": detections})
    elapsed = time.perf_counter() - start
    write_jsonl(output_jsonl, rows)
    return elapsed


def install_nanodet_compat_shims() -> None:
    import torch

    module = types.ModuleType("torch._six")
    module.string_classes = (str, bytes)
    sys.modules["torch._six"] = module

    original_load = torch.load

    def compat_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_load(*args, **kwargs)

    torch.load = compat_load

    original_module_to = torch.nn.Module.to

    def cpu_safe_module_to(self, *args, **kwargs):
        if args and str(args[0]).startswith("cuda") and not torch.cuda.is_available():
            args = ("cpu",) + tuple(args[1:])
        if str(kwargs.get("device", "")).startswith("cuda") and not torch.cuda.is_available():
            kwargs["device"] = "cpu"
        return original_module_to(self, *args, **kwargs)

    torch.nn.Module.to = cpu_safe_module_to

    original_tensor_to = torch.Tensor.to

    def cpu_safe_tensor_to(self, *args, **kwargs):
        if args and str(args[0]).startswith("cuda") and not torch.cuda.is_available():
            args = ("cpu",) + tuple(args[1:])
        if str(kwargs.get("device", "")).startswith("cuda") and not torch.cuda.is_available():
            kwargs["device"] = "cpu"
        return original_tensor_to(self, *args, **kwargs)

    torch.Tensor.to = cpu_safe_tensor_to


def nanodet_no_pretrain_config(config_path: Path, output_dir: Path, input_size: int | None = None) -> Path:
    import yaml

    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    backbone = config.setdefault("model", {}).setdefault("arch", {}).setdefault("backbone", {})
    backbone["pretrain"] = False
    if input_size is not None:
        config.setdefault("data", {}).setdefault("val", {})["input_size"] = [input_size, input_size]
    suffix = f"_input{input_size}" if input_size is not None else ""
    output = output_dir / f"{config_path.stem}{suffix}_no_pretrain.yml"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    return output


def predict_nanodet(
    images: list[Path],
    config_path: Path,
    checkpoint_path: Path,
    output_jsonl: Path,
    conf: float,
    tmp_dir: Path,
    input_size: int | None,
) -> float:
    install_nanodet_compat_shims()
    nanodet_root = ROOT / "third_party" / "nanodet"
    if str(nanodet_root) not in sys.path:
        sys.path.insert(0, str(nanodet_root))

    from demo.demo import Predictor
    from nanodet.util import Logger, cfg, load_config

    runtime_config = nanodet_no_pretrain_config(config_path, tmp_dir, input_size)
    load_config(cfg, str(runtime_config))
    predictor = Predictor(cfg, str(checkpoint_path), Logger(-1, use_tensorboard=False), device="cpu")

    rows: list[dict[str, object]] = []
    start = time.perf_counter()
    for image_path in images:
        _meta, results = predictor.inference(str(image_path))
        frame_detections = []
        for class_id, boxes in results[0].items():
            for box in boxes:
                values = [float(value) for value in box]
                if len(values) < 5 or values[4] < conf:
                    continue
                frame_detections.append(
                    {
                        "class_id": int(class_id),
                        "score": values[4],
                        "bbox": values[:4],
                    }
                )
        rows.append({"frame_id": frame_id_from_path(image_path), "detections": frame_detections})
    elapsed = time.perf_counter() - start
    write_jsonl(output_jsonl, rows)
    return elapsed


def probe_picodet(
    image_path: Path,
    model_dir: Path,
    output_dir: Path,
    timeout_seconds: int,
) -> tuple[str, str]:
    infer_script = ROOT / "third_party" / "PaddleDetection" / "deploy" / "python" / "infer.py"
    if not infer_script.exists():
        return "missing", f"{infer_script} not found"
    output_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.setdefault("HOME", str(ROOT / "tmp" / "picodet_home"))
    env.setdefault("USERPROFILE", str(ROOT / "tmp" / "picodet_userprofile"))
    env.setdefault("APPDATA", str(ROOT / "tmp" / "picodet_appdata"))
    env.setdefault("PADDLE_HOME", str(ROOT / "tmp" / "picodet_paddle_home"))
    env.setdefault("MPLCONFIGDIR", str(ROOT / ".plot_mplconfig"))
    env.setdefault("FLAGS_use_mkldnn", "0")
    env.setdefault("FLAGS_use_onednn", "0")
    env.setdefault("ONEDNN_VERBOSE", "0")

    argv = [
        str(infer_script),
        "--model_dir",
        str(model_dir),
        "--image_file",
        str(image_path),
        "--output_dir",
        str(output_dir),
        "--device",
        "CPU",
        "--threshold",
        "0.25",
    ]
    wrapper = f"""
import runpy
import sys

import numpy as np

if not hasattr(np, "sctypes"):
    np.sctypes = {{
        "int": [np.int8, np.int16, np.int32, np.int64],
        "uint": [np.uint8, np.uint16, np.uint32, np.uint64],
        "float": [np.float16, np.float32, np.float64],
        "complex": [np.complex64, np.complex128],
        "others": [np.bool_, np.bytes_, np.str_],
    }}

sys.argv = {argv!r}
runpy.run_path({str(infer_script)!r}, run_name="__main__")
"""
    command = [sys.executable, "-c", wrapper]
    try:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=env,
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return "timeout", f"timed out after {timeout_seconds}s"
    combined = "\n".join(part for part in (completed.stdout, completed.stderr) if part).strip()
    tail = combined[-1200:].replace("\r", "")
    if completed.returncode == 0:
        return "available_unparsed", "PaddleDetection ran on one image, but structured boxes are not parsed by this script yet."
    return "failed_local", tail or f"return code {completed.returncode}"


def average_precision(gt: dict[int, list[Box]], pred: dict[int, list[Box]], label: str, threshold: float) -> float | None:
    gt_by_frame = {
        frame_id: [box for box in boxes if box.label == label]
        for frame_id, boxes in gt.items()
    }
    total_gt = sum(len(boxes) for boxes in gt_by_frame.values())
    if total_gt == 0:
        return None

    candidates = []
    for frame_id, boxes in pred.items():
        for box in boxes:
            if box.label == label:
                candidates.append((box.score, frame_id, box))
    candidates.sort(key=lambda item: item[0], reverse=True)
    if not candidates:
        return 0.0

    matched: dict[int, set[int]] = defaultdict(set)
    tp: list[float] = []
    fp: list[float] = []
    for _score, frame_id, candidate in candidates:
        options = [
            (idx, candidate_iou)
            for idx, gt_box in enumerate(gt_by_frame[frame_id])
            if idx not in matched[frame_id]
            for candidate_iou in [iou_box(candidate, gt_box)]
        ]
        if options:
            best_idx, best_iou = max(options, key=lambda item: item[1])
            if best_iou >= threshold:
                matched[frame_id].add(best_idx)
                tp.append(1.0)
                fp.append(0.0)
                continue
        tp.append(0.0)
        fp.append(1.0)

    cum_tp = []
    cum_fp = []
    running_tp = running_fp = 0.0
    for hit, miss in zip(tp, fp):
        running_tp += hit
        running_fp += miss
        cum_tp.append(running_tp)
        cum_fp.append(running_fp)
    recalls = [value / total_gt for value in cum_tp]
    precisions = [
        cum_tp[idx] / (cum_tp[idx] + cum_fp[idx]) if cum_tp[idx] + cum_fp[idx] else 0.0
        for idx in range(len(cum_tp))
    ]
    recalls = [0.0] + recalls + [1.0]
    precisions = [0.0] + precisions + [0.0]
    for idx in range(len(precisions) - 2, -1, -1):
        precisions[idx] = max(precisions[idx], precisions[idx + 1])
    ap = 0.0
    for idx in range(1, len(recalls)):
        if recalls[idx] != recalls[idx - 1]:
            ap += (recalls[idx] - recalls[idx - 1]) * precisions[idx]
    return ap


def iou_box(a: Box, b: Box) -> float:
    ax1, ay1, ax2, ay2 = a.xyxy
    bx1, by1, bx2, by2 = b.xyxy
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


def map_metrics(gt: dict[int, list[Box]], pred: dict[int, list[Box]]) -> dict[str, float]:
    labels = sorted({box.label for boxes in gt.values() for box in boxes})
    ap50_values = []
    map_values = []
    thresholds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    for label in labels:
        ap50 = average_precision(gt, pred, label, 0.50)
        if ap50 is not None:
            ap50_values.append(ap50)
        class_aps = [average_precision(gt, pred, label, threshold) for threshold in thresholds]
        class_aps = [value for value in class_aps if value is not None]
        if class_aps:
            map_values.append(sum(class_aps) / len(class_aps))
    return {
        "ap50": sum(ap50_values) / len(ap50_values) if ap50_values else 0.0,
        "map50_95": sum(map_values) / len(map_values) if map_values else 0.0,
    }


def per_frame_rows(
    method_id: str,
    method_name: str,
    gt: dict[int, list[Box]],
    pred: dict[int, list[Box]],
    threshold: float,
) -> list[dict[str, object]]:
    rows = []
    for frame_id, gt_boxes in gt.items():
        tp, fp, fn, ious = score_frame(gt_boxes, pred[frame_id], threshold)
        rows.append(
            {
                "run": method_id,
                "method": method_name,
                "frame_id": frame_id,
                "gt_boxes": len(gt_boxes),
                "pred_boxes": len(pred[frame_id]),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "mean_matched_iou": sum(ious) / len(ious) if ious else 0.0,
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def export_summary_figure(rows: list[dict[str, object]], output: Path) -> None:
    if not rows:
        return
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt

    metrics = [
        ("precision", "Precision", "#3973AC"),
        ("recall", "Recall", "#D9822B"),
        ("f1", "F1", "#3F8F62"),
        ("ap50", "AP50", "#7A59A3"),
        ("map50_95", "mAP50-95", "#777777"),
    ]
    names = [str(row["method"]) for row in rows]
    x = list(range(len(rows)))
    width = 0.15
    fig, ax = plt.subplots(figsize=(9.2, 4.2), constrained_layout=True)
    for metric_index, (key, label, color) in enumerate(metrics):
        offset = (metric_index - (len(metrics) - 1) / 2) * width
        values = [100.0 * float(row.get(key, 0.0)) for row in rows]
        bars = ax.bar([item + offset for item in x], values, width=width, label=label, color=color)
        for bar, value in zip(bars, values):
            if math.isfinite(value):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    value + 1.0,
                    f"{value:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=6.2,
                    rotation=90,
                )
    ax.set_title("Annotated-frame detection accuracy", loc="left", fontweight="bold")
    ax.text(
        0,
        1.01,
        f"{rows[0]['frames']} manually annotated frames, {rows[0]['gt_boxes']} GT boxes, class-aware IoU@0.50 matching",
        transform=ax.transAxes,
        fontsize=8,
        color="#4D5560",
    )
    ax.set_ylabel("Score (%)")
    ax.set_ylim(0, 108)
    ax.set_xticks(x, names, rotation=20, ha="right")
    ax.yaxis.grid(True, color="#D9DEE4", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.22), fontsize=8)
    output.parent.mkdir(parents=True, exist_ok=True)
    for suffix, kwargs in ((".png", {"dpi": 300}), ((".svg"), {}), ((".pdf"), {})):
        fig.savefig(output.with_suffix(suffix), bbox_inches="tight", **kwargs)
    plt.close(fig)


def add_evaluation_row(
    rows: list[dict[str, object]],
    per_frame: list[dict[str, object]],
    method_id: str,
    method_name: str,
    gt: dict[int, list[Box]],
    predictions_path: Path,
    threshold: float,
) -> None:
    predictions = read_predictions(predictions_path, set(gt))
    row = metric_row(method_id, method_name, gt, predictions, threshold)
    row.update(map_metrics(gt, predictions))
    rows.append(row)
    per_frame.extend(per_frame_rows(method_id, method_name, gt, predictions, threshold))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations-dir", type=Path, default=ROOT / "data" / "annotations" / "sample3_50frames_640")
    parser.add_argument("--suite-dir", type=Path, default=ROOT / "experiments" / "final_thermal_20260728_132853")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "experiments" / "results" / "sota_annotated_accuracy")
    parser.add_argument("--predictions-dir", type=Path, default=ROOT / "experiments" / "logs" / "sota_baselines" / "annotated_accuracy")
    parser.add_argument("--yolo-model", type=Path, default=ROOT / "models" / "baselines" / "yolov8n.pt")
    parser.add_argument("--nanodet-config", type=Path, default=ROOT / "third_party" / "nanodet" / "config" / "nanodet-plus-m_320.yml")
    parser.add_argument("--nanodet-checkpoint", type=Path, default=ROOT / "models" / "baselines" / "nanodet-plus-m_320.ckpt")
    parser.add_argument("--nanodet-input-size", type=int, default=320)
    parser.add_argument("--picodet-model-dir", type=Path, default=ROOT / "models" / "baselines" / "picodet_s_320_coco_lcnet_portable")
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--picodet-timeout", type=int, default=60)
    parser.add_argument(
        "--extra-prediction",
        action="append",
        default=[],
        help=(
            "Add an existing runtime_detections.jsonl to the table as "
            "'run_id|Display name|path'. Can be used multiple times."
        ),
    )
    parser.add_argument("--skip-yolo", action="store_true")
    parser.add_argument("--skip-nanodet", action="store_true")
    parser.add_argument("--skip-picodet-probe", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.predictions_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = ROOT / "tmp" / "sota_annotated_accuracy"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    images = annotated_images(args.annotations_dir)
    gt = read_voc_annotations(args.annotations_dir)
    status_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    per_frame: list[dict[str, object]] = []

    for method_id, method_name, relative_path in PROJECT_METHODS:
        path = args.suite_dir / relative_path
        if not path.exists():
            status_rows.append({"method": method_id, "status": "missing", "detail": str(path)})
            continue
        add_evaluation_row(metric_rows, per_frame, method_id, method_name, gt, path, args.iou)
        status_rows.append({"method": method_id, "status": "ok", "detail": str(path)})

    for spec in args.extra_prediction:
        parts = spec.split("|", 2)
        if len(parts) != 3:
            status_rows.append({"method": spec, "status": "invalid_extra_prediction", "detail": "expected run_id|Display name|path"})
            continue
        method_id, method_name, path_text = parts
        path = Path(path_text)
        if not path.is_absolute():
            path = ROOT / path
        if not path.exists():
            status_rows.append({"method": method_id, "status": "missing", "detail": str(path)})
            continue
        add_evaluation_row(metric_rows, per_frame, method_id, method_name, gt, path, args.iou)
        status_rows.append({"method": method_id, "status": "ok", "detail": str(path)})

    if not args.skip_yolo:
        path = args.predictions_dir / "yolov8n_640_annotated_frames.jsonl"
        try:
            elapsed = predict_yolov8n(images, args.yolo_model, path, args.conf)
            add_evaluation_row(metric_rows, per_frame, "yolov8n_640", "YOLOv8n 640", gt, path, args.iou)
            status_rows.append({"method": "yolov8n_640", "status": "ok", "detail": f"{elapsed:.3f}s for {len(images)} frames; {path}"})
        except Exception as exc:
            status_rows.append({"method": "yolov8n_640", "status": "failed_local", "detail": repr(exc)})

    if not args.skip_nanodet:
        nanodet_id = f"nanodet_plus_m_input{args.nanodet_input_size}"
        path = args.predictions_dir / f"{nanodet_id}_annotated_frames.jsonl"
        try:
            elapsed = predict_nanodet(
                images,
                args.nanodet_config,
                args.nanodet_checkpoint,
                path,
                args.conf,
                tmp_dir,
                args.nanodet_input_size,
            )
            add_evaluation_row(
                metric_rows,
                per_frame,
                nanodet_id,
                f"NanoDet-Plus-m input{args.nanodet_input_size}",
                gt,
                path,
                args.iou,
            )
            status_rows.append({"method": nanodet_id, "status": "ok", "detail": f"{elapsed:.3f}s for {len(images)} frames; {path}"})
        except Exception as exc:
            status_rows.append({"method": nanodet_id, "status": "failed_local", "detail": repr(exc)})

    if not args.skip_picodet_probe:
        status, detail = probe_picodet(images[0], args.picodet_model_dir, tmp_dir / "picodet_probe", args.picodet_timeout)
        status_rows.append({"method": "pp_picodet_s_320", "status": status, "detail": detail})

    write_csv(args.output_dir / "metrics.csv", metric_rows)
    write_csv(args.output_dir / "per_frame.csv", per_frame)
    write_csv(args.output_dir / "status.csv", status_rows)
    export_summary_figure(metric_rows, args.output_dir / "annotated_accuracy_summary")
    print(json.dumps(metric_rows, indent=2))
    print(f"metrics: {args.output_dir / 'metrics.csv'}")
    print(f"status: {args.output_dir / 'status.csv'}")


if __name__ == "__main__":
    main()
