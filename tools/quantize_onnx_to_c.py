#!/usr/bin/env python3
"""
Offline ONNX quantization + C array export tool for RT-DETR.

This script does two offline steps:

1. FP32 ONNX -> INT8 quantized ONNX
2. Quantized ONNX -> C source/header byte array

Important:
- This does NOT rewrite RT-DETR operators as hand-written C code.
- It embeds the quantized ONNX model as a C byte array.
- C/C++ inference should still use ONNX Runtime C/C++ API.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


class RTDETRCalibrationDataReader:
    """
    CalibrationDataReader-compatible object for ONNX Runtime static quantization.

    It follows the same preprocessing convention as the current Python RT-DETR engine:
    - BGR OpenCV image
    - resize to resolution x resolution
    - BGR -> RGB
    - float32 / 255.0
    - NCHW
    - batch dimension
    - optional orig_target_sizes input
    """

    def __init__(
        self,
        image_dir: str | Path,
        input_names: list[str],
        resolution: int,
        max_images: int,
    ) -> None:
        self.image_dir = Path(image_dir)
        self.input_names = list(input_names)
        self.resolution = int(resolution)
        self.max_images = int(max_images)

        self.image_paths = self._collect_images(self.image_dir, self.max_images)
        if not self.image_paths:
            raise FileNotFoundError(
                f"No calibration images found in: {self.image_dir}"
            )

        self._iterator: Iterator[Path] = iter(self.image_paths)

    @staticmethod
    def _collect_images(image_dir: Path, max_images: int) -> list[Path]:
        if not image_dir.exists():
            raise FileNotFoundError(f"Calibration directory does not exist: {image_dir}")

        images = [
            path
            for path in sorted(image_dir.rglob("*"))
            if path.suffix.lower() in IMAGE_SUFFIXES
        ]
        if max_images > 0:
            images = images[:max_images]
        return images

    def get_next(self) -> dict[str, np.ndarray] | None:
        while True:
            try:
                image_path = next(self._iterator)
            except StopIteration:
                return None

            frame = cv2.imread(str(image_path))
            if frame is None:
                print(f"[warn] Could not read calibration image: {image_path}")
                continue

            blob = self._preprocess(frame, self.resolution)
            orig_target_sizes = np.array(
                [[self.resolution, self.resolution]],
                dtype=np.int64,
            )

            feeds: dict[str, np.ndarray] = {}
            for name in self.input_names:
                if name == "images":
                    feeds[name] = blob
                elif name == "orig_target_sizes":
                    feeds[name] = orig_target_sizes
                else:
                    raise ValueError(
                        f"Unsupported ONNX input during calibration: {name}. "
                        f"Expected 'images' or 'orig_target_sizes'."
                    )

            return feeds

    @staticmethod
    def _preprocess(frame: np.ndarray, resolution: int) -> np.ndarray:
        resized = cv2.resize(frame, (resolution, resolution))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        blob = rgb.astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))
        return np.expand_dims(blob, axis=0)


def sanitize_c_symbol(name: str) -> str:
    """
    Convert an arbitrary string into a valid C identifier.
    """
    symbol = re.sub(r"[^0-9a-zA-Z_]", "_", name)
    if not symbol:
        symbol = "embedded_onnx_model"
    if symbol[0].isdigit():
        symbol = "_" + symbol
    return symbol


def get_onnx_input_names(model_path: Path) -> list[str]:
    import onnxruntime as ort

    session = ort.InferenceSession(
        str(model_path),
        providers=["CPUExecutionProvider"],
    )
    return [input_info.name for input_info in session.get_inputs()]


def quantize_dynamic_onnx(
    input_model: Path,
    output_model: Path,
) -> None:
    """
    Dynamic quantization.

    This is easy to test because it does not require calibration images.
    It usually quantizes MatMul/Gemm weights, but for CNN-heavy models it may
    provide limited acceleration.
    """
    from onnxruntime.quantization import QuantType, quantize_dynamic

    output_model.parent.mkdir(parents=True, exist_ok=True)

    quantize_dynamic(
        model_input=str(input_model),
        model_output=str(output_model),
        weight_type=QuantType.QInt8,
        per_channel=True,
        op_types_to_quantize=["MatMul", "Gemm"],
    )


def quantize_static_onnx(
    input_model: Path,
    output_model: Path,
    calibration_dir: Path,
    resolution: int,
    max_calibration_images: int,
) -> None:
    """
    Static INT8 quantization with calibration data.

    Recommended for deployment testing because it can quantize Conv/MatMul/Gemm.
    The output is a QDQ-format quantized ONNX model.
    """
    from onnxruntime.quantization import (
        CalibrationMethod,
        QuantFormat,
        QuantType,
        quantize_static,
    )

    output_model.parent.mkdir(parents=True, exist_ok=True)

    input_names = get_onnx_input_names(input_model)
    print(f"[info] ONNX input names: {input_names}")

    reader = RTDETRCalibrationDataReader(
        image_dir=calibration_dir,
        input_names=input_names,
        resolution=resolution,
        max_images=max_calibration_images,
    )

    quantize_static(
        model_input=str(input_model),
        model_output=str(output_model),
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        calibrate_method=CalibrationMethod.MinMax,
        per_channel=True,
        op_types_to_quantize=["Conv", "MatMul", "Gemm"],
        extra_options={
            "ActivationSymmetric": False,
            "WeightSymmetric": True,
        },
    )


def write_binary_as_c_array(
    binary_path: Path,
    c_path: Path,
    h_path: Path,
    symbol_name: str,
    bytes_per_line: int = 12,
) -> None:
    """
    Convert any binary file into a C byte array.

    The generated files expose:

        extern const unsigned char <symbol_name>[];
        extern const size_t <symbol_name>_len;
    """
    symbol_name = sanitize_c_symbol(symbol_name)

    c_path.parent.mkdir(parents=True, exist_ok=True)
    h_path.parent.mkdir(parents=True, exist_ok=True)

    data = binary_path.read_bytes()
    guard = f"{symbol_name.upper()}_H"

    h_text = "\n".join(
        [
            f"#ifndef {guard}",
            f"#define {guard}",
            "",
            "#include <stddef.h>",
            "#include <stdint.h>",
            "",
            f"extern const unsigned char {symbol_name}[];",
            f"extern const size_t {symbol_name}_len;",
            "",
            f"#endif  /* {guard} */",
            "",
        ]
    )

    c_lines: list[str] = []
    for start in range(0, len(data), bytes_per_line):
        chunk = data[start : start + bytes_per_line]
        c_lines.append(
            "    " + ", ".join(f"0x{byte:02x}" for byte in chunk) + ","
        )

    c_text = "\n".join(
        [
            f'#include "{h_path.name}"',
            "",
            f"const unsigned char {symbol_name}[] = {{",
            *c_lines,
            "};",
            "",
            f"const size_t {symbol_name}_len = sizeof({symbol_name});",
            "",
        ]
    )

    h_path.write_text(h_text, encoding="utf-8")
    c_path.write_text(c_text, encoding="utf-8")

    print(f"[info] Wrote C source: {c_path}")
    print(f"[info] Wrote C header: {h_path}")
    print(f"[info] Embedded model size: {len(data) / (1024 * 1024):.2f} MB")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quantize RT-DETR ONNX model and export it as C byte array."
    )

    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Input FP32 ONNX model path.",
    )
    parser.add_argument(
        "--out-onnx",
        type=Path,
        required=True,
        help="Output quantized ONNX model path.",
    )
    parser.add_argument(
        "--mode",
        choices=["static", "dynamic", "c-only"],
        default="static",
        help=(
            "static: calibration-based INT8 quantization; "
            "dynamic: no calibration images; "
            "c-only: skip quantization and only convert --out-onnx to C."
        ),
    )
    parser.add_argument(
        "--calib-dir",
        type=Path,
        default=None,
        help="Calibration image directory. Required for static quantization.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=640,
        help="Model input resolution used for calibration preprocessing.",
    )
    parser.add_argument(
        "--max-calib",
        type=int,
        default=128,
        help="Maximum number of calibration images. Use 0 for all images.",
    )
    parser.add_argument(
        "--c-out",
        type=Path,
        default=None,
        help="Output .c file path. If omitted, C export is skipped.",
    )
    parser.add_argument(
        "--h-out",
        type=Path,
        default=None,
        help="Output .h file path. If omitted, C export is skipped.",
    )
    parser.add_argument(
        "--symbol",
        default="rtdetr_quantized_onnx",
        help="C symbol name for embedded ONNX byte array.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_model = args.model
    output_model = args.out_onnx

    if not input_model.exists():
        raise FileNotFoundError(f"Input model does not exist: {input_model}")

    if args.mode == "static":
        if args.calib_dir is None:
            raise ValueError("--calib-dir is required when --mode static")

        print("[info] Running static INT8 quantization...")
        print(f"[info] Input model: {input_model}")
        print(f"[info] Output model: {output_model}")
        print(f"[info] Calibration dir: {args.calib_dir}")
        print(f"[info] Calibration resolution: {args.resolution}")
        print(f"[info] Max calibration images: {args.max_calib}")

        quantize_static_onnx(
            input_model=input_model,
            output_model=output_model,
            calibration_dir=args.calib_dir,
            resolution=args.resolution,
            max_calibration_images=args.max_calib,
        )

    elif args.mode == "dynamic":
        print("[info] Running dynamic quantization...")
        print(f"[info] Input model: {input_model}")
        print(f"[info] Output model: {output_model}")

        quantize_dynamic_onnx(
            input_model=input_model,
            output_model=output_model,
        )

    elif args.mode == "c-only":
        if not output_model.exists():
            raise FileNotFoundError(
                f"--mode c-only expects --out-onnx to already exist: {output_model}"
            )
        print("[info] Skipping quantization; exporting existing ONNX to C.")

    else:
        raise ValueError(f"Unsupported mode: {args.mode}")

    if args.c_out is not None or args.h_out is not None:
        if args.c_out is None or args.h_out is None:
            raise ValueError("--c-out and --h-out must be provided together")

        print("[info] Exporting ONNX binary to C byte array...")
        write_binary_as_c_array(
            binary_path=output_model,
            c_path=args.c_out,
            h_path=args.h_out,
            symbol_name=args.symbol,
        )

    print("[done] Finished.")


if __name__ == "__main__":
    main()