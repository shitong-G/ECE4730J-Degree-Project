import argparse
from pathlib import Path

import cv2
import numpy as np


BAYER_CODES = {
    "RGGB": cv2.COLOR_BayerRG2BGR,
    "BGGR": cv2.COLOR_BayerBG2BGR,
    "GRBG": cv2.COLOR_BayerGR2BGR,
    "GBRG": cv2.COLOR_BayerGB2BGR,
}


BAYER_CODES_EA = {
    "RGGB": getattr(cv2, "COLOR_BayerRG2BGR_EA", cv2.COLOR_BayerRG2BGR),
    "BGGR": getattr(cv2, "COLOR_BayerBG2BGR_EA", cv2.COLOR_BayerBG2BGR),
    "GRBG": getattr(cv2, "COLOR_BayerGR2BGR_EA", cv2.COLOR_BayerGR2BGR),
    "GBRG": getattr(cv2, "COLOR_BayerGB2BGR_EA", cv2.COLOR_BayerGB2BGR),
}


def infer_stride(raw_size, width, height):
    if raw_size % height != 0:
        raise ValueError(f"Cannot infer stride: raw_size={raw_size}, height={height}")
    stride = raw_size // height
    if stride < width:
        raise ValueError(f"Invalid stride={stride}, width={width}")
    return stride


def unpack_rg10(raw16):
    max_val = int(raw16.max())
    print(f"raw16 min={raw16.min()}, max={raw16.max()}")

    # 有些驱动把 10-bit 数据放在低 10 bit，有些会左对齐到高位。
    # 如果最大值明显大于 10-bit 范围，先右移 6 bit。
    if max_val > 4095:
        print("Detected likely left-aligned 10-bit data, applying >> 6")
        raw10 = raw16 >> 6
    else:
        print("Detected likely low-bit 10-bit data, no shift")
        raw10 = raw16

    raw10 = np.clip(raw10, 0, 1023).astype(np.uint16)
    print(f"raw10 min={raw10.min()}, max={raw10.max()}")
    return raw10


def normalize_to_u8(raw10):
    lo = np.percentile(raw10, 0.5)
    hi = np.percentile(raw10, 99.5)

    print(f"percentile lo={lo:.2f}, hi={hi:.2f}")

    if hi <= lo:
        lo = float(raw10.min())
        hi = float(raw10.max())

    raw = np.clip(raw10.astype(np.float32), lo, hi)
    raw8 = ((raw - lo) / (hi - lo + 1e-6) * 255.0).astype(np.uint8)
    return raw8


def gray_world_wb(bgr):
    bgr_f = bgr.astype(np.float32)
    means = bgr_f.reshape(-1, 3).mean(axis=0)
    gray = means.mean()

    scale = gray / (means + 1e-6)
    bgr_f *= scale.reshape(1, 1, 3)
    return np.clip(bgr_f, 0, 255).astype(np.uint8)


def sharpen(bgr):
    blur = cv2.GaussianBlur(bgr, (0, 0), 1.0)
    sharp = cv2.addWeighted(bgr, 1.5, blur, -0.5, 0)
    return sharp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--outdir", default="bayer_debug")
    parser.add_argument("--stride-pixels", type=int, default=None)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    raw16_flat = np.fromfile(args.input, dtype=np.uint16)

    stride = args.stride_pixels
    if stride is None:
        stride = infer_stride(raw16_flat.size, args.width, args.height)

    print(f"width={args.width}, height={args.height}, stride_pixels={stride}")

    raw16 = raw16_flat.reshape(args.height, stride)[:, : args.width]
    raw10 = unpack_rg10(raw16)
    raw8 = normalize_to_u8(raw10)

    cv2.imwrite(str(outdir / "00_raw_gray.jpg"), raw8)

    for name, code in BAYER_CODES.items():
        bgr = cv2.cvtColor(raw8, code)
        wb = gray_world_wb(bgr)
        sharp = sharpen(wb)

        cv2.imwrite(str(outdir / f"01_{name}_plain.jpg"), bgr)
        cv2.imwrite(str(outdir / f"02_{name}_white_balance.jpg"), wb)
        cv2.imwrite(str(outdir / f"03_{name}_wb_sharp.jpg"), sharp)

    for name, code in BAYER_CODES_EA.items():
        bgr = cv2.cvtColor(raw8, code)
        wb = gray_world_wb(bgr)
        sharp = sharpen(wb)

        cv2.imwrite(str(outdir / f"04_{name}_EA_plain.jpg"), bgr)
        cv2.imwrite(str(outdir / f"05_{name}_EA_white_balance.jpg"), wb)
        cv2.imwrite(str(outdir / f"06_{name}_EA_wb_sharp.jpg"), sharp)

    print(f"Saved debug images to: {outdir}")


if __name__ == "__main__":
    main()
