import argparse
from pathlib import Path

import cv2
import numpy as np


def normalize_u8(x, lo_pct=0.5, hi_pct=99.5):
    x = x.astype(np.float32)
    lo = np.percentile(x, lo_pct)
    hi = np.percentile(x, hi_pct)
    if hi <= lo:
        lo, hi = float(x.min()), float(x.max())
    y = np.clip(x, lo, hi)
    return ((y - lo) / (hi - lo + 1e-6) * 255.0).astype(np.uint8)


def gray_world_bgr(bgr):
    f = bgr.astype(np.float32)
    means = f.reshape(-1, 3).mean(axis=0)
    gray = means.mean()
    gains = gray / (means + 1e-6)
    out = f * gains.reshape(1, 1, 3)
    return np.clip(out, 0, 255).astype(np.uint8), means, gains


def stats(name, x):
    return (
        f"{name}: "
        f"min={int(x.min())}, "
        f"p1={np.percentile(x, 1):.2f}, "
        f"mean={x.mean():.2f}, "
        f"p99={np.percentile(x, 99):.2f}, "
        f"max={int(x.max())}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--outdir", default="bayer_channel_check")
    parser.add_argument("--black-level", type=int, default=64)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    raw_flat = np.fromfile(args.input, dtype=np.uint16)

    if raw_flat.size % args.height != 0:
        raise ValueError(
            f"Cannot infer stride: total uint16 values={raw_flat.size}, "
            f"height={args.height}"
        )

    stride = raw_flat.size // args.height
    print(f"width={args.width}, height={args.height}, stride_pixels={stride}")

    raw16 = raw_flat.reshape(args.height, stride)[:, : args.width]

    print(stats("raw16", raw16))

    if raw16.max() > 4095:
        raw10 = raw16 >> 6
        print("bit alignment: raw16 >> 6")
    else:
        raw10 = raw16.copy()
        print("bit alignment: raw16 low 10-bit")

    raw10 = np.clip(raw10.astype(np.int32) - args.black_level, 0, 1023).astype(np.uint16)

    print(stats("raw10_after_black_level", raw10))

    # media-ctl 显示 SRGGB10_1X10，理论排列：
    # row0: R G R G ...
    # row1: G B G B ...
    R = raw10[0::2, 0::2]
    G1 = raw10[0::2, 1::2]
    G2 = raw10[1::2, 0::2]
    B = raw10[1::2, 1::2]
    G = ((G1.astype(np.float32) + G2.astype(np.float32)) * 0.5).astype(np.uint16)

    lines = [
        f"stride_pixels={stride}",
        stats("R", R),
        stats("G1", G1),
        stats("G2", G2),
        stats("G", G),
        stats("B", B),
    ]

    print("\n".join(lines))
    (outdir / "channel_stats.txt").write_text("\n".join(lines))

    cv2.imwrite(str(outdir / "00_raw10_gray.jpg"), normalize_u8(raw10))
    cv2.imwrite(str(outdir / "01_R_plane.jpg"), normalize_u8(R))
    cv2.imwrite(str(outdir / "02_G1_plane.jpg"), normalize_u8(G1))
    cv2.imwrite(str(outdir / "03_G2_plane.jpg"), normalize_u8(G2))
    cv2.imwrite(str(outdir / "04_B_plane.jpg"), normalize_u8(B))

    # 直接用 Bayer 四个子通道拼一个半分辨率彩色图，绕过 OpenCV Bayer code 的歧义
    R8 = normalize_u8(R)
    G8 = normalize_u8(G)
    B8 = normalize_u8(B)

    direct_bgr = cv2.merge([B8, G8, R8])
    cv2.imwrite(str(outdir / "10_direct_planes_color_no_wb.jpg"), direct_bgr)

    direct_bgr_wb, means, gains = gray_world_bgr(direct_bgr)
    cv2.imwrite(str(outdir / "11_direct_planes_color_gray_world_wb.jpg"), direct_bgr_wb)

    with open(outdir / "direct_color_stats.txt", "w") as f:
        f.write(f"BGR means before WB: {means}\n")
        f.write(f"BGR gains: {gains}\n")

    # 同时输出 OpenCV demosaic 的几个候选，方便对比
    raw8 = normalize_u8(raw10)

    candidates = {
        "BayerRG2BGR": cv2.COLOR_BayerRG2BGR,
        "BayerBG2BGR": cv2.COLOR_BayerBG2BGR,
        "BayerGR2BGR": cv2.COLOR_BayerGR2BGR,
        "BayerGB2BGR": cv2.COLOR_BayerGB2BGR,
    }

    for name, code in candidates.items():
        bgr = cv2.cvtColor(raw8, code)
        bgr_wb, _, _ = gray_world_bgr(bgr)
        cv2.imwrite(str(outdir / f"20_{name}_no_wb.jpg"), bgr)
        cv2.imwrite(str(outdir / f"21_{name}_gray_world_wb.jpg"), bgr_wb)

    print(f"Saved results to {outdir}")


if __name__ == "__main__":
    main()
