import argparse
import numpy as np
import cv2


BAYER_CODES = {
    "RGGB": cv2.COLOR_BayerBG2BGR,
    "BGGR": cv2.COLOR_BayerRG2BGR,
    "GRBG": cv2.COLOR_BayerGB2BGR,
    "GBRG": cv2.COLOR_BayerGR2BGR,
}


def normalize_to_uint8(raw16: np.ndarray) -> np.ndarray:
    # 用百分位拉伸，避免几个异常亮点导致整张图发黑
    lo = np.percentile(raw16, 0.5)
    hi = np.percentile(raw16, 99.5)

    if hi <= lo:
        hi = raw16.max()
        lo = raw16.min()

    raw = np.clip(raw16.astype(np.float32), lo, hi)
    raw8 = ((raw - lo) / (hi - lo + 1e-6) * 255.0).astype(np.uint8)
    return raw8


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--output", default="preview.jpg")
    parser.add_argument("--bayer", default="RGGB", choices=BAYER_CODES.keys())
    parser.add_argument(
        "--stride-pixels",
        type=int,
        default=None,
        help="Pixels per row including padding. If omitted, infer from file size.",
    )
    args = parser.parse_args()

    raw16_flat = np.fromfile(args.input, dtype=np.uint16)

    if args.stride_pixels is None:
        if raw16_flat.size % args.height != 0:
            raise ValueError(
                f"Cannot infer stride: total pixels {raw16_flat.size} "
                f"is not divisible by height {args.height}."
            )
        stride_pixels = raw16_flat.size // args.height
    else:
        stride_pixels = args.stride_pixels

    if stride_pixels < args.width:
        raise ValueError(
            f"Invalid stride_pixels={stride_pixels}, smaller than width={args.width}."
        )

    expected_with_stride = stride_pixels * args.height

    if raw16_flat.size != expected_with_stride:
        raise ValueError(
            f"Size mismatch: got {raw16_flat.size} uint16 values, "
            f"expected {expected_with_stride} = {stride_pixels}*{args.height}."
        )

    print(f"width={args.width}, height={args.height}, stride_pixels={stride_pixels}")
    print(f"raw min={raw16_flat.min()}, max={raw16_flat.max()}")

    raw16_stride = raw16_flat.reshape((args.height, stride_pixels))

    # 裁掉每行右侧 padding
    raw16 = raw16_stride[:, : args.width]

    raw8 = normalize_to_uint8(raw16)

    bgr = cv2.cvtColor(raw8, BAYER_CODES[args.bayer])

    cv2.imwrite(args.output, bgr)
    cv2.imwrite("raw_gray_preview.jpg", raw8)

    print(f"Saved color image to {args.output}")
    print("Saved raw grayscale preview to raw_gray_preview.jpg")


if __name__ == "__main__":
    main()
