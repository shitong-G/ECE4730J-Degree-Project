import argparse
import cv2
import numpy as np


def read_rg10_raw(path, width, height):
    raw_flat = np.fromfile(path, dtype=np.uint16)

    if raw_flat.size % height != 0:
        raise ValueError(
            f"Cannot infer stride: raw values={raw_flat.size}, height={height}"
        )

    stride_pixels = raw_flat.size // height
    if stride_pixels < width:
        raise ValueError(f"stride_pixels={stride_pixels} < width={width}")

    raw16 = raw_flat.reshape(height, stride_pixels)[:, :width]

    print(f"width={width}, height={height}, stride_pixels={stride_pixels}")
    print(f"raw16 min={raw16.min()}, max={raw16.max()}")

    if raw16.max() > 4095:
        print("Detected left-aligned 10-bit data, using raw16 >> 6")
        raw10 = raw16 >> 6
    else:
        print("Detected low-bit 10-bit data, using raw16 directly")
        raw10 = raw16

    return raw10.astype(np.float32)


def apply_gamma(rgb01, gamma):
    rgb01 = np.clip(rgb01, 0.0, 1.0)
    return np.power(rgb01, 1.0 / gamma)


def boost_saturation(bgr8, saturation):
    hsv = cv2.cvtColor(bgr8, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] *= saturation
    hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def sharpen(bgr8, amount):
    if amount <= 0:
        return bgr8
    blur = cv2.GaussianBlur(bgr8, (0, 0), 1.0)
    return cv2.addWeighted(bgr8, 1.0 + amount, blur, -amount, 0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--output", default="preview_lite_isp.jpg")

    parser.add_argument("--black-level", type=float, default=64.0)
    parser.add_argument("--r-gain", type=float, default=1.0)
    parser.add_argument("--g-gain", type=float, default=1.0)
    parser.add_argument("--b-gain", type=float, default=1.0)

    parser.add_argument("--low-percentile", type=float, default=0.5)
    parser.add_argument("--high-percentile", type=float, default=99.5)
    parser.add_argument("--gamma", type=float, default=2.2)
    parser.add_argument("--saturation", type=float, default=1.6)
    parser.add_argument("--sharpen", type=float, default=0.2)
    parser.add_argument("--resize-original", action="store_true")

    args = parser.parse_args()

    raw10 = read_rg10_raw(args.input, args.width, args.height)

    # black level correction
    raw10 = np.clip(raw10 - args.black_level, 0, 1023)

    # SRGGB10_1X10:
    # row0: R G R G ...
    # row1: G B G B ...
    R = raw10[0::2, 0::2]
    G1 = raw10[0::2, 1::2]
    G2 = raw10[1::2, 0::2]
    B = raw10[1::2, 1::2]
    G = 0.5 * (G1 + G2)

    # manual gains
    R = R * args.r_gain
    G = G * args.g_gain
    B = B * args.b_gain

    # merge as BGR because OpenCV uses BGR
    bgr = np.stack([B, G, R], axis=-1)

    print(f"B mean={B.mean():.2f}, G mean={G.mean():.2f}, R mean={R.mean():.2f}")

    # use one shared normalization range to preserve color ratio
    lo = np.percentile(bgr, args.low_percentile)
    hi = np.percentile(bgr, args.high_percentile)
    print(f"normalize lo={lo:.2f}, hi={hi:.2f}")

    bgr01 = np.clip((bgr - lo) / (hi - lo + 1e-6), 0.0, 1.0)

    # raw linear -> display-like gamma
    bgr01 = apply_gamma(bgr01, args.gamma)

    bgr8 = (bgr01 * 255.0).astype(np.uint8)

    # saturation boost
    bgr8 = boost_saturation(bgr8, args.saturation)

    # slight sharpening
    bgr8 = sharpen(bgr8, args.sharpen)

    if args.resize_original:
        bgr8 = cv2.resize(
            bgr8,
            (args.width, args.height),
            interpolation=cv2.INTER_CUBIC,
        )

    cv2.imwrite(args.output, bgr8)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
