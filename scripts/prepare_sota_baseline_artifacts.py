#!/usr/bin/env python3
"""Download lightweight baseline detector artifacts used by comparison scripts."""

from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models" / "baselines"


ARTIFACTS = {
    "pp_picodet_s_320": {
        "url": "https://paddledet.bj.bcebos.com/deploy/Inference/picodet_s_320_coco_lcnet.tar",
        "path": MODELS_DIR / "picodet_s_320_coco_lcnet.tar",
        "extract_dir": MODELS_DIR / "picodet_s_320_coco_lcnet_portable",
        "strip_prefix": "picodet_s_320_coco_lcnet/",
        "expected": MODELS_DIR / "picodet_s_320_coco_lcnet_portable" / "infer_cfg.yml",
    },
    "pp_picodet_l_640": {
        "url": "https://paddledet.bj.bcebos.com/deploy/Inference/picodet_l_640_coco_lcnet.tar",
        "path": MODELS_DIR / "picodet_l_640_coco_lcnet.tar",
        "extract_dir": MODELS_DIR / "picodet_l_640_coco_lcnet_portable",
        "strip_prefix": "picodet_l_640_coco_lcnet/",
        "expected": MODELS_DIR / "picodet_l_640_coco_lcnet_portable" / "infer_cfg.yml",
    },
    "nanodet_plus_m_320": {
        "gdrive_id": "1YvuEhahlgqxIhJu7bsL-fhaqubKcCWQc",
        "path": MODELS_DIR / "nanodet-plus-m_320.ckpt",
        "extract_dir": None,
        "expected": MODELS_DIR / "nanodet-plus-m_320.ckpt",
        "fallback_note": (
            "Install gdown and rerun, or download the NanoDet-Plus-m-320 "
            "Checkpoint from the official NanoDet model-zoo Google Drive link."
        ),
    },
    "nanodet_plus_m_320_fastdeploy_onnx": {
        "url": "https://bj.bcebos.com/paddlehub/fastdeploy/nanodet-plus-m_320.onnx",
        "path": MODELS_DIR / "nanodet_plus_m_320_fastdeploy.onnx",
        "extract_dir": None,
        "expected": MODELS_DIR / "nanodet_plus_m_320_fastdeploy.onnx",
    },
}


def download(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        print(f"exists: {path}")
        return
    print(f"download: {url}")
    print(f"      to: {path}")
    with urllib.request.urlopen(url, timeout=120) as response:
        with path.open("wb") as f:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)


def extract_tar(
    path: Path,
    output_dir: Path,
    expected: Path,
    strip_prefix: str = "",
) -> None:
    if expected.exists():
        print(f"exists: {expected}")
        return
    print(f"extract: {path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, "r:*") as tar:
        for member in tar.getmembers():
            name = member.name
            if strip_prefix:
                if not name.startswith(strip_prefix):
                    continue
                name = name[len(strip_prefix):]
            if not name or member.isdir():
                continue
            target = (output_dir / name).resolve()
            output_root = output_dir.resolve()
            if output_root not in target.parents and target != output_root:
                raise RuntimeError(f"Refusing to extract outside output dir: {member.name}")
            target.parent.mkdir(parents=True, exist_ok=True)
            source = tar.extractfile(member)
            if source is None:
                continue
            target.write_bytes(source.read())


def download_gdrive(file_id: str, path: Path, fallback_note: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        print(f"exists: {path}")
        return
    if importlib.util.find_spec("gdown") is None:
        print(f"missing: {path}")
        print(fallback_note)
        print(
            "Command: python -m pip install gdown && "
            f"python -m gdown {file_id} -O {path}"
        )
        return
    command = [sys.executable, "-m", "gdown", file_id, "-O", str(path)]
    print("$ " + " ".join(command))
    subprocess.run(command, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        default="all",
        help="Comma-separated keys or all. Keys: " + ",".join(sorted(ARTIFACTS)),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    keys = sorted(ARTIFACTS) if args.models == "all" else [
        item.strip() for item in args.models.split(",") if item.strip()
    ]
    for key in keys:
        if key not in ARTIFACTS:
            raise KeyError(f"Unknown artifact key: {key}")
        item = ARTIFACTS[key]
        path = Path(item["path"])
        if "gdrive_id" in item:
            download_gdrive(
                str(item["gdrive_id"]),
                path,
                str(item.get("fallback_note", "")),
            )
        else:
            download(str(item["url"]), path)
        extract_dir = item.get("extract_dir")
        if extract_dir is not None:
            extract_tar(
                path,
                Path(extract_dir),
                Path(item["expected"]),
                str(item.get("strip_prefix", "")),
            )


if __name__ == "__main__":
    main()
