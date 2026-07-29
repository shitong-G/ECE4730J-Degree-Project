#!/usr/bin/env python3
"""Print setup and same-video validation commands for baseline detectors."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover - depends on caller environment
    yaml = None


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "configs" / "sota_baselines.yaml"


def load_manifest(path: Path) -> dict[str, Any]:
    if yaml is None:
        raise SystemExit(
            "PyYAML is required to read configs/sota_baselines.yaml. "
            "Install project requirements with: python -m pip install -r requirements.txt"
        )
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid manifest: {path}")
    return data


def print_block(title: str, lines: list[str]) -> None:
    print(f"\n## {title}")
    for line in lines:
        print(line)


def model_validation_command(
    model_key: str,
    model: dict[str, Any],
    video: str,
    output_root: str,
    host: str,
    threads: int,
) -> list[str]:
    size = int(model["recommended_input_size"])
    artifacts = model.get("artifacts", {})
    prefix = f"{host}_{model_key}"
    family = model.get("family", "")

    if family == "ultralytics-yolo":
        return [
            "yolo predict \\",
            f"  model={artifacts.get('onnx')} \\",
            f"  source={video} \\",
            f"  imgsz={size} \\",
            "  device=cpu \\",
            f"  project={output_root} \\",
            f"  name={prefix} \\",
            "  save=True",
        ]

    if family == "paddle-picodet":
        return [
            "python third_party/PaddleDetection/deploy/python/infer.py \\",
            f"  --model_dir={artifacts.get('paddle_inference_dir')} \\",
            f"  --video_file={video} \\",
            "  --device=CPU \\",
            f"  --cpu_threads={threads} \\",
            f"  --output_dir={output_root}/{prefix}",
        ]

    if family == "nanodet-plus":
        return [
            "python third_party/nanodet/demo/demo.py video \\",
            f"  --config {model.get('pretrained', {}).get('config')} \\",
            f"  --model {artifacts.get('torch')} \\",
            f"  --path {video}",
        ]

    return [f"# No validation command registered for {model_key}"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--host", choices=["local", "pi"], default="local")
    parser.add_argument("--model", default="all", help="Model key or 'all'.")
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    video = str(manifest.get("video", "data/sample.mp4"))
    output_root = str(
        manifest.get("visualization_root", "experiments/visualizations/sota_baselines")
    )
    threads = int(manifest.get("default_threads", 4))
    models = manifest.get("baselines", {})
    if args.model != "all":
        models = {args.model: models[args.model]}

    print(f"# Host: {args.host}")
    print(f"# Video: {video}")

    for key, model in models.items():
        name = model.get("display_name", key)
        print_block(
            f"{name} Artifacts",
            [
                f"# source: {model.get('source_repo')}",
                f"# preferred local backend: {model.get('local_backend', {}).get('preferred')}",
                f"# preferred Pi backend: {model.get('raspberry_pi_backend', {}).get('preferred')}",
                f"# artifacts: {model.get('artifacts')}",
            ],
        )
        print_block(f"{name} Export", list(model.get("export_commands", [])))
        print_block(
            f"{name} Validate",
            model_validation_command(
                key,
                model,
                video,
                output_root,
                args.host,
                threads,
            ),
        )


if __name__ == "__main__":
    main()
