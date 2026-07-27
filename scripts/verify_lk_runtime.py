#!/usr/bin/env python3
"""Print the LK tracker code/config actually used by runtime scripts."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scene_runtime.runtime.config import load_config
from scene_runtime.tracking.lk_tracker import SparseLKBoxTracker
import scene_runtime.tracking.lk_tracker as lk_tracker_module


LK_KEYS = [
    "lk_grid_size",
    "lk_robust_mad_multiplier",
    "lk_redetect_interval",
    "lk_redetect_min_points",
    "lk_edge_refresh_margin_ratio",
    "lk_min_refresh_point_span_ratio",
    "lk_edge_exit_frames",
    "lk_edge_exit_min_area_ratio",
    "lk_exit_refresh_min_area_ratio",
    "lk_large_track_refresh_frames",
    "lk_large_track_refresh_area_ratio",
    "lk_quality_confirm_enabled",
    "lk_quality_confirm_frames",
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_report(path: Path) -> None:
    exists = path.exists()
    print(f"{path}: {'exists' if exists else 'missing'}")
    if exists:
        stat = path.stat()
        print(f"  size={stat.st_size}")
        print(f"  sha256={_sha256(path)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "raspberry_pi4.yaml")
    parser.add_argument("--strategy", default="scene_track_lk")
    args = parser.parse_args()

    config = load_config(args.config, args.strategy)
    tracking = config.get("tracking", {})
    module_path = Path(inspect.getfile(lk_tracker_module)).resolve()

    print("Runtime LK verification")
    print("=======================")
    print(f"python={sys.executable}")
    print(f"cwd={Path.cwd()}")
    print(f"repo_root={ROOT}")
    print(f"strategy={args.strategy}")
    print(f"lk_tracker_module={module_path}")
    print()

    print("Files")
    print("-----")
    for relative in [
        "src/scene_runtime/tracking/lk_tracker.py",
        "src/scene_runtime/runtime/loop.py",
        f"configs/strategies/{args.strategy}.yaml",
        "scripts/run_live_dashboard.py",
        "scripts/run_experiment.py",
    ]:
        _file_report(ROOT / relative)
    print()

    source = module_path.read_text(encoding="utf-8")
    print("Required markers")
    print("----------------")
    for marker in [
        "lk_track_exit_or_disappearance",
        "large_track_refresh_frames",
        "large_track_refresh_area_ratio",
        "_has_exit_refresh_failure",
        "_robust_translation",
    ]:
        print(f"{marker}: {'present' if marker in source else 'MISSING'}")
    print()

    print("Effective tracking config")
    print("-------------------------")
    for key in LK_KEYS:
        print(f"{key}={tracking.get(key, '<default>')}")
    print()

    tracker = SparseLKBoxTracker(
        max_corners=int(tracking.get("lk_max_corners", 40)),
        min_valid_points=int(tracking.get("lk_min_valid_points", 5)),
        min_survival_ratio=float(tracking.get("lk_min_survival_ratio", 0.35)),
        max_forward_backward_error=float(tracking.get("lk_max_forward_backward_error", 1.5)),
        max_failure_ratio=float(tracking.get("lk_max_failure_ratio", 0.30)),
        redetect_interval=int(tracking.get("lk_redetect_interval", 5)),
        redetect_min_points=int(tracking.get("lk_redetect_min_points", 8)),
        win_size=int(tracking.get("lk_win_size", 15)),
        max_level=int(tracking.get("lk_max_level", 2)),
        max_iterations=int(tracking.get("lk_max_iterations", 15)),
        grid_size=int(tracking.get("lk_grid_size", 3)),
        robust_mad_multiplier=float(tracking.get("lk_robust_mad_multiplier", 2.5)),
        edge_refresh_margin_ratio=float(tracking.get("lk_edge_refresh_margin_ratio", 0.02)),
        min_refresh_point_span_ratio=float(
            tracking.get("lk_min_refresh_point_span_ratio", 0.18)
        ),
        edge_exit_frames=int(tracking.get("lk_edge_exit_frames", 8)),
        edge_exit_min_area_ratio=float(tracking.get("lk_edge_exit_min_area_ratio", 0.03)),
        exit_refresh_min_area_ratio=float(
            tracking.get("lk_exit_refresh_min_area_ratio", 0.01)
        ),
        large_track_refresh_frames=int(tracking.get("lk_large_track_refresh_frames", 30)),
        large_track_refresh_area_ratio=float(
            tracking.get("lk_large_track_refresh_area_ratio", 0.08)
        ),
    )

    print("Instantiated tracker")
    print("--------------------")
    for attr in [
        "grid_size",
        "robust_mad_multiplier",
        "edge_exit_frames",
        "edge_exit_min_area_ratio",
        "exit_refresh_min_area_ratio",
        "large_track_refresh_frames",
        "large_track_refresh_area_ratio",
    ]:
        print(f"{attr}={getattr(tracker, attr)}")


if __name__ == "__main__":
    main()
