#!/usr/bin/env python3
"""Check OS support for applying runtime governor and affinity actions."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scene_runtime.controller.actions import RuntimeAction
from scene_runtime.device.action_applier import RuntimeActionApplier


CPU_ROOT = Path("/sys/devices/system/cpu")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe runtime action support")
    parser.add_argument("--governor", default="performance")
    parser.add_argument("--affinity", default="0,1,2,3")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Try to apply requested governor and affinity, then read back state",
    )
    return parser.parse_args()


def _read(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return None


def _print_cpu_state() -> None:
    paths = sorted(CPU_ROOT.glob("cpu[0-9]*/cpufreq/scaling_governor"))
    if not paths:
        print("cpufreq governor sysfs: not found")
        return

    print("cpufreq governor sysfs:")
    for governor_path in paths:
        cpu = governor_path.parts[-3]
        available = _read(governor_path.with_name("scaling_available_governors"))
        current = _read(governor_path)
        writable = os.access(governor_path, os.W_OK)
        print(
            f"  {cpu}: current={current} writable={writable} "
            f"available={available}"
        )


def _print_identity() -> None:
    euid = getattr(os, "geteuid", lambda: None)()
    user = os.environ.get("USER") or os.environ.get("USERNAME") or "unknown"
    print(f"user={user} euid={euid}")
    print(f"platform={sys.platform}")


def _parse_affinity(text: str) -> list[int]:
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def main() -> None:
    args = parse_args()
    _print_identity()
    _print_cpu_state()

    if hasattr(os, "sched_getaffinity"):
        try:
            current = sorted(os.sched_getaffinity(0))
            print("process affinity:", ",".join(str(cpu) for cpu in current))
        except OSError as exc:
            print(f"process affinity: read failed: {exc}")
    else:
        print("process affinity: sched_getaffinity unavailable")

    if not args.apply:
        print("dry probe only; pass --apply to test writes")
        return

    action = RuntimeAction(
        mode="probe",
        input_resolution=480,
        inference_interval=1,
        cpu_threads=4,
        governor=args.governor,
        cpu_affinity=_parse_affinity(args.affinity),
    )
    state = RuntimeActionApplier(enabled=True).apply(action)
    print("apply result:")
    for key, value in state.__dict__.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
