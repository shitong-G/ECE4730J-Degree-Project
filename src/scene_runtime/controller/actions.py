"""Runtime action dataclass emitted by the decision controller."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RuntimeAction:
    """
    Runtime configuration applied to inference and system resources.

    Attributes
    ----------
    mode:
        High-level mode label, e.g. ``balanced``, ``low_power``, ``high_quality``.
    input_resolution:
        Square input side length for detector preprocess.
    inference_interval:
        Run inference every N frames (1 = every frame).
    cpu_threads:
        ONNX Runtime / OpenMP thread count.
    cpu_affinity:
        Optional list of CPU core indices to pin.
    governor:
        Optional cpufreq governor name (applied externally on Pi).
    decoder_layers:
        Optional RT-DETR decoder layer count when model supports dynamic depth.
    query_budget:
        Optional DETR query budget when supported by exported ONNX.
    """

    mode: str
    input_resolution: int
    inference_interval: int
    cpu_threads: int
    cpu_affinity: list[int] | None = None
    governor: str | None = None
    decoder_layers: int | None = None
    query_budget: int | None = None
