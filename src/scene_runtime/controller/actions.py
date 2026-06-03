"""Runtime action dataclass emitted by the decision controller."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RuntimeAction:
    """
    Step 5 output — runtime knobs applied at Step 6 (inference / skip).

    Figure mapping (Scene-Thermal Co-Adaptation on Raspberry Pi):
    - ``decoder_layers`` → **Dynamic Decoder** depth (e.g. skip layers 4–6 if scene simple)
    - ``query_budget`` → **Uncertainty-Minimal Query Selection (Top-K)**
    - ``inference_interval``, ``input_resolution``, ``cpu_*``, ``governor`` → schedule /
      edge deploy resources (outside the RT-DETR diagram block, same runtime manager)
    """

    mode: str
    input_resolution: int
    inference_interval: int
    cpu_threads: int
    cpu_affinity: list[int] | None = None
    governor: str | None = None
    decoder_layers: int | None = None
    query_budget: int | None = None
