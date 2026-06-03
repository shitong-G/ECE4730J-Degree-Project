# Project Overview

## Title

**Dynamic RT-DETR Inference with Scene-Thermal Co-Adaptation on Raspberry Pi**

## Motivation

Edge devices such as the Raspberry Pi must run object detection under varying scene complexity and thermal constraints. A fixed inference configuration wastes power on simple scenes or degrades accuracy on busy scenes. This project implements a **scene-aware and thermal-aware embedded runtime manager** that dynamically adjusts inference and system resources while using RT-DETR as the representative detection workload.

The main contribution is **not** a new detector architecture, but an **adaptive runtime** that co-adapts to:

1. **Scene workload** — estimated from low-cost visual and detection-history signals.
2. **Device state** — CPU temperature, frequency, throttling, latency, and FPS.

## Architecture

```
Camera / Video
    ↓
Scene Workload Estimator
    ↓
Device State Monitor
    ↓
Runtime Decision Controller
    ↓
Runtime Action
    ↓
RT-DETR Inference Engine
    ↓
Logs + Metrics
```

## Modules

| Module | Responsibility |
|--------|----------------|
| Scene Workload Estimator | Classify scene as light / medium / heavy |
| Device State Monitor | Read Pi thermal and CPU state |
| Runtime Decision Controller | Map state → runtime configuration |
| RT-DETR Inference Engine | ONNX Runtime detection |
| Runtime Loop | Orchestrate pipeline and logging |

## Upstream Dependency

RT-DETR is referenced via `third_party/RT-DETR` (git clone) or exported ONNX models. See [third_party/README.md](../third_party/README.md).

## Experiment Strategies

Seven compared strategies are defined under `configs/strategies/`:

- `default`, `static_affinity`, `fixed_low_power`, `fixed_frame_skip`
- `thermal_only`, `scene_only`, `scene_thermal_coadaptive`

See [experiment_protocol.md](experiment_protocol.md) for run procedures.
