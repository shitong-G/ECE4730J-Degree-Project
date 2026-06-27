# Project Overview

## Title

**Dynamic RT-DETR Inference with Scene-Thermal Co-Adaptation on Raspberry Pi**

## Motivation

Edge devices such as the Raspberry Pi must run object detection under varying scene complexity and thermal constraints. This project implements a **scene-aware and thermal-aware embedded runtime manager** around RT-DETR: the runtime observes the scene and SoC temperature, then adjusts frame schedule, requested resolution, decoder/query hints, and edge-resource hints.

The main contribution is **not** a new full detector backbone, but the **co-adaptation control plane** on the Pi. Real ONNX inference and Raspberry Pi experiment logs are already available locally; full scene-aware adaptation still needs workload classification to be completed.

## Per-frame workflow

See [README.md](../README.md) for the full table. Summary:

1. Capture frame  
2. Scene workload features  
3. Device / SoC state  
4. Classify runtime state  
5. Select `RuntimeAction`  
6. RT-DETR infer or skip  
7. Log and update for next frame  

Implemented in `src/scene_runtime/runtime/loop.py`.

## Architecture (control plane)

```
Camera / Video
    ↓
Scene Workload Estimator     ← figure: Scene Complexity
    ↓
Device State Monitor         ← figure: SoC Temp Sensor (+ feedback)
    ↓
Classify runtime state
    ↓
Runtime Decision Controller  ← figure: Layer Router & Schedule
    ↓
RuntimeAction                ← decoder_layers, query_budget, interval, …
    ↓
RT-DETR Inference Engine     ← figure: Dynamic Decoder + Top-K queries (in ONNX)
    ↓
Logs + Metrics               ← feeds next-frame decisions
```

## Modules

| Module | Responsibility |
|--------|----------------|
| Scene Workload Estimator | Lightweight features; workload light/medium/heavy |
| Device State Monitor | SoC temp, frequency, throttling |
| Runtime Decision Controller | Runtime state + Layer Router / schedule |
| RT-DETR Inference Engine | ONNX inference under selected action |
| Runtime Loop | 7-step orchestration |

## Status

**Current:** real ONNX Runtime inference is wired, Pi-side logs/results exist locally, and the thermal controller has normal/warm/hot/critical behavior with hysteresis and hold frames.

**Still open:** `SceneWorkloadEstimator.classify_workload()` returns `medium`, so scene-aware control is not complete. `decoder_layers`, `query_budget`, `cpu_threads`, and governor values are logged as policy outputs until downstream ONNX/OS enforcement is implemented and verified.

## Upstream Dependency

RT-DETR: `third_party/RT-DETR` — see [third_party/README.md](../third_party/README.md).

## Experiment Strategies

Seven strategies under `configs/strategies/`. See [experiment_protocol.md](experiment_protocol.md).
