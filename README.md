# Dynamic RT-DETR Inference with Scene-Thermal Co-Adaptation on Raspberry Pi

Edge vision system that runs **RT-DETR** object detection on Raspberry Pi under a **scene-aware and thermal-aware embedded runtime manager**. The research contribution is the adaptive runtime—not a new detector architecture.

## Motivation

Fixed inference settings waste power on simple scenes and under-serve complex ones. Raspberry Pi devices also throttle under heat. This project **co-adapts** inference interval, resolution, CPU threads, and related knobs using:

- **Scene workload** (light / medium / heavy) from cheap visual + detection-history signals
- **Device state** (temperature, frequency, throttling, FPS, latency)

Upstream detector: [RT-DETR](https://github.com/lyuwenyu/RT-DETR) (cloned to `third_party/RT-DETR`, not vendored in repo root).

## Architecture

### Per-frame runtime workflow (backbone — `RuntimeLoop`)

Each frame follows this order (implemented in `src/scene_runtime/runtime/loop.py`):

| Step | Action | Code module |
|------|--------|-------------|
| 1 | Capture current frame | `FrameSource` / camera / video |
| 2 | Extract lightweight scene workload features | `SceneWorkloadEstimator` |
| 3 | Read Raspberry Pi device state (SoC temp, freq, …) | `DeviceStateMonitor` |
| 4 | Classify runtime state (scene × thermal) | `RuntimeDecisionController.classify_runtime_state()` |
| 5 | Select runtime action (schedule + query/layer budget) | `RuntimeDecisionController.decide()` → `RuntimeAction` |
| 6 | Run RT-DETR inference **or** skip/reuse per `inference_interval` | `ONNXRTDETREngine` |
| 7 | Log performance; update history for next decision | `RuntimeLogger`, `DetectionHistory`, `MetricsTracker` |

```
Frame in
  → [2] Scene features / workload
  → [3] SoC state (temp feedback re-read each frame)
  → [4] Runtime state
  → [5] RuntimeAction
  → [6] RT-DETR (ONNX) or skip
  → [7] CSV log + metrics
```

### Mapping to the thesis figure (RT-DETR + Co-Adaptation)

The diagram’s **in-model** blocks (Backbone, Encoder, Decoder) live inside RT-DETR/ONNX. This repo implements the **edge runtime manager** that drives them:

| Figure block | Repo backbone (now) | Feature TODO |
|--------------|---------------------|--------------|
| Scene Complexity (MLP) | Lightweight OpenCV/NumPy features + `classify_workload()` stub | Calibrated scene MLP/rules (Member 1) |
| SoC Temp Sensor + feedback | `DeviceStateMonitor` sysfs / `vcgencmd` | Pi validation, apply governor/affinity (Member 2) |
| Layer Router & Schedule | `RuntimeAction.decoder_layers`, `inference_interval`, … | Scene×thermal layer schedule (Member 3) |
| Uncertainty-Minimal Query (Top-K) | `RuntimeAction.query_budget` (logged, not applied) | Top-K / query budget in ONNX (Member 4) |
| Dynamic Decoder (skip L4–6 if simple) | `decoder_layers` field in action + logs | Early-exit / partial decoder export (Member 4) |
| Raspberry Pi edge deploy | `RuntimeLoop` + configs | Full Pi experiments (Member 4) |

SoC temperature measured at Step 3 feeds Step 4 on the **next** frame (closed-loop thermal feedback), matching the dashed feedback arrow in the figure.

## Project status

### Backbone (done)

Infrastructure only — enough to run **dry-run** experiments and develop in parallel:

| Area | Backbone deliverable |
|------|----------------------|
| Repo & docs | Layout, configs, 7 strategy YAMLs, protocols, contribution guide |
| Scene | `SceneWorkloadEstimator` API, visual feature helpers (OpenCV/NumPy), `DetectionHistory` |
| Device | `DeviceStateMonitor` API, sysfs / `vcgencmd` readers (graceful off-Pi) |
| Controller | `RuntimeAction`, fixed-strategy YAML loading, adaptive path → safe defaults |
| Inference | `BaseInferenceEngine`, `ONNXRTDETREngine` skeleton, **dry-run** fake detections |
| Runtime | `RuntimeLoop`, CSV logger schema, CLI scripts, minimal unit tests |

### Features (team TODO)

Real functionality for the thesis — **not implemented** in the backbone (see code `TODO(Member N)`).

## TODO — 4-member split

| Member | Branch | Features to implement |
|--------|--------|------------------------|
| **1 — Scene** | `feature/scene-estimator` | Calibrated `classify_workload()` (light/medium/heavy); threshold tuning on labeled videos; optional learned classifier; scene-only strategy validation |
| **2 — Device** | `feature/device-monitor` | Pi4/Pi5 validation of temp/freq/throttling; apply **governor** & **CPU affinity** from `RuntimeAction`; per-board thermal YAML; optional power (INA219) |
| **3 — Controller** | `feature/controller` | **Layer Router & Schedule**: scene×thermal → `decoder_layers`, `query_budget`, interval, resolution; implement `classify_runtime_state` hints; tune all 7 strategies |
| **4 — Inference & experiments** | `feature/inference-engine`, `experiment/pi4-baseline` | RT-DETR ONNX + **Dynamic Decoder** / **Top-K query** from `RuntimeAction`; real `postprocess.py`; Pi runs & evaluation |

**Suggested order:** Member 4 (ONNX) → Members 1 & 2 (parallel) → Member 3 (policies) → Member 4 (full evaluation).

**Shared:** `pytest`, one `--dry-run` smoke per PR; do not change log CSV columns without team agreement.

## Repository Layout

```
repo-root/
  README.md
  configs/              # Device profiles + experiment strategies
  docs/                 # Setup, protocols, contribution guide
  third_party/          # RT-DETR clone instructions
  scripts/              # Setup, experiment, plotting CLIs
  src/scene_runtime/    # Core Python package
  experiments/          # Protocols, logs, results
  tests/
```

## Quick Start (Laptop Dry-Run)

```bash
# 1. Environment
bash scripts/setup_env.sh
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 2. (Optional) Clone RT-DETR for future ONNX export
bash scripts/clone_rtdetr.sh

# 3. Run 1-minute dry-run experiment
python scripts/run_experiment.py --dry-run --strategy scene_thermal_coadaptive --duration-min 1

# 4. Plot example or your log
python scripts/plot_results.py --input experiments/logs/example.csv
```

## Run Experiments
```bash
python scripts/run_experiment.py --config configs/raspberry_pi4.yaml --strategy native_rtdetr --video data/sample.mp4 --duration-min 15
```
For detailed Experiment intructions, see Experiments.md 

## Logs
Dry-run mode simulates inference latency and fake detections—no ONNX model required.

## Raspberry Pi Deployment

See [docs/setup_raspberry_pi.md](docs/setup_raspberry_pi.md).

```bash
python scripts/run_experiment.py \
  --config configs/raspberry_pi4.yaml \
  --strategy scene_thermal_coadaptive \
  --video data/sample.mp4 \
  --duration-min 15
```

Export ONNX first: `bash scripts/export_model_onnx.sh` (see `third_party/README.md`).

## Experiment Strategies

| Strategy | Description |
|----------|-------------|
| `native_rtdetr` | Non-adaptive RT-DETR baseline |
| `default` | Balanced adaptive policy |
| `static_affinity` | Fixed config + CPU affinity |
| `fixed_low_power` | Static low power |
| `fixed_frame_skip` | Static frame skip |
| `thermal_only` | Thermal adaptation only |
| `scene_only` | Scene adaptation only |
| `scene_thermal_coadaptive` | **Proposed** full co-adaptation |

Configs: `configs/strategies/<name>.yaml`

Fixed strategies (`fixed_*`, `static_affinity`) use YAML values end-to-end. Adaptive strategies (`thermal_only`, `scene_only`, `scene_thermal_coadaptive`, `default`) run through placeholder balanced defaults until Member 3 implements policies.

## Log Format

CSV columns (see `src/scene_runtime/runtime/logger.py`):

`timestamp`, `frame_id`, `strategy`, `workload`, `temp_c`, `freq_mhz_avg`, `power_w`, `latency_ms`, `fps`, `input_resolution`, `inference_interval`, `cpu_threads`, `governor`, `decoder_layers`, `query_budget`, `detection_count`, `confidence_mean`

## Clone RT-DETR

```bash
bash scripts/clone_rtdetr.sh
# => third_party/RT-DETR/
```

Do **not** copy RT-DETR sources into `src/`. Use exported ONNX with `ONNXRTDETREngine`.

## Development

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

### Branch Naming

- `feature/scene-estimator`
- `feature/device-monitor`
- `feature/controller`
- `feature/inference-engine`
- `experiment/pi4-baseline`

### Commit Messages

`feat:`, `fix:`, `docs:`, `exp:`, `refactor:`

### Issue Labels

`module:scene`, `module:device`, `module:controller`, `module:inference`, `module:experiment`, `priority:high`, `good-first-issue`

See [docs/contribution_guide.md](docs/contribution_guide.md).

## Team Collaboration Workflow

1. Pick a module branch and open a PR against `main`
2. Keep hardware code in `src/scene_runtime/device/`
3. Run `pytest` and a `--dry-run` smoke test before review
4. Archive experiment CSV + config YAML with results under `experiments/results/`
5. Use protocol YAMLs in `experiments/protocols/` for reproducible durations

## License

MIT — see [LICENSE](LICENSE).