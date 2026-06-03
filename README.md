# Dynamic RT-DETR Inference with Scene-Thermal Co-Adaptation on Raspberry Pi

Edge vision system that runs **RT-DETR** object detection on Raspberry Pi under a **scene-aware and thermal-aware embedded runtime manager**. The research contribution is the adaptive runtime—not a new detector architecture.

## Motivation

Fixed inference settings waste power on simple scenes and under-serve complex ones. Raspberry Pi devices also throttle under heat. This project **co-adapts** inference interval, resolution, CPU threads, and related knobs using:

- **Scene workload** (light / medium / heavy) from cheap visual + detection-history signals
- **Device state** (temperature, frequency, throttling, FPS, latency)

Upstream detector: [RT-DETR](https://github.com/lyuwenyu/RT-DETR) (cloned to `third_party/RT-DETR`, not vendored in repo root).

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
| **3 — Controller** | `feature/controller` | Co-adaptive rules in `runtime_controller.py` (scene × thermal); wire actions into loop; tune all 7 strategies; latency-aware rules via `recent_metrics` |
| **4 — Inference & experiments** | `feature/inference-engine`, `experiment/pi4-baseline` | RT-DETR clone + ONNX export; real `postprocess.py`; ONNX thread/options from action; Pi real-video runs; 15/30/60 min campaigns; analysis plots & report tables |

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
