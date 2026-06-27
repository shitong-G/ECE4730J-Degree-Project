# Dynamic RT-DETR Inference with Scene-Thermal Co-Adaptation on Raspberry Pi

Edge vision runtime for running **RT-DETR** object detection on Raspberry Pi under a scene-aware and thermal-aware controller. The project contribution is the embedded runtime manager and experiment pipeline around RT-DETR, not a new detector architecture.

## Current Status

The project is past the dry-run skeleton stage:

| Area | Status |
| --- | --- |
| ONNX inference | Real ONNX Runtime inference is implemented in `src/scene_runtime/inference/onnx_engine.py` via `onnxruntime.InferenceSession(...).run(...)`. |
| Models | `models/rtdetr_r18_lite_pi4.onnx` and `models/rtdetr_r18_pi4.onnx` are present locally. They are ignored by git. |
| Raspberry Pi data | Real experiment CSV/profile logs and result plots exist under `experiments/logs/` and `experiments/results/`. These folders are mostly ignored by git, so they may not appear in normal repository status. |
| Thermal adaptation | The controller has `normal` / `warm` / `hot` / `critical` states, hysteresis, hold frames, and critical thermal pressure logic. It emits lower resolution, larger inference interval, fewer CPU threads, lower query budget, and governor hints as temperature rises. |
| Runtime knobs that are definitely enforced | `inference_interval` is enforced by the runtime loop and reliably skips inference frames. ONNX input size is enforced only when the exported ONNX model supports the requested shape; otherwise the fixed model input shape wins. |
| Runtime knobs that are still mostly hints | `cpu_threads`, `governor`, `decoder_layers`, and `query_budget` are selected and logged, but they are not yet dynamically applied to ONNX Runtime sessions, OS governor, or an exported partial decoder/query graph. |
| Scene adaptation | `SceneWorkloadEstimator` extracts edge, motion, frame-difference, entropy, and detection-history features, but `classify_workload()` still returns `"medium"` for every frame. Full scene-aware adaptation is therefore not complete. |

In short: **real inference, Pi-side experiment logs, and thermal adaptation are in place; full scene-thermal co-adaptation still needs scene workload classification and deeper runtime knob enforcement.**

## Motivation

Fixed inference settings waste compute on simple scenes and can overheat edge devices under sustained load. This project co-adapts runtime behavior using:

- Scene workload: intended `light` / `medium` / `heavy` classification from cheap visual and detection-history signals.
- Device state: SoC temperature, CPU frequency, throttling state where available, FPS, and latency.
- Runtime actions: input resolution, inference interval, CPU-thread hints, governor hints, decoder-layer hints, and query-budget hints.

Upstream detector: [RT-DETR](https://github.com/lyuwenyu/RT-DETR), cloned separately under `third_party/RT-DETR`.

## Architecture

Implemented per-frame workflow in `src/scene_runtime/runtime/loop.py`:

| Step | Action | Module |
| --- | --- | --- |
| 1 | Capture frame | `FrameSource` / camera / video |
| 2 | Extract lightweight scene features | `SceneWorkloadEstimator` |
| 3 | Read Raspberry Pi state | `DeviceStateMonitor` |
| 4 | Fuse scene and thermal state | `RuntimeDecisionController.classify_runtime_state()` |
| 5 | Select runtime action | `RuntimeDecisionController.decide()` |
| 6 | Run RT-DETR ONNX inference or skip/reuse detections | `ONNXRTDETREngine` |
| 7 | Log metrics and update history | `RuntimeLogger`, `ProfileLogger`, `DetectionHistory` |

```text
Frame
  -> scene features
  -> device state
  -> runtime state
  -> RuntimeAction
  -> ONNX inference or frame skip
  -> CSV/profile logs and next-frame history
```

## Mapping to Thesis Blocks

| Thesis block | Current repository implementation | Remaining work |
| --- | --- | --- |
| Scene Complexity | OpenCV/NumPy features in `SceneWorkloadEstimator` | Implement calibrated `light` / `medium` / `heavy` classification. |
| SoC Temp Sensor + feedback | `DeviceStateMonitor` plus thermal guard in `RuntimeDecisionController` | Validate throttling flags and clock readings across Pi setups. |
| Layer Router & Schedule | `RuntimeAction.decoder_layers`, `inference_interval`, `input_resolution` | Only frame skipping is fully enforced; dynamic decoder export is still needed. |
| Uncertainty-Minimal Query / Top-K | `RuntimeAction.query_budget` | Currently logged as a policy output; not applied inside ONNX graph. |
| Dynamic Decoder | `decoder_layers` field | Requires partial decoder/early-exit ONNX support. |
| Raspberry Pi deployment | `scripts/run_experiment.py`, configs, logs, plots | Final controlled experiment matrix and report figures. |

## Repository Layout

```text
repo-root/
  README.md
  Experiments.md
  PLAN.md
  configs/              # Device profiles and strategy YAMLs
  data/                 # Local videos, ignored by git
  docs/                 # Setup and protocol notes
  experiments/          # Protocols, ignored logs, ignored results
  models/               # Local ONNX models, ignored by git
  scripts/              # Experiment, export, plotting, summary CLIs
  src/scene_runtime/    # Runtime package
  tests/
  third_party/          # RT-DETR clone location
```

## Quick Start

Dry-run smoke test:

```bash
python scripts/run_experiment.py --dry-run --strategy scene_thermal_coadaptive --duration-min 1
```

Real ONNX smoke test on Raspberry Pi:

```bash
python scripts/run_experiment.py \
  --config configs/raspberry_pi4.yaml \
  --strategy fixed_low_power \
  --video data/sample.mp4 \
  --duration-min 1
```

Real baseline run:

```bash
python scripts/run_experiment.py \
  --config configs/raspberry_pi4.yaml \
  --strategy native_rtdetr \
  --video data/sample.mp4 \
  --duration-min 15
```

Plot a run:

```bash
python scripts/plot_results.py --input experiments/logs/<run>.csv
```

Generate synthetic thermal logs for analysis testing:

```bash
python scripts/generate_thermal_test_logs.py \
  --output-dir experiments/logs/synthetic_thermal \
  --duration-sec 900 \
  --strategies native_rtdetr,fixed_frame_skip,fixed_low_power,thermal_only
```

Run the thermal experiment suite on Raspberry Pi:

```bash
python scripts/run_thermal_experiment_suite.py \
  --config configs/raspberry_pi4.yaml \
  --video data/sample.mp4 \
  --loop-video \
  --duration-min 15
```

## Experiment Evidence Already Present Locally

Local ignored artifacts include:

- `data/sample.mp4`
- `models/rtdetr_r18_lite_pi4.onnx`
- `models/rtdetr_r18_pi4.onnx`
- `experiments/logs/baseline_native/`
- `experiments/logs/thermal_stress/`
- `experiments/results/baseline_native/`
- `experiments/results/thermal_stress/`
- timestamped `scene_thermal_coadaptive_*.csv` runs

One verified profile file, `experiments/logs/baseline_native/native_rtdetr_lite_run1_profile.csv`, has real `onnx_run_ms` values with an average around **3224.8 ms** over 1054 profiled inference rows. This confirms the run is not dry-run simulation.

## Strategies

| Strategy | Description | Current reliability |
| --- | --- | --- |
| `native_rtdetr` | Non-adaptive RT-DETR baseline | Reliable fixed baseline |
| `static_affinity` | Fixed runtime with affinity/governor hints | Runtime values logged; OS affinity/governor enforcement should be verified |
| `fixed_low_power` | Static low-power baseline | Reliable for frame skipping and requested resolution |
| `fixed_frame_skip` | Static frame-skipping baseline | Reliable |
| `thermal_only` | Temperature-aware adaptive policy | Mostly functional through thermal controller; non-frame-skip knobs are partly hints |
| `scene_only` | Scene-aware adaptive policy | Blocked by fixed `"medium"` workload classification |
| `scene_thermal_coadaptive` | Intended full co-adaptive strategy | Thermal part works; scene part is not complete |
| `default` | Balanced adaptive policy | Useful fallback/smoke strategy |

Strategy YAML files live under `configs/strategies/`.

## Log Schema

Main CSV logs currently include:

```text
timestamp, frame_id, strategy, workload,
thermal_state, raw_thermal_state, control_thermal_state, action_mode,
temp_c, freq_mhz_avg, arm_clock_mhz, power_w,
throttling_raw, under_voltage, arm_freq_capped, currently_throttled, soft_temp_limit,
did_infer, latency_ms, fps, loop_fps, effective_inference_fps, actual_inference_fps,
input_resolution, inference_interval, cpu_threads, governor,
decoder_layers, query_budget, detection_count, confidence_mean
```

Profile CSV logs include timing breakdowns such as:

```text
frame_total_ms, scene_ms, device_ms, runtime_state_ms, decision_ms,
infer_outer_ms, preprocess_ms, build_feed_ms, onnx_run_ms,
postprocess_ms, infer_total_ms, summary_ms, main_log_write_ms
```

## Important Implementation Notes

- `inference_interval` is the strongest currently enforced adaptive knob.
- `input_resolution` is passed into preprocessing, but fixed-shape ONNX models override requested spatial sizes through `_resolve_input_resolution()`.
- `cpu_threads` should be wired into `onnxruntime.SessionOptions` if it is to affect real inference.
- `governor` and CPU affinity require OS-level application and verification.
- `decoder_layers` and `query_budget` require model/export support before they can change RT-DETR computation.
- `classify_workload()` must stop returning constant `"medium"` before claiming full scene-aware adaptation.

Check ONNX input shape:

```bash
python - <<'PY'
import onnxruntime as ort

sess = ort.InferenceSession("models/rtdetr_r18_lite_pi4.onnx")
for i in sess.get_inputs():
    print(i.name, i.shape)
PY
```

## Next Plan

1. Implement and calibrate scene workload classification in `src/scene_runtime/scene/workload_estimator.py`.
2. Run a short workload validation set with low/medium/high complexity clips and confirm the CSV `workload` column changes.
3. Verify ONNX input shape and decide between dynamic-shape inference or multiple fixed-resolution ONNX exports.
4. Make CPU-thread control real by creating ONNX Runtime sessions with `SessionOptions`, likely preloading sessions for a small set of thread counts instead of rebuilding per frame.
5. Add optional OS-control support for governor and affinity, with clear permission requirements and state verification.
6. Decide whether `decoder_layers` and `query_budget` stay as thesis-level future work or become real via partial decoder/query-budget ONNX exports.
7. Re-run the experiment matrix: `native_rtdetr`, `fixed_frame_skip`, `fixed_low_power`, `thermal_only`, `scene_only`, and `scene_thermal_coadaptive`.
8. Summarize results into tables and figures for README/report: temperature, FPS, ARM clock, `did_infer`, workload, action mode, and `onnx_run_ms`.

## Development

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

See also:

- `Experiments.md` for the experiment operation guide.
- `docs/setup_raspberry_pi.md` for Raspberry Pi setup.
- `docs/experiment_protocol.md` for protocol notes.

## License

MIT, see `LICENSE`.
