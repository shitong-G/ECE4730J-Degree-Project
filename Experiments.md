# Full Experiment Operation Guide for RT-DETR Scene-Thermal Runtime on Raspberry Pi 4B

## 1. Purpose

This document explains how to run one complete experiment for the RT-DETR scene-thermal runtime project on Raspberry Pi 4B. It also summarizes all currently available runtime strategies and how each strategy is configured.

The project goal is to deploy RT-DETR on Raspberry Pi 4B and evaluate runtime control policies that adapt inference behavior based on:

* device temperature,
* scene workload,
* frame skipping,
* input resolution,
* CPU thread hints,
* CPU governor hints,
* and detection workload.

The current experiment framework supports both real ONNX inference and `--dry-run` simulation.

---

## 2. Repository Structure Relevant to Experiments

The experiment-related files are organized as follows:

```text
configs/
  default.yaml
  raspberry_pi4.yaml
  raspberry_pi4_1gb.yaml
  strategies/
    native_rtdetr.yaml
    default.yaml
    static_affinity.yaml
    fixed_low_power.yaml
    fixed_frame_skip.yaml
    thermal_only.yaml
    scene_only.yaml
    scene_thermal_coadaptive.yaml

scripts/
  run_experiment.py
  plot_results.py
  export_model_onnx.sh

src/
  scene_runtime/
    controller/
    device/
    inference/
    runtime/
    scene/

experiments/
  protocols/
  logs/
  results/
```

The main experiment entry point is:

```bash
python scripts/run_experiment.py
```

The plotting entry point is:

```bash
python scripts/plot_results.py
```

---

## 3. Basic Experiment Pipeline

A full experiment follows this runtime pipeline:

```text
Input video frame
  ↓
Scene workload estimation
  ↓
Device state monitoring
  ↓
Runtime state classification
  ↓
Runtime action decision
  ↓
RT-DETR inference or frame skip
  ↓
Detection summary
  ↓
CSV/profile logging
  ↓
Result plotting
```

Each frame produces one row in the main CSV log. When inference is actually executed, the runtime also records profiling information in a separate profile CSV.

---

## 4. Environment Preparation

### 4.1 Clone the repository

```bash
git clone https://github.com/shitong-G/ECE4730J-Degree-Project.git
cd ECE4730J-Degree-Project
```

### 4.2 Create and activate Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 4.3 Install Python dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

The main dependencies are:

```text
numpy
opencv-python
PyYAML
onnxruntime
matplotlib
pandas
```

### 4.4 Install Raspberry Pi system packages

On Raspberry Pi OS or Ubuntu for Raspberry Pi:

```bash
sudo apt update
sudo apt install -y python3-pip python3-venv libopencv-dev v4l-utils
```

For thermal throttling diagnostics, install:

```bash
sudo apt install -y libraspberrypi-bin
```

This provides `vcgencmd`, which is used to read actual ARM clock and throttling flags.

---

## 5. ONNX Model Preparation

The Raspberry Pi config expects the RT-DETR Lite ONNX model at:

```text
models/rtdetr_r18_lite_pi4.onnx
```

The default export script exports RT-DETRv2 R18 Lite:

```bash
bash scripts/export_model_onnx.sh
```

The export script also supports:

```bash
bash scripts/export_model_onnx.sh lite
bash scripts/export_model_onnx.sh rtdetrv2-s
bash scripts/export_model_onnx.sh v1
```

Recommended model for Raspberry Pi 4B:

```text
RT-DETRv2 R18 Lite
```

Expected output path:

```text
models/rtdetr_r18_lite_pi4.onnx
```

For the 1GB Raspberry Pi 4B, the ONNX model should be exported on a workstation rather than directly on the Pi.

---

## 6. Config Files

### 6.1 `configs/default.yaml`

The default config defines general runtime defaults:

```yaml
project:
  name: scene-runtime
  strategy: default

runtime:
  default_input_resolution: 640
  default_inference_interval: 1
  default_cpu_threads: 4
  dry_run_latency_ms: 45.0

thermal:
  normal_max_c: 65.0
  warm_max_c: 75.0

scene:
  light_edge_density_max: 0.08
  light_motion_max: 0.05
  heavy_edge_density_min: 0.18
  heavy_motion_min: 0.15
  heavy_detection_count_min: 8

logging:
  output_dir: experiments/logs
  format: csv

inference:
  model_path: null
  onnx_providers: ["CPUExecutionProvider"]
```

This config is the base profile. Device-specific configs extend it.

---

### 6.2 `configs/raspberry_pi4.yaml`

This is the main Raspberry Pi 4 profile:

```yaml
device:
  platform: raspberry_pi4
  cpu_count: 4

runtime:
  default_input_resolution: 480
  default_inference_interval: 2
  default_cpu_threads: 3

thermal:
  normal_max_c: 60.0
  warm_max_c: 72.0

inference:
  model_path: models/rtdetr_r18_lite_pi4.onnx
```

This means:

```text
normal: temp < 60°C
warm:   60°C <= temp < 72°C
hot:    temp >= 72°C
```

The default runtime action under this config is:

```text
input_resolution = 480
inference_interval = 2
cpu_threads = 3
```

---

### 6.3 `configs/raspberry_pi4_1gb.yaml`

This is the constrained-memory Raspberry Pi 4 profile:

```yaml
device:
  platform: raspberry_pi4
  cpu_count: 4

runtime:
  default_input_resolution: 320
  default_inference_interval: 4
  default_cpu_threads: 2

thermal:
  normal_max_c: 58.0
  warm_max_c: 68.0

inference:
  model_path: models/rtdetr_r18_lite_pi4.onnx
  onnx_providers: ["CPUExecutionProvider"]
```

This profile is designed for memory-constrained runs.

Use it when:

```text
- Raspberry Pi has only 1GB RAM,
- ONNX Runtime is close to memory limit,
- the process is killed by the OS,
- or sustained inference is unstable.
```

---

## 7. Current Strategy List

The main experiment script currently supports the following strategies:

```text
native_rtdetr
default
static_affinity
fixed_low_power
fixed_frame_skip
thermal_only
scene_only
scene_thermal_coadaptive
```

Each strategy is selected by:

```bash
--strategy <strategy_name>
```

Example:

```bash
python scripts/run_experiment.py \
  --config configs/raspberry_pi4.yaml \
  --strategy scene_thermal_coadaptive \
  --video data/sample.mp4 \
  --duration-min 15
```

---

## 8. Strategy Configuration Summary

### 8.1 Strategy overview table

| Strategy                   | Uses Scene | Uses Thermal | Fixed Interval | Fixed Resolution | Fixed Threads | CPU Affinity | Governor    | Main Purpose                              |
| -------------------------- | ---------: | -----------: | -------------: | ---------------: | ------------: | ------------ | ----------- | ----------------------------------------- |
| `native_rtdetr`            |         No |           No |              1 |              640 |             4 | null         | performance | Native high-performance RT-DETR baseline  |
| `default`                  |        Yes |          Yes |           null |             null |          null | null         | null        | Balanced adaptive policy placeholder      |
| `static_affinity`          |         No |           No |              1 |              640 |             3 | `[0,1,2]`    | performance | Fixed runtime with CPU affinity           |
| `fixed_low_power`          |         No |           No |              4 |              320 |             2 | `[0,1]`      | powersave   | Static low-power baseline                 |
| `fixed_frame_skip`         |         No |           No |              3 |              480 |             4 | null         | null        | Static frame-skipping baseline            |
| `thermal_only`             |         No |          Yes |           null |             null |          null | null         | null        | Adaptive policy using only temperature    |
| `scene_only`               |        Yes |           No |           null |             null |          null | null         | null        | Adaptive policy using only scene workload |
| `scene_thermal_coadaptive` |        Yes |          Yes |           null |             null |          null | null         | null        | Full scene-thermal adaptive policy        |

---

## 9. Detailed Strategy Explanation

### 9.1 `native_rtdetr`

Configuration:

```yaml
strategy:
  name: native_rtdetr
  description: Native RT-DETR baseline without scene/thermal adaptation or frame skipping

policy:
  use_scene: false
  use_thermal: false
  fixed_inference_interval: 1
  fixed_input_resolution: 640
  fixed_cpu_threads: 4
  fixed_cpu_affinity: null
  fixed_governor: performance
```

Behavior:

```text
- Runs inference every frame.
- Uses 640 input resolution.
- Uses 4 CPU threads.
- Suggests performance governor.
- Does not adapt to scene complexity.
- Does not adapt to temperature.
```

Purpose:

```text
This is the high-performance baseline.
It is useful for measuring raw RT-DETR behavior and thermal throttling risk.
```

Recommended experiment:

```bash
python scripts/run_experiment.py \
  --config configs/raspberry_pi4.yaml \
  --strategy native_rtdetr \
  --video data/sample.mp4 \
  --duration-min 15
```

---

### 9.2 `default`

Configuration:

```yaml
strategy:
  name: default
  description: Balanced rule-based policy with thermal and scene inputs

policy:
  use_scene: true
  use_thermal: true
  fixed_inference_interval: null
  fixed_input_resolution: null
  fixed_cpu_threads: null
  fixed_cpu_affinity: null
  fixed_governor: null
```

Behavior:

```text
- Enables scene input.
- Enables thermal input.
- Does not force fixed runtime values.
- Falls back to the controller's rule-based action.
```

Current important note:

```text
At the current repository state, the adaptive rule table is still a placeholder.
Therefore, this strategy currently behaves like a balanced default policy.
```

With `configs/raspberry_pi4.yaml`, current default action is approximately:

```text
input_resolution = 480
inference_interval = 2
cpu_threads = 3
governor = ondemand
query_budget = 200
```

Recommended experiment:

```bash
python scripts/run_experiment.py \
  --config configs/raspberry_pi4.yaml \
  --strategy default \
  --video data/sample.mp4 \
  --duration-min 15
```

---

### 9.3 `static_affinity`

Configuration:

```yaml
strategy:
  name: static_affinity
  description: Fixed runtime with CPU affinity pinned to cores 0-2

policy:
  use_scene: false
  use_thermal: false
  fixed_inference_interval: 1
  fixed_input_resolution: 640
  fixed_cpu_threads: 3
  fixed_cpu_affinity: [0, 1, 2]
  fixed_governor: performance
```

Behavior:

```text
- Runs inference every frame.
- Uses 640 input resolution.
- Uses 3 CPU threads.
- Suggests CPU affinity on cores 0, 1, and 2.
- Suggests performance governor.
- Does not adapt to scene or temperature.
```

Purpose:

```text
This strategy is useful for comparing whether limiting CPU core usage improves thermal stability while keeping high input resolution.
```

Important note:

```text
The current controller emits CPU affinity as part of RuntimeAction, but the runtime code may not yet apply CPU affinity at the OS level. Verify before claiming that affinity is actually enforced.
```

Recommended experiment:

```bash
python scripts/run_experiment.py \
  --config configs/raspberry_pi4.yaml \
  --strategy static_affinity \
  --video data/sample.mp4 \
  --duration-min 15
```

---

### 9.4 `fixed_low_power`

Configuration:

```yaml
strategy:
  name: fixed_low_power
  description: Static low-power configuration regardless of scene or thermal state

policy:
  use_scene: false
  use_thermal: false
  fixed_inference_interval: 4
  fixed_input_resolution: 320
  fixed_cpu_threads: 2
  fixed_cpu_affinity: [0, 1]
  fixed_governor: powersave
```

Behavior:

```text
- Runs inference every 4 frames.
- Uses 320 input resolution.
- Uses 2 CPU threads.
- Suggests CPU affinity on cores 0 and 1.
- Suggests powersave governor.
- Does not adapt to scene or temperature.
```

Purpose:

```text
This is the low-power baseline.
It should produce lower heat and lower compute load, but may reduce detection responsiveness.
```

Recommended experiment:

```bash
python scripts/run_experiment.py \
  --config configs/raspberry_pi4.yaml \
  --strategy fixed_low_power \
  --video data/sample.mp4 \
  --duration-min 15
```

For Raspberry Pi 4B 1GB, this is the safest real-inference strategy:

```bash
python scripts/run_experiment.py \
  --config configs/raspberry_pi4_1gb.yaml \
  --strategy fixed_low_power \
  --video data/sample.mp4 \
  --duration-min 15
```

---

### 9.5 `fixed_frame_skip`

Configuration:

```yaml
strategy:
  name: fixed_frame_skip
  description: Fixed frame skipping (interval=3) at medium resolution

policy:
  use_scene: false
  use_thermal: false
  fixed_inference_interval: 3
  fixed_input_resolution: 480
  fixed_cpu_threads: 4
  fixed_cpu_affinity: null
  fixed_governor: null
```

Behavior:

```text
- Runs inference every 3 frames.
- Uses 480 input resolution.
- Uses 4 CPU threads.
- Does not force CPU affinity.
- Does not force governor.
- Does not adapt to scene or temperature.
```

Purpose:

```text
This strategy isolates the effect of frame skipping.
It is useful for testing how much thermal pressure can be reduced by lowering inference frequency only.
```

Recommended experiment:

```bash
python scripts/run_experiment.py \
  --config configs/raspberry_pi4.yaml \
  --strategy fixed_frame_skip \
  --video data/sample.mp4 \
  --duration-min 15
```

---

### 9.6 `thermal_only`

Configuration:

```yaml
strategy:
  name: thermal_only
  description: Adapt runtime based on device temperature only; ignore scene workload

policy:
  use_scene: false
  use_thermal: true
  fixed_inference_interval: null
  fixed_input_resolution: null
  fixed_cpu_threads: null
  fixed_cpu_affinity: null
  fixed_governor: null
```

Behavior:

```text
- Ignores scene workload.
- Uses device thermal state.
- Does not force fixed interval, resolution, or thread count.
- Should be controlled by the runtime decision controller.
```

Current important note:

```text
At the current repository state, the adaptive rule table is still a placeholder.
Therefore, this strategy may currently return the balanced default action instead of a true temperature-dependent action.
```

Expected future behavior:

```text
normal temperature → higher performance
warm temperature   → moderate frame skipping or lower resolution
hot temperature    → aggressive frame skipping or lower resolution
```

Recommended experiment:

```bash
python scripts/run_experiment.py \
  --config configs/raspberry_pi4.yaml \
  --strategy thermal_only \
  --video data/sample.mp4 \
  --duration-min 15
```

---

### 9.7 `scene_only`

Configuration:

```yaml
strategy:
  name: scene_only
  description: Adapt runtime based on scene workload only; ignore thermal state

policy:
  use_scene: true
  use_thermal: false
  fixed_inference_interval: null
  fixed_input_resolution: null
  fixed_cpu_threads: null
  fixed_cpu_affinity: null
  fixed_governor: null
```

Behavior:

```text
- Uses scene workload.
- Ignores device thermal state.
- Does not force fixed interval, resolution, or thread count.
- Should be controlled by the runtime decision controller.
```

Current important note:

```text
At the current repository state, SceneWorkloadEstimator.classify_workload() still returns "medium" for all frames.
Therefore, scene_only currently cannot fully demonstrate real scene-aware adaptation until workload classification is implemented.
```

Expected future behavior:

```text
light scene  → lower resolution or larger inference interval
medium scene → balanced behavior
heavy scene  → higher resolution or smaller inference interval
```

Recommended experiment:

```bash
python scripts/run_experiment.py \
  --config configs/raspberry_pi4.yaml \
  --strategy scene_only \
  --video data/sample.mp4 \
  --duration-min 15
```

---

### 9.8 `scene_thermal_coadaptive`

Configuration:

```yaml
strategy:
  name: scene_thermal_coadaptive
  description: Full co-adaptive policy using both scene workload and device thermal state

policy:
  use_scene: true
  use_thermal: true
  fixed_inference_interval: null
  fixed_input_resolution: null
  fixed_cpu_threads: null
  fixed_cpu_affinity: null
  fixed_governor: null
```

Behavior:

```text
- Uses scene workload.
- Uses device thermal state.
- Does not force fixed interval, resolution, or thread count.
- Should be the main adaptive strategy for the project.
```

Current important note:

```text
This is the intended final strategy, but two current placeholders limit it:

1. SceneWorkloadEstimator.classify_workload() currently returns "medium".
2. RuntimeDecisionController._rule_based_action() currently returns a balanced placeholder action.

Therefore, this strategy currently runs through the full control pipeline, but its adaptive behavior is not fully implemented yet.
```

Expected future behavior:

```text
normal + heavy scene → preserve accuracy
normal + light scene → save compute
warm + medium scene  → balanced thermal saving
hot + any scene      → reduce load to avoid throttling
```

Recommended experiment:

```bash
python scripts/run_experiment.py \
  --config configs/raspberry_pi4.yaml \
  --strategy scene_thermal_coadaptive \
  --video data/sample.mp4 \
  --duration-min 15
```

---

## 10. Current Adaptive Policy Limitation

The current repository already has the control-plane structure:

```text
scene features
device state
runtime state
decision controller
runtime action
inference or skip
logging
```

However, the adaptive policy itself is not fully implemented yet.

### 10.1 Workload classification limitation

Current behavior:

```text
SceneWorkloadEstimator.classify_workload() always returns "medium".
```

Effect:

```text
scene_only and scene_thermal_coadaptive cannot yet fully reflect real scene complexity.
```

### 10.2 Rule-based controller limitation

Current behavior:

```text
RuntimeDecisionController._rule_based_action() returns a balanced placeholder action.
```

Effect:

```text
thermal_only, scene_only, default, and scene_thermal_coadaptive currently use the adaptive pipeline,
but they may not yet produce truly different actions for normal/warm/hot or light/medium/heavy states.
```

### 10.3 Fixed strategies are reliable baselines

The following strategies have explicit fixed values and are therefore reliable for current experiments:

```text
native_rtdetr
static_affinity
fixed_low_power
fixed_frame_skip
```

The following strategies are structurally ready but depend on future adaptive rule implementation:

```text
default
thermal_only
scene_only
scene_thermal_coadaptive
```

---

## 11. What Actually Affects Runtime Right Now

### 11.1 Inference interval

The `inference_interval` field is currently applied in the runtime loop.

Example:

```text
inference_interval = 1 → infer every frame
inference_interval = 3 → infer once every 3 frames
inference_interval = 4 → infer once every 4 frames
```

This is currently the most reliable runtime knob.

---

### 11.2 Input resolution

The `input_resolution` field is passed to the ONNX inference engine.

However, if the ONNX model has fixed spatial input size, the engine will use the fixed ONNX input size instead of the requested runtime resolution.

Check the ONNX input shape with:

```bash
python - <<'PY'
import onnxruntime as ort

sess = ort.InferenceSession("models/rtdetr_r18_lite_pi4.onnx")
for i in sess.get_inputs():
    print(i.name, i.shape)
PY
```

If the output is fixed, for example:

```text
images [1, 3, 640, 640]
```

then runtime resolution changes may not actually change ONNX inference size.

If the output has dynamic dimensions, for example:

```text
images [1, 3, 'height', 'width']
```

then runtime resolution changes may work as expected.

---

### 11.3 CPU threads

The `cpu_threads` field is emitted as part of `RuntimeAction`.

However, ONNX Runtime thread count is usually set when creating the `InferenceSession`. If the session is only created once, changing `cpu_threads` per frame may not actually affect ONNX Runtime unless explicitly implemented.

Treat `cpu_threads` as a logged policy hint unless the code is updated to create sessions with `ort.SessionOptions`.

---

### 11.4 CPU governor and affinity

The strategy files include:

```text
fixed_governor
fixed_cpu_affinity
```

However, these should be treated as policy hints unless the runtime explicitly applies them at the OS level.

Manual governor control example:

```bash
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
```

Manual powersave example:

```bash
echo powersave | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
```

Do not claim that governor or affinity is dynamically enforced unless this has been verified in the runtime code and system state.

---

## 12. Smoke Test Before a Full Experiment

Before running a long real ONNX experiment, run a dry-run smoke test.

```bash
source .venv/bin/activate

python scripts/run_experiment.py \
  --dry-run \
  --strategy scene_thermal_coadaptive \
  --duration-min 1
```

Expected result:

```text
Experiment finished.
strategy: scene_thermal_coadaptive
dry_run:  True
log:      experiments/logs/<strategy>_<timestamp>.csv
```

Dry-run mode:

```text
- does not load the ONNX model,
- uses synthetic input if no video is provided,
- simulates inference latency,
- generates fake detections,
- verifies the runtime loop and logging pipeline.
```

---

## 13. Real ONNX Smoke Test

After dry-run works, run a short real inference test:

```bash
python scripts/run_experiment.py \
  --config configs/raspberry_pi4.yaml \
  --strategy fixed_low_power \
  --video data/sample.mp4 \
  --duration-min 1
```

This verifies:

```text
- ONNX model can be loaded,
- video file can be read,
- inference can run,
- CSV logs are generated,
- no immediate memory crash occurs.
```

For Raspberry Pi 4B 1GB:

```bash
python scripts/run_experiment.py \
  --config configs/raspberry_pi4_1gb.yaml \
  --strategy fixed_low_power \
  --video data/sample.mp4 \
  --duration-min 1
```

---

## 14. Running One Complete Experiment

A complete experiment should run long enough to observe temperature rise and possible throttling behavior.

Recommended duration:

```text
15 minutes minimum
30 minutes preferred if time allows
```

Example command:

```bash
python scripts/run_experiment.py \
  --config configs/raspberry_pi4.yaml \
  --strategy native_rtdetr \
  --video data/sample.mp4 \
  --duration-min 15
```

This will produce a log file similar to:

```text
experiments/logs/native_rtdetr_YYYYMMDD_HHMMSS.csv
```

It will also produce a profile log:

```text
experiments/logs/native_rtdetr_YYYYMMDD_HHMMSS_profile.csv
```

---

## 15. Running the Full Strategy Suite

To compare all currently available strategies, run:

```bash
for strategy in \
  native_rtdetr \
  static_affinity \
  fixed_frame_skip \
  fixed_low_power \
  thermal_only \
  scene_only \
  scene_thermal_coadaptive
do
  python scripts/run_experiment.py \
    --config configs/raspberry_pi4.yaml \
    --strategy "$strategy" \
    --video data/sample.mp4 \
    --duration-min 15
done
```

Optional: include `default` as well:

```bash
python scripts/run_experiment.py \
  --config configs/raspberry_pi4.yaml \
  --strategy default \
  --video data/sample.mp4 \
  --duration-min 15
```

Recommended practical order:

```text
1. fixed_low_power
2. fixed_frame_skip
3. native_rtdetr
4. static_affinity
5. thermal_only
6. scene_only
7. scene_thermal_coadaptive
```

Reason:

```text
Start with safer low-power runs, then move to more thermally aggressive baselines.
```

---

## 16. Plotting Results

After each experiment, generate plots:

```bash
python scripts/plot_results.py \
  --input experiments/logs/<run>.csv
```

Example:

```bash
python scripts/plot_results.py \
  --input experiments/logs/native_rtdetr_20260617_120000.csv
```

Output figures are saved to:

```text
experiments/results/
```

The plotting script currently generates:

```text
summary plot:
- inference latency
- CPU temperature
- scene workload
- FPS

CPU frequency plot:
- arm_clock_mhz if available
- otherwise freq_mhz_avg
```

---

## 17. Log Files and Metrics

### 17.1 Main CSV log

Each experiment generates a main CSV log with columns:

```text
timestamp
frame_id
strategy
workload
temp_c
freq_mhz_avg
arm_clock_mhz
power_w
latency_ms
fps
input_resolution
inference_interval
cpu_threads
governor
decoder_layers
query_budget
detection_count
confidence_mean
```

### 17.2 Profile CSV log

Each experiment also generates a profile CSV with timing breakdowns such as:

```text
frame_total_ms
scene_ms
device_ms
runtime_state_ms
decision_ms
infer_outer_ms
preprocess_ms
build_feed_ms
onnx_run_ms
postprocess_ms
infer_total_ms
summary_ms
main_log_write_ms
```

The profile CSV is useful for identifying runtime overhead outside the ONNX model.

---

## 18. Key Metrics to Compare

For each strategy, compare:

```text
mean FPS
median FPS
mean latency_ms
median latency_ms
max temperature
mean temperature
time above 80°C
arm_clock_mhz trend
freq_mhz_avg trend
detection_count
confidence_mean
number of inferred frames
number of skipped frames
```

Important thermal metrics:

```text
temp_c
arm_clock_mhz
freq_mhz_avg
throttling flags if available
```

Important performance metrics:

```text
latency_ms
fps
inference_interval
input_resolution
```

Important detection metrics:

```text
detection_count
confidence_mean
```

---

## 19. Recommended Experiment Matrix

### 19.1 Baseline comparison

| Experiment | Strategy           | Goal                                                            |
| ---------- | ------------------ | --------------------------------------------------------------- |
| A          | `native_rtdetr`    | Measure native high-performance behavior and thermal throttling |
| B          | `fixed_frame_skip` | Measure effect of frame skipping                                |
| C          | `fixed_low_power`  | Measure low-power stable baseline                               |
| D          | `static_affinity`  | Measure fixed CPU core limitation behavior                      |

### 19.2 Adaptive comparison

| Experiment | Strategy                   | Goal                                       |
| ---------- | -------------------------- | ------------------------------------------ |
| E          | `thermal_only`             | Evaluate temperature-only control pipeline |
| F          | `scene_only`               | Evaluate scene-only control pipeline       |
| G          | `scene_thermal_coadaptive` | Evaluate full co-adaptive runtime pipeline |

Current limitation:

```text
The adaptive comparison becomes meaningful only after workload classification
and rule-based adaptive actions are fully implemented.
```

---

## 20. Recommended Complete Experiment Procedure

### Step 1: Prepare the Pi

```bash
sudo apt update
sudo apt install -y python3-pip python3-venv libopencv-dev v4l-utils libraspberrypi-bin
```

### Step 2: Activate environment

```bash
cd ECE4730J-Degree-Project
source .venv/bin/activate
```

### Step 3: Confirm model exists

```bash
ls -lh models/rtdetr_r18_lite_pi4.onnx
```

### Step 4: Check ONNX input shape

```bash
python - <<'PY'
import onnxruntime as ort

sess = ort.InferenceSession("models/rtdetr_r18_lite_pi4.onnx")
for i in sess.get_inputs():
    print(i.name, i.shape)
PY
```

### Step 5: Run dry-run smoke test

```bash
python scripts/run_experiment.py \
  --dry-run \
  --strategy scene_thermal_coadaptive \
  --duration-min 1
```

### Step 6: Run real ONNX smoke test

```bash
python scripts/run_experiment.py \
  --config configs/raspberry_pi4.yaml \
  --strategy fixed_low_power \
  --video data/sample.mp4 \
  --duration-min 1
```

### Step 7: Run a full 15-minute experiment

```bash
python scripts/run_experiment.py \
  --config configs/raspberry_pi4.yaml \
  --strategy native_rtdetr \
  --video data/sample.mp4 \
  --duration-min 15
```

### Step 8: Plot results

```bash
python scripts/plot_results.py \
  --input experiments/logs/<run>.csv
```

### Step 9: Repeat for other strategies

```bash
for strategy in \
  native_rtdetr \
  static_affinity \
  fixed_frame_skip \
  fixed_low_power \
  thermal_only \
  scene_only \
  scene_thermal_coadaptive
do
  python scripts/run_experiment.py \
    --config configs/raspberry_pi4.yaml \
    --strategy "$strategy" \
    --video data/sample.mp4 \
    --duration-min 15
done
```

### Step 10: Compare generated plots

Check:

```text
experiments/results/
```

Compare:

```text
temperature curve
FPS curve
latency curve
ARM clock curve
workload curve
```

---

## 21. Suggested Naming Convention for Logs

For reproducibility, use explicit output names:

```bash
python scripts/run_experiment.py \
  --config configs/raspberry_pi4.yaml \
  --strategy native_rtdetr \
  --video data/sample.mp4 \
  --duration-min 15 \
  --output experiments/logs/exp01_native_rtdetr_15min.csv
```

Example naming:

```text
exp01_native_rtdetr_15min.csv
exp02_fixed_frame_skip_15min.csv
exp03_fixed_low_power_15min.csv
exp04_static_affinity_15min.csv
exp05_thermal_only_15min.csv
exp06_scene_only_15min.csv
exp07_scene_thermal_coadaptive_15min.csv
```

This makes plotting and comparison easier.

---

## 22. Current Best Baseline Set

For the current repository state, the most reliable experiment set is:

```text
native_rtdetr
fixed_frame_skip
fixed_low_power
static_affinity
```

Reason:

```text
These strategies use explicit fixed runtime values.
They do not depend on placeholder adaptive rules.
```

After implementing true adaptive rules, the most important comparison becomes:

```text
native_rtdetr
fixed_low_power
thermal_only
scene_only
scene_thermal_coadaptive
```

---

## 23. What to Report After One Complete Experiment

For each strategy, report:

```text
Strategy name
Input video
Duration
Config file
Model path
Mean FPS
Mean latency
Max temperature
Mean temperature
Minimum ARM clock
Whether throttling was observed
Average detection count
Average confidence
```

Recommended table format:

| Strategy                 | Mean FPS | Mean Latency ms | Max Temp °C | Min ARM Clock MHz | Mean Detections | Notes                      |
| ------------------------ | -------: | --------------: | ----------: | ----------------: | --------------: | -------------------------- |
| native_rtdetr            |      TBD |             TBD |         TBD |               TBD |             TBD | High-performance baseline  |
| fixed_frame_skip         |      TBD |             TBD |         TBD |               TBD |             TBD | Lower inference frequency  |
| fixed_low_power          |      TBD |             TBD |         TBD |               TBD |             TBD | Lowest compute load        |
| static_affinity          |      TBD |             TBD |         TBD |               TBD |             TBD | Limited CPU cores          |
| thermal_only             |      TBD |             TBD |         TBD |               TBD |             TBD | Temperature-aware pipeline |
| scene_only               |      TBD |             TBD |         TBD |               TBD |             TBD | Scene-aware pipeline       |
| scene_thermal_coadaptive |      TBD |             TBD |         TBD |               TBD |             TBD | Full adaptive pipeline     |

---

## 24. Recommended Next Code Improvements Before Final Experiments

Before using adaptive results in the final report, implement:

```text
1. SceneWorkloadEstimator.classify_workload()
2. RuntimeDecisionController._rule_based_action()
3. action_mode logging
4. optional thermal_state logging
5. optional throttling flag logging
```

The most important missing log field is:

```text
action_mode
```

Reason:

```text
strategy tells which experiment is running,
but action_mode tells which runtime action was selected on each frame.
```

For example:

```text
strategy = scene_thermal_coadaptive
action_mode = hot_cooldown
```

This is essential for proving that runtime switching actually happened.

---

## 25. Interpretation Guide

### 25.1 If temperature reaches around 85°C

This indicates high thermal pressure and possible CPU throttling.

Check:

```text
arm_clock_mhz
freq_mhz_avg
throttling flags
fps drop
latency increase
```

### 25.2 If `arm_clock_mhz` drops

This is strong evidence that the Raspberry Pi firmware is lowering the actual ARM clock.

This is more useful than only checking Linux `scaling_cur_freq`.

### 25.3 If `native_rtdetr` FPS degrades over time

This supports the claim:

```text
Sustained RT-DETR inference causes thermal pressure and performance degradation on Raspberry Pi 4B.
```

### 25.4 If `fixed_low_power` keeps temperature stable

This supports the claim:

```text
Reducing resolution and inference frequency can improve long-duration stability.
```

### 25.5 If adaptive policies do not change much

Check whether:

```text
classify_workload() is still returning medium
_rule_based_action() is still returning balanced_placeholder
input_resolution is overridden by fixed ONNX shape
cpu_threads/governor are only logged but not applied
```

---

## 26. Recommended Final Experiment Story

The final project report can use the following structure:

```text
1. Deploy RT-DETR on Raspberry Pi 4B.
2. Profile model runtime and observe that CNN backbone dominates latency.
3. Replace R18 backbone with R18-lite to reduce compute cost.
4. Run native RT-DETR long-duration video inference.
5. Observe temperature rise and performance degradation near thermal throttling.
6. Compare fixed runtime baselines:
   - native_rtdetr
   - fixed_frame_skip
   - fixed_low_power
   - static_affinity
7. Implement scene-thermal adaptive controller.
8. Compare:
   - thermal_only
   - scene_only
   - scene_thermal_coadaptive
9. Show that adaptive runtime can reduce thermal pressure and stabilize FPS.
```

---

## 27. Final Checklist for One Complete Experiment

Before running:

```text
[ ] Raspberry Pi has stable power supply
[ ] Cooling condition is recorded
[ ] Python virtual environment is activated
[ ] requirements.txt dependencies are installed
[ ] ONNX model exists at configured path
[ ] input video exists
[ ] dry-run smoke test passes
[ ] real ONNX smoke test passes
[ ] enough disk space is available for logs
```

During running:

```text
[ ] Do not run other heavy processes
[ ] Keep cooling setup unchanged
[ ] Use the same video for all strategies
[ ] Use the same duration for all strategies
[ ] Record ambient condition if possible
```

After running:

```text
[ ] Main CSV log exists
[ ] Profile CSV log exists
[ ] Plot script succeeds
[ ] Temperature plot generated
[ ] FPS plot generated
[ ] CPU frequency plot generated
[ ] Compare metrics across strategies
```

---

## 28. Summary

The repository already provides a complete experiment skeleton:

```text
config loading
strategy selection
scene feature extraction
device monitoring
runtime decision
inference / frame skip
logging
plotting
```

The currently reliable fixed strategies are:

```text
native_rtdetr
static_affinity
fixed_low_power
fixed_frame_skip
```

The currently available adaptive strategies are:

```text
default
thermal_only
scene_only
scene_thermal_coadaptive
```

However, the adaptive strategies still depend on unfinished placeholder logic in:

```text
SceneWorkloadEstimator.classify_workload()
RuntimeDecisionController._rule_based_action()
```

Therefore, for immediate experiments, use fixed strategies to establish baselines. For the final project contribution, implement the missing adaptive logic and then compare `scene_thermal_coadaptive` against the fixed baselines.
