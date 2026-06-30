# Experiments

This document records the current end-to-end experiment flow for Raspberry Pi
RT-DETR runtime adaptation. The main objective is to maximize sustained
performance while avoiding thermal throttling and accounting for detection
quality loss.

## Directory Layout

```text
experiments/
  protocols/     # YAML experiment definitions
  logs/          # Per-run CSV/JSONL logs
  results/       # Aggregated plots, summaries, quality tables
  visualizations # Optional annotated frames/videos
```

## Experiment Flow

1. Verify runtime-action support on the Pi.
2. Calibrate resolution quality with 640-resolution RT-DETR as pseudo-label teacher.
3. Run the core system/scene ablation strategies with cooldown between runs.
4. Monitor runs through the live dashboard when needed.
5. Summarize quality-adjusted sustained performance.
6. Run temporal tracking quality analysis to compare stale-box reuse with LK.
7. For annotated benchmark datasets, report real mAP/precision/recall separately.

## Smoke Test

Use this to verify the Python path and dry-run pipeline:

```bash
.venv/bin/python scripts/run_experiment.py \
  --config configs/raspberry_pi4.yaml \
  --strategy thermal_balanced \
  --dry-run \
  --duration-min 1
```

## Runtime Support

Check whether governor and CPU affinity can be applied:

```bash
.venv/bin/python scripts/check_runtime_action_support.py
```

With root privileges, verify actual governor writes:

```bash
sudo -E .venv/bin/python scripts/check_runtime_action_support.py \
  --apply \
  --governor performance \
  --affinity 0,1,2,3
```

Expected result on Raspberry Pi: affinity should apply as the normal user, while
governor usually requires `sudo`.

## Stage 1: Resolution Quality

When ground-truth labels are unavailable, estimate the quality cost of lower
resolutions using 640-resolution RT-DETR as a pseudo-label teacher:

```bash
sudo -E .venv/bin/python scripts/evaluate_resolution_quality.py \
  --video data/sample.mp4 \
  --model-template 'models/rtdetr_r18_lite_pi4_{resolution}.onnx' \
  --teacher-resolution 640 \
  --student-resolutions 480,320 \
  --threads 4 \
  --max-frames 120 \
  --frame-stride 5
```

Outputs:

```text
experiments/results/quality/resolution_quality_summary.csv
experiments/results/quality/resolution_quality_matches.csv
```

The summary provides `pseudo_recall`, `precision_proxy`, `mean_matched_iou`,
`mean_confidence_drop`, and `detection_count_ratio`. These values are later used
to compute quality-adjusted FPS.

## Stage 2: Core Strategy Suite

Run the main ablation suite:

```bash
sudo -E .venv/bin/python scripts/run_thermal_experiment_suite.py \
  --config configs/raspberry_pi4.yaml \
  --video data/sample.mp4 \
  --loop-video \
  --duration-min 15 \
  --repeats 3 \
  --strategies native_rtdetr,thermal_interval_first,scene_track_lk,scene_thermal_interval_lk \
  --cooldown-temp-c 55 \
  --cooldown-poll-sec 10 \
  --max-cooldown-min 30 \
  --log-detections \
  --enable-thread-sessions \
  --thread-session-counts 1,2,3,4 \
  --apply-runtime-actions
```

Strategies:

```text
native_rtdetr                  Native RT-DETR: 640 / interval=1 / threads=4 / performance
thermal_interval_first         Thermal-aware system control: keep 640, reduce duty cycle first
scene_track_lk                 Scene-aware temporal reuse: event-triggered RT-DETR + LK box tracking
scene_thermal_interval_lk      Full method: thermal interval-first budget + event-triggered scene/LK reuse
```

For direct comparison with the earlier resolution-first thermal policy, add
`thermal_balanced` and `scene_thermal_coadaptive` to the strategy list. These
older policies lower resolution earlier and are useful as ablation baselines.

The log records performance and thermal/power proxy fields:

```text
latency_ms
fps
loop_fps
effective_inference_fps
actual_inference_fps
input_resolution
resolved_input_resolution
inference_interval
cpu_threads
governor
requested_governor
applied_governor
freq_mhz_avg
arm_clock_mhz
power_w
temp_c
currently_throttled
arm_freq_capped
soft_temp_limit
```

With `--log-detections`, each CSV also gets a sidecar file:

```text
<run>_detections.jsonl
```

Each JSONL row stores `frame_id`, strategy, `did_infer`, tracking state,
resolution, and all current boxes with `class_id`, `score`, and `bbox`.

`power_w` depends on whether the Pi exposes a readable power sensor. If it is
empty, use `arm_clock_mhz`, `freq_mhz_avg`, `temp_c`, and throttling flags as
thermal/power proxies.

## Stage 3: Live Dashboard

Run one strategy with a browser dashboard:

```bash
sudo -E .venv/bin/python scripts/run_live_dashboard.py \
  --config configs/raspberry_pi4.yaml \
  --strategy thermal_balanced \
  --video data/sample.mp4 \
  --loop-video \
  --duration-min 15 \
  --host 0.0.0.0 \
  --port 8000 \
  --enable-thread-sessions \
  --thread-session-counts 1,2,3,4 \
  --apply-runtime-actions
```

Open this from another device on the same network:

```text
http://<raspberry-pi-ip>:8000
```

The dashboard shows:

```text
live detection stream
temperature
latency
actual inference FPS
resolution
governor / applied governor
thread setting
throttling flags
rolling thermal and latency curves
tracking state, when LK is enabled
```

For lower network load:

```bash
--jpeg-width 640 --jpeg-quality 65
```

For curves only:

```bash
--no-video-stream
```

The repeated suite can also use the dashboard:

```bash
sudo -E .venv/bin/python scripts/run_thermal_experiment_suite.py \
  --config configs/raspberry_pi4.yaml \
  --video data/sample.mp4 \
  --loop-video \
  --duration-min 15 \
  --repeats 3 \
  --strategies native_rtdetr,thermal_interval_first,scene_track_lk,scene_thermal_interval_lk \
  --cooldown-temp-c 55 \
  --log-detections \
  --enable-thread-sessions \
  --thread-session-counts 1,2,3,4 \
  --apply-runtime-actions \
  --dashboard \
  --dashboard-port 8000
```

## Stage 4A: Trade-Off Summary

After the suite finishes, score logs with quality-adjusted FPS and thermal
penalties:

```bash
.venv/bin/python scripts/evaluate_experiment_quality_tradeoff.py \
  experiments/logs/thermal_suite/<run>_r01_01_native_rtdetr.csv \
  experiments/logs/thermal_suite/<run>_r01_02_thermal_interval_first.csv \
  experiments/logs/thermal_suite/<run>_r01_03_scene_track_lk.csv \
  experiments/logs/thermal_suite/<run>_r01_04_scene_thermal_interval_lk.csv \
  --quality-summary experiments/results/quality/resolution_quality_summary.csv \
  --output experiments/results/quality_tradeoff_summary.csv
```

Important output fields:

```text
raw_detector_fps
quality_adjusted_fps
output_fps
quality_adjusted_output_fps
sustained_utility
sustained_output_utility
latency_ms_mean
latency_ms_p95
temp_c_mean
temp_c_max
time_above_80c_ratio
soft_temp_limit_ratio
currently_throttled_ratio
arm_freq_capped_ratio
power_w_mean
resolution_duration_sec
```

Interpretation:

```text
raw_detector_fps
  Throughput without quality penalty.

quality_adjusted_fps
  Throughput weighted by estimated detection quality at each resolution.

output_fps
  Detector frames plus valid LK-tracked output frames. This is the main
  performance metric for scene_track_lk and scene_thermal_interval_lk.

quality_adjusted_output_fps
  Output FPS weighted by resolution quality and LK tracking diagnostics.

sustained_utility
  Quality-adjusted FPS penalized by high-temperature time and throttling flags.

sustained_output_utility
  Output-quality counterpart of sustained_utility for tracking-enabled policies.
```

## Stage 4B: Box-Level Strategy Quality

If `--log-detections` was enabled, compare each strategy's per-frame boxes
against the native RT-DETR run as a pseudo-label teacher:

```bash
.venv/bin/python scripts/evaluate_strategy_detection_quality.py \
  --teacher experiments/logs/thermal_suite/<run>_r01_01_native_rtdetr_detections.jsonl \
  --students \
    experiments/logs/thermal_suite/<run>_r01_02_thermal_interval_first_detections.jsonl \
    experiments/logs/thermal_suite/<run>_r01_03_scene_track_lk_detections.jsonl \
    experiments/logs/thermal_suite/<run>_r01_04_scene_thermal_interval_lk_detections.jsonl \
  --student-csvs \
    experiments/logs/thermal_suite/<run>_r01_02_thermal_interval_first.csv \
    experiments/logs/thermal_suite/<run>_r01_03_scene_track_lk.csv \
    experiments/logs/thermal_suite/<run>_r01_04_scene_thermal_interval_lk.csv \
  --output experiments/results/strategy_detection_quality_summary.csv \
  --matches-output experiments/results/strategy_detection_quality_frames.csv
```

Important output fields:

```text
pseudo_recall
precision_proxy
mean_matched_iou
mean_center_error_norm
detection_count_ratio
infer_frame_pseudo_recall
noninfer_frame_pseudo_recall
latency_ms_mean
actual_inference_fps_mean
temp_c_max
currently_throttled_ratio
```

This is the closest unlabeled-video proxy for comparing native, scene-aware,
thermal-aware, and scene-thermal co-adaptive output quality frame by frame.

## Stage 5: Scene Tracking Ablation

`scene_track_lk` and `scene_thermal_interval_lk` enable event-triggered LK
tracking from their YAML configs. They do not use a fixed workload-to-interval
mapping. Instead:

```text
first frame                         -> RT-DETR
healthy tracks                      -> LK tracking
LK failure ratio too high           -> RT-DETR refresh
unexplained motion outside boxes    -> RT-DETR refresh
scene change / camera motion        -> RT-DETR refresh
safety_refresh_frames reached       -> RT-DETR refresh
```

In `scene_thermal_interval_lk`, thermal control may impose a minimum detector
gap through `inference_interval`, so urgent scene refreshes can be deferred
briefly while the SoC is hot. To isolate the scene-aware contribution, compare
native RT-DETR with scene-only LK tracking:

```bash
sudo -E .venv/bin/python scripts/run_experiment.py \
  --config configs/raspberry_pi4.yaml \
  --strategy native_rtdetr \
  --video data/sample.mp4 \
  --loop-video \
  --duration-min 15 \
  --output experiments/logs/scene_ablation_native.csv \
  --log-detections \
  --enable-thread-sessions \
  --thread-session-counts 1,2,3,4 \
  --apply-runtime-actions
```

Then run the scene-aware temporal reuse policy:

```bash
sudo -E .venv/bin/python scripts/run_experiment.py \
  --config configs/raspberry_pi4.yaml \
  --strategy scene_track_lk \
  --video data/sample.mp4 \
  --loop-video \
  --duration-min 15 \
  --output experiments/logs/scene_ablation_track_lk.csv \
  --log-detections \
  --enable-thread-sessions \
  --thread-session-counts 1,2,3,4 \
  --apply-runtime-actions
```

LK-related CSV fields:

```text
tracking_mode
tracking_reason
tracking_ms
tracking_failure_ratio
tracking_mean_quality
tracking_should_refresh
```

## Stage 6: Temporal Tracking Quality

Use a teacher/student temporal evaluation to compare plain frame skipping with
LK tracking. The teacher runs RT-DETR on every processed frame. Students run
RT-DETR every `--keyframe-interval` frames and fill intermediate frames either
by reusing the last boxes or by LK tracking.

```bash
sudo -E .venv/bin/python scripts/evaluate_tracking_quality.py \
  --video data/sample.mp4 \
  --model models/rtdetr_r18_lite_pi4_640.onnx \
  --resolution 640 \
  --threads 4 \
  --keyframe-interval 4 \
  --max-frames 120 \
  --frame-stride 1 \
  --policies skip_reuse,lk_track
```

Outputs:

```text
experiments/results/tracking_quality/tracking_quality_summary.csv
experiments/results/tracking_quality/tracking_quality_frames.csv
```

Important metrics:

```text
detector_invocation_rate
mean_total_latency_ms
mean_detector_latency_ms
pseudo_recall
precision_proxy
skip_frame_pseudo_recall
skip_frame_precision_proxy
mean_skip_frame_iou
mean_skip_frame_center_error_norm
detection_count_ratio
skip_detection_count_ratio
```

Expected interpretation:

```text
skip_reuse
  Same detector-call reduction as LK, but boxes become stale on skipped frames.

lk_track
  Same keyframe interval, with boxes updated by LK on skipped frames. A useful
  LK tracker should improve skip-frame IoU and center error at low extra latency.
```

## Stage 7: Detect-Only vs Detect+Track Visualization

For qualitative comparison on a short video, use the standalone visualization
script in two modes.

Native RT-DETR every processed frame:

```bash
.venv/bin/python scripts/detect_track_lk.py \
  --mode detect_only \
  --video data/sample2.mp4 \
  --model models/rtdetr_r18_lite_pi4_640.onnx \
  --output-dir experiments/visualizations/sample_detect_only \
  --resolution 640 \
  --threads 4 \
  --max-frames 300
```

Event-triggered RT-DETR + LK tracking:

```bash
.venv/bin/python scripts/detect_track_lk.py \
  --mode detect_track \
  --video data/sample2.mp4 \
  --model models/rtdetr_r18_lite_pi4_640.onnx \
  --output-dir experiments/visualizations/sample_detect_track \
  --resolution 640 \
  --threads 4 \
  --max-frames 300
```

Compare:

```text
experiments/visualizations/sample_detect_only/summary.txt
experiments/visualizations/sample_detect_track/summary.txt
experiments/visualizations/*/events.csv
experiments/visualizations/*/contact_sheet.jpg
```

Key fields:

```text
detector_calls
detector_invocation_rate
average_total_latency_ms
average_detect_frame_latency_ms
mode
reason
tracker_failure_ratio
tracker_mean_quality
outside_ratio
```

## Stage 8: Real Accuracy Benchmarks

For datasets with ground-truth annotations, report true accuracy metrics
separately from pseudo-label quality:

```text
mAP@0.5
mAP@0.5:0.95
precision
recall
per-class AP
temporal stability / box jitter, if evaluating tracking
```

Use pseudo-label metrics only for relative quality-cost analysis on unlabeled
experiment videos such as `data/sample.mp4`.

## Post-Run Plotting

Generate timeline plots for an individual run:

```bash
.venv/bin/python scripts/plot_results.py \
  --input experiments/logs/<run>.csv \
  --output-dir experiments/results
```

Generate detailed inference plots:

```bash
.venv/bin/python scripts/plot_inference_details.py \
  --input experiments/logs/<run>.csv
```

Summarize a single CSV:

```bash
.venv/bin/python scripts/summarize_baseline.py \
  --input experiments/logs/<run>.csv \
  --output-dir experiments/results/summaries \
  --label <run>
```

Use `--skip-plot` in the thermal suite if pandas/matplotlib are not installed
on the Pi.
