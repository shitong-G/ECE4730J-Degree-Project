# Post-Experiment Analysis Guide

This guide explains how to generate plots and statistics after one experiment run.
When adding a new script that plots, summarizes, or analyzes experiment logs,
update this file with its purpose, inputs, outputs, and an example command.

## Inputs

Experiment runs write logs to:

```bash
experiments/logs/
```

For a run named:

```bash
experiments/logs/<run>.csv
```

the runtime also writes a profiling log:

```bash
experiments/logs/<run>_profile.csv
```

The main CSV contains per-frame runtime values, including latency, FPS, scene
workload, device state, and controller-selected parameters. The profile CSV
contains timing breakdowns for the frame source, runtime loop, and inference
engine.

Profile timing fields:

- `serial_total_ms`: approximate serial per-frame time, equal to frame-source
  time plus runtime-loop time. In producer-consumer mode this is the
  serial-equivalent cost, not wall-clock blocking time.
- `source_total_ms`: frame-source time before the runtime loop receives the
  frame.
- `source_wait_ms`: intentional capture wait, for example when a capture
  interval is configured.
- `capture_ms`: raw camera capture or video-frame read time.
- `isp_ms`: IMX219 raw-to-BGR lite ISP conversion time.
- `source_resize_ms`: resize time applied inside the frame source before
  runtime processing.
- `source_save_ms`: time spent writing `camera_latest.jpg` when latest-frame
  saving is enabled.
- `source_runtime_resize_ms`: resize time applied to the runtime input frame by
  `--frame-width` and `--frame-height` before scene analysis, LK tracking, and
  RT-DETR.
- `source_consumer_wait_ms`: time the runtime waited for a produced frame.
- `source_frame_age_ms`: how old the latest produced frame was when runtime
  consumed it.
- `source_dropped_frames`: produced frames skipped because runtime consumed
  only the newest frame.
- `frame_total_ms`: runtime-loop time after a frame has entered the system.

FPS fields:

- `loop_fps`: how many video frames the runtime loop processes per second,
  including skipped frames.
- `effective_inference_fps`: estimated full-inference throughput. This is
  `loop_fps / inference_interval`, so it stays comparable when frame skipping
  changes.
- `fps`: legacy field kept for compatibility; treat it as loop FPS.

The runtime computes FPS using `runtime.metrics_window_frames`. Raspberry Pi 4
configs use a longer window to reduce skip/infer oscillation in the log itself.
Plots intended for inference analysis use only `did_infer=True` rows by default.

Latency fields:

- `latency_ms` is nonzero only on frames where RT-DETR ran, either full-frame
  inference or ROI refresh inference.
- Plotting scripts filter out skipped-frame latency zeros when plotting
  detector latency.

LK ROI refresh fields:

- `tracking_reason=roi_refresh_lk_tracking_quality_degraded` means LK lost a
  local subset of tracks and the runtime used a lower-resolution ROI detector
  refresh instead of a full-frame detector refresh. This path is disabled by
  default with `roi_refresh_lk_quality_enabled: false`, because repeated Pi
  experiments showed unstable long-tail latency for LK-quality ROI refresh.
  Re-enable it only for controlled comparison experiments.
- When LK-quality ROI refresh is explicitly enabled, it remains conservative:
  it only allows one failed box, failure ratio at or below `0.40`, and ROI area
  at or below `18%` of the full frame. Broader LK degradation falls back to
  full-frame RT-DETR.
- `tracking_reason=roi_refresh_unexplained_motion_outside_tracks` means the
  residual-motion gate found local motion outside tracked boxes and the runtime
  used ROI detector refresh for that region. This is the default ROI path kept
  enabled after the latest experiments because it produced the most stable ROI
  latency improvement.
- Motion-triggered ROI refresh is guarded by `roi_refresh_max_area_ratio`.
  The current default for `scene_thermal_interval_lk` and `scene_track_lk` is
  `0.25`; larger ROI candidates are rejected before ROI inference and the
  runtime falls back to full-frame RT-DETR.
- The main CSV records ROI reviewer details in `roi_refresh_candidate`,
  `roi_refresh_applied`, `roi_refresh_reason`, `roi_refresh_reject_reason`,
  `roi_refresh_area_ratio`, `roi_refresh_width_px`,
  `roi_refresh_height_px`, and `roi_refresh_max_area_ratio`.
- `tracking_reason=forced_refresh_*` means the ROI reviewer rejected the local
  refresh path and the runtime fell back to full-frame RT-DETR.
- `tracking_reason=lk_quality_degraded_confirming` means LK reported a soft
  quality-degraded frame, but the runtime deferred full-frame RT-DETR for one
  frame to check whether the degradation persists. The main CSV records
  `lk_quality_confirm_count`, `lk_quality_confirm_deferred`,
  `lk_quality_confirm_total_deferred`, `tracking_track_count_before`,
  `tracking_track_count_after`, and `tracking_failed_box_count` for this check.
  Catastrophic loss, such as no surviving tracks or failure ratio above
  `lk_quality_confirm_max_failure_ratio`, is not deferred.
- `tracking_reason=track_healthy_safety_refresh_deferred` means the periodic
  full-frame safety refresh was due, but LK tracking still looked healthy, so
  the runtime deferred the full-frame detector call.
- `tracking_reason=forced_refresh_long_interval_safety_refresh_hard_limit`
  means the safety refresh reached its hard upper frame limit and the runtime
  forced a full-frame detector call even if tracking still looked healthy.
- The ROI reviewer rejects unsafe cases before running ROI inference. Once ROI
  inference starts, the runtime does not run a second full-frame RT-DETR in the
  same frame, even if the ROI returns no detections. This keeps critical-path
  detector cost bounded by either one ROI inference or one full-frame inference.
- ROI refresh keeps detection boxes mapped to the full-frame detection
  coordinate system, so dashboard visualization and LK tracking can continue
  without changing downstream code.

Fan fields:

- `fan_enabled`: whether the runtime requested active fan cooling on that row.
- `fan_duty_cycle`: `1.0` for DC full-on control, `0.0` when off, or the PWM
  duty cycle when `fan.strategy: pwm`.
- `fan_mode`: `dc`, `pwm`, `off`, `disabled`, or `dc_no_gpio` /
  `pwm_no_gpio` when GPIO output could not be initialized.

Generated plots and summaries should be written to:

```bash
experiments/results/
```

## PowerShell And Raspberry Pi Paths

Use these fixed paths when exchanging files between Windows and the Raspberry Pi.

Windows project root:

```powershell
D:\___SJTU___\Undergraduate_Course\Senior 3\Scene_Thermal_Aware_RT_DETR\ECE4730J-Degree-Project-main
```

Raspberry Pi project root:

```bash
~/ECE4730J-Degree-Project
```

Raspberry Pi SSH target:

```powershell
ece473g1@ece473g1-desktop
```

In the commands below, replace `<run>` with the log stem, such as:

```text
scene_thermal_coadaptive_20260617_223528
```

The corresponding log files are:

```text
<run>.csv
<run>_profile.csv
```

## 1. Summary Plots

Use this script for a quick overview of one run.

```bash
python scripts/plot_results.py --input experiments/logs/<run>.csv
```

Example:

```bash
python scripts/plot_results.py \
  --input experiments/logs/scene_thermal_coadaptive_20260617_223528.csv
```

Outputs:

```bash
experiments/results/<run>_summary.png
experiments/results/<run>_cpu_freq.png   # only when frequency data exists
```

PowerShell: copy the needed log from the Raspberry Pi to Windows:

```powershell
scp "ece473g1@ece473g1-desktop:~/ECE4730J-Degree-Project/experiments/logs/<run>.csv" "D:\___SJTU___\Undergraduate_Course\Senior 3\Scene_Thermal_Aware_RT_DETR\ECE4730J-Degree-Project-main\experiments\logs\"
```

PowerShell: run the plot locally:

```powershell
cd "D:\___SJTU___\Undergraduate_Course\Senior 3\Scene_Thermal_Aware_RT_DETR\ECE4730J-Degree-Project-main"
.\.venv\Scripts\python.exe scripts\plot_results.py --input experiments\logs\<run>.csv
```

PowerShell: copy generated summary plot back to the Raspberry Pi:

```powershell
scp "D:\___SJTU___\Undergraduate_Course\Senior 3\Scene_Thermal_Aware_RT_DETR\ECE4730J-Degree-Project-main\experiments\results\<run>_summary.png" "ece473g1@ece473g1-desktop:~/ECE4730J-Degree-Project/experiments/results/"
```

PowerShell: copy generated CPU frequency plot back to the Raspberry Pi, only if it exists:

```powershell
scp "D:\___SJTU___\Undergraduate_Course\Senior 3\Scene_Thermal_Aware_RT_DETR\ECE4730J-Degree-Project-main\experiments\results\<run>_cpu_freq.png" "ece473g1@ece473g1-desktop:~/ECE4730J-Degree-Project/experiments/results/"
```

The summary figure includes:

- full inference latency, excluding skipped-frame zero latency rows
- CPU temperature when available
- scene workload
- effective inference FPS
- fan enabled state and fan duty cycle when available
- runtime workload knobs, including interval and input resolution

The x-axis is elapsed experiment time in minutes when the log contains
`timestamp`. It falls back to `frame_id` only for logs without timestamps.
Numeric trend lines use raw values by default to preserve local fluctuation.
Use `--smooth-window <N>` to add rolling smoothing.

Optional frame limit:

```bash
python scripts/plot_results.py \
  --input experiments/logs/<run>.csv \
  --max-frame-id 260
```

Add smoothing:

```bash
python scripts/plot_results.py \
  --input experiments/logs/<run>.csv \
  --smooth-window 5
```

## 2. Per-Inference Detailed Dashboard

Use this script when you want to inspect every full inference event. It filters
to frames where inference actually ran, then combines the main CSV with the
matching profile CSV.

```bash
python scripts/plot_inference_details.py --input experiments/logs/<run>.csv
```

Example:

```bash
python scripts/plot_inference_details.py \
  --input experiments/logs/scene_thermal_coadaptive_20260617_223528.csv
```

Default output:

```bash
experiments/results/<run>_inference_details.jpg
```

PowerShell: copy the main log and profile log from the Raspberry Pi to Windows:

```powershell
scp "ece473g1@ece473g1-desktop:~/ECE4730J-Degree-Project/experiments/logs/<run>.csv" "D:\___SJTU___\Undergraduate_Course\Senior 3\Scene_Thermal_Aware_RT_DETR\ECE4730J-Degree-Project-main\experiments\logs\"
scp "ece473g1@ece473g1-desktop:~/ECE4730J-Degree-Project/experiments/logs/<run>_profile.csv" "D:\___SJTU___\Undergraduate_Course\Senior 3\Scene_Thermal_Aware_RT_DETR\ECE4730J-Degree-Project-main\experiments\logs\"
```

PowerShell: run the detailed plot locally:

```powershell
cd "D:\___SJTU___\Undergraduate_Course\Senior 3\Scene_Thermal_Aware_RT_DETR\ECE4730J-Degree-Project-main"
.\.venv\Scripts\python.exe scripts\plot_inference_details.py --input experiments\logs\<run>.csv
```

PowerShell: copy generated JPG dashboard back to the Raspberry Pi:

```powershell
scp "D:\___SJTU___\Undergraduate_Course\Senior 3\Scene_Thermal_Aware_RT_DETR\ECE4730J-Degree-Project-main\experiments\results\<run>_inference_details.jpg" "ece473g1@ece473g1-desktop:~/ECE4730J-Degree-Project/experiments/results/"
```

The JPG dashboard includes:

- logged inference latency and profiled inference latency
- preprocess, feed build, ONNX run, and postprocess timing
- frame-source timing, including camera capture, lite ISP, resize, and save time
- runtime loop timing after the frame enters the system
- effective inference FPS
- temperature, power, CPU frequency, and ARM clock when available
- input resolution and query budget
- inference interval, CPU threads, and decoder layers
- fan enabled state, fan duty cycle, and fan mode when available
- workload, thermal state, action mode, and governor
- detection count and mean confidence

The x-axis is elapsed experiment time in minutes when the log contains
`timestamp`. It falls back to `frame_id` only for logs without timestamps.
Numeric trend lines use raw inference-row values by default. Use
`--smooth-window <N>` to add rolling smoothing.

Use a custom output path:

```bash
python scripts/plot_inference_details.py \
  --input experiments/logs/<run>.csv \
  --output experiments/results/<name>.jpg
```

Limit the number of inference events shown:

```bash
python scripts/plot_inference_details.py \
  --input experiments/logs/<run>.csv \
  --max-inferences 200
```

If the profile CSV is not named `<run>_profile.csv`, pass it explicitly:

```bash
python scripts/plot_inference_details.py \
  --input experiments/logs/<run>.csv \
  --profile experiments/logs/<profile_run>.csv
```

## 3. Baseline Statistics

Use this script to produce numeric summaries for reports or tables.

```bash
python scripts/summarize_baseline.py --input experiments/logs/<run>.csv
```

Example:

```bash
python scripts/summarize_baseline.py \
  --input experiments/logs/native_rtdetr_20260617_194557.csv \
  --label native_rtdetr_pi4
```

Default output directory:

```bash
experiments/results/baseline_native/
```

Outputs:

```bash
experiments/results/baseline_native/<label>_summary.json
experiments/results/baseline_native/<label>_summary.csv
```

PowerShell: copy the needed log from the Raspberry Pi to Windows:

```powershell
scp "ece473g1@ece473g1-desktop:~/ECE4730J-Degree-Project/experiments/logs/<run>.csv" "D:\___SJTU___\Undergraduate_Course\Senior 3\Scene_Thermal_Aware_RT_DETR\ECE4730J-Degree-Project-main\experiments\logs\"
```

PowerShell: run the statistics script locally:

```powershell
cd "D:\___SJTU___\Undergraduate_Course\Senior 3\Scene_Thermal_Aware_RT_DETR\ECE4730J-Degree-Project-main"
.\.venv\Scripts\python.exe scripts\summarize_baseline.py --input experiments\logs\<run>.csv --label <label>
```

PowerShell: copy generated statistics back to the Raspberry Pi:

```powershell
ssh ece473g1@ece473g1-desktop "mkdir -p ~/ECE4730J-Degree-Project/experiments/results/baseline_native"
scp "D:\___SJTU___\Undergraduate_Course\Senior 3\Scene_Thermal_Aware_RT_DETR\ECE4730J-Degree-Project-main\experiments\results\baseline_native\<label>_summary.json" "ece473g1@ece473g1-desktop:~/ECE4730J-Degree-Project/experiments/results/baseline_native/"
scp "D:\___SJTU___\Undergraduate_Course\Senior 3\Scene_Thermal_Aware_RT_DETR\ECE4730J-Degree-Project-main\experiments\results\baseline_native\<label>_summary.csv" "ece473g1@ece473g1-desktop:~/ECE4730J-Degree-Project/experiments/results/baseline_native/"
```

The summary includes:

- wall time
- total frames
- inference frames
- skipped frames
- latency mean, median, p95, p99, and max
- legacy FPS mean, median, min, and max
- loop FPS mean
- effective inference FPS mean, median, min, and max
- temperature start, end, mean, and max when available
- frequency and ARM clock statistics when available
- fan enabled ratio and fan duty cycle statistics when available
- detection and confidence averages

Use a custom output directory:

```bash
python scripts/summarize_baseline.py \
  --input experiments/logs/<run>.csv \
  --output-dir experiments/results/<analysis_name> \
  --label <label>
```

## 4. Stage Latency Summary

`scripts/run_experiment.py` and `scripts/run_live_dashboard.py` automatically
print this summary when an experiment finishes. Use this script manually when
you want to re-check an older log. It reads the main CSV and automatically looks
for the matching `<run>_profile.csv`.

For repeatability checks, `scripts/run_live_dashboard.py` can run the same
experiment multiple times. During the run it prints only compact progress and
log paths; after all repeated runs finish, it prints all full reports together
so the terminal tail still contains every report for copy/paste:

```bash
python scripts/run_live_dashboard.py \
  --config configs/raspberry_pi4.yaml \
  --strategy scene_thermal_interval_lk \
  --video data/sample.mp4 \
  --loop-video \
  --duration-min 10 \
  --repeat-runs 3
```

When `--output` is not set, each run gets a fresh timestamped log name with a
`_runNN` suffix. When `--output experiments/logs/my_run.csv` is set, repeated
runs write `my_run_run01.csv`, `my_run_run02.csv`, and so on. Add
`--repeat-cooldown-sec <seconds>` if you want a fixed pause between runs.

```bash
python scripts/summarize_stage_latency.py --input experiments/logs/<run>.csv
```

Example:

```bash
python scripts/summarize_stage_latency.py \
  --input experiments/logs/scene_thermal_interval_lk_live_20260701_143000.csv
```

PowerShell: copy the main log and profile log from the Raspberry Pi to Windows:

```powershell
scp "ece473g1@ece473g1-desktop:~/ECE4730J-Degree-Project/experiments/logs/<run>.csv" "D:\___SJTU___\Undergraduate_Course\Senior 3\Scene_Thermal_Aware_RT_DETR\ECE4730J-Degree-Project-main\experiments\logs\"
scp "ece473g1@ece473g1-desktop:~/ECE4730J-Degree-Project/experiments/logs/<run>_profile.csv" "D:\___SJTU___\Undergraduate_Course\Senior 3\Scene_Thermal_Aware_RT_DETR\ECE4730J-Degree-Project-main\experiments\logs\"
```

PowerShell: run the summary locally:

```powershell
cd "D:\___SJTU___\Undergraduate_Course\Senior 3\Scene_Thermal_Aware_RT_DETR\ECE4730J-Degree-Project-main"
.\.venv\Scripts\python.exe scripts\summarize_stage_latency.py --input experiments\logs\<run>.csv
```

Useful options:

Disable the automatic summary during a run:

```bash
python scripts/run_live_dashboard.py \
  --config configs/raspberry_pi4.yaml \
  --strategy scene_thermal_interval_lk \
  --camera imx219-raw \
  --no-stage-summary
```

```bash
python scripts/summarize_stage_latency.py \
  --input experiments/logs/<run>.csv \
  --inference-only
```

```bash
python scripts/summarize_stage_latency.py \
  --input experiments/logs/<run>.csv \
  --profile experiments/logs/<custom_profile>.csv
```

The script prints the 10 slowest inference rows by default. Change the number
or disable this table with:

```bash
python scripts/summarize_stage_latency.py \
  --input experiments/logs/<run>.csv \
  --top-slow-inferences 20
```

```bash
python scripts/summarize_stage_latency.py \
  --input experiments/logs/<run>.csv \
  --top-slow-inferences 0
```

The script also prints the 10 slowest device-monitor rows by default. Change or
disable this table with:

```bash
python scripts/summarize_stage_latency.py \
  --input experiments/logs/<run>.csv \
  --top-slow-device 20
```

```bash
python scripts/summarize_stage_latency.py \
  --input experiments/logs/<run>.csv \
  --top-slow-device 0
```

The printed summary includes:

- overall RT-DETR latency from the main CSV
- inference latency grouped by `tracking_reason`, including ROI refresh and
  full-frame forced-refresh groups
- LK-quality refresh confirmation counts, including deferred rows and
  full-frame LK-quality refresh rows when the CSV contains confirmation fields
- ROI candidate details, including rejection reason, latency, area ratio, and
  ROI width/height when the CSV contains ROI reviewer fields
- the slowest inference rows, including frame id, latency, ONNX runtime,
  scene/device stage time, temperature, frequency, throttling flags, ROI area,
  and tracking reason
- ONNX session selection time separately from ONNX execution time when available.
  New experiments warm up configured ONNX sessions before the timed loop so the
  first use of 320/480/640 models does not appear as an inference latency
  spike.
- Raspberry Pi configs also run a short pre-logging warmup before frame 0 is
  written to CSV: the first decoded frame is used for unlogged 640 full-frame
  and 320 ROI inference after a brief governor-settle delay, then the same frame
  is processed normally as logged frame 0. This is intended to remove CPU
  frequency ramp-up and ORT cold-start noise from the first measured inference.
- ROI slow-fuse behavior is intentionally permissive: the current default only
  pauses ROI after `roi_slow_fuse_consecutive_limit` consecutive ROI refreshes
  exceed `roi_slow_fuse_threshold_ms`. A fast ROI refresh resets the slow count.
  This keeps normal 320 ROI refreshes active while still guarding against
  repeated extreme ONNX tail latency.
- Current ROI configuration enables motion-triggered 320-resolution ROI refresh
  and warms up only the 640 full-frame session plus the 320 ROI session. The
  unused 480 session is not warmed up, which keeps memory pressure lower than
  the earlier 320/480/640 warmup experiment.
- loop FPS and inference FPS
- serial total frame time when available
- camera/video capture time
- IMX219 lite ISP time
- source resize and latest-frame save time
- producer-consumer wait, frame age, and dropped-frame counts when enabled
- source recoverable error counts, used for IMX219 empty raw or transient
  capture failures that were logged and recovered without stopping the run
- ONNX long-tail diagnostic context for inference rows. New profile logs record
  load average, available memory, process thread count, and background-camera
  active/pending/count/skip/error state immediately before and after each RT-DETR
  call. The summary groups ONNX runtime by background-camera overlap so a slow
  ROI row can be checked against camera activity without opening the CSV.
- the slowest inference rows include compact diagnostic columns: `bgS` and
  `bgE` show whether the background camera was active at inference start/end,
  `bgCap` shows camera captures completed during that inference, `bgSkip`
  shows skipped camera triggers during that inference, and `load1` / `memMB`
  show host load and available memory at inference start.
- the slowest device-monitor rows include `device_ms`, `firmware_poll_ms`,
  temperature, CPU/ARM frequency, `arm_clock_stale`, `throttling_stale`,
  throttling flags, and tracking reason. Use this table to check whether a
  device-monitor spike came from firmware polling, stale cached firmware state,
  or unrelated scheduler delay.
- runtime time after the frame enters the system
- scene, device, decision, inference, and log-write stage timings

Device monitor timing:

- `Device monitor` is the per-frame cost of `DeviceStateMonitor.snapshot()`.
  It reads CPU temperature from thermal sysfs, CPU frequency from cpufreq
  sysfs, ARM clock from `vcgencmd measure_clock arm`, power placeholder, and
  throttling flags from `vcgencmd get_throttled`.
- The runtime calls `DeviceStateMonitor.snapshot()` once per processed frame,
  before controller decision and fan update. Temperature and thermal state are
  therefore available to the strategy every frame.
- The controller primarily uses `temp_c` / `thermal_state`; throttling flags can
  elevate thermal pressure when firmware reports soft-temp-limit, capped ARM
  frequency, or active throttling. `arm_clock_mhz` is mainly diagnostic.
- The sysfs reads are normally cheap. The slow outliers observed on Raspberry
  Pi were most likely from `vcgencmd` subprocess calls waiting for their old
  timeout. The runtime now reads firmware state in a background monitor thread
  at most once per `thermal.firmware_poll_interval_sec` seconds, default `1.0`,
  uses a short `0.2 s` timeout for each `vcgencmd` call, and keeps the last
  valid firmware values for up to `thermal.firmware_cache_max_age_sec`, default
  `10.0`. The main loop reads only the cached values and does not wait for
  `vcgencmd`.
- Main CSV fields `arm_clock_stale`, `throttling_stale`, and
  `firmware_poll_ms` show whether the current row used stale cached firmware
  state and how much time the most recent background firmware poll cost. Even if
  a firmware read times out, temperature-based control still uses fresh sysfs
  temperature on the same frame, and the frame itself is not blocked by
  `vcgencmd`.
- `vcgencmd get_throttled` history bits are logged as
  `under_voltage_occurred`, `arm_freq_capped_occurred`,
  `throttled_occurred`, and `soft_temp_limit_occurred`. For example,
  `throttled=0x50000` means under-voltage and throttling occurred previously,
  even if the device is not throttled at the moment the command is run.

## 5. IMX219 Camera Timing Benchmark

Use this script on the Raspberry Pi to measure how long one IMX219 raw camera
frame takes, split into raw capture, raw conversion, resize, and optional JPEG
save.

Measure the current color-tuned lite ISP path:

```bash
sudo -E .venv/bin/python scripts/benchmark_imx219_camera.py \
  --count 30 \
  --warmup 3 \
  --imx219-runtime-mode lite-isp \
  --imx219-frame-width 640 \
  --imx219-frame-height 480 \
  --imx219-raw-output /dev/shm/camera_bench.raw \
  --no-imx219-save-latest \
  --csv-output experiments/logs/imx219_lite_isp_benchmark.csv
```

Measure the fast grayscale runtime path, which skips color tuning:

```bash
sudo -E .venv/bin/python scripts/benchmark_imx219_camera.py \
  --count 30 \
  --warmup 3 \
  --imx219-runtime-mode gray \
  --imx219-frame-width 640 \
  --imx219-frame-height 480 \
  --imx219-raw-output /dev/shm/camera_bench.raw \
  --no-imx219-save-latest \
  --csv-output experiments/logs/imx219_gray_benchmark.csv
```

PowerShell: copy benchmark CSV files from the Raspberry Pi to Windows:

```powershell
scp "ece473g1@ece473g1-desktop:~/ECE4730J-Degree-Project/experiments/logs/imx219_lite_isp_benchmark.csv" "D:\___SJTU___\Undergraduate_Course\Senior 3\Scene_Thermal_Aware_RT_DETR\ECE4730J-Degree-Project-main\experiments\logs\"
scp "ece473g1@ece473g1-desktop:~/ECE4730J-Degree-Project/experiments/logs/imx219_gray_benchmark.csv" "D:\___SJTU___\Undergraduate_Course\Senior 3\Scene_Thermal_Aware_RT_DETR\ECE4730J-Degree-Project-main\experiments\logs\"
```

Use `--imx219-save-latest` only when you need to include JPEG writing cost in
the measurement. The normal runtime can use `--no-imx219-save-latest` to avoid
that extra file I/O.

For controlled full-runtime tests with a background camera, the default trigger
mode is:

```bash
--background-camera-trigger during-tracking
```

This sends one asynchronous camera-capture request when LK tracking starts, so
the camera can capture while tracking runs. If the previous camera capture is
still active, the new request is skipped instead of queued. This avoids
capturing during normal RT-DETR inference frames and avoids accumulated camera
work. In this mode, the background camera ignores
`--imx219-capture-interval-sec` so each accepted request starts immediately.
Use `--background-camera-trigger post-tracking` for synchronous capture after
tracking, or `--background-camera-trigger continuous` when intentionally
measuring the older always-on camera load.

`run_live_dashboard.py` suppresses verbose `v4l2-ctl` capture output. During a
background-camera run it prints compact progress only every 20 completed camera
captures, including completed captures, skipped camera triggers, and total
trigger requests. The final repeated-run report still prints the total captured
frames and skipped triggers for each run.

## Suggested Workflow After A Test

After running an experiment:

```bash
python scripts/run_experiment.py \
  --config configs/raspberry_pi4.yaml \
  --strategy scene_thermal_coadaptive \
  --video data/sample.mp4 \
  --loop-video \
  --frame-width 640 \
  --frame-height 480 \
  --duration-min 15
```

Use `--loop-video` when the input video is shorter than the planned experiment
duration. The runtime keeps the same process alive and restarts the video from
the first frame when it reaches EOF.

Use `--frame-width` and `--frame-height` to resize video input before it enters
scene analysis, LK tracking, and RT-DETR. This is useful when comparing the
sample video with camera runs at a fixed workload, for example 640x480.

For `scene_thermal_coadaptive` on Raspberry Pi configs, active fan cooling is
enabled by default with DC GPIO output:

```yaml
fan:
  enabled: true
  strategy: dc
  gpio_mode: BCM
  on_temp_c: 68.0
  off_temp_c: 63.0
  dc_high_pin: 17
  dc_low_pin: 27
```

DC mode turns the fan on by setting `dc_high_pin` high and `dc_low_pin` low.
When cooling is off, both pins are set low. PWM mode is already implemented;
switch to it by changing `strategy: pwm` and setting `pwm_pin`,
`min_duty_cycle`, `max_duty_cycle`, and `pwm_full_temp_c`.

To simulate thermal policy behavior without waiting for the device to heat up:

```bash
python scripts/run_experiment.py \
  --config configs/raspberry_pi4.yaml \
  --strategy scene_thermal_coadaptive \
  --dry-run \
  --video data/sample.mp4 \
  --loop-video \
  --duration-min 1 \
  --thermal-temp-c 78
```

To find the passive-cooling lower bound, run the fixed ultra-low workload
strategy. This strategy does not use scene adaptation, thermal adaptation, or
fan control; it keeps a fixed 320 px input, interval 16, one CPU thread,
`powersave`, decoder layer hint 2, and query budget hint 40.

```bash
python scripts/run_experiment.py \
  --config configs/raspberry_pi4.yaml \
  --strategy fixed_ultra_low_power \
  --video data/sample.mp4 \
  --loop-video \
  --duration-min 15
```

find the newest log:

```bash
ls -lt experiments/logs/
```

then run:

```bash
python scripts/plot_results.py --input experiments/logs/<run>.csv
python scripts/plot_inference_details.py --input experiments/logs/<run>.csv
python scripts/summarize_baseline.py --input experiments/logs/<run>.csv --label <run>
```

Check the generated files:

```bash
ls -lt experiments/results/
```

## Copy Code Changes To Raspberry Pi

When code or documentation changes locally, copy each changed file separately
from PowerShell. Keep the same relative path under the Raspberry Pi project root.

Template:

```powershell
scp "D:\___SJTU___\Undergraduate_Course\Senior 3\Scene_Thermal_Aware_RT_DETR\ECE4730J-Degree-Project-main\<relative-path>" "ece473g1@ece473g1-desktop:~/ECE4730J-Degree-Project/<relative-path>"
```

Example:

```powershell
scp "D:\___SJTU___\Undergraduate_Course\Senior 3\Scene_Thermal_Aware_RT_DETR\ECE4730J-Degree-Project-main\scripts\plot_inference_details.py" "ece473g1@ece473g1-desktop:~/ECE4730J-Degree-Project/scripts/plot_inference_details.py"
scp "D:\___SJTU___\Undergraduate_Course\Senior 3\Scene_Thermal_Aware_RT_DETR\ECE4730J-Degree-Project-main\docs\post_experiment_analysis.md" "ece473g1@ece473g1-desktop:~/ECE4730J-Degree-Project/docs/post_experiment_analysis.md"
```

## Maintenance Rule

Every future script that reads experiment logs and produces a plot, summary,
table, or report must be added to this guide. Include:

- the command name
- required inputs
- optional arguments
- output file names
- a short explanation of when to use it
- PowerShell `scp` commands for copying required Raspberry Pi inputs to Windows
- PowerShell `scp` commands for copying generated plots or summaries back to the Raspberry Pi

After every code change, report:

- the changed files
- one `scp` command per changed file for copying it from Windows to the Raspberry Pi

When an implementation change is kept because an experiment report shows a
positive effect, also update:

```bash
docs/experiment_change_log.md
```

Record the change, the representative experiment report values, and the reason
the change remains in the codebase. Keep unvalidated changes in that file's
pending section until a Raspberry Pi report confirms the effect.
