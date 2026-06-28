# Thermal-Aware Experiment Design

## Goal

Evaluate whether thermal-aware scheduling can reduce sustained high-temperature exposure and throttling risk while preserving useful inference throughput.

## Strategies

Run the following strategies under identical video, cooling, and duration:

```text
native_rtdetr
fixed_frame_skip
fixed_low_power
thermal_only
thermal_balanced
```

Interpretation:

- `native_rtdetr`: no thermal control, highest load baseline.
- `fixed_frame_skip`: static inference-rate reduction.
- `fixed_low_power`: static low-power baseline.
- `thermal_only`: aggressive safety controller.
- `thermal_balanced`: performance-oriented thermal-budget controller. It keeps
  more threads/resolution in warm and low-hot states, then steps down only near
  critical temperature.

## Recommended Command

```bash
python scripts/run_thermal_experiment_suite.py \
  --config configs/raspberry_pi4.yaml \
  --video data/sample.mp4 \
  --loop-video \
  --duration-min 15 \
  --repeats 3 \
  --cooldown-temp-c 50 \
  --cooldown-poll-sec 15 \
  --max-cooldown-min 30 \
  --enable-thread-sessions \
  --thread-session-counts 1,2,3,4 \
  --apply-runtime-actions
```

Use `--skip-plot` if pandas/matplotlib are unavailable on the Pi.

For a long SSH-safe run:

```bash
nohup python -u scripts/run_thermal_experiment_suite.py \
  --config configs/raspberry_pi4.yaml \
  --video data/sample.mp4 \
  --loop-video \
  --duration-min 15 \
  --repeats 3 \
  --cooldown-temp-c 50 \
  --cooldown-poll-sec 15 \
  --max-cooldown-min 30 \
  --enable-thread-sessions \
  --thread-session-counts 1,2,3,4 \
  --apply-runtime-actions \
  > thermal_suite_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

To test real CPU-thread control through ONNX Runtime sessions, use
`--enable-thread-sessions --thread-session-counts 1,2,3,4`.
This preloads one ONNX Runtime session per thread count and selects the session
matching `RuntimeAction.cpu_threads`. On low-memory boards, use fewer sessions,
for example `--thread-session-counts 1,2`.

To test best-effort OS control, add:

```bash
--apply-runtime-actions
```

This attempts to apply CPU governor and process affinity. Governor control
usually requires root permissions; failures are logged through requested/applied
fields and `governor_apply_error` rather than aborting the experiment.

Before running the suite, diagnose OS support:

```bash
python scripts/check_runtime_action_support.py
python scripts/check_runtime_action_support.py --apply --governor performance --affinity 0,1,2,3
```

If governor writes fail with permission/readback errors, run the experiment with
root privileges or grant passwordless write access to the cpufreq sysfs nodes.
On a standard Raspberry Pi OS install, process affinity can work as a normal
user, but `scaling_governor` normally cannot be written without `sudo`.

## Required Controls

- Use the same video and loop setting for every run.
- Keep cooling hardware and placement unchanged.
- Start each run only after CPU temperature is at or below the cooldown target.
- Record ambient condition if possible.
- Do not run other heavy processes during experiments.

## Primary Metrics

Report these from the generated summary CSV/JSON:

```text
temp_c_max
temp_c_mean
time_above_70c_sec
time_above_75c_sec
time_above_80c_sec
currently_throttled_ratio
soft_temp_limit_ratio
skip_ratio
actual_inference_fps_mean
latency_ms_mean
latency_ms_p95
arm_clock_mhz_min
action_mode_counts
decision_reason_counts
thermal_pressure_level_counts
governor_applied_ratio
governor_apply_error_counts
cpu_affinity_applied_ratio
cpu_affinity_apply_error_counts
```

## Expected Claims

Safe claim if supported:

```text
Thermal-aware scheduling reduces high-temperature exposure and throttling risk
relative to static baselines by dynamically increasing the inference interval.
```

Balanced-controller claim if supported:

```text
thermal_balanced preserves higher actual inference FPS than thermal_only while
keeping throttling ratio near zero.
```

Avoid claiming lower average temperature unless the repeated controlled runs show it clearly.
