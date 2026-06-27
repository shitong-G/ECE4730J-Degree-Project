# Experiment Protocol

## Protocol Files

| Protocol | Duration | Purpose |
| -------- | -------- | ------- |
| `experiments/protocols/15min_smoke_test.yaml` | 15 min | Quick validation |
| `experiments/protocols/30min_sustained_test.yaml` | 30 min | Sustained load |
| `experiments/protocols/60min_sustained_test.yaml` | 60 min | Thermal stress |

## Standard Run

```bash
python scripts/run_experiment.py \
  --config configs/raspberry_pi4.yaml \
  --strategy <strategy_name> \
  --video data/sample.mp4 \
  --duration-min 30 \
  --output experiments/logs/<run_id>.csv
```

## Native RT-DETR Baseline

Use `native_rtdetr` as the non-adaptive baseline: no scene policy, no thermal policy,
no frame skipping, fixed CPU settings, and the exported RT-DETR-R18 ONNX model.

For Raspberry Pi 4 (1GB), run at least three repeats with cooldown between runs:

```bash
mkdir -p experiments/logs/baseline_native experiments/results/baseline_native

python scripts/run_baseline.py \
  --config configs/raspberry_pi4_1gb.yaml \
  --strategy native_rtdetr \
  --video data/sample.mp4 \
  --duration-min 5 \
  --output experiments/logs/baseline_native/native_rtdetr_run1.csv

python scripts/summarize_baseline.py \
  --input experiments/logs/baseline_native/native_rtdetr_run1.csv \
  --output-dir experiments/results/baseline_native \
  --label native_rtdetr_run1

python scripts/plot_results.py \
  --input experiments/logs/baseline_native/native_rtdetr_run1.csv \
  --output-dir experiments/results/baseline_native
```

If the 1GB board shows out-of-memory kills or extreme swap stalls, reduce the duration
for the first smoke run to 1 minute, but keep the final baseline duration consistent
across all strategies.

## Strategies to Compare

Run each strategy on the **same video** and **same duration** for fair comparison:

1. `native_rtdetr` — non-adaptive native RT-DETR baseline
2. `fixed_low_power` — baseline static low power
3. `fixed_frame_skip` — baseline static skip
4. `static_affinity` — baseline affinity pinning
5. `thermal_only` — thermal adaptation only
6. `scene_only` — scene adaptation only
7. `scene_thermal_coadaptive` — full co-adaptation (proposed)
8. `default` — balanced default policy

Current interpretation:

- `native_rtdetr`, `fixed_low_power`, and `fixed_frame_skip` are reliable baselines.
- `thermal_only` is meaningful for validating the current temperature controller.
- `scene_only` and the scene part of `scene_thermal_coadaptive` require `SceneWorkloadEstimator.classify_workload()` to be implemented before making strong scene-aware claims.
- `cpu_threads`, governor, `decoder_layers`, and `query_budget` should be reported as policy outputs unless their ONNX/OS enforcement is separately verified.

## Metrics to Record

Per-frame logs include: workload, raw/control thermal state, action mode, whether inference ran, throttling flags, temperature, frequency, latency, FPS, actual inference FPS, resolution, interval, threads, detection count, and confidence statistics.

For baseline tables, report both raw per-frame CSV and summary metrics from
`scripts/summarize_baseline.py`: wall time, frame count, inference frame count,
latency mean/median/p95/p99/max, FPS mean/median/min/max, temperature start/end/mean/max,
average CPU frequency, detection count, and confidence mean.

## Analysis

```bash
python scripts/plot_results.py --input experiments/logs/<run_id>.csv
```

## Reproducibility Checklist

- [ ] Record Pi model, OS version, kernel
- [ ] Record model ONNX path and export commit
- [ ] Record ambient temperature
- [ ] Use identical video and protocol duration
- [ ] Archive log CSV and config YAML with results
