# Experiment Protocol

## Protocol Files

| Protocol | Duration | Purpose |
|----------|----------|---------|
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

## Strategies to Compare

Run each strategy on the **same video** and **same duration** for fair comparison:

1. `fixed_low_power` — baseline static low power
2. `fixed_frame_skip` — baseline static skip
3. `static_affinity` — baseline affinity pinning
4. `thermal_only` — thermal adaptation only
5. `scene_only` — scene adaptation only
6. `scene_thermal_coadaptive` — full co-adaptation (proposed)
7. `default` — balanced default policy

## Metrics to Record

Per-frame logs include: workload, temperature, frequency, latency, FPS, resolution, interval, threads, detection count, confidence statistics.

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
