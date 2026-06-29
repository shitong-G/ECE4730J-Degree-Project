# Experiments

## Directory Layout

```
experiments/
  protocols/     # YAML experiment definitions
  logs/          # Per-run CSV/JSONL (gitignored except .gitkeep)
  results/       # Aggregated plots and tables
```

## Quick Smoke Test

```bash
python scripts/run_experiment.py --dry-run --strategy scene_thermal_coadaptive --duration-min 1
```

## Logs

Logs are written to `experiments/logs/` by default. See root `README.md` for the column schema.

## Results

After a run, generate plots:

```bash
python scripts/plot_results.py --input experiments/logs/<run>.csv
```

Output figures go to `experiments/results/`.

For all post-run plotting and statistical analysis scripts, see
`docs/post_experiment_analysis.md`.

## Thermal Suite

For repeated thermal-aware comparison on Raspberry Pi, wait for the board to cool
before each next run:

```bash
python scripts/run_thermal_experiment_suite.py \
  --config configs/raspberry_pi4.yaml \
  --video data/sample.mp4 \
  --loop-video \
  --duration-min 15 \
  --repeats 3 \
  --cooldown-temp-c 55 \
  --cooldown-poll-sec 15 \
  --max-cooldown-min 30
```

The default thermal suite compares:

```text
native_rtdetr
fixed_frame_skip
fixed_low_power
thermal_only
thermal_balanced
```

Use `--skip-plot` if pandas/matplotlib are not installed on the Pi.

## Live Dashboard

Run an experiment with a browser dashboard for remote monitoring:

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

Open `http://<raspberry-pi-ip>:8000` from another device on the same network.
The page shows the live detection stream, temperature, latency, FPS, resolution,
governor/thread state, throttling flags, and rolling performance curves.

For lower network load, add:

```bash
--jpeg-width 640 --jpeg-quality 65
```

If only curves are needed, disable video streaming:

```bash
--no-video-stream
```

The repeated thermal suite can also use the same live page:

```bash
sudo -E .venv/bin/python scripts/run_thermal_experiment_suite.py \
  --config configs/raspberry_pi4.yaml \
  --video data/sample.mp4 \
  --loop-video \
  --duration-min 15 \
  --repeats 3 \
  --cooldown-temp-c 55 \
  --enable-thread-sessions \
  --thread-session-counts 1,2,3,4 \
  --apply-runtime-actions \
  --dashboard \
  --dashboard-port 8000
```

Optional real runtime knobs:

```bash
--enable-thread-sessions --thread-session-counts 1,2,3,4
--apply-runtime-actions
```

Thread sessions make `cpu_threads` select among pre-created ONNX Runtime
sessions. Runtime actions apply governor/affinity best-effort and log
requested/applied state.
