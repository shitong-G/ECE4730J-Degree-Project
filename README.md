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

## Run Experiments
```bash
python scripts/run_experiment.py --config configs/raspberry_pi4.yaml --strategy native_rtdetr --video data/sample.mp4 --duration-min 15
```
For detailed Experiment intructions, see Experiments.md 

## Logs

Logs are written to `experiments/logs/` by default. See root `README.md` for the column schema.

## Results

After a run, generate plots:

```bash
python scripts/plot_results.py --input experiments/logs/<run>.csv
```

Output figures go to `experiments/results/`.
