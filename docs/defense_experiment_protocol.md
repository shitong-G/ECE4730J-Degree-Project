# Defense experiment protocol

## What to run

The full defense matrix is implemented by
`scripts/run_defense_experiment_suite.py`. Conditions shared by multiple
questions are run once and reused.

| Run | Defense group | Condition | Detector model(s) | Fan during formal run |
|---|---|---|---|---|
| 1 | A, B | Native FP32, every frame | native 640 | off |
| 2 | B, C | INT8, every frame | INT8 640 | off |
| 3 | C | Fixed detector interval 2, no LK | INT8 640 | off |
| 4 | C | Fixed detector interval 5, no LK | INT8 640 | off |
| 5 | C | Fixed detector interval 10, no LK | INT8 640 | off |
| 6 | C | Periodic detector interval 5 + LK | INT8 640 | off |
| 7 | C, D | Event-triggered LK, ROI disabled | INT8 640 | off |
| 8 | D | Event LK + ROI, full query budget baseline (Q=300) | dynamic-query INT8 320/640 | off |
| 9 | D | Event LK + ROI, fixed reduced query budget (Q=64) | dynamic-query INT8 320/640 | off |
| 10 | D | Event LK + ROI, thermal-adaptive Q=64/48/40/32 | dynamic-query INT8 320/640 | off |
| 11 | D, E | Full controller with runtime query-budget allocation | dynamic-query INT8 320/480/640 | off |
| 12 | E | Native FP32 + threshold/PWM fan | native 640 | on |
| 13 | E | Full software controller + threshold/PWM fan | dynamic-query INT8 320/480/640 | on |

The adaptive query controller uses `Q=64/48/40/32` for normal/warm/hot/critical
thermal states.  Historical native runs provide an empirical lower-bound
indicator: the mean final detection count was 17.43 per frame, with P95/P99 of
23/24 and a maximum of 26 over 1,185 frames.  Query slots are candidates before
classification and filtering, so the schedule reserves additional capacity for
duplicate candidates, low-confidence objects, transient scene changes, and
imperfect query ranking.  The Q=32 state is therefore evaluated as graceful
degradation under thermal stress, not as a claim of full accuracy.  Q=300
remains an explicit full-budget ablation.

Group meanings:

- A: sustained native FP32 thermal baseline.
- B: FP32 versus INT8.
- C: every-frame, fixed skip, periodic LK, and event-triggered LK.
- D: LK, ROI, real graph query-budget, and software thermal-controller ablation.
- E: fan-only, software-only, and software-plus-fan comparison.

## Controlled protocol

- Every formal condition lasts 20 minutes by default.
- Before every condition, the cooldown fan runs at 100% until CPU temperature
  is at most `50.0 + 0.5 °C`, then the fan releases GPIO.
- Before the suite and after every formal condition, the runner rejects current
  or historical undervoltage (`get_throttled` bits 0/16). Fix the PSU/cable and
  reboot before collecting thesis data.
- Only the first condition performs unlogged RT-DETR warmup. It always runs at
  least one detector inference and stops after reaching 50 °C.
- The passive conditions explicitly disable their fan throughout the formal
  measurement.
- Fan conditions use temperature-only thresholds and PWM; controller action
  modes cannot switch the fan on by themselves.
- Per-frame boxes are saved for pseudo-label quality analysis.
- The suite applies the configured governor/affinity by default so every
  ablation uses the requested performance state. Use
  `--no-apply-runtime-actions` only for a separate native-OS experiment.

The fan preflight spins the fan for one second before the suite. This verifies
real GPIO PWM and is not part of any formal run.

## Raspberry Pi command

Generate the dynamic-query variants once. The tool rewires both the decoder
uncertainty-minimal TopK and the final prediction TopK; it does not overwrite
the static INT8 models.

Strict query-budget runs validate the dynamic-query metadata and reject a model
unless both `/model/decoder/TopK` and `/postprocessor/TopK` use the
`query_budget` graph input. Only `query_budget_mode=graph_input` is counted as
decoder-computation reduction; `postprocess_only` is explicitly warned and is
not an acceleration result.

```bash
for R in 320 480 640; do
  .venv/bin/python tools/make_dynamic_query_onnx.py \
    --model "models/rtdetr_r18_lite_pi4_${R}_int8.onnx" \
    --output "models/rtdetr_r18_lite_pi4_${R}_int8_dynamic_q.onnx"
done
```

An individual graph can be checked before a run with:

```bash
python tools/validate_dynamic_query_onnx.py \
  --model models/rtdetr_r18_lite_pi4_640_int8_dynamic_q.onnx
```

```bash
sudo -E .venv/bin/python -u scripts/run_defense_experiment_suite.py \
  --config configs/raspberry_pi4.yaml \
  --video data/sample.mp4 \
  --groups ABCDE \
  --duration-min 20 \
  --cooldown-temp-c 50 \
  --cooldown-tolerance-c 0.5 \
  --first-warmup-temp-c 50 \
  --native-model models/rtdetr_r18_lite_pi4_640.onnx \
  --quantized-model-320 models/rtdetr_r18_lite_pi4_320_int8.onnx \
  --quantized-model-480 models/rtdetr_r18_lite_pi4_480_int8.onnx \
  --quantized-model-640 models/rtdetr_r18_lite_pi4_640_int8.onnx \
  --dynamic-query-model-320 models/rtdetr_r18_lite_pi4_320_int8_dynamic_q.onnx \
  --dynamic-query-model-480 models/rtdetr_r18_lite_pi4_480_int8_dynamic_q.onnx \
  --dynamic-query-model-640 models/rtdetr_r18_lite_pi4_640_int8_dynamic_q.onnx
```

Inspect commands without starting an experiment:

```bash
python scripts/run_defense_experiment_suite.py --groups ABCDE --plan-only
```

The query-budget sensitivity sweep is a separate ablation.  It holds the
scene/LK/ROI policy and resolution family fixed and runs exactly
`Q=32,48,64,100,300`:

```bash
sudo -E .venv/bin/python -u scripts/run_query_budget_sweep.py \
  --config configs/raspberry_pi4.yaml \
  --video data/sample.mp4 \
  --duration-min 20 \
  --teacher-cycle-frames 428
```

Its `analysis/` directory contains `quality_summary.csv`,
`query_budget_summary.csv`, `08_query_budget_tradeoff.png`, and
`query_budget_timeline.png`.

The 13 formal runs alone require 260 minutes. Cooldown and model startup make
the wall-clock time longer. If the immediate goal is only the five-condition
quantization/LK/ROI/fan ablation, use `scripts/run_core_ablation_suite.py`
instead.

If a child run finishes but the suite stops during the power preflight, resume
the same directory after fixing the power supply and rebooting:

```bash
sudo -E .venv/bin/python -u scripts/run_defense_experiment_suite.py \
  --resume-dir experiments/defense_suite/defense_YYYYMMDD_HHMMSS \
  --groups ABCDE \
  --config configs/raspberry_pi4.yaml \
  --video data/sample.mp4
```

Successful conditions are retained and only incomplete conditions are started.
The resume command must use the same model paths and experiment parameters as
the original run. If the power check fails immediately after a condition, use
`--resume-rerun-from-index N` for that condition; its old directory is moved to
an `_invalidated_power_*` backup before rerunning.

## Saved results

The default root is:

```text
experiments/defense_suite/defense_YYYYMMDD_HHMMSS/
```

Every condition contains:

- `runtime.csv`: per-frame performance, thermal, fan, LK, ROI, and detector-use
  metrics.
- `runtime_profile.csv`: per-stage timing.
- `runtime_detections.jsonl`: per-frame class, score, and bounding boxes.
- `temperature_trace.csv`: independent wall-clock temperature trace.
- `cooldown_trace.csv`: pre-condition temperature and cooldown-fan trace.
- `run_manifest.json`: exact command, hashes, system state, and temperatures.

`runtime.csv` distinguishes:

- `detector_invocation_count/ratio`: all full and ROI detector calls.
- `full_detector_invocation_count/ratio`: full-frame detector calls.
- `roi_detector_invocation_count/ratio`: ROI detector calls.
- `detector_call_type`: `full`, `roi`, or empty on a non-detector frame.
- `detector_call_resolution`: the actual 320/480/640 model selected for that
  detector call.
- `query_budget_requested/applied`: requested and graph-applied query counts.
- `query_budget_mode`: only `graph_input` represents decoder-compute reduction;
  `postprocess_only` is output truncation and must not be reported as an
  inference acceleration mechanism.
- `query_budget_source` and `query_budget_temperature_state`: fixed, action, or
  temperature-driven budget selection and its hysteretic thermal state.
- `query_output_count`: raw ONNX output width before the score threshold; for a
  correctly converted graph this must equal the graph-applied budget.
- `query_budget_ratio`: `query_budget_applied / 300`, the normalized query
  computation proxy. `onnx_run_ms` records the actual ONNX execution stage for
  the query sweep.

The suite-level `analysis/` directory contains:

- `quality_summary.csv` and `quality_frames.csv`.
- `defense_summary.csv`, including early-versus-late latency/FPS/clock drift.
- `01_temperature_vs_time.png`.
- `02_latency_vs_time.png`.
- `03_fps_vs_time.png`.
- `04_detector_invocation_ratio.png`.
- `05_latency_accuracy_tradeoff.png`.
- `06_temperature_performance_tradeoff.png`.
- `07_ablation_bar_chart.png`.
- `08_query_budget_tradeoff.png`.
- `query_budget_timeline.png`, showing thermal state, selected Q, and detector
  invocation over time.
- `query_budget_summary.csv`, containing latency, FPS, temperature increase,
  pseudo recall, IoU, detector invocation ratio, and query-budget ratio.

Pseudo recall, precision proxy, and matched IoU are agreement with the native
FP32 teacher, not ground-truth mAP. State this explicitly in the thesis and
defense.
