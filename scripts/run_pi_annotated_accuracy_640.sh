#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-run-all}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="${PYTHON:-python3}"
VENV="${VENV:-.venv}"
OUT_ROOT="${OUT_ROOT:-experiments/logs/pi_annotated_accuracy_640}"
RESULT_ROOT="${RESULT_ROOT:-experiments/results/pi_annotated_accuracy_640}"
VIDEO="${VIDEO:-data/sample3.mp4}"
ANNOTATIONS="${ANNOTATIONS:-data/annotations/sample3_50frames_640}"
MAX_FRAMES="${MAX_FRAMES:-214}"
THREADS="${THREADS:-4}"

run_setup() {
  "$PY" -m venv "$VENV"
  # shellcheck disable=SC1091
  source "$VENV/bin/activate"
  python -m pip install --upgrade pip
  python -m pip install \
    numpy==1.26.4 opencv-python-headless PyYAML onnxruntime \
    ultralytics onnx onnxslim gdown \
    torch torchvision omegaconf termcolor pycocotools pytorch-lightning==1.9.5 \
    torchmetrics pyaml imagesize tabulate "setuptools<81"

  if [ ! -d third_party/nanodet ]; then
    git clone --depth 1 https://github.com/RangiLyu/nanodet.git third_party/nanodet
  fi
  if [ ! -d third_party/PaddleDetection ]; then
    git clone --depth 1 --branch release/2.7 https://github.com/PaddlePaddle/PaddleDetection.git third_party/PaddleDetection
  fi
}

run_download() {
  # shellcheck disable=SC1091
  source "$VENV/bin/activate"
  mkdir -p models/baselines
  python scripts/prepare_sota_baseline_artifacts.py --models nanodet_plus_m_320 pp_picodet_l_640
  python - <<'PY'
from ultralytics import YOLO
model = YOLO("yolov8n.pt")
model.export(format="onnx", imgsz=640, opset=12, simplify=True)
PY
  mv -f yolov8n.pt models/baselines/yolov8n.pt 2>/dev/null || true
  mv -f yolov8n.onnx models/baselines/yolov8n_640.onnx 2>/dev/null || true
}

run_project_methods() {
  # shellcheck disable=SC1091
  source "$VENV/bin/activate"
  mkdir -p "$OUT_ROOT"/{rtdetr_native_640,lk_tracking_640,proposed_software_640}

  python scripts/run_experiment.py \
    --config configs/raspberry_pi4.yaml \
    --strategy native_rtdetr \
    --video "$VIDEO" \
    --max-frames "$MAX_FRAMES" \
    --output "$OUT_ROOT/rtdetr_native_640/runtime.csv" \
    --log-detections \
    --model models/rtdetr_r18_lite_pi4_640.onnx \
    --fixed-input-resolution 640 \
    --enable-thread-sessions \
    --thread-session-counts "$THREADS" \
    --apply-runtime-actions

  python scripts/run_experiment.py \
    --config configs/raspberry_pi4.yaml \
    --strategy scene_track_lk \
    --video "$VIDEO" \
    --max-frames "$MAX_FRAMES" \
    --output "$OUT_ROOT/lk_tracking_640/runtime.csv" \
    --log-detections \
    --model models/rtdetr_r18_lite_pi4_640_int8_dynamic_q.onnx \
    --fixed-input-resolution 640 \
    --query-budget-override 300 \
    --query-budget-mode strict \
    --enable-thread-sessions \
    --thread-session-counts "$THREADS" \
    --apply-runtime-actions

  python scripts/run_experiment.py \
    --config configs/raspberry_pi4.yaml \
    --strategy scene_thermal_interval_lk \
    --video "$VIDEO" \
    --max-frames "$MAX_FRAMES" \
    --output "$OUT_ROOT/proposed_software_640/runtime.csv" \
    --log-detections \
    --model-paths-by-resolution "320=models/rtdetr_r18_lite_pi4_320_int8_dynamic_q.onnx,640=models/rtdetr_r18_lite_pi4_640_int8_dynamic_q.onnx" \
    --query-budget-override 300 \
    --query-budget-mode strict \
    --enable-thread-sessions \
    --thread-session-counts "$THREADS" \
    --apply-runtime-actions
}

run_accuracy() {
  # shellcheck disable=SC1091
  source "$VENV/bin/activate"
  export PYTHONPATH="$ROOT/src:$ROOT/third_party/nanodet:$ROOT/third_party/PaddleDetection:${PYTHONPATH:-}"
  export MPLCONFIGDIR="$ROOT/.plot_mplconfig"
  export YOLO_CONFIG_DIR="$ROOT/Ultralytics"
  mkdir -p "$RESULT_ROOT"

  python scripts/run_annotated_sota_accuracy.py \
    --suite-dir experiments/empty_suite_for_extra_predictions \
    --annotations-dir "$ANNOTATIONS" \
    --output-dir "$RESULT_ROOT" \
    --predictions-dir "$OUT_ROOT/sota_frame_predictions" \
    --nanodet-input-size 640 \
    --extra-prediction "rtdetr_native_pi640|RT-DETR native Pi 640|$OUT_ROOT/rtdetr_native_640/runtime_detections.jsonl" \
    --extra-prediction "lk_tracking_pi640|LK tracking Pi 640|$OUT_ROOT/lk_tracking_640/runtime_detections.jsonl" \
    --extra-prediction "proposed_software_pi640|Proposed software Pi 640|$OUT_ROOT/proposed_software_640/runtime_detections.jsonl"
}

run_picodet_visual_probe() {
  # Optional: requires Paddle/PaddleLite-compatible environment on the Pi.
  # It writes visual detections; structured PicoDet boxes are not consumed by
  # run_annotated_sota_accuracy.py yet.
  # shellcheck disable=SC1091
  source "$VENV/bin/activate"
  python third_party/PaddleDetection/deploy/python/infer.py \
    --model_dir models/baselines/picodet_l_640_coco_lcnet_portable \
    --image_dir "$ANNOTATIONS" \
    --output_dir "$OUT_ROOT/picodet_l_640_visual_probe" \
    --device CPU \
    --threshold 0.25
}

case "$MODE" in
  setup) run_setup ;;
  download) run_download ;;
  run-project) run_project_methods ;;
  accuracy) run_accuracy ;;
  picodet-probe) run_picodet_visual_probe ;;
  run-all)
    run_setup
    run_download
    run_project_methods
    run_accuracy
    ;;
  *)
    echo "usage: $0 {setup|download|run-project|accuracy|picodet-probe|run-all}" >&2
    exit 2
    ;;
esac
