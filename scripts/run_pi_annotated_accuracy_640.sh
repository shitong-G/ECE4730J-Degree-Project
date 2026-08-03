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
INSTALL_TORCH="${INSTALL_TORCH:-cpu}"

run_setup() {
  if [ "$INSTALL_TORCH" = "apt" ]; then
    "$PY" -m venv --system-site-packages "$VENV"
  else
    "$PY" -m venv "$VENV"
  fi
  # shellcheck disable=SC1091
  source "$VENV/bin/activate"
  python -m pip install --upgrade pip
  python -m pip install \
    numpy==1.26.4 opencv-python-headless PyYAML onnxruntime \
    onnx onnxslim gdown \
    omegaconf termcolor pycocotools pyaml imagesize tabulate six \
    "setuptools<81"

  if [ "$INSTALL_TORCH" = "apt" ]; then
    # Do not let pip resolve Lightning's torch dependency into the venv.
    # Raspberry Pi must use the distro ARM build of PyTorch in this mode.
    python -m pip install --no-deps pytorch-lightning==1.9.5 torchmetrics
  else
    python -m pip install pytorch-lightning==1.9.5 torchmetrics
  fi

  if [ "$INSTALL_TORCH" = "cpu" ]; then
    python -m pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
  elif [ "$INSTALL_TORCH" = "apt" ]; then
    echo "INSTALL_TORCH=apt selected; install torch outside the venv first:"
    echo "  sudo apt update && sudo apt install -y python3-torch python3-torchvision"
    python - <<'PY'
from pathlib import Path

try:
    import torch
    torch_file = Path(torch.__file__).resolve()
    print(f"using system torch: {torch.__version__} from {torch_file}")
    if ".venv" in torch_file.parts:
        raise SystemExit(
            "INSTALL_TORCH=apt is selected, but torch is still imported from "
            f"the virtualenv: {torch_file}. Run: deactivate; rm -rf .venv; "
            "sudo apt install -y python3-torch python3-torchvision; "
            "INSTALL_TORCH=apt ./scripts/run_pi_annotated_accuracy_640.sh setup"
        )
except Exception as exc:
    raise SystemExit(
        "INSTALL_TORCH=apt requires apt-installed python3-torch visible inside "
        f"the venv; import failed: {exc}"
    )
PY
  elif [ "$INSTALL_TORCH" = "skip" ]; then
    echo "Skipping torch/torchvision install. NanoDet and Ultralytics .pt export need torch."
  else
    echo "Unknown INSTALL_TORCH=$INSTALL_TORCH; use cpu, apt, or skip." >&2
    exit 2
  fi

  # Install Ultralytics without dependency resolution so pip does not pull a
  # CUDA torch stack on CPU-only Raspberry Pi systems.
  python -m pip install --no-deps ultralytics ultralytics-thop

  clone_or_download() {
    local url="$1"
    local branch="$2"
    local target="$3"
    local zip_url="$4"
    if [ -d "$target" ]; then
      return
    fi
    git -c http.version=HTTP/1.1 clone --depth 1 --branch "$branch" "$url" "$target" && return
    echo "git clone failed for $target; falling back to codeload zip"
    mkdir -p "$(dirname "$target")" tmp/downloads
    local zip_path="tmp/downloads/$(basename "$target").zip"
    python - <<PY
from pathlib import Path
import urllib.request
url = "$zip_url"
path = Path("$zip_path")
path.parent.mkdir(parents=True, exist_ok=True)
urllib.request.urlretrieve(url, path)
PY
    python - <<PY
from pathlib import Path
import shutil
import zipfile
zip_path = Path("$zip_path")
target = Path("$target")
tmp = Path("tmp/downloads/extract_$(basename "$target")")
if tmp.exists():
    shutil.rmtree(tmp)
tmp.mkdir(parents=True)
with zipfile.ZipFile(zip_path) as zf:
    zf.extractall(tmp)
roots = [p for p in tmp.iterdir() if p.is_dir()]
if not roots:
    raise SystemExit(f"no directory extracted from {zip_path}")
if target.exists():
    shutil.rmtree(target)
shutil.move(str(roots[0]), str(target))
PY
  }

  if [ ! -d third_party/nanodet ]; then
    clone_or_download \
      https://github.com/RangiLyu/nanodet.git \
      main \
      third_party/nanodet \
      https://codeload.github.com/RangiLyu/nanodet/zip/refs/heads/main
  fi
  if [ ! -d third_party/PaddleDetection ]; then
    clone_or_download \
      https://github.com/PaddlePaddle/PaddleDetection.git \
      release/2.7 \
      third_party/PaddleDetection \
      https://codeload.github.com/PaddlePaddle/PaddleDetection/zip/refs/heads/release/2.7
  fi
}

run_download() {
  # shellcheck disable=SC1091
  source "$VENV/bin/activate"
  mkdir -p models/baselines
  python scripts/prepare_sota_baseline_artifacts.py --models nanodet_plus_m_320,pp_picodet_l_640
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

run_sota_benchmarks() {
  # shellcheck disable=SC1091
  source "$VENV/bin/activate"
  export PYTHONPATH="$ROOT/src:$ROOT/third_party/nanodet:$ROOT/third_party/PaddleDetection:${PYTHONPATH:-}"
  export MPLCONFIGDIR="$ROOT/.plot_mplconfig"
  export YOLO_CONFIG_DIR="$ROOT/Ultralytics"
  python scripts/run_sota_thermal_matrix.py \
    --video "$VIDEO" \
    --max-frames "${SOTA_MAX_FRAMES:-0}" \
    --duration-min "${SOTA_DURATION_MIN:-20}" \
    --repeats "${SOTA_REPEATS:-3}" \
    --threads "$THREADS" \
    --models "${SOTA_MODELS:-yolov8n_640,nanodet_plus_m_input640,picodet_l_640}" \
    --output-dir experiments/sota_thermal_matrix \
    --visualization-dir experiments/visualizations/sota_thermal_matrix
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
  run-sota) run_sota_benchmarks ;;
  accuracy) run_accuracy ;;
  picodet-probe) run_picodet_visual_probe ;;
  run-all)
    run_setup
    run_download
    run_project_methods
    run_accuracy
    ;;
  *)
    echo "usage: $0 {setup|download|run-project|run-sota|accuracy|picodet-probe|run-all}" >&2
    exit 2
    ;;
esac
