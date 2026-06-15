#!/usr/bin/env bash
# Export RT-DETR ONNX for Raspberry Pi deployment.
#
# Default: RT-DETRv2 R18 lite (sp1, fewest decoder sampling points, best for Pi).
#
# Usage:
#   bash scripts/export_model_onnx.sh              # lite (default)
#   bash scripts/export_model_onnx.sh lite
#   bash scripts/export_model_onnx.sh rtdetrv2-s   # RT-DETRv2-S (full sp4)
#   bash scripts/export_model_onnx.sh v1           # legacy RT-DETR v1 R18
#
# Requires a workstation env with torch + torchvision (see docs/setup_raspberry_pi.md).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RTDETR="${ROOT}/third_party/RT-DETR"
OUT_DIR="${ROOT}/models"
CKPT_DIR="${ROOT}/third_party/checkpoints"
VARIANT="${1:-lite}"

if [ ! -d "$RTDETR" ]; then
  echo "RT-DETR not found. Run: bash scripts/clone_rtdetr.sh"
  exit 1
fi

mkdir -p "$OUT_DIR" "$CKPT_DIR"

case "$VARIANT" in
  lite|r18-lite|rtdetrv2-lite)
    PYTORCH_DIR="${RTDETR}/rtdetrv2_pytorch"
    CONFIG="configs/rtdetrv2/rtdetrv2_r18vd_sp1_120e_coco.yml"
    CKPT_URL="https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetrv2_r18vd_sp1_120e_coco.pth"
    CKPT_FILE="${CKPT_DIR}/rtdetrv2_r18vd_sp1_120e_coco.pth"
    OUT_FILE="${OUT_DIR}/rtdetr_r18_lite_pi4.onnx"
    LABEL="RT-DETRv2 R18 lite (sp1)"
    ;;
  s|rtdetrv2-s|r18)
    PYTORCH_DIR="${RTDETR}/rtdetrv2_pytorch"
    CONFIG="configs/rtdetrv2/rtdetrv2_r18vd_120e_coco.yml"
    CKPT_URL="https://github.com/lyuwenyu/storage/releases/download/v0.2/rtdetrv2_r18vd_120e_coco_rerun_48.1.pth"
    CKPT_FILE="${CKPT_DIR}/rtdetrv2_r18vd_120e_coco_rerun_48.1.pth"
    OUT_FILE="${OUT_DIR}/rtdetr_r18_pi4.onnx"
    LABEL="RT-DETRv2-S (R18)"
    ;;
  v1|rtdetr-v1)
    PYTORCH_DIR="${RTDETR}/rtdetr_pytorch"
    CONFIG="configs/rtdetr/rtdetr_r18vd_6x_coco.yml"
    CKPT_URL="https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r18vd_dec3_6x_coco_from_paddle.pth"
    CKPT_FILE="${CKPT_DIR}/rtdetr_r18vd_dec3_6x_coco_from_paddle.pth"
    OUT_FILE="${OUT_DIR}/rtdetr_r18_pi4.onnx"
    LABEL="RT-DETR v1 R18"
    ;;
  *)
    echo "Unknown variant: $VARIANT"
    echo "Supported: lite | rtdetrv2-s | v1"
    exit 1
    ;;
esac

if [ ! -f "$CKPT_FILE" ]; then
  echo "Downloading checkpoint for $LABEL ..."
  if command -v wget >/dev/null 2>&1; then
    wget -c "$CKPT_URL" -O "$CKPT_FILE"
  elif command -v curl >/dev/null 2>&1; then
    curl -L "$CKPT_URL" -o "$CKPT_FILE"
  else
    echo "Install wget or curl to download checkpoints."
    exit 1
  fi
fi

if ! python -c "import torch" >/dev/null 2>&1; then
  echo "PyTorch not found in current Python env."
  echo "Activate your export env first, e.g.:"
  echo "  conda activate rtdetr-export"
  echo "  bash scripts/setup_rtdetr_export_env.sh"
  exit 1
fi

# RT-DETRv2 pulls in dataset code at import time; ensure export-only deps exist.
if [[ "$PYTORCH_DIR" == *rtdetrv2_pytorch* ]]; then
  echo "Checking RT-DETRv2 export dependencies ..."
  python -m pip install -q "numpy<2" "faster-coco-eval>=1.6.6" scipy PyYAML onnx
  python - <<'PY'
import numpy as np
major = int(np.__version__.split(".")[0])
if major >= 2:
    raise SystemExit(
        f"numpy {np.__version__} is incompatible with torch 2.1. "
        "Run: pip install 'numpy<2'"
    )
import faster_coco_eval  # noqa: F401
print(f"  numpy {np.__version__}, faster_coco_eval OK")
PY
fi

echo "Exporting $LABEL -> $OUT_FILE"
cd "$PYTORCH_DIR"

python tools/export_onnx.py \
  -c "$CONFIG" \
  -r "$CKPT_FILE" \
  -o "$OUT_FILE" \
  --check

echo ""
echo "Done."
echo "  Model : $OUT_FILE"
echo "  Update configs inference.model_path if needed:"
echo "    inference:"
echo "      model_path: models/$(basename "$OUT_FILE")"
