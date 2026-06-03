#!/usr/bin/env bash
# Export RT-DETR ONNX model using upstream clone.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RTDETR="${ROOT}/third_party/RT-DETR"
OUT_DIR="${ROOT}/models"

if [ ! -d "$RTDETR" ]; then
  echo "RT-DETR not found. Run: bash scripts/clone_rtdetr.sh"
  exit 1
fi

mkdir -p "$OUT_DIR"

echo "TODO: implement ONNX export using RT-DETR tools in $RTDETR"
echo "Placeholder steps:"
echo "  1. Install RT-DETR dependencies in a separate env if needed"
echo "  2. Export RT-DETR-R18 (recommended for Pi) to $OUT_DIR/rtdetr_r18.onnx"
echo "  3. Update configs/raspberry_pi4.yaml inference.model_path"
echo ""
echo "See: https://github.com/lyuwenyu/RT-DETR for official export instructions."
