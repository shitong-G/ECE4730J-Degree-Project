#!/usr/bin/env bash
# One-time setup for RT-DETR ONNX export on a workstation.
#
# Usage:
#   conda create -n rtdetr-export python=3.10 -y
#   conda activate rtdetr-export
#   bash scripts/setup_rtdetr_export_env.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "Installing export dependencies from requirements-export.txt ..."
python -m pip install --upgrade pip
python -m pip install -r "${ROOT}/requirements-export.txt"

echo ""
echo "Verifying imports ..."
python - <<'PY'
import numpy
import torch
import torchvision
import faster_coco_eval  # noqa: F401
import onnx

assert int(numpy.__version__.split(".")[0]) < 2, f"numpy must be <2, got {numpy.__version__}"
print(f"numpy      {numpy.__version__}")
print(f"torch      {torch.__version__}")
print(f"torchvision {torchvision.__version__}")
print(f"onnx       {onnx.__version__}")
print("Export env ready.")
PY

echo ""
echo "Next:"
echo "  bash scripts/clone_rtdetr.sh"
echo "  bash scripts/export_model_onnx.sh"
