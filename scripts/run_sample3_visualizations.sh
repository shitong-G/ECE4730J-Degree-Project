#!/usr/bin/env bash
# Generate annotated sample3 videos using the three INT8 RT-DETR models and
# event-triggered LK tracking. Run from WSL: bash scripts/run_sample3_visualizations.sh
set -eo pipefail

# The WSL profile adds this directory interactively; use the absolute path so
# this script also works from a non-interactive WSL shell.
source /home/gst/anaconda3/bin/activate 4903

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

OUTPUT_DIR="experiments/visualizations/sample3"
mkdir -p "$OUTPUT_DIR"

for resolution in 640 480 320; do
  python scripts/visualize_detections.py \
    --video data/sample3.mp4 \
    --model "models/rtdetr_r18_lite_pi4_${resolution}_int8.onnx" \
    --resolution "$resolution" \
    --max-frames 0 \
    --output "$OUTPUT_DIR/sample3_int8_${resolution}.mp4"
done

LK_DIR="$OUTPUT_DIR/lk_int8_640"
python scripts/detect_track_lk.py \
  --mode detect_track \
  --video data/sample3.mp4 \
  --model models/rtdetr_r18_lite_pi4_640_int8.onnx \
  --resolution 640 \
  --max-frames 0 \
  --output-dir "$LK_DIR"

FPS="$(python - <<'PY'
import cv2
capture = cv2.VideoCapture('data/sample3.mp4')
print(capture.get(cv2.CAP_PROP_FPS) or 25.0)
capture.release()
PY
)"
ffmpeg -y -framerate "$FPS" -i "$LK_DIR/frames/frame_%06d.jpg" \
  -c:v libx264 -pix_fmt yuv420p "$OUTPUT_DIR/sample3_int8_640_lk.mp4"
