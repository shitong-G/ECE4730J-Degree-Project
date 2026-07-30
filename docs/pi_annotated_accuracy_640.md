# Raspberry Pi annotated-frame accuracy at 640 input

This protocol keeps the annotated-frame accuracy comparison on one evaluation
surface:

- video: `data/sample3.mp4`
- annotations: `data/annotations/sample3_50frames_640`
- labelled frame range: `frame_000000` through `frame_000213`
- detector input: 640 for RT-DETR, YOLOv8n, and NanoDet inference
- metric: class-aware one-to-one matching at IoU 0.50 plus AP50 and mAP50-95

## Model choices

Use these names in the report:

- `RT-DETR native Pi 640`: `models/rtdetr_r18_lite_pi4_640.onnx`
- `LK tracking Pi 640`: `models/rtdetr_r18_lite_pi4_640_int8_dynamic_q.onnx`, fixed input 640, query budget 300
- `Proposed software Pi 640`: thermal/LK policy with 320/640 dynamic-query model family
- `YOLOv8n 640`: Ultralytics `yolov8n.pt`, `imgsz=640`
- `NanoDet-Plus-m input640`: official NanoDet-Plus-m checkpoint, with validation/inference input changed to 640
- `PicoDet-L 640`: official PaddleDetection 640 model; use this for a strict 640 PicoDet comparison, not PicoDet-S 320

NanoDet note: the official NanoDet-Plus-m checkpoints are 320 and 416. The
script can run the fully convolutional 320 checkpoint at input 640, but the
paper/report should label it as `input640`, not as a 640-trained checkpoint.

PicoDet note: structured PicoDet boxes are not consumed by the accuracy script
yet. The Pi script includes a visual probe for `picodet_l_640_coco_lcnet`.
For numerical accuracy, add a parser for PaddleDetection's saved prediction
outputs or convert the exported model to an ONNX path with structured outputs.

## One-command Pi run

From the repository root on the Raspberry Pi:

```bash
chmod +x scripts/run_pi_annotated_accuracy_640.sh
./scripts/run_pi_annotated_accuracy_640.sh run-all
```

Outputs:

- `experiments/logs/pi_annotated_accuracy_640/*/runtime.csv`
- `experiments/logs/pi_annotated_accuracy_640/*/runtime_detections.jsonl`
- `experiments/results/pi_annotated_accuracy_640/metrics.csv`
- `experiments/results/pi_annotated_accuracy_640/annotated_accuracy_summary.png`

## Step-by-step commands

Setup Python dependencies and clone third-party repos:

```bash
./scripts/run_pi_annotated_accuracy_640.sh setup
```

The Raspberry Pi is CPU-only. The setup script therefore installs Ultralytics
with `--no-deps` and installs PyTorch from the CPU wheel index:

```bash
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
python -m pip install --no-deps ultralytics ultralytics-thop
```

If pip previously pulled CUDA packages such as `nvidia-cublas-*`,
`nvidia-cudnn-*`, `nvidia-cuda-*`, or `triton`, remove them before rerunning:

```bash
source .venv/bin/activate
python -m pip uninstall -y \
  triton cuda-toolkit cuda-bindings cuda-pathfinder \
  nvidia-cublas-cu13 nvidia-cuda-cupti-cu13 nvidia-cuda-nvrtc-cu13 \
  nvidia-cuda-runtime-cu13 nvidia-cudnn-cu13 nvidia-cufft-cu13 \
  nvidia-cufile-cu13 nvidia-curand-cu13 nvidia-cusolver-cu13 \
  nvidia-cusparse-cu13 nvidia-cusparselt-cu13 nvidia-nccl-cu13 \
  nvidia-nvjitlink-cu13 nvidia-nvshmem-cu13 nvidia-nvtx-cu13
```

If the CPU PyTorch wheel index does not provide a compatible wheel for your Pi
OS/Python combination, use the OS package and tell the script not to install
torch inside the venv:

```bash
sudo apt update
sudo apt install -y python3-torch python3-torchvision
INSTALL_TORCH=skip ./scripts/run_pi_annotated_accuracy_640.sh setup
```

Download/open-source weights:

```bash
./scripts/run_pi_annotated_accuracy_640.sh download
```

The download step prepares:

- YOLOv8n: Ultralytics auto-download/export to `models/baselines/yolov8n_640.onnx`
- NanoDet-Plus-m: Google Drive checkpoint ID `1YvuEhahlgqxIhJu7bsL-fhaqubKcCWQc`
- PicoDet-L 640: `https://paddledet.bj.bcebos.com/deploy/Inference/picodet_l_640_coco_lcnet.tar`

Run the three project methods on the annotated frame span:

```bash
./scripts/run_pi_annotated_accuracy_640.sh run-project
```

Run only the SOTA baselines for non-accuracy system metrics:

```bash
./scripts/run_pi_annotated_accuracy_640.sh run-sota
```

This calls `scripts/run_sota_thermal_matrix.py`, following the same formal
style as `scripts/run_final_thermal_matrix.py`: repeated runs, balanced order,
start-temperature conditioning, per-run `temperature_trace.csv`, per-run
`run_manifest.json`, and a suite-level `summary.csv`.

Default SOTA conditions:

- `yolov8n_640`
- `nanodet_plus_m_input640`
- `picodet_l_640`

Useful controls:

```bash
SOTA_REPEATS=1 SOTA_MAX_FRAMES=600 ./scripts/run_pi_annotated_accuracy_640.sh run-sota
SOTA_MODELS=yolov8n_640,nanodet_plus_m_input640 ./scripts/run_pi_annotated_accuracy_640.sh run-sota
```

Outputs:

- `experiments/sota_thermal_matrix/<run-id>/summary.csv`
- `experiments/sota_thermal_matrix/<run-id>/manifest.json`
- `experiments/sota_thermal_matrix/<run-id>/r*/runtime.csv`
- `experiments/sota_thermal_matrix/<run-id>/r*/temperature_trace.csv`
- `experiments/sota_thermal_matrix/<run-id>/r*/run_manifest.json`

Run YOLOv8n/NanoDet on the annotated PNG frames and evaluate all available
`runtime_detections.jsonl` files:

```bash
./scripts/run_pi_annotated_accuracy_640.sh accuracy
```

Optional PicoDet-L 640 visual check:

```bash
./scripts/run_pi_annotated_accuracy_640.sh picodet-probe
```

## Direct accuracy command

If the project method detections already exist, run only the unified evaluator:

```bash
source .venv/bin/activate
export PYTHONPATH="$PWD/src:$PWD/third_party/nanodet:$PWD/third_party/PaddleDetection:$PYTHONPATH"
python scripts/run_annotated_sota_accuracy.py \
  --suite-dir experiments/empty_suite_for_extra_predictions \
  --annotations-dir data/annotations/sample3_50frames_640 \
  --output-dir experiments/results/pi_annotated_accuracy_640 \
  --predictions-dir experiments/logs/pi_annotated_accuracy_640/sota_frame_predictions \
  --nanodet-input-size 640 \
  --extra-prediction "rtdetr_native_pi640|RT-DETR native Pi 640|experiments/logs/pi_annotated_accuracy_640/rtdetr_native_640/runtime_detections.jsonl" \
  --extra-prediction "lk_tracking_pi640|LK tracking Pi 640|experiments/logs/pi_annotated_accuracy_640/lk_tracking_640/runtime_detections.jsonl" \
  --extra-prediction "proposed_software_pi640|Proposed software Pi 640|experiments/logs/pi_annotated_accuracy_640/proposed_software_640/runtime_detections.jsonl"
```
