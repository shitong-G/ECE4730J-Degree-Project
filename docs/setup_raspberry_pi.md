# Raspberry Pi Setup

## Hardware

- Raspberry Pi 4 (4GB+) or Pi 5 for normal deployment
- Raspberry Pi 4 (1GB) for constrained-memory deployment with `configs/raspberry_pi4_1gb.yaml`
- Camera module or pre-recorded test video
- Adequate power supply (official PSU recommended)

## OS Preparation

For Raspberry Pi 4 (1GB), use **64-bit Raspberry Pi OS Lite** without desktop UI. The desktop environment leaves too little memory for ONNX Runtime + OpenCV + RT-DETR.

```bash
sudo apt update
sudo apt install -y python3-pip python3-venv libopencv-dev v4l-utils
```

Optional for throttling diagnostics:

```bash
sudo apt install -y libraspberrypi-bin
```

Recommended on Raspberry Pi 4 (1GB): enable larger swap before real ONNX inference.

```bash
sudo dphys-swapfile swapoff
sudo sed -i 's/CONF_SWAPSIZE=.*/CONF_SWAPSIZE=2048/' /etc/dphys-swapfile
sudo dphys-swapfile setup
sudo dphys-swapfile swapon
```

Optional memory saving on 1GB boards:

```bash
echo "gpu_mem=16" | sudo tee -a /boot/firmware/config.txt
sudo reboot
```

## Clone Repository and RT-DETR

```bash
git clone <your-repo-url> scene-runtime
cd scene-runtime
bash scripts/setup_env.sh
bash scripts/clone_rtdetr.sh
```

## Export ONNX Model (on workstation or Pi)

Export on a workstation for Raspberry Pi 4 (1GB). Do not export or train RT-DETR on the 1GB Pi.

```bash
bash scripts/export_model_onnx.sh
# Place exported model at path configured in configs/raspberry_pi4.yaml
```

For the 1GB profile, place the exported lightweight RT-DETR-R18 ONNX model at:

```bash
models/rtdetr_r18_pi4.onnx
```

## Run on Pi 4GB+ / Pi 5

```bash
source .venv/bin/activate
python scripts/run_experiment.py \
  --config configs/raspberry_pi4.yaml \
  --strategy scene_thermal_coadaptive \
  --video data/sample.mp4 \
  --duration-min 15
```

## Run on Raspberry Pi 4 (1GB)

First validate the runtime without loading RT-DETR:

```bash
source .venv/bin/activate
python scripts/run_experiment.py \
  --config configs/raspberry_pi4_1gb.yaml \
  --strategy fixed_low_power \
  --dry-run \
  --duration-min 1
```

Then run a short real ONNX smoke test:

```bash
python scripts/run_experiment.py \
  --config configs/raspberry_pi4_1gb.yaml \
  --strategy fixed_low_power \
  --video data/sample.mp4 \
  --duration-min 5
```

Use the 1GB profile for sustained tests only after the smoke test is stable:

```bash
python scripts/run_experiment.py \
  --config configs/raspberry_pi4_1gb.yaml \
  --strategy fixed_low_power \
  --video data/sample.mp4 \
  --duration-min 15
```

The 1GB profile uses lower memory settings: 320 input resolution, inference interval 4, and 2 CPU threads. If the process is still killed by the OS, reduce `runtime.default_input_resolution` to 256 and `runtime.default_cpu_threads` to 1 in `configs/raspberry_pi4_1gb.yaml`.

## CPU Governor (optional)

The runtime controller may suggest governor changes. Apply manually if needed:

```bash
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
```

## Thermal Monitoring

Temperature is read from `/sys/class/thermal/thermal_zone0/temp`. Throttling via `vcgencmd get_throttled` when available.
