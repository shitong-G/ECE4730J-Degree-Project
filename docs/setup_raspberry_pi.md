# Raspberry Pi Setup

## Hardware

- Raspberry Pi 4 (4GB+) or Pi 5
- Camera module or pre-recorded test video
- Adequate power supply (official PSU recommended)

## OS Preparation

```bash
sudo apt update
sudo apt install -y python3-pip python3-venv libopencv-dev v4l-utils
```

Optional for throttling diagnostics:

```bash
sudo apt install -y libraspberrypi-bin
```

## Clone Repository and RT-DETR

```bash
git clone <your-repo-url> scene-runtime
cd scene-runtime
bash scripts/setup_env.sh
bash scripts/clone_rtdetr.sh
```

## Export ONNX Model (on workstation or Pi)

```bash
bash scripts/export_model_onnx.sh
# Place exported model at path configured in configs/raspberry_pi4.yaml
```

## Run on Pi

```bash
source .venv/bin/activate
python scripts/run_experiment.py \
  --config configs/raspberry_pi4.yaml \
  --strategy scene_thermal_coadaptive \
  --video data/sample.mp4 \
  --duration-min 15
```

## CPU Governor (optional)

The runtime controller may suggest governor changes. Apply manually if needed:

```bash
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
```

## Thermal Monitoring

Temperature is read from `/sys/class/thermal/thermal_zone0/temp`. Throttling via `vcgencmd get_throttled` when available.
