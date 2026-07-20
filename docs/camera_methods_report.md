# Camera Integration and Optimization Report

## 1. Objective

The camera work aims to integrate a Raspberry Pi Camera Module v2, based on the
IMX219 sensor, into the Scene and Thermal Co-adaptive RT-DETR runtime. The
camera should support two different roles:

- A real runtime frame source that can replace `data/sample.mp4`.
- A background peripheral workload used for system-level experiments while the
  controlled runtime input remains `data/sample.mp4`.

The second role is important for controlled evaluation. It allows the camera to
remain active while the model still receives a fixed video input, so different
experiments use the same visual content.

## 2. Hardware and System Context

The tested platform is a Raspberry Pi running Ubuntu 22.04. The camera is a
Raspberry Pi Camera Module v2 connected through the CSI interface. Because the
standard `python3-picamera2` package was not available from the configured apt
repositories, the final camera pipeline follows the working local CSI/V4L2
method instead of relying only on Picamera2.

The implemented raw camera path uses:

- `media-ctl` to configure the IMX219 media pipeline.
- `v4l2-ctl` to capture one RG10 raw frame from the video device.
- A custom lightweight ISP script to convert RG10 Bayer raw data into an image
  usable by OpenCV, the dashboard, RT-DETR, and LK tracking.

Default IMX219 capture parameters used in the runtime are:

- Raw sensor size: `1640x1232`
- Pixel format: `RG10`
- Stride: `1648` pixels
- Runtime output frame size: `640x480`
- Raw output path for fast temporary storage: `/dev/shm/camera_latest.raw` or
  `/dev/shm/camera_bench.raw`

## 3. Implemented Camera Methods

### 3.1 Raw IMX219 Frame Capture

The raw camera source is implemented in `src/scene_runtime/utils/video.py` as
the `imx219-raw` backend of `FrameSource`.

For each frame, the runtime performs the following steps:

1. Configure the sensor format with `media-ctl`.
2. Capture one RG10 frame with `v4l2-ctl --stream-count=1`.
3. Read the raw file as `uint16`.
4. Convert the raw RG10 image into a BGR image.
5. Resize the converted frame to the configured runtime size, usually `640x480`.
6. Optionally save the latest JPEG for inspection.

The frame-source profile logs these stages as:

- `capture_ms`
- `isp_ms`
- `source_resize_ms`
- `source_save_ms`
- `source_total_ms`

### 3.2 Lite ISP Color Conversion

The color conversion is implemented in
`scripts/convert_imx219_rg10_lite_isp.py`.

The `lite-isp` path performs:

- RG10 raw loading and 10-bit alignment handling.
- Black-level subtraction.
- Bayer channel extraction for the IMX219 RGGB pattern.
- Manual white-balance gains for R, G, and B.
- Percentile-based normalization.
- Gamma correction.
- HSV saturation boosting.
- Optional sharpening.
- Optional resize back to the original raw resolution.

The visually tuned command found to give acceptable colors was:

```bash
python scripts/convert_imx219_rg10_lite_isp.py \
  --input frame_1640x1232_RG10_test.raw \
  --width 1640 \
  --height 1232 \
  --output preview_lite_isp.jpg \
  --black-level 64 \
  --gamma 2.2 \
  --saturation 1.8 \
  --sharpen 0.2 \
  --resize-original \
  --r-gain 1.2 \
  --b-gain 1.2 \
  --g-gain 0.8
```

This path is useful for human-visible preview images, but it is relatively
expensive on the Raspberry Pi CPU.

### 3.3 Fast Grayscale Runtime Conversion

Because RT-DETR and LK tracking do not necessarily require a visually pleasing
color image, a faster `gray` mode was added.

The grayscale path:

- Reads the RG10 raw frame.
- Applies black-level subtraction and simple linear normalization.
- Converts the result to an 8-bit grayscale image.
- Expands grayscale to 3-channel BGR so existing downstream code can use the
  same frame interface.

This mode skips the expensive color-tuning stages, especially percentile color
normalization, saturation boosting, and sharpening. It is selected by:

```bash
--imx219-runtime-mode gray
```

The original color path remains available as:

```bash
--imx219-runtime-mode lite-isp
```

### 3.4 Single-Shot Camera-to-RT-DETR Test

Earlier bring-up scripts were added to validate the camera independently of the
full runtime:

- `capture_imx219_frame.py`
- `camera_rtdetr_once.py`
- `scripts/convert_imx219_rg10_lite_isp.py`

The intended one-shot workflow was:

```bash
python camera_rtdetr_once.py
```

and it produced:

- `camera_latest.raw`: raw RG10 camera capture
- `camera_latest.jpg`: converted camera image
- `camera_rtdetr_result.jpg`: RT-DETR detection result with boxes

This confirmed that the camera image could be converted into a valid model
input.

### 3.5 Live Dashboard Camera Mode

The live dashboard runner supports camera input through:

```bash
--camera imx219-raw
```

In this mode, camera frames become the actual runtime input. The existing
runtime strategy still decides whether to run RT-DETR or use LK tracking; the
camera itself does not force inference on every captured frame.

The dashboard also exposes frame-source timing values, including capture,
conversion, resize, and serial frame time.

### 3.6 Background Camera Mode

A separate background camera mode was added:

```bash
--background-camera imx219-raw
```

In this mode:

- The real runtime input can remain `--video data/sample.mp4`.
- The IMX219 camera can run continuously, or it can be triggered only after LK
  tracking frames.
- Captured camera frames are discarded and are not sent into RT-DETR or LK
  tracking.
- The experiment output reports how many background camera frames were captured
  and prints the latest background camera timing.

This was introduced to keep experiments comparable while still measuring the
system impact of an active camera peripheral.

The default trigger mode is now:

```bash
--background-camera-trigger during-tracking
```

With this mode, the runtime sends an asynchronous capture request when LK
tracking starts. A background camera worker consumes that request and captures
one IMX219 frame while LK tracking continues in the main runtime thread. If a
previous camera capture is still active, the new request is skipped instead of
being queued. This keeps camera work from accumulating and spilling into later
RT-DETR inference frames.

Two older behaviors are still available:

```bash
--background-camera-trigger post-tracking
```

This captures one frame synchronously after each LK tracking frame.

```bash
--background-camera-trigger continuous
```

This runs the camera continuously as the older always-on background load.

In both `during-tracking` and `post-tracking` modes,
`--imx219-capture-interval-sec` is ignored for the background camera because
each requested capture should start immediately.

### 3.7 Runtime Input Resizing

To make video-based and camera-based tests more comparable, runtime video input
can be resized before scene analysis, LK tracking, and RT-DETR:

```bash
--frame-width 640 --frame-height 480
```

This applies to the runtime input, such as `data/sample.mp4`, not to the
background camera workload. The resize cost is logged as:

```text
source_runtime_resize_ms
```

## 4. Benchmark and Profiling Scripts

### 4.1 Camera Timing Benchmark

`scripts/benchmark_imx219_camera.py` was added to measure camera-only
performance. It reports:

- Total per-frame camera source time.
- Raw capture time.
- Raw conversion time.
- Resize time.
- Optional JPEG save time.
- Effective camera FPS.

Color path benchmark:

```bash
sudo -E .venv/bin/python scripts/benchmark_imx219_camera.py \
  --count 30 \
  --warmup 3 \
  --imx219-runtime-mode lite-isp \
  --imx219-frame-width 640 \
  --imx219-frame-height 480 \
  --imx219-raw-output /dev/shm/camera_bench.raw \
  --no-imx219-save-latest \
  --csv-output experiments/logs/imx219_lite_isp_benchmark.csv
```

Fast grayscale benchmark:

```bash
sudo -E .venv/bin/python scripts/benchmark_imx219_camera.py \
  --count 30 \
  --warmup 3 \
  --imx219-runtime-mode gray \
  --imx219-frame-width 640 \
  --imx219-frame-height 480 \
  --imx219-raw-output /dev/shm/camera_bench.raw \
  --no-imx219-save-latest \
  --csv-output experiments/logs/imx219_gray_benchmark.csv
```

### 4.2 Stage Latency Summary

The runtime now writes a profile CSV next to the main experiment CSV. The
summary script:

```bash
python scripts/summarize_stage_latency.py --input experiments/logs/<run>.csv
```

prints stage-level latency, including frame-source timing, runtime-loop timing,
LK tracking latency, and RT-DETR inference latency. It is automatically run at
the end of `scripts/run_experiment.py` and `scripts/run_live_dashboard.py`
unless disabled with:

```bash
--no-stage-summary
```

## 5. Experimental Observations

### 5.1 Camera-Only Speed

The camera-only benchmark showed the following results.

Color `lite-isp` path:

```text
Measured frames: 30
Camera FPS:      2.254
total mean:      458.816 ms
capture mean:    177.359 ms
convert mean:    264.895 ms
resize mean:      16.419 ms
```

Fast `gray` path:

```text
Measured frames: 30
Camera FPS:      3.896
total mean:      265.658 ms
capture mean:    179.803 ms
convert mean:     70.662 ms
resize mean:      15.052 ms
```

The raw capture cost is almost unchanged at about `180 ms`. The main
improvement comes from reducing conversion time from about `265 ms` to about
`71 ms`.

### 5.2 Background Camera Impact on Full Runtime

When the background camera used the color `lite-isp` path, the latest reported
background timing was approximately:

```text
capture = 172.6 ms
isp     = 348.8 ms
source  = 537.2 ms
```

With `gray` mode, the latest reported background timing became approximately:

```text
capture = 242.1 ms
isp     = 37.3 ms
source  = 292.6 ms
```

This confirms that grayscale mode substantially reduces conversion cost.

However, full model experiments showed that the camera is not free even in
grayscale mode. Without the background camera, average RT-DETR inference was
around three seconds. With grayscale background capture once per second, average
RT-DETR latency increased to more than four seconds. Therefore, even the
optimized camera path still competes with RT-DETR for CPU, cache, memory
bandwidth, and process scheduling.

### 5.3 Interpretation

The camera pipeline has two different cost components:

- An unavoidable raw capture cost of about `180 ms` per frame in the current
  one-frame `v4l2-ctl` capture approach.
- A conversion cost that depends strongly on the selected runtime mode.

The `gray` path addresses the second component but not the first. When the
camera runs concurrently with ONNX inference, it can still increase RT-DETR
latency because both workloads are CPU-bound on Raspberry Pi.

## 6. Current Recommended Commands

### 6.1 Controlled Full Runtime With Background Camera

Use `data/sample.mp4` as the runtime input and keep the camera active in the
background:

```bash
sudo -E .venv/bin/python scripts/run_live_dashboard.py \
  --config configs/raspberry_pi4.yaml \
  --strategy scene_thermal_interval_lk \
  --video data/sample.mp4 \
  --loop-video \
  --background-camera imx219-raw \
  --background-camera-trigger during-tracking \
  --imx219-runtime-mode gray \
  --frame-width 640 \
  --frame-height 480 \
  --duration-min 10 \
  --host 0.0.0.0 \
  --port 8000 \
  --imx219-frame-width 640 \
  --imx219-frame-height 480 \
  --imx219-raw-output /dev/shm/camera_latest.raw \
  --no-imx219-save-latest \
  --jpeg-width 640 \
  --jpeg-quality 55 \
  --log-detections
```

### 6.2 No-Camera Baseline

Use this command to measure the same runtime without camera interference:

```bash
sudo -E .venv/bin/python scripts/run_live_dashboard.py \
  --config configs/raspberry_pi4.yaml \
  --strategy scene_thermal_interval_lk \
  --video data/sample.mp4 \
  --loop-video \
  --frame-width 640 \
  --frame-height 480 \
  --duration-min 10 \
  --host 0.0.0.0 \
  --port 8000 \
  --jpeg-width 640 \
  --jpeg-quality 55 \
  --log-detections
```

## 7. Limitations

The current camera pipeline still has several limitations:

- It captures one frame at a time through `v4l2-ctl`, which introduces process
  launch and I/O overhead.
- Raw capture is still about `180 ms` per frame in the current setup.
- The background camera currently runs independently from RT-DETR scheduling,
  so it may capture during full inference and increase ONNX latency.
- The `gray` path is efficient but not visually suitable for human-facing
  dashboard preview.
- The color `lite-isp` path is visually better but too expensive for frequent
  runtime use on Raspberry Pi.
- The current implementation does not yet produce separate synchronized outputs
  for fast grayscale inference and slower color dashboard preview.

## 8. Recommended Next Steps

The next camera-focused improvements should prioritize scheduling rather than
only image conversion:

1. Pause or reduce background camera capture while RT-DETR inference is running.
2. Allow higher camera frequency during LK tracking or skipped-inference
   periods.
3. Use `gray` as the default runtime camera mode for inference/tracking.
4. Run color `lite-isp` only at a lower rate for dashboard preview or saved
   inspection frames.
5. Consider replacing per-frame `v4l2-ctl` calls with a persistent capture
   process or streaming API if the project needs higher camera FPS.
6. Keep no-camera, low-frequency-camera, and full-speed-camera experiments as
   separate baselines to isolate camera workload from tracking and RT-DETR
   workload.

## 9. Summary

The camera pipeline is functional and can provide valid IMX219 frames to the
existing RT-DETR runtime. The main implemented strategies are raw CSI capture,
custom lite ISP conversion, grayscale fast conversion, live dashboard camera
input, background camera draining, runtime input resizing, and dedicated
camera-stage profiling.

The most important finding is that color conversion is optimizable, but the
camera workload still affects the full model. Grayscale conversion reduces
camera-only frame time from about `459 ms` to about `266 ms`, but background
camera operation can still increase RT-DETR latency. Future work should
therefore focus on camera scheduling and decoupled preview/inference pipelines.
