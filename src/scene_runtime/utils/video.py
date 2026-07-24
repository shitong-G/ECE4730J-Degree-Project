"""Video and camera frame sources."""

from __future__ import annotations

import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


class FrameSource:
    """
    Iterable frame source from video file, camera index, or synthetic dry-run.

    Parameters
    ----------
    video:
        Path to video file, camera index as int string, or None for synthetic.
    synthetic:
        Generate blank frames when no video (for dry-run without sample video).
    max_frames:
        Optional cap on frames for short tests.
    loop:
        Restart a video file from the first frame when EOF is reached.
    camera_backend:
        Set to ``"csi"`` to read Raspberry Pi Camera Module v2 through Picamera2.
        Set to ``"imx219-raw"`` to capture IMX219 RG10 raw frames with V4L2.
    """

    def __init__(
        self,
        video: str | Path | int | None = None,
        *,
        synthetic: bool = False,
        synthetic_size: tuple[int, int] = (640, 480),
        max_frames: int | None = None,
        loop: bool = False,
        frame_size: tuple[int, int] | None = None,
        camera_backend: str | None = None,
        camera_size: tuple[int, int] = (640, 480),
        camera_framerate: int = 30,
        imx219_media_device: str = "/dev/media3",
        imx219_video_device: str = "/dev/video0",
        imx219_sensor_entity: str = "imx219 10-0010",
        imx219_sensor_format: str = "SRGGB10_1X10",
        imx219_pixel_format: str = "RG10",
        imx219_width: int = 1640,
        imx219_height: int = 1232,
        imx219_stride_pixels: int | None = 1648,
        imx219_frame_width: int = 640,
        imx219_frame_height: int = 480,
        imx219_raw_output: str | Path = "camera_latest.raw",
        imx219_jpg_output: str | Path = "camera_latest.jpg",
        imx219_save_latest: bool = True,
        imx219_stream_mmap: int = 3,
        imx219_capture_interval_sec: float = 0.0,
        imx219_black_level: float = 64.0,
        imx219_r_gain: float = 1.2,
        imx219_g_gain: float = 0.8,
        imx219_b_gain: float = 1.2,
        imx219_gamma: float = 2.2,
        imx219_saturation: float = 1.8,
        imx219_sharpen: float = 0.2,
        imx219_resize_original: bool = True,
        imx219_runtime_mode: str = "lite-isp",
        frame_source_mode: str = "serial",
    ) -> None:
        self._video = video
        self._synthetic = synthetic
        self._synthetic_size = synthetic_size
        self._max_frames = max_frames
        self._loop = loop
        self._frame_size = frame_size
        self._camera_backend = camera_backend
        self._frame_source_mode = str(frame_source_mode)
        self._camera_size = camera_size
        self._camera_framerate = int(camera_framerate)
        self._imx219_media_device = str(imx219_media_device)
        self._imx219_video_device = str(imx219_video_device)
        self._imx219_sensor_entity = str(imx219_sensor_entity)
        self._imx219_sensor_format = str(imx219_sensor_format)
        self._imx219_pixel_format = str(imx219_pixel_format)
        self._imx219_width = int(imx219_width)
        self._imx219_height = int(imx219_height)
        self._imx219_stride_pixels = (
            int(imx219_stride_pixels) if imx219_stride_pixels is not None else None
        )
        self._imx219_frame_width = int(imx219_frame_width)
        self._imx219_frame_height = int(imx219_frame_height)
        self._imx219_raw_output = Path(imx219_raw_output)
        self._imx219_jpg_output = Path(imx219_jpg_output)
        self._imx219_save_latest = bool(imx219_save_latest)
        self._imx219_stream_mmap = int(imx219_stream_mmap)
        self._imx219_capture_interval_sec = max(0.0, float(imx219_capture_interval_sec))
        self._imx219_next_capture_time = 0.0
        self._imx219_isp_kwargs = {
            "black_level": float(imx219_black_level),
            "r_gain": float(imx219_r_gain),
            "g_gain": float(imx219_g_gain),
            "b_gain": float(imx219_b_gain),
            "gamma": float(imx219_gamma),
            "saturation": float(imx219_saturation),
            "sharpen_amount": float(imx219_sharpen),
            "resize_original": bool(imx219_resize_original),
        }
        self._imx219_gray_kwargs = {
            "black_level": float(imx219_black_level),
            "gamma": 1.0,
        }
        self._imx219_runtime_mode = str(imx219_runtime_mode)
        if self._imx219_runtime_mode not in {"lite-isp", "gray"}:
            raise ValueError(
                "imx219_runtime_mode must be 'lite-isp' or 'gray', "
                f"got {self._imx219_runtime_mode!r}"
            )
        self._cap: cv2.VideoCapture | None = None
        self._picam2 = None
        self._imx219_converter = None
        self._imx219_gray_converter = None
        self._count = 0
        self._last_profile = self._empty_profile()
        self._producer_thread: threading.Thread | None = None
        self._producer_condition = threading.Condition()
        self._producer_stop = False
        self._producer_error: BaseException | None = None
        self._producer_error_count = 0
        self._latest_frame: np.ndarray | None = None
        self._latest_profile: dict[str, float] | None = None
        self._latest_seq = 0
        self._consumed_seq = 0

    @property
    def last_profile(self) -> dict[str, float]:
        """Return timing for the most recently yielded frame source step."""
        return dict(self._last_profile)

    @staticmethod
    def _empty_profile() -> dict[str, float]:
        return {
            "source_total_ms": 0.0,
            "source_wait_ms": 0.0,
            "capture_ms": 0.0,
            "isp_ms": 0.0,
            "source_resize_ms": 0.0,
            "source_save_ms": 0.0,
            "source_runtime_resize_ms": 0.0,
            "source_consumer_wait_ms": 0.0,
            "source_frame_age_ms": 0.0,
            "source_dropped_frames": 0.0,
            "source_error_count": 0.0,
        }

    @staticmethod
    def _elapsed_ms(t0: float) -> float:
        return (time.perf_counter() - t0) * 1000.0

    def _resize_runtime_frame(self, frame: np.ndarray, profile: dict[str, float]) -> np.ndarray:
        if self._frame_size is None:
            return frame
        width, height = self._frame_size
        width = int(width)
        height = int(height)
        if width <= 0 or height <= 0:
            return frame
        if frame.shape[1] == width and frame.shape[0] == height:
            return frame
        t0 = time.perf_counter()
        resized = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        profile["source_runtime_resize_ms"] = self._elapsed_ms(t0)
        return resized

    def _open(self) -> None:
        if self._synthetic or self._video is None:
            if self._camera_backend == "csi":
                self._open_csi_camera()
            elif self._camera_backend == "imx219-raw":
                self._open_imx219_raw_camera()
                if self._use_latest_thread():
                    self._start_latest_frame_producer()
            return
        if isinstance(self._video, int):
            self._cap = cv2.VideoCapture(self._video)
        else:
            path = Path(self._video)
            if path.exists():
                self._cap = cv2.VideoCapture(str(path))
            else:
                # Fall back to synthetic if file missing in dry-run dev
                self._synthetic = True

    def _open_csi_camera(self) -> None:
        try:
            from picamera2 import Picamera2  # type: ignore[import-not-found]
        except Exception as exc:
            raise RuntimeError(
                "Picamera2 is required for --camera csi. Install it on the Pi "
                "with: sudo apt install -y python3-picamera2"
            ) from exc
        width, height = self._camera_size
        picam2 = Picamera2()
        config = picam2.create_video_configuration(
            main={"size": (int(width), int(height)), "format": "RGB888"},
            controls={"FrameRate": self._camera_framerate},
        )
        picam2.configure(config)
        picam2.start()
        self._picam2 = picam2

    def _open_imx219_raw_camera(self) -> None:
        self._require_tool("media-ctl")
        self._require_tool("v4l2-ctl")
        self._imx219_raw_output.parent.mkdir(parents=True, exist_ok=True)
        self._imx219_jpg_output.parent.mkdir(parents=True, exist_ok=True)
        self._load_imx219_converter()
        sensor_fmt = (
            f'"{self._imx219_sensor_entity}":0 '
            f"[fmt:{self._imx219_sensor_format}/"
            f"{self._imx219_width}x{self._imx219_height} field:none]"
        )
        self._run_command(
            [
                "media-ctl",
                "-d",
                self._imx219_media_device,
                "--set-v4l2",
                sensor_fmt,
            ]
        )
        self._run_command(
            [
                "v4l2-ctl",
                "-d",
                self._imx219_video_device,
                (
                    "--set-fmt-video="
                    f"width={self._imx219_width},height={self._imx219_height},"
                    f"pixelformat={self._imx219_pixel_format}"
                ),
            ]
        )

    def _load_imx219_converters(self) -> None:
        if self._imx219_converter is not None and self._imx219_gray_converter is not None:
            return
        root = Path(__file__).resolve().parents[3]
        scripts_dir = root / "scripts"
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))
        try:
            from convert_imx219_rg10_lite_isp import rg10_to_bgr, rg10_to_gray_bgr
        except Exception as exc:
            raise RuntimeError(
                "Could not import scripts/convert_imx219_rg10_lite_isp.py"
            ) from exc
        self._imx219_converter = rg10_to_bgr
        self._imx219_gray_converter = rg10_to_gray_bgr

    def _load_imx219_converter(self):
        self._load_imx219_converters()
        if self._imx219_converter is None:
            raise RuntimeError("IMX219 lite ISP converter is not available")
        return self._imx219_converter

    def _load_imx219_gray_converter(self):
        self._load_imx219_converters()
        if self._imx219_gray_converter is None:
            raise RuntimeError("IMX219 grayscale converter is not available")
        return self._imx219_gray_converter

    @staticmethod
    def _require_tool(name: str) -> None:
        if shutil.which(name) is None:
            raise FileNotFoundError(f"Required command not found: {name}")

    @staticmethod
    def _run_command(cmd: list[str]) -> None:
        result = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            details = "\n".join(
                part
                for part in (
                    result.stdout.strip(),
                    result.stderr.strip(),
                )
                if part
            )
            message = f"Command failed with exit code {result.returncode}: {' '.join(cmd)}"
            if details:
                message = f"{message}\n{details}"
            raise RuntimeError(message)

    def _use_latest_thread(self) -> bool:
        return (
            self._camera_backend == "imx219-raw"
            and self._frame_source_mode in {"latest-thread", "producer-consumer"}
        )

    def _start_latest_frame_producer(self) -> None:
        if self._producer_thread is not None:
            return
        self._producer_stop = False
        self._producer_error = None
        self._producer_thread = threading.Thread(
            target=self._producer_loop,
            name="imx219-frame-producer",
            daemon=True,
        )
        self._producer_thread.start()

    def _producer_loop(self) -> None:
        while not self._producer_stop:
            try:
                frame = self._capture_imx219_raw_frame()
                profile = self.last_profile
                profile["_source_ready_time"] = time.perf_counter()
                profile["source_error_count"] = float(self._producer_error_count)
                with self._producer_condition:
                    self._latest_seq += 1
                    self._latest_frame = frame
                    self._latest_profile = profile
                    self._producer_error = None
                    self._producer_condition.notify_all()
            except BaseException as exc:
                self._producer_error_count += 1
                with self._producer_condition:
                    self._producer_error = exc
                    self._producer_condition.notify_all()
                time.sleep(0.25)

    def _next_latest_frame(self) -> np.ndarray | None:
        wait_t0 = time.perf_counter()
        with self._producer_condition:
            while (
                not self._producer_stop
                and self._producer_error is None
                and self._latest_seq <= self._consumed_seq
            ):
                self._producer_condition.wait(timeout=0.2)

            if self._producer_error is not None and self._latest_frame is None:
                raise RuntimeError("Frame producer failed") from self._producer_error
            if self._latest_frame is None or self._latest_profile is None:
                return None

            dropped = max(0, self._latest_seq - self._consumed_seq - 1)
            self._consumed_seq = self._latest_seq
            frame = self._latest_frame
            profile = dict(self._latest_profile)

        profile["source_consumer_wait_ms"] = self._elapsed_ms(wait_t0)
        ready_time = profile.pop("_source_ready_time", None)
        if ready_time is not None:
            profile["source_frame_age_ms"] = max(
                0.0,
                (time.perf_counter() - float(ready_time)) * 1000.0,
            )
        profile["source_dropped_frames"] = float(dropped)
        profile["source_error_count"] = float(self._producer_error_count)
        self._last_profile = profile
        return frame

    def _capture_imx219_raw_frame(self) -> np.ndarray:
        profile = self._empty_profile()
        source_t0 = time.perf_counter()

        now = time.perf_counter()
        if self._imx219_next_capture_time > now:
            wait_t0 = time.perf_counter()
            time.sleep(self._imx219_next_capture_time - now)
            profile["source_wait_ms"] = self._elapsed_ms(wait_t0)

        capture_started = time.perf_counter()
        capture_t0 = time.perf_counter()
        try:
            self._imx219_raw_output.unlink(missing_ok=True)
        except OSError:
            pass
        self._run_command(
            [
                "v4l2-ctl",
                "-d",
                self._imx219_video_device,
                f"--stream-mmap={self._imx219_stream_mmap}",
                "--stream-count=1",
                f"--stream-to={self._imx219_raw_output}",
            ]
        )
        profile["capture_ms"] = self._elapsed_ms(capture_t0)
        if (
            not self._imx219_raw_output.exists()
            or self._imx219_raw_output.stat().st_size == 0
        ):
            raise RuntimeError(
                f"IMX219 raw capture produced an empty file: {self._imx219_raw_output}"
            )
        isp_t0 = time.perf_counter()
        if self._imx219_runtime_mode == "gray":
            converter = self._load_imx219_gray_converter()
            frame = converter(
                input_path=self._imx219_raw_output,
                width=self._imx219_width,
                height=self._imx219_height,
                stride_pixels=self._imx219_stride_pixels,
                verbose=False,
                **self._imx219_gray_kwargs,
            )
        else:
            converter = self._load_imx219_converter()
            frame = converter(
                input_path=self._imx219_raw_output,
                width=self._imx219_width,
                height=self._imx219_height,
                stride_pixels=self._imx219_stride_pixels,
                verbose=False,
                **self._imx219_isp_kwargs,
            )
        profile["isp_ms"] = self._elapsed_ms(isp_t0)

        if self._imx219_frame_width > 0 and self._imx219_frame_height > 0:
            target_size = (self._imx219_frame_width, self._imx219_frame_height)
            if frame.shape[1] != target_size[0] or frame.shape[0] != target_size[1]:
                resize_t0 = time.perf_counter()
                frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)
                profile["source_resize_ms"] = self._elapsed_ms(resize_t0)

        if self._imx219_save_latest:
            save_t0 = time.perf_counter()
            cv2.imwrite(str(self._imx219_jpg_output), frame)
            profile["source_save_ms"] = self._elapsed_ms(save_t0)
        if self._imx219_capture_interval_sec > 0:
            self._imx219_next_capture_time = capture_started + self._imx219_capture_interval_sec
        profile["source_total_ms"] = self._elapsed_ms(source_t0)
        self._last_profile = profile
        return frame

    def __iter__(self) -> Iterator[np.ndarray]:
        self._open()
        while True:
            if self._max_frames is not None and self._count >= self._max_frames:
                break
            if self._use_latest_thread():
                frame = self._next_latest_frame()
                if frame is None:
                    break
            elif self._picam2 is not None:
                source_t0 = time.perf_counter()
                capture_t0 = time.perf_counter()
                rgb = self._picam2.capture_array()
                capture_ms = self._elapsed_ms(capture_t0)
                frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                profile = self._empty_profile()
                profile["capture_ms"] = capture_ms
                frame = self._resize_runtime_frame(frame, profile)
                profile["source_total_ms"] = self._elapsed_ms(source_t0)
                self._last_profile = profile
            elif self._camera_backend == "imx219-raw":
                frame = self._capture_imx219_raw_frame()
            elif self._synthetic or self._cap is None:
                source_t0 = time.perf_counter()
                frame = self._synthetic_frame()
                profile = self._empty_profile()
                frame = self._resize_runtime_frame(frame, profile)
                profile["source_total_ms"] = self._elapsed_ms(source_t0)
                self._last_profile = profile
            else:
                source_t0 = time.perf_counter()
                capture_t0 = time.perf_counter()
                ok, frame = self._cap.read()
                capture_ms = self._elapsed_ms(capture_t0)
                if not ok:
                    if not self._loop:
                        break
                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    capture_t0 = time.perf_counter()
                    ok, frame = self._cap.read()
                    capture_ms += self._elapsed_ms(capture_t0)
                    if not ok:
                        break
                profile = self._empty_profile()
                profile["capture_ms"] = capture_ms
                frame = self._resize_runtime_frame(frame, profile)
                profile["source_total_ms"] = self._elapsed_ms(source_t0)
                self._last_profile = profile
            yield frame
            self._count += 1

    def _synthetic_frame(self) -> np.ndarray:
        w, h = self._synthetic_size
        # Slight variation so visual features are non-trivial
        base = (self._count * 3) % 255
        frame = np.full((h, w, 3), base, dtype=np.uint8)
        cv2.rectangle(
            frame,
            (50 + (self._count % 100), 50),
            (200, 200),
            (255 - base, 128, 64),
            2,
        )
        return frame

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        if self._picam2 is not None:
            self._picam2.stop()
            self._picam2.close()
            self._picam2 = None
        if self._producer_thread is not None:
            self._producer_stop = True
            with self._producer_condition:
                self._producer_condition.notify_all()
            self._producer_thread.join(timeout=2.0)
            self._producer_thread = None
