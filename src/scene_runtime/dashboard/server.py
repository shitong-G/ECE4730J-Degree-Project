"""Dependency-light HTTP dashboard for live runtime monitoring."""

from __future__ import annotations

import json
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import numpy as np

from scene_runtime.inference.postprocess import Detection


COCO80 = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]


INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Scene Runtime Live Dashboard</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #101418;
      --panel: #171d23;
      --panel2: #1d242b;
      --line: #2d3842;
      --text: #edf2f7;
      --muted: #a4b0bc;
      --green: #60d394;
      --amber: #f4c95d;
      --red: #ef6f6c;
      --cyan: #48cae4;
      --blue: #8ab4f8;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      padding: 14px 18px;
      border-bottom: 1px solid var(--line);
      background: #12171c;
      position: sticky;
      top: 0;
      z-index: 10;
    }
    h1 { margin: 0; font-size: 18px; font-weight: 650; letter-spacing: 0; }
    .status { color: var(--muted); font-size: 13px; white-space: nowrap; }
    main {
      display: grid;
      grid-template-columns: minmax(360px, 1.25fr) minmax(340px, .75fr);
      gap: 14px;
      padding: 14px;
    }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
    }
    .panel h2 {
      margin: 0;
      padding: 11px 13px;
      font-size: 13px;
      font-weight: 650;
      color: var(--muted);
      border-bottom: 1px solid var(--line);
    }
    .video-wrap {
      background: #050709;
      min-height: 280px;
      display: grid;
      place-items: center;
    }
    .video-wrap img {
      display: block;
      width: 100%;
      max-height: 72vh;
      object-fit: contain;
    }
    .grid {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 10px;
      padding: 12px;
    }
    .metric {
      min-height: 74px;
      background: var(--panel2);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px;
    }
    .label { color: var(--muted); font-size: 12px; margin-bottom: 7px; }
    .value { font-size: 23px; line-height: 1.05; font-weight: 700; overflow-wrap: anywhere; }
    .unit { color: var(--muted); font-size: 12px; margin-left: 3px; }
    canvas { width: 100%; height: 176px; display: block; }
    .chart { padding: 10px 12px 12px; }
    .table {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 1px;
      background: var(--line);
    }
    .cell {
      background: var(--panel);
      padding: 9px 11px;
      min-height: 38px;
      font-size: 13px;
    }
    .cell span { color: var(--muted); display: block; font-size: 11px; margin-bottom: 3px; }
    @media (max-width: 920px) {
      main { grid-template-columns: 1fr; }
      .grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .status { white-space: normal; text-align: right; }
    }
  </style>
</head>
<body>
  <header>
    <h1>Scene Runtime Live Dashboard</h1>
    <div id="status" class="status">connecting...</div>
  </header>
  <main>
    <section class="panel">
      <h2>Live Detections</h2>
      <div class="video-wrap"><img id="stream" src="/stream.mjpg" alt="Live detection stream" /></div>
    </section>
    <section class="panel">
      <h2>Current Runtime</h2>
      <div class="grid">
        <div class="metric"><div class="label">Temperature</div><div class="value" id="temp">--</div></div>
        <div class="metric"><div class="label">Latency</div><div class="value" id="latency">--</div></div>
        <div class="metric"><div class="label">Actual FPS</div><div class="value" id="actualFps">--</div></div>
        <div class="metric"><div class="label">Resolution</div><div class="value" id="res">--</div></div>
      </div>
      <div class="table" id="details"></div>
    </section>
    <section class="panel">
      <h2>Thermal / FPS</h2>
      <div class="chart"><canvas id="chartThermal" width="900" height="250"></canvas></div>
    </section>
    <section class="panel">
      <h2>Latency / ONNX Runtime</h2>
      <div class="chart"><canvas id="chartLatency" width="900" height="250"></canvas></div>
    </section>
  </main>
  <script>
    const fmt = (v, d=1) => Number.isFinite(Number(v)) ? Number(v).toFixed(d) : "--";
    const byId = (id) => document.getElementById(id);

    function drawChart(canvas, series) {
      const ctx = canvas.getContext("2d");
      const w = canvas.width, h = canvas.height;
      ctx.clearRect(0, 0, w, h);
      ctx.fillStyle = "#151b21";
      ctx.fillRect(0, 0, w, h);
      ctx.strokeStyle = "#2d3842";
      ctx.lineWidth = 1;
      for (let i = 1; i < 4; i++) {
        const y = (h - 28) * i / 4 + 12;
        ctx.beginPath(); ctx.moveTo(36, y); ctx.lineTo(w - 12, y); ctx.stroke();
      }
      const all = series.flatMap(s => s.values).filter(v => Number.isFinite(Number(v)));
      if (!all.length) return;
      const min = Math.min(...all), max = Math.max(...all);
      const pad = Math.max((max - min) * 0.12, 1);
      const lo = min - pad, hi = max + pad;
      const n = Math.max(...series.map(s => s.values.length));
      ctx.font = "12px system-ui";
      ctx.fillStyle = "#a4b0bc";
      ctx.fillText(hi.toFixed(1), 6, 18);
      ctx.fillText(lo.toFixed(1), 6, h - 10);
      series.forEach(s => {
        ctx.strokeStyle = s.color;
        ctx.lineWidth = 2;
        ctx.beginPath();
        s.values.forEach((v, i) => {
          if (!Number.isFinite(Number(v))) return;
          const x = 38 + (w - 52) * (n <= 1 ? 0 : i / (n - 1));
          const y = 12 + (h - 30) * (1 - (Number(v) - lo) / (hi - lo));
          if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        });
        ctx.stroke();
        ctx.fillStyle = s.color;
        ctx.fillText(s.name, 44 + series.indexOf(s) * 145, 20);
      });
    }

    function updateDetails(s) {
      const rows = [
        ["Strategy", s.strategy],
        ["Action", s.action_mode],
        ["Thermal state", s.control_thermal_state || s.thermal_state],
        ["Decision", s.decision_reason],
        ["Governor", `${s.governor || "--"} -> ${s.applied_governor || "--"}`],
        ["Threads", s.cpu_threads],
        ["Interval", s.inference_interval],
        ["Detections", s.detection_count],
        ["Freq MHz", fmt(s.freq_mhz_avg, 0)],
        ["ARM clock", fmt(s.arm_clock_mhz, 0)],
        ["Soft temp limit", s.soft_temp_limit],
        ["Throttled", s.currently_throttled],
      ];
      byId("details").innerHTML = rows.map(([k, v]) => `<div class="cell"><span>${k}</span>${v ?? "--"}</div>`).join("");
    }

    async function tick() {
      try {
        const [state, history] = await Promise.all([
          fetch("/api/state", {cache: "no-store"}).then(r => r.json()),
          fetch("/api/history", {cache: "no-store"}).then(r => r.json()),
        ]);
        const age = Date.now() / 1000 - (state.updated_at || 0);
        byId("status").textContent = `${state.host || location.host} | frame ${state.frame_id ?? "--"} | ${age.toFixed(1)}s ago`;
        byId("temp").innerHTML = `${fmt(state.temp_c)}<span class="unit">C</span>`;
        byId("latency").innerHTML = `${fmt(state.latency_ms, 0)}<span class="unit">ms</span>`;
        byId("actualFps").textContent = fmt(state.actual_inference_fps, 3);
        byId("res").textContent = state.resolved_input_resolution || state.input_resolution || "--";
        updateDetails(state);
        drawChart(byId("chartThermal"), [
          {name: "temp C", color: "#f4c95d", values: history.map(x => x.temp_c)},
          {name: "actual FPS", color: "#60d394", values: history.map(x => x.actual_inference_fps)},
          {name: "resolution/10", color: "#8ab4f8", values: history.map(x => (x.resolved_input_resolution || x.input_resolution || 0) / 10)},
        ]);
        drawChart(byId("chartLatency"), [
          {name: "latency ms", color: "#ef6f6c", values: history.map(x => x.latency_ms)},
          {name: "onnx ms", color: "#48cae4", values: history.map(x => x.onnx_run_ms)},
        ]);
      } catch (err) {
        byId("status").textContent = "waiting for dashboard data...";
      }
    }
    setInterval(tick, 1000);
    tick();
  </script>
</body>
</html>
"""


@dataclass
class LiveDashboardState:
    """Thread-safe storage for live metrics and the latest annotated frame."""

    max_history: int = 600
    jpeg_quality: int = 80
    jpeg_width: int = 960
    score_threshold: float = 0.5
    show_stream: bool = True
    latest: dict[str, Any] = field(default_factory=dict)
    history: deque[dict[str, Any]] = field(init=False)
    jpeg: bytes | None = None
    condition: threading.Condition = field(default_factory=threading.Condition)

    def __post_init__(self) -> None:
        self.history = deque(maxlen=self.max_history)

    def publish(
        self,
        payload: dict[str, Any],
        frame: np.ndarray | None,
        detections: list[Detection],
        input_resolution: int | None,
    ) -> None:
        now = time.time()
        cleaned = _json_safe(payload)
        cleaned["updated_at"] = now
        with self.condition:
            self.latest = cleaned
            self.history.append(cleaned)
            if self.show_stream and frame is not None:
                encoded = encode_annotated_jpeg(
                    frame,
                    detections,
                    input_resolution or int(cleaned.get("input_resolution") or 640),
                    cleaned,
                    width=self.jpeg_width,
                    quality=self.jpeg_quality,
                    score_threshold=self.score_threshold,
                )
                if encoded is not None:
                    self.jpeg = encoded
            self.condition.notify_all()

    def snapshot(self) -> dict[str, Any]:
        with self.condition:
            return dict(self.latest)

    def history_snapshot(self) -> list[dict[str, Any]]:
        with self.condition:
            return list(self.history)


class LiveDashboardServer:
    """Small HTTP server exposing dashboard pages and live state."""

    def __init__(self, state: LiveDashboardState, host: str = "0.0.0.0", port: int = 8000) -> None:
        self.state = state
        self.host = host
        self.port = port
        handler = self._make_handler(state)
        self._httpd = ThreadingHTTPServer((host, port), handler)
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()
        self._thread.join(timeout=2.0)

    @staticmethod
    def _make_handler(state: LiveDashboardState) -> type[BaseHTTPRequestHandler]:
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, fmt: str, *args: Any) -> None:
                return

            def do_GET(self) -> None:
                if self.path in ("/", "/index.html"):
                    self._send_bytes(INDEX_HTML.encode("utf-8"), "text/html; charset=utf-8")
                    return
                if self.path.startswith("/api/state"):
                    self._send_json(state.snapshot())
                    return
                if self.path.startswith("/api/history"):
                    self._send_json(state.history_snapshot())
                    return
                if self.path.startswith("/stream.mjpg"):
                    self._send_stream()
                    return
                self.send_error(404)

            def _send_bytes(self, data: bytes, content_type: str) -> None:
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Cache-Control", "no-store")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def _send_json(self, data: Any) -> None:
                self._send_bytes(json.dumps(data).encode("utf-8"), "application/json")

            def _send_stream(self) -> None:
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                last_jpeg: bytes | None = None
                try:
                    while True:
                        with state.condition:
                            state.condition.wait(timeout=2.0)
                            jpeg = state.jpeg
                        if jpeg is None or jpeg is last_jpeg:
                            continue
                        last_jpeg = jpeg
                        self.wfile.write(b"--frame\r\n")
                        self.wfile.write(b"Content-Type: image/jpeg\r\n")
                        self.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode("ascii"))
                        self.wfile.write(jpeg)
                        self.wfile.write(b"\r\n")
                except (BrokenPipeError, ConnectionResetError):
                    return

        return Handler


def encode_annotated_jpeg(
    frame: np.ndarray,
    detections: list[Detection],
    input_resolution: int,
    payload: dict[str, Any],
    *,
    width: int,
    quality: int,
    score_threshold: float,
) -> bytes | None:
    try:
        import cv2
    except ImportError:
        return None

    annotated = frame.copy()
    h, w = annotated.shape[:2]
    if width > 0 and w > width:
        scale = width / float(w)
        annotated = cv2.resize(annotated, (width, max(1, int(h * scale))))
        h, w = annotated.shape[:2]

    sx = w / float(input_resolution)
    sy = h / float(input_resolution)
    for detection in detections:
        if detection.score < score_threshold:
            continue
        x1, y1, x2, y2 = detection.bbox
        pt1 = (max(0, int(x1 * sx)), max(0, int(y1 * sy)))
        pt2 = (min(w - 1, int(x2 * sx)), min(h - 1, int(y2 * sy)))
        color = _color(detection.class_id)
        cv2.rectangle(annotated, pt1, pt2, color, 2)
        label = f"{_label(detection.class_id)} {detection.score:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
        y = max(0, pt1[1] - th - 7)
        cv2.rectangle(annotated, (pt1[0], y), (pt1[0] + tw + 7, y + th + 7), color, -1)
        cv2.putText(
            annotated,
            label,
            (pt1[0] + 3, y + th + 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (8, 10, 12),
            1,
            cv2.LINE_AA,
        )

    overlay = (
        f"res={payload.get('resolved_input_resolution') or payload.get('input_resolution')} "
        f"lat={_fmt(payload.get('latency_ms'), 0)}ms "
        f"temp={_fmt(payload.get('temp_c'), 1)}C "
        f"fps={_fmt(payload.get('actual_inference_fps'), 3)}"
    )
    cv2.rectangle(annotated, (0, 0), (min(w, 760), 34), (12, 16, 20), -1)
    cv2.putText(
        annotated,
        overlay,
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (245, 248, 250),
        2,
        cv2.LINE_AA,
    )
    ok, encoded = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        return None
    return encoded.tobytes()


def _label(class_id: int) -> str:
    if 0 <= class_id < len(COCO80):
        return COCO80[class_id]
    return str(class_id)


def _color(class_id: int) -> tuple[int, int, int]:
    palette = [
        (65, 210, 128), (245, 190, 75), (85, 170, 255), (235, 105, 105),
        (185, 125, 255), (70, 215, 215), (230, 120, 200), (135, 220, 80),
    ]
    return palette[class_id % len(palette)]


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and (value != value or value in (float("inf"), float("-inf"))):
        return None
    return value


def _fmt(value: Any, digits: int) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "--"
