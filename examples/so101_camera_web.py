#!/usr/bin/env python

"""Preview the two SO-101 USB cameras in a browser.

Run with the Python environment used for recording::

    python examples/so101_camera_web.py --host 0.0.0.0 --port 8765

The page is available at http://<robot-host>:8765/. Stop the viewer before
starting a separate recording process, since both programs open the cameras.
"""

from __future__ import annotations

import argparse
import json
import logging
import threading
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

import cv2


CAMERA_DEVICES = {
    "front": (
        "前置相机",
        "/dev/v4l/by-id/usb-icSpring_icspring_camera_202404160005-video-index0",
        "MJPG",
    ),
    "handeye": (
        "手眼相机",
        "/dev/v4l/by-id/usb-icSpring_icspring_camera-video-index0",
        "YUYV",
    ),
}

PAGE = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>SO-101 相机实时预览</title>
  <style>
    :root { color-scheme: dark; font-family: system-ui, -apple-system, sans-serif; }
    * { box-sizing: border-box; }
    body { margin: 0; background: #0b1018; color: #e9f0fb; }
    main { max-width: 1480px; margin: auto; padding: 28px clamp(16px, 3vw, 40px); }
    header { display: flex; flex-wrap: wrap; align-items: end; justify-content: space-between; gap: 12px; margin-bottom: 24px; }
    h1 { font-size: clamp(24px, 3vw, 34px); letter-spacing: -.03em; margin: 0 0 6px; }
    p { margin: 0; color: #9caec5; }
    .badge { padding: 8px 12px; border: 1px solid #34455c; border-radius: 999px; color: #b7c9e0; font-size: 13px; }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 490px), 1fr)); gap: 22px; }
    article { background: #141d2a; border: 1px solid #29384d; border-radius: 16px; overflow: hidden; box-shadow: 0 16px 40px #0004; }
    .bar { display: flex; align-items: center; justify-content: space-between; gap: 12px; padding: 15px 18px; }
    h2 { margin: 0; font-size: 18px; }
    .state { display: inline-flex; align-items: center; gap: 7px; font-size: 13px; color: #a8bad1; }
    .dot { width: 9px; height: 9px; border-radius: 50%; background: #ef6871; }
    .state.online .dot { background: #54d9a4; box-shadow: 0 0 12px #54d9a488; }
    img { display: block; width: 100%; aspect-ratio: 4 / 3; object-fit: contain; background: #070b11; }
    .footer { display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: 12px; padding: 14px 18px; color: #9caec5; font-size: 13px; }
    button { border: 1px solid #4b6f9a; border-radius: 9px; background: #233e60; color: #e9f0fb; padding: 8px 12px; font: inherit; cursor: pointer; }
    button:hover { background: #31527c; }
    #message { margin-top: 18px; min-height: 20px; font-size: 13px; color: #ecb57a; }
  </style>
</head>
<body>
  <main>
    <header>
      <div><h1>SO-101 相机实时预览</h1><p>前置与手眼画面 · 640 × 480</p></div>
      <span class="badge">192.168.1.9 · 本地网络</span>
    </header>
    <div class="grid">
      <article>
        <div class="bar"><h2>前置相机</h2><span id="state-front" class="state"><span class="dot"></span><span>连接中</span></span></div>
        <img id="image-front" src="/stream/front" alt="前置相机实时画面">
        <div class="footer"><span id="meta-front">等待画面…</span><button onclick="saveSnapshot('front')">保存当前帧</button></div>
      </article>
      <article>
        <div class="bar"><h2>手眼相机</h2><span id="state-handeye" class="state"><span class="dot"></span><span>连接中</span></span></div>
        <img id="image-handeye" src="/stream/handeye" alt="手眼相机实时画面">
        <div class="footer"><span id="meta-handeye">等待画面…</span><button onclick="saveSnapshot('handeye')">保存当前帧</button></div>
      </article>
    </div>
    <div id="message" role="status"></div>
  </main>
  <script>
    for (const name of ['front', 'handeye']) {
      const image = document.getElementById(`image-${name}`);
      image.onerror = () => setTimeout(() => {
        image.src = `/stream/${name}?retry=${Date.now()}`;
      }, 1500);
    }

    async function refreshStatus() {
      try {
        const response = await fetch('/api/status', {cache: 'no-store'});
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const statuses = await response.json();
        for (const name of ['front', 'handeye']) {
          const status = statuses[name];
          const state = document.getElementById(`state-${name}`);
          state.classList.toggle('online', status.online);
          state.lastElementChild.textContent = status.online ? '在线' : '离线';
          document.getElementById(`meta-${name}`).textContent = status.online
            ? `${status.width} × ${status.height} · ${status.fps.toFixed(1)} fps · ${status.age_s.toFixed(1)} 秒前`
            : (status.error || '等待相机画面…');
        }
        document.getElementById('message').textContent = '';
      } catch (error) {
        document.getElementById('message').textContent = `状态更新失败：${error.message}`;
      }
    }

    async function saveSnapshot(name) {
      try {
        const response = await fetch(`/snapshot/${name}.jpg`, {cache: 'no-store'});
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const url = URL.createObjectURL(await response.blob());
        const link = document.createElement('a');
        link.href = url;
        link.download = `so101_${name}_${new Date().toISOString().replace(/[:.]/g, '-')}.jpg`;
        link.click();
        setTimeout(() => URL.revokeObjectURL(url), 1000);
      } catch (error) {
        document.getElementById('message').textContent = `保存失败：${error.message}`;
      }
    }

    refreshStatus();
    setInterval(refreshStatus, 2000);
  </script>
</body>
</html>
"""


@dataclass(frozen=True)
class CameraSpec:
    name: str
    label: str
    device: str
    fourcc: str


class CameraWorker:
    def __init__(
        self,
        spec: CameraSpec,
        width: int,
        height: int,
        fps: int,
        jpeg_quality: int,
        controls: list[tuple[int, float]] | None = None,
    ):
        self.spec = spec
        self.width = width
        self.height = height
        self.requested_fps = fps
        self.jpeg_quality = jpeg_quality
        self.controls = controls or []
        self.condition = threading.Condition()
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, name=f"camera-{spec.name}", daemon=True)
        self.jpeg: bytes | None = None
        self.sequence = 0
        self.updated_at = 0.0
        self.actual_fps = 0.0
        self.frame_width = 0
        self.frame_height = 0
        self.error: str | None = "连接中"

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        with self.condition:
            self.condition.notify_all()
        self.thread.join(timeout=3)

    def _set_error(self, message: str) -> None:
        logging.warning("%s: %s", self.spec.label, message)
        with self.condition:
            self.error = message
            self.condition.notify_all()

    def _run(self) -> None:
        while not self.stop_event.is_set():
            if not Path(self.spec.device).exists():
                self._set_error("未找到相机设备，等待重新连接")
                self.stop_event.wait(2)
                continue

            capture = cv2.VideoCapture(self.spec.device, cv2.CAP_V4L2)
            if not capture.isOpened():
                self._set_error("无法打开相机，可能正被录制程序占用")
                capture.release()
                self.stop_event.wait(2)
                continue

            capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.spec.fourcc))
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            capture.set(cv2.CAP_PROP_FPS, self.requested_fps)
            for prop, val in self.controls:
                ok = capture.set(prop, val)
                logging.info(
                    "%s ctrl %d set %s ok=%s get=%s",
                    self.spec.label, prop, val, ok, capture.get(prop),
                )
            logging.info("%s 已打开：%s", self.spec.label, self.spec.device)

            window_start = time.monotonic()
            window_frames = 0
            try:
                while not self.stop_event.is_set():
                    ok, frame = capture.read()
                    if not ok:
                        self._set_error("读取画面失败，正在重连")
                        break

                    ok, encoded = cv2.imencode(
                        ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
                    )
                    if not ok:
                        self._set_error("JPEG 编码失败，正在重试")
                        break

                    now = time.monotonic()
                    window_frames += 1
                    with self.condition:
                        self.jpeg = encoded.tobytes()
                        self.sequence += 1
                        self.updated_at = now
                        self.frame_height, self.frame_width = frame.shape[:2]
                        self.error = None
                        if now - window_start >= 2:
                            self.actual_fps = window_frames / (now - window_start)
                            window_start = now
                            window_frames = 0
                        self.condition.notify_all()
            finally:
                capture.release()

            self.stop_event.wait(1)

    def status(self) -> dict[str, object]:
        with self.condition:
            age = time.monotonic() - self.updated_at if self.updated_at else None
            return {
                "online": age is not None and age < 3,
                "fps": round(self.actual_fps, 1),
                "age_s": round(age, 1) if age is not None else None,
                "width": self.frame_width,
                "height": self.frame_height,
                "error": self.error,
            }

    def latest(self) -> bytes | None:
        with self.condition:
            if not self.updated_at or time.monotonic() - self.updated_at >= 3:
                return None
            return self.jpeg

    def wait_for_frame(self, last_sequence: int) -> tuple[int, bytes | None]:
        with self.condition:
            self.condition.wait_for(
                lambda: self.sequence > last_sequence or self.stop_event.is_set(), timeout=5
            )
            if self.sequence > last_sequence and time.monotonic() - self.updated_at < 3:
                return self.sequence, self.jpeg
            return last_sequence, None


class CameraHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address: tuple[str, int], workers: dict[str, CameraWorker]):
        super().__init__(address, CameraHandler)
        self.workers = workers


class CameraHandler(BaseHTTPRequestHandler):
    server: CameraHTTPServer

    def do_GET(self) -> None:
        path = urlsplit(self.path).path
        if path == "/":
            self._send(PAGE.encode(), "text/html; charset=utf-8")
        elif path == "/api/status":
            payload = {name: worker.status() for name, worker in self.server.workers.items()}
            self._send(json.dumps(payload, ensure_ascii=False).encode(), "application/json; charset=utf-8")
        elif path.startswith("/snapshot/") and path.endswith(".jpg"):
            name = path.removeprefix("/snapshot/").removesuffix(".jpg")
            worker = self.server.workers.get(name)
            if worker is None:
                self.send_error(404, "Unknown camera")
                return
            jpeg = worker.latest()
            if jpeg is None:
                self.send_error(503, "Camera is offline")
                return
            self._send(jpeg, "image/jpeg")
        elif path.startswith("/stream/"):
            name = path.removeprefix("/stream/")
            worker = self.server.workers.get(name)
            if worker is None:
                self.send_error(404, "Unknown camera")
                return
            self._stream(worker)
        else:
            self.send_error(404, "Not found")

    def _send(self, body: bytes, content_type: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _stream(self, worker: CameraWorker) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()

        sequence = 0
        try:
            while True:
                sequence, jpeg = worker.wait_for_frame(sequence)
                if jpeg is None:
                    continue
                self.wfile.write(
                    b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                    + str(len(jpeg)).encode()
                    + b"\r\n\r\n"
                    + jpeg
                    + b"\r\n"
                )
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass

    def log_message(self, format: str, *args: object) -> None:
        logging.info("%s %s", self.address_string(), format % args)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1", help="Listen address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--jpeg-quality", type=int, default=80)
    parser.add_argument("--auto-exposure", dest="auto_exposure", action="store_true", default=True)
    parser.add_argument("--no-auto-exposure", dest="auto_exposure", action="store_false")
    parser.add_argument("--exposure", type=float, default=None, help="V4L2 absolute exposure")
    parser.add_argument("--gain", type=float, default=None)
    parser.add_argument("--brightness", type=float, default=None)
    args = parser.parse_args()

    if not 1 <= args.jpeg_quality <= 100:
        parser.error("--jpeg-quality must be between 1 and 100")

    controls: list[tuple[int, float]] = []
    if args.exposure is not None or not args.auto_exposure:
        # This camera only accepts manual mode via auto_exposure=1 (0.25/2 are rejected).
        controls.append((cv2.CAP_PROP_AUTO_EXPOSURE, 1.0))
    if args.exposure is not None:
        controls.append((cv2.CAP_PROP_EXPOSURE, args.exposure))
    if args.gain is not None:
        controls.append((cv2.CAP_PROP_GAIN, args.gain))
    if args.brightness is not None:
        controls.append((cv2.CAP_PROP_BRIGHTNESS, args.brightness))

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    workers = {
        name: CameraWorker(CameraSpec(name, *device), args.width, args.height, args.fps, args.jpeg_quality, controls)
        for name, device in CAMERA_DEVICES.items()
    }
    server = CameraHTTPServer((args.host, args.port), workers)
    for worker in workers.values():
        worker.start()

    logging.info("SO-101 camera viewer: http://%s:%s/", args.host, args.port)
    try:
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        for worker in workers.values():
            worker.stop()


if __name__ == "__main__":
    main()
