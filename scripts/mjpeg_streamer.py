#!/usr/bin/env python3
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

import cv2
import numpy as np


class MjpegStreamer(object):
    def __init__(
        self,
        host="127.0.0.1",
        port=8090,
        stream_path="/stream.mjpg",
        snapshot_path="/snapshot.jpg",
        fps=15.0,
        jpeg_quality=85,
        title="XAI VLM Live",
    ):
        self.host = str(host)
        self.port = int(port)
        self.stream_path = _normalize_path(stream_path)
        self.snapshot_path = _normalize_path(snapshot_path)
        self.fps = max(1.0, float(fps))
        self.jpeg_quality = max(1, min(100, int(jpeg_quality)))
        self.title = str(title)
        self._frame = None
        self._frame_stamp = None
        self._lock = threading.Lock()
        self._server = None
        self._thread = None
        self._stopped = threading.Event()

    @property
    def stream_url(self):
        return "http://{}:{}{}".format(self.host, self.port, self.stream_path)

    @property
    def page_url(self):
        return "http://{}:{}/".format(self.host, self.port)

    def start(self):
        if self._server is not None:
            return True

        streamer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, fmt, *args):
                return

            def do_GET(self):
                streamer._handle_get(self)

            def do_OPTIONS(self):
                self.send_response(204)
                streamer._send_cors_headers(self)
                self.end_headers()

        try:
            self._server = ThreadingHTTPServer((self.host, self.port), Handler)
        except OSError:
            self._server = None
            return False

        self._thread = threading.Thread(target=self._server.serve_forever, name="mjpeg-streamer", daemon=True)
        self._thread.start()
        return True

    def stop(self):
        self._stopped.set()
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self._thread = None

    def update(self, frame_bgr):
        if frame_bgr is None:
            return
        with self._lock:
            self._frame = frame_bgr.copy()
            self._frame_stamp = time.time()

    def _handle_get(self, handler):
        path = urlparse(handler.path).path
        if path in ("", "/"):
            return self._serve_index(handler)
        if path == self.stream_path:
            return self._serve_stream(handler)
        if path == self.snapshot_path:
            return self._serve_snapshot(handler)
        if path == "/health":
            return self._serve_health(handler)
        handler.send_error(404)

    def _serve_index(self, handler):
        body = (
            "<!doctype html><html><head><meta charset=\"utf-8\">"
            "<title>{title}</title>"
            "<style>body{{margin:0;background:#111827;color:#f8fafc;font-family:sans-serif}}"
            "main{{display:grid;gap:12px;padding:16px}}img{{max-width:100%;height:auto;background:#020617}}"
            "code{{color:#93c5fd}}</style></head><body><main>"
            "<h1>{title}</h1><img src=\"{stream_path}\" alt=\"live stream\">"
            "<code>{url}</code></main></body></html>"
        ).format(title=self.title, stream_path=self.stream_path, url=self.stream_url)
        data = body.encode("utf-8")
        handler.send_response(200)
        self._send_cors_headers(handler)
        handler.send_header("Content-Type", "text/html; charset=utf-8")
        handler.send_header("Content-Length", str(len(data)))
        handler.end_headers()
        handler.wfile.write(data)

    def _serve_health(self, handler):
        with self._lock:
            has_frame = self._frame is not None
            frame_stamp = self._frame_stamp
        payload = {
            "ok": True,
            "has_frame": has_frame,
            "frame_age_s": None if frame_stamp is None else round(time.time() - frame_stamp, 3),
            "stream_url": self.stream_url,
        }
        data = json.dumps(payload).encode("utf-8")
        handler.send_response(200)
        self._send_cors_headers(handler)
        handler.send_header("Content-Type", "application/json")
        handler.send_header("Content-Length", str(len(data)))
        handler.end_headers()
        handler.wfile.write(data)

    def _serve_snapshot(self, handler):
        jpeg = self._latest_jpeg()
        handler.send_response(200)
        self._send_cors_headers(handler)
        handler.send_header("Content-Type", "image/jpeg")
        handler.send_header("Cache-Control", "no-store")
        handler.send_header("Content-Length", str(len(jpeg)))
        handler.end_headers()
        handler.wfile.write(jpeg)

    def _serve_stream(self, handler):
        handler.send_response(200)
        self._send_cors_headers(handler)
        handler.send_header("Age", "0")
        handler.send_header("Cache-Control", "no-cache, private")
        handler.send_header("Pragma", "no-cache")
        handler.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        handler.end_headers()

        delay_s = 1.0 / self.fps
        while not self._stopped.is_set():
            jpeg = self._latest_jpeg()
            try:
                handler.wfile.write(b"--frame\r\n")
                handler.wfile.write(b"Content-Type: image/jpeg\r\n")
                handler.wfile.write("Content-Length: {}\r\n\r\n".format(len(jpeg)).encode("ascii"))
                handler.wfile.write(jpeg)
                handler.wfile.write(b"\r\n")
            except (BrokenPipeError, ConnectionResetError):
                break
            time.sleep(delay_s)

    def _latest_jpeg(self):
        with self._lock:
            frame = None if self._frame is None else self._frame.copy()
        if frame is None:
            frame = _placeholder_frame(self.title)
        ok, encoded = cv2.imencode(
            ".jpg",
            frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
        )
        if not ok:
            placeholder = _placeholder_frame("JPEG encode failed")
            ok, encoded = cv2.imencode(".jpg", placeholder)
        return encoded.tobytes()

    def _send_cors_headers(self, handler):
        handler.send_header("Access-Control-Allow-Origin", "*")
        handler.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        handler.send_header("Access-Control-Allow-Headers", "Content-Type")


def _normalize_path(path):
    path = str(path or "/stream.mjpg").strip()
    if not path.startswith("/"):
        path = "/" + path
    return path


def _placeholder_frame(title):
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    frame[:, :] = (28, 35, 49)
    cv2.putText(frame, title[:36], (28, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (240, 248, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "waiting for VLM frame", (28, 205), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (147, 197, 253), 2, cv2.LINE_AA)
    return frame
