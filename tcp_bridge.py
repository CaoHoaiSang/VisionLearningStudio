import socket
import struct
import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class BridgeStats:
    client: str = "-"
    frames: int = 0
    fps: float = 0.0
    processing_ms: float = 0.0
    last_error: str = "-"


class TcpVisionBridge:
    def __init__(self, process_frame, on_frame=None, on_status=None):
        self.process_frame, self.on_frame, self.on_status = process_frame, on_frame, on_status
        self.stats, self._stop = BridgeStats(), threading.Event()
        self._server = self._client = self._thread = None

    @property
    def running(self):
        return bool(self._thread and self._thread.is_alive())

    def start(self, host="0.0.0.0", port=6001):
        if self.running:
            return
        self._stop.clear()
        self.stats = BridgeStats()
        self._thread = threading.Thread(target=self._serve, args=(host, int(port)), daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        for sock in (self._client, self._server):
            try:
                if sock:
                    sock.close()
            except OSError:
                pass
        if self._thread and self._thread is not threading.current_thread():
            self._thread.join(timeout=1)
        self._thread = None

    def _status(self, text):
        if self.on_status:
            self.on_status(text, self.stats)

    def _serve(self, host, port):
        try:
            self._server = socket.socket()
            self._server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self._server.bind((host, port))
            self._server.listen(1)
            self._server.settimeout(.5)
            self._status(f"Đang nghe {host}:{port}")
            while not self._stop.is_set():
                try:
                    client, address = self._server.accept()
                except socket.timeout:
                    continue
                self._client = client
                client.settimeout(3)
                self.stats.client = f"{address[0]}:{address[1]}"
                self._status(f"Đã kết nối {self.stats.client}")
                started = time.perf_counter()
                try:
                    while not self._stop.is_set():
                        size = struct.unpack("<I", self._recv(client, 4))[0]
                        if not 0 < size <= 25 * 1024 * 1024:
                            raise ValueError("Kích thước JPEG không hợp lệ")
                        image = cv2.imdecode(np.frombuffer(self._recv(client, size), np.uint8), cv2.IMREAD_COLOR)
                        if image is None:
                            raise ValueError("Không giải mã được JPEG")
                        t0 = time.perf_counter()
                        detections, overlay = self.process_frame(image)
                        self.stats.processing_ms = (time.perf_counter() - t0) * 1000
                        self.stats.frames += 1
                        self.stats.fps = self.stats.frames / max(.001, time.perf_counter() - started)
                        payload = bytearray(struct.pack("<I", min(64, len(detections))))
                        for item in detections[:64]:
                            payload.extend(struct.pack("<ifffff", *item.to_robot_tuple()))
                        client.sendall(payload)
                        if self.on_frame:
                            self.on_frame(overlay, detections, self.stats)
                except (OSError, ConnectionError, ValueError) as exc:
                    if not self._stop.is_set():
                        self.stats.last_error = str(exc)
                        self._status(f"Mất kết nối: {exc}")
                finally:
                    client.close()
                    self.stats.client = "-"
        except OSError as exc:
            self.stats.last_error = str(exc)
            self._status(f"Không mở được cầu nối: {exc}")
        finally:
            self._status("Cầu nối đã dừng")

    @staticmethod
    def _recv(sock, length):
        data = bytearray()
        while len(data) < length:
            chunk = sock.recv(length - len(data))
            if not chunk:
                raise ConnectionError("Client đã đóng kết nối")
            data.extend(chunk)
        return bytes(data)
