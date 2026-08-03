from __future__ import annotations

import threading
import time

import cv2
import numpy as np


try:
    from pypylon import pylon
except Exception:
    pylon = None


class BaslerCamera:
    def __init__(self, on_frame, on_status):
        self.on_frame = on_frame
        self.on_status = on_status
        self.camera = None
        self.converter = None
        self._stop = threading.Event()
        self._thread = None

    @property
    def available(self):
        return pylon is not None

    @property
    def running(self):
        return bool(self._thread and self._thread.is_alive())

    def enumerate(self):
        if pylon is None:
            return []
        devices = pylon.TlFactory.GetInstance().EnumerateDevices()
        return [
            {
                "model": device.GetModelName(),
                "serial": device.GetSerialNumber(),
                "ip": getattr(device, "GetIpAddress", lambda: "")(),
                "device": device,
            }
            for device in devices
        ]

    def connect(self, serial=None):
        if pylon is None:
            raise RuntimeError("Chưa cài pypylon/Basler Runtime")
        devices = self.enumerate()
        if not devices:
            raise RuntimeError("Không tìm thấy camera Basler")
        selected = next((d for d in devices if d["serial"] == serial), devices[0])
        self.camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateDevice(selected["device"]))
        self.camera.Open()
        self.converter = pylon.ImageFormatConverter()
        self.converter.OutputPixelFormat = pylon.PixelType_BGR8packed
        self.converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
        self.on_status(f"Đã kết nối {selected['model']} · {selected['serial']} · {selected['ip']}")

    def apply(self, exposure_us, gain_db, width=0, height=0, frame_rate=0, trigger_mode="Off"):
        if not self.camera or not self.camera.IsOpen():
            raise RuntimeError("Camera chưa kết nối")
        if self.running:
            raise RuntimeError("Hãy dừng LIVE trước khi đổi ROI/Trigger")
        for auto_name in ("ExposureAuto", "GainAuto"):
            node = getattr(self.camera, auto_name, None)
            if node and node.IsWritable():
                node.SetValue("Off")
        for name, value in (("Width", width), ("Height", height)):
            if float(value) > 0:
                self._set_clamped(name, float(value))
        self._set_clamped("ExposureTime", float(exposure_us))
        self._set_clamped("Gain", float(gain_db))
        rate_enable = getattr(self.camera, "AcquisitionFrameRateEnable", None)
        if rate_enable and rate_enable.IsWritable():
            rate_enable.SetValue(float(frame_rate) > 0)
        if float(frame_rate) > 0:
            self._set_clamped("AcquisitionFrameRate", float(frame_rate))
        trigger = getattr(self.camera, "TriggerMode", None)
        if trigger and trigger.IsWritable():
            trigger.SetValue("On" if trigger_mode != "Off" else "Off")
        if trigger_mode == "Software":
            source = getattr(self.camera, "TriggerSource", None)
            if source and source.IsWritable():
                source.SetValue("Software")
        elif trigger_mode == "Line1":
            source = getattr(self.camera, "TriggerSource", None)
            if source and source.IsWritable():
                source.SetValue("Line1")
        return self.read_settings()

    def _set_clamped(self, name, value):
        node = getattr(self.camera, name, None)
        if node and node.IsWritable():
            value = max(node.GetMin(), min(node.GetMax(), value))
            increment = getattr(node, "GetInc", lambda: 0)()
            if increment:
                value = node.GetMin() + round((value - node.GetMin()) / increment) * increment
            node.SetValue(value)

    def read_settings(self):
        result = {}
        for name in ("ExposureTime", "Gain", "Width", "Height", "AcquisitionFrameRate", "TriggerMode", "TriggerSource"):
            node = getattr(self.camera, name, None)
            if node and node.IsReadable():
                result[name] = node.GetValue()
        return result

    def software_trigger(self):
        if not self.running:
            raise RuntimeError("Camera chưa LIVE")
        trigger = getattr(self.camera, "TriggerSoftware", None)
        if trigger is None:
            raise RuntimeError("Camera không hỗ trợ Software Trigger")
        trigger.Execute()

    def start(self):
        if self.running:
            return
        if not self.camera or not self.camera.IsOpen():
            self.connect()
        self._stop.clear()
        self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self.camera and self.camera.IsGrabbing():
            self.camera.StopGrabbing()
        if self._thread and self._thread is not threading.current_thread():
            self._thread.join(timeout=1)
        self._thread = None

    def close(self):
        self.stop()
        if self.camera and self.camera.IsOpen():
            self.camera.Close()
        self.camera = None

    def _loop(self):
        last = time.perf_counter()
        while not self._stop.is_set() and self.camera.IsGrabbing():
            grab = self.camera.RetrieveResult(500, pylon.TimeoutHandling_Return)
            if not grab or not grab.GrabSucceeded():
                continue
            image = self.converter.Convert(grab).GetArray()
            now = time.perf_counter()
            fps = 1.0 / max(1e-6, now - last)
            last = now
            self.on_frame(np.asarray(image), fps)
            grab.Release()
