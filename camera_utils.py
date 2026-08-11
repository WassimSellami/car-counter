"""Helpers for opening Windows camera devices with backend fallbacks."""

from __future__ import annotations

import time

import cv2


def open_camera(index: int) -> cv2.VideoCapture:
    """Open a camera using the first backend that works on this machine."""
    backends = [cv2.CAP_MSMF, cv2.CAP_DSHOW, cv2.CAP_ANY]
    last_camera = None

    for backend in backends:
        camera = cv2.VideoCapture(index, backend)
        if camera.isOpened():
            camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            deadline = time.perf_counter() + 2.0
            while time.perf_counter() < deadline:
                success, frame = camera.read()
                if success and frame is not None and frame.size > 0:
                    return camera
                time.sleep(0.05)
            camera.release()
            continue
        camera.release()
        last_camera = camera

    assert last_camera is not None
    return last_camera