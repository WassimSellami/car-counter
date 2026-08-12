"""Helpers for opening Windows camera devices with backend fallbacks."""

from __future__ import annotations

import time

import cv2


def open_camera(
    index: int, width: int | None = None, height: int | None = None
) -> cv2.VideoCapture:
    """Open a camera after applying and validating an optional capture mode.

    Some Windows camera drivers accept a resolution change through one backend,
    but fail on the next ``read``. Validate the requested mode before choosing a
    backend so DirectShow can be used when Media Foundation is unreliable.
    """
    backends = [cv2.CAP_MSMF, cv2.CAP_DSHOW, cv2.CAP_ANY]
    last_camera = None

    for backend in backends:
        camera = cv2.VideoCapture(index, backend)
        if camera.isOpened():
            try:
                camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                if width is not None:
                    camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                if height is not None:
                    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                deadline = time.perf_counter() + 2.0
                while time.perf_counter() < deadline:
                    success, frame = camera.read()
                    if success and frame is not None and frame.size > 0:
                        return camera
                    time.sleep(0.05)
            except cv2.error:
                # This driver/backend cannot produce the requested mode.
                # Release it and let the next backend try.
                pass
            camera.release()
            last_camera = camera
            continue
        camera.release()
        last_camera = camera

    assert last_camera is not None
    return last_camera
