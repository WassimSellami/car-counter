"""Crop the local DroidCam feed and send ordered batches to the Vast counter."""

from queue import Empty, Full, Queue
from threading import Event, Thread
import os
from pathlib import Path
import time

import cv2
import numpy as np
import requests

from camera_utils import open_camera
from constants import CAMERA_HEIGHT, CAMERA_SOURCE, CAMERA_WIDTH, CLOUD_BATCH_SIZE


SEND_WIDTH = 640
JPEG_QUALITY = 70


def load_dotenv() -> None:
    env_file = Path(".env")
    if not env_file.is_file():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        if line and not line.lstrip().startswith("#") and "=" in line:
            name, value = line.split("=", maxsplit=1)
            os.environ.setdefault(name.strip(), value.strip().strip('"').strip("'"))


def select_road() -> tuple[np.ndarray, tuple[int, int]]:
    """Capture one camera frame, then choose TL, TR, BR, BL road corners."""
    camera = open_camera(CAMERA_SOURCE, CAMERA_WIDTH, CAMERA_HEIGHT)
    if not camera.isOpened():
        raise RuntimeError(f"Could not open camera {CAMERA_SOURCE}")
    try:
        while True:
            success, frame = camera.read()
            if not success:
                raise RuntimeError("Could not read the camera")
            cv2.putText(frame, "Press S to capture a road snapshot", (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow("Camera preview", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("s"), ord("S")):
                break
            if key == ord("q"):
                raise SystemExit(0)
    finally:
        camera.release()
        cv2.destroyWindow("Camera preview")

    points: list[tuple[int, int]] = []
    window = "Select road: click TL, TR, BR, BL; then Enter"
    def on_mouse(event: int, x: int, y: int, _flags: int, _params: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append((x, y))
    cv2.namedWindow(window)
    cv2.setMouseCallback(window, on_mouse)
    while True:
        preview = frame.copy()
        for index, point in enumerate(points, start=1):
            cv2.circle(preview, point, 6, (0, 255, 0), -1)
            cv2.putText(preview, str(index), (point[0] + 8, point[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        if len(points) > 1:
            cv2.polylines(preview, [np.int32(points)], len(points) == 4, (0, 255, 0), 2)
        cv2.imshow(window, preview)
        key = cv2.waitKey(1) & 0xFF
        if key in (13, 32) and len(points) == 4:
            source = np.float32(points)
            top, bottom = np.linalg.norm(source[1] - source[0]), np.linalg.norm(source[2] - source[3])
            left, right = np.linalg.norm(source[3] - source[0]), np.linalg.norm(source[2] - source[1])
            cv2.destroyWindow(window)
            return source, (round(max(top, bottom)), round(max(left, right)))
        if key == ord("r"):
            points.clear()


def main() -> None:
    load_dotenv()
    base_url = os.environ.get("CLOUD_COUNTER_URL", "").rstrip("/")
    api_key = os.environ.get("CLOUD_INFERENCE_API_KEY", "")
    if not base_url or not api_key:
        raise RuntimeError("Set CLOUD_COUNTER_URL and CLOUD_INFERENCE_API_KEY in .env")

    road_points, road_size = select_road()
    camera = open_camera(CAMERA_SOURCE, CAMERA_WIDTH, CAMERA_HEIGHT)
    if not camera.isOpened():
        raise RuntimeError(f"Could not open camera {CAMERA_SOURCE}")
    destination = np.float32([[0, 0], [road_size[0] - 1, 0], [road_size[0] - 1, road_size[1] - 1], [0, road_size[1] - 1]])
    projection = cv2.getPerspectiveTransform(road_points, destination)
    queue: Queue[np.ndarray] = Queue(maxsize=CLOUD_BATCH_SIZE * 4)
    stop_event = Event()

    def capture() -> None:
        while not stop_event.is_set():
            success, frame = camera.read()
            if not success:
                stop_event.set()
                return
            frame = cv2.warpPerspective(frame, projection, road_size)
            try:
                queue.put(frame, timeout=0.01)
            except Full:
                try:
                    queue.get_nowait()
                except Empty:
                    pass
                queue.put(frame)

    Thread(target=capture, daemon=True).start()
    session = requests.Session()
    try:
        while not stop_event.is_set():
            batch = []
            while len(batch) < CLOUD_BATCH_SIZE and not stop_event.is_set():
                try:
                    batch.append(queue.get(timeout=0.1))
                except Empty:
                    continue
            if not batch:
                continue
            files = []
            for frame in batch:
                height, width = frame.shape[:2]
                resized = cv2.resize(frame, (min(SEND_WIDTH, width), round(height * min(SEND_WIDTH, width) / width)), interpolation=cv2.INTER_AREA)
                encoded, jpeg = cv2.imencode(".jpg", resized, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                if encoded:
                    files.append(("frames", ("frame.jpg", jpeg.tobytes(), "image/jpeg")))
            if files:
                response = session.post(f"{base_url}/ingest-batch", files=files, headers={"X-API-Key": api_key}, timeout=30)
                response.raise_for_status()
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        camera.release()
        session.close()


if __name__ == "__main__":
    main()
