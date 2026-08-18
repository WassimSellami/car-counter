from __future__ import annotations

import os
import time
from pathlib import Path

import cv2
import requests

from camera_utils import open_camera
from constants import CAMERA_HEIGHT, CAMERA_SOURCE, CAMERA_WIDTH


INFERENCE_WIDTH = 640
JPEG_QUALITY = 60


def load_dotenv() -> None:
    """Load local cloud settings without requiring command-line arguments."""
    env_file = Path(".env")
    if not env_file.is_file():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        if line and not line.lstrip().startswith("#") and "=" in line:
            name, value = line.split("=", maxsplit=1)
            os.environ.setdefault(name.strip(), value.strip().strip('"').strip("'"))


def select_road_crop(camera: cv2.VideoCapture) -> tuple[int, int, int, int]:
    """Show a live preview until S captures one frame for crop selection."""
    window_name = "Camera preview: focus here, press S to select road"
    cv2.namedWindow(window_name)
    try:
        while True:
            success, frame = camera.read()
            if not success:
                raise RuntimeError("Could not read a frame for crop selection.")
            cv2.putText(
                frame,
                "Press S to capture this frame and select the road crop",
                (12, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
            cv2.imshow(window_name, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("s"), ord("S")):
                return select_crop(frame)
            if key == ord("q"):
                raise SystemExit(0)
    finally:
        cv2.destroyWindow(window_name)


def select_crop(frame) -> tuple[int, int, int, int]:
    """Return a mouse-dragged crop; Enter or Space confirms it."""
    window_name = "Select road crop: drag, then press Enter or Space"
    start: tuple[int, int] | None = None
    selection: tuple[int, int, int, int] | None = None

    def on_mouse(event: int, x: int, y: int, _flags: int, _params: object) -> None:
        nonlocal start, selection
        if event == cv2.EVENT_LBUTTONDOWN:
            start = (x, y)
            selection = None
        elif event == cv2.EVENT_MOUSEMOVE and start is not None:
            selection = min(start[0], x), min(start[1], y), abs(x - start[0]), abs(y - start[1])
        elif event == cv2.EVENT_LBUTTONUP and start is not None:
            selection = min(start[0], x), min(start[1], y), abs(x - start[0]), abs(y - start[1])
            start = None

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, on_mouse)
    while True:
        preview = frame.copy()
        if selection is not None:
            x, y, width, height = selection
            cv2.rectangle(preview, (x, y), (x + width, y + height), (0, 255, 0), 2)
        cv2.imshow(window_name, preview)
        key = cv2.waitKey(1) & 0xFF
        if key in (13, 32) and selection is not None and selection[2] and selection[3]:
            cv2.destroyWindow(window_name)
            return selection
        if key == ord("c"):
            cv2.destroyWindow(window_name)
            height, width = frame.shape[:2]
            return 0, 0, width, height


def infer_frame(
    session: requests.Session,
    frame,
    url: str,
    headers: dict[str, str],
    maximum_width: int,
    jpeg_quality: int,
) -> tuple[list[dict], int, int, float, float]:
    """Send one frame and return its boxes plus the cloud round-trip time."""
    frame_height, frame_width = frame.shape[:2]
    inference_width = min(maximum_width, frame_width)
    inference_height = round(frame_height * inference_width / frame_width)
    inference_frame = cv2.resize(
        frame, (inference_width, inference_height), interpolation=cv2.INTER_AREA
    )
    encoded, jpeg = cv2.imencode(
        ".jpg", inference_frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
    )
    if not encoded:
        raise RuntimeError("Could not encode the camera frame as JPEG.")

    started_at = time.perf_counter()
    response = session.post(
        url,
        files={"frame": ("frame.jpg", jpeg.tobytes(), "image/jpeg")},
        headers=headers,
        timeout=15,
    )
    response.raise_for_status()
    return (
        response.json(),
        inference_width,
        inference_height,
        time.perf_counter() - started_at,
        float(response.headers.get("X-Yolo-Ms", 0)),
    )


def infer_frame_batch(
    session: requests.Session,
    frames: list,
    url: str,
    headers: dict[str, str],
    maximum_width: int,
    jpeg_quality: int,
) -> tuple[list[list[dict]], list[tuple[int, int]], float, float]:
    """Send up to three ordered frames for one batched cloud inference."""
    files = []
    dimensions = []
    for frame in frames:
        frame_height, frame_width = frame.shape[:2]
        inference_width = min(maximum_width, frame_width)
        inference_height = round(frame_height * inference_width / frame_width)
        inference_frame = cv2.resize(
            frame, (inference_width, inference_height), interpolation=cv2.INTER_AREA
        )
        encoded, jpeg = cv2.imencode(
            ".jpg", inference_frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
        )
        if not encoded:
            raise RuntimeError("Could not encode a camera frame as JPEG.")
        files.append(("frames", ("frame.jpg", jpeg.tobytes(), "image/jpeg")))
        dimensions.append((inference_width, inference_height))

    started_at = time.perf_counter()
    response = session.post(url, files=files, headers=headers, timeout=15)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict) or "frames" not in payload:
        raise RuntimeError("Cloud server is outdated. Upload and restart vast_inference_server.py.")
    return (
        payload["frames"],
        dimensions,
        time.perf_counter() - started_at,
        float(response.headers.get("X-Yolo-Ms", 0)),
    )


def main() -> None:
    load_dotenv()
    url = os.environ.get("CLOUD_INFERENCE_URL")
    if not url:
        raise RuntimeError("Add CLOUD_INFERENCE_URL=https://.../infer to .env")
    api_key = os.environ.get("CLOUD_INFERENCE_API_KEY")
    headers = {"X-API-Key": api_key} if api_key else {}
    camera = open_camera(CAMERA_SOURCE, CAMERA_WIDTH, CAMERA_HEIGHT)
    session = requests.Session()

    if not camera.isOpened():
        raise RuntimeError(f"Could not open camera source {CAMERA_SOURCE}.")

    try:
        crop = select_road_crop(camera)

        while True:
            success, frame = camera.read()
            if not success:
                raise RuntimeError("Could not read a frame from the camera.")

            if crop is not None:
                x, y, width, height = crop
                frame = frame[y : y + height, x : x + width]

            frame_height, frame_width = frame.shape[:2]
            try:
                detections, detection_width, detection_height, elapsed, yolo_ms = infer_frame(
                    session,
                    frame,
                    url,
                    headers,
                    INFERENCE_WIDTH,
                    JPEG_QUALITY,
                )
            except requests.RequestException:
                cv2.putText(
                    frame,
                    "Cloud unavailable - reconnecting...",
                    (12, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                )
                cv2.imshow("Cloud detection (press q to close)", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    return
                time.sleep(0.25)
                continue

            for detection in detections:
                x1, y1, x2, y2 = detection["xyxy"]
                cv2.rectangle(
                    frame,
                    (round(x1 * frame_width / detection_width), round(y1 * frame_height / detection_height)),
                    (round(x2 * frame_width / detection_width), round(y2 * frame_height / detection_height)),
                    (0, 255, 0),
                    2,
                )

            cv2.putText(
                frame,
                f"Cloud {1 / elapsed:.1f} FPS | Delay {elapsed * 1000:.0f} ms | "
                f"YOLO {yolo_ms:.0f} ms",
                (12, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )

            cv2.imshow("Cloud detection (press q to close)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                return
    finally:
        camera.release()
        session.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
