"""Read the phone's H.264 RTMP stream, crop it, and send local batches to Vast."""
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
import requests


COUNTER_URL = os.environ.get("COUNTER_INTERNAL_URL", "http://127.0.0.1:8000").rstrip("/")
API_KEY = os.environ["CLOUD_INFERENCE_API_KEY"]
STREAM_URL = os.environ.get("PHONE_STREAM_URL", "rtmp://127.0.0.1:10100/phone")
CONFIG_PATH = Path(os.environ.get("PHONE_CROP_CONFIG", "/workspace/car-counter/phone_crop.json"))
BATCH_SIZE = 10


def projection(frame):
    config = json.loads(CONFIG_PATH.read_text())
    points = config["points"]
    h, w = frame.shape[:2]
    source = np.float32([[x * w, y * h] for x, y in zip(points[::2], points[1::2])])
    top = max(2, round(((source[1] - source[0]) @ (source[1] - source[0])) ** .5))
    bottom = max(2, round(((source[2] - source[3]) @ (source[2] - source[3])) ** .5))
    left = max(2, round(((source[3] - source[0]) @ (source[3] - source[0])) ** .5))
    right = max(2, round(((source[2] - source[1]) @ (source[2] - source[1])) ** .5))
    out_w, out_h = max(top, bottom), max(left, right)
    matrix = cv2.getPerspectiveTransform(source, np.float32([[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]]))
    return matrix, (out_w, out_h)


def main():
    matrix = size = None
    changed = 0.0
    camera = cv2.VideoCapture(STREAM_URL)
    session = requests.Session()
    batch = []
    while True:
        ok, frame = camera.read()
        if not ok:
            camera.release()
            time.sleep(1)
            camera = cv2.VideoCapture(STREAM_URL)
            continue
        if not CONFIG_PATH.exists():
            print("Waiting for crop configuration from the phone", flush=True)
            time.sleep(1)
            continue
        modified = CONFIG_PATH.stat().st_mtime
        if matrix is None or modified != changed:
            matrix, size = projection(frame)
            changed = modified
        frame = cv2.warpPerspective(frame, matrix, size)
        encoded, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if encoded:
            batch.append(jpeg.tobytes())
        if len(batch) < BATCH_SIZE:
            continue
        files = [("frames", ("frame.jpg", image, "image/jpeg")) for image in batch]
        batch.clear()
        try:
            session.post(f"{COUNTER_URL}/ingest-batch", files=files, headers={"X-API-Key": API_KEY}, timeout=10).raise_for_status()
        except requests.RequestException as error:
            print(f"Counter upload failed: {error}", flush=True)


if __name__ == "__main__":
    main()
