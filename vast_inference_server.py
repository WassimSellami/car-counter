import os
import time
from pathlib import Path

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, Header, HTTPException, Response, UploadFile
from ultralytics.trackers.byte_tracker import BYTETracker
from ultralytics.utils import IterableSimpleNamespace, YAML
from ultralytics import YOLO


API_KEY = os.environ.get("CLOUD_INFERENCE_API_KEY")
MODEL_PATH = "yolo11m.pt"
TRACKER_PATH = Path(__file__).with_name("vehicle_bytetrack.yaml")

if not API_KEY:
    raise RuntimeError("Set CLOUD_INFERENCE_API_KEY before starting the server.")

model = YOLO(MODEL_PATH)
model.to("cuda")
tracker = BYTETracker(IterableSimpleNamespace(**YAML.load(TRACKER_PATH)))
app = FastAPI()


def _track_detections(result, image: np.ndarray) -> list[dict]:
    boxes = result.boxes
    tracks = tracker.update(boxes.cpu().numpy(), image) if boxes is not None else []
    return [
        {
            "xyxy": [round(float(value)) for value in track[:4]],
            "track_id": int(track[4]),
            "confidence": round(float(track[5]), 4),
            "class_id": int(track[6]),
        }
        for track in tracks
    ]


@app.post("/infer")
async def infer(
    frame: UploadFile,
    response: Response,
    x_api_key: str = Header(default=""),
):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

    image = cv2.imdecode(np.frombuffer(await frame.read(), np.uint8), cv2.IMREAD_COLOR)
    started_at = time.perf_counter()
    result = model(
        image,
        classes=[1, 2, 5, 7],
        conf=0.05,
        imgsz=640,
        max_det=15,
        verbose=False,
    )[0]
    response.headers["X-Yolo-Ms"] = str(round((time.perf_counter() - started_at) * 1000, 1))
    return _track_detections(result, image)


@app.post("/infer-batch")
async def infer_batch(
    frames: list[UploadFile],
    response: Response,
    x_api_key: str = Header(default=""),
):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    if not 1 <= len(frames) <= 12:
        raise HTTPException(status_code=400, detail="Send between 1 and 12 frames")

    images = [cv2.imdecode(np.frombuffer(await frame.read(), np.uint8), cv2.IMREAD_COLOR) for frame in frames]
    started_at = time.perf_counter()
    results = model(
        images,
        classes=[1, 2, 5, 7],
        conf=0.05,
        imgsz=640,
        max_det=15,
        batch=len(images),
        verbose=False,
    )
    response.headers["X-Yolo-Ms"] = str(round((time.perf_counter() - started_at) * 1000, 1))
    return {"frames": [_track_detections(result, image) for result, image in zip(results, images)]}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
