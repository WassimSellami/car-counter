import os
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, Header, HTTPException, Response, UploadFile
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from ultralytics.trackers.byte_tracker import BYTETracker
from ultralytics.utils import IterableSimpleNamespace, YAML
from ultralytics import YOLO


API_KEY = os.environ.get("CLOUD_INFERENCE_API_KEY")
MODEL_PATH = "yolo11m.pt"
# ViT-L/14 is substantially more capable than the previous ViT-B/32 CLIP model.
COLOR_MODEL_PATH = "openai/clip-vit-large-patch14"
TRACKER_PATH = Path(__file__).with_name("vehicle_bytetrack.yaml")
COLOR_LABELS = ("black", "white", "grey", "silver", "red", "blue", "green", "yellow", "orange", "brown")
CAR_CLASS_ID = 2
COLOR_SAMPLE_INTERVAL = 4
COLOR_FREEZE_VOTES = 6

if not API_KEY:
    raise RuntimeError("Set CLOUD_INFERENCE_API_KEY before starting the server.")

model = YOLO(MODEL_PATH)
model.to("cuda")
color_processor = CLIPProcessor.from_pretrained(COLOR_MODEL_PATH)
color_model = CLIPModel.from_pretrained(COLOR_MODEL_PATH).to("cuda").eval()
tracker = BYTETracker(IterableSimpleNamespace(**YAML.load(TRACKER_PATH)))
track_color_votes: dict[int, Counter[str]] = defaultdict(Counter)
track_color_frames: Counter[int] = Counter()
frozen_track_colors: dict[int, str] = {}
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


def _add_vehicle_colors(images: list[np.ndarray], detection_batches: list[list[dict]]) -> None:
    """Sample car tracks sparsely, then freeze their stable colour."""
    crops: list[Image.Image] = []
    sampled_detections: list[dict] = []
    car_detections: list[dict] = []
    for image, frame_detections in zip(images, detection_batches):
        height, width = image.shape[:2]
        for detection in frame_detections:
            # Colour is requested only for cars/vans. Do not spend GPU time on
            # trucks, buses, or bicycles.
            if detection["class_id"] != CAR_CLASS_ID:
                continue
            track_id = detection["track_id"]
            car_detections.append(detection)
            if track_id in frozen_track_colors:
                detection["color"] = frozen_track_colors[track_id]
                continue
            track_color_frames[track_id] += 1
            # First frame plus every fourth frame: at most three samples for a
            # vehicle across our usual ten-frame cloud batch.
            if track_color_frames[track_id] != 1 and track_color_frames[track_id] % COLOR_SAMPLE_INTERVAL:
                continue
            x1, y1, x2, y2 = detection["xyxy"]
            x1, x2 = sorted((max(0, min(x1, width)), max(0, min(x2, width))))
            y1, y2 = sorted((max(0, min(y1, height)), max(0, min(y2, height))))
            if x2 - x1 < 8 or y2 - y1 < 8:
                detection["color"] = "unknown"
                continue
            crops.append(Image.fromarray(cv2.cvtColor(image[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)))
            sampled_detections.append(detection)

    for start in range(0, len(crops), 32):
        inputs = color_processor(
            text=[f"a {color} vehicle" for color in COLOR_LABELS],
            images=crops[start:start + 32],
            return_tensors="pt",
            padding=True,
        ).to("cuda")
        with torch.inference_mode():
            probabilities = color_model(**inputs).logits_per_image.softmax(dim=1).cpu().numpy()
        for detection, scores in zip(sampled_detections[start:start + 32], probabilities):
            color_index = int(scores.argmax())
            color = COLOR_LABELS[color_index]
            track_id = detection["track_id"]
            track_color_votes[track_id][color] += 1
            chosen_color, votes = track_color_votes[track_id].most_common(1)[0]
            if votes >= COLOR_FREEZE_VOTES:
                frozen_track_colors[track_id] = chosen_color
            detection["color"] = chosen_color
            detection["color_confidence"] = round(float(scores[color_index]), 3)

    # Return the current leading vote on every frame, even when it was not
    # sampled by CLIP in this request.
    for detection in car_detections:
        track_id = detection["track_id"]
        if track_id in frozen_track_colors:
            detection["color"] = frozen_track_colors[track_id]
        elif track_color_votes[track_id]:
            detection["color"] = track_color_votes[track_id].most_common(1)[0][0]


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
    detections = _track_detections(result, image)
    clip_started_at = time.perf_counter()
    _add_vehicle_colors([image], [detections])
    response.headers["X-Clip-Ms"] = str(round((time.perf_counter() - clip_started_at) * 1000, 1))
    return detections


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
    detection_batches = [_track_detections(result, image) for result, image in zip(results, images)]
    clip_started_at = time.perf_counter()
    _add_vehicle_colors(images, detection_batches)
    response.headers["X-Clip-Ms"] = str(round((time.perf_counter() - clip_started_at) * 1000, 1))
    return {"frames": detection_batches}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
