"""Vast-side vehicle counter: inference, counting, CSV/Supabase and MJPEG output."""

from collections import Counter, defaultdict, deque
import asyncio
from datetime import date, datetime
import csv
import json
import os
from pathlib import Path
from threading import Lock, Thread
import time

import cv2
import numpy as np
import requests
import torch
import uvicorn
from fastapi import FastAPI, Header, HTTPException, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from ultralytics import YOLO
from ultralytics.trackers.byte_tracker import BYTETracker
from ultralytics.utils import IterableSimpleNamespace, YAML


API_KEY = os.environ["CLOUD_INFERENCE_API_KEY"]
MODEL_PATH = os.environ.get("MODEL_PATH", "yolo11m.pt")
OUTPUT_ROOT = Path(os.environ.get("COUNTER_OUTPUT_DIR", "/workspace/outputs"))
PHONE_CROP_CONFIG = Path(os.environ.get("PHONE_CROP_CONFIG", "/workspace/car-counter/phone_crop.json"))
UPLOAD_TO_SUPABASE = os.environ.get("UPLOAD_TO_SUPABASE", "false").lower() == "true"
INTO_PASSAU_IS_DOWN = os.environ.get("INTO_PASSAU_IS_DOWN", os.environ.get("INTO_PASSAU_IS_RIGHT", "true")).lower() == "true"
COLOR_LABELS = ("black", "white", "grey", "silver", "red", "blue", "green", "yellow", "orange", "brown")
COLOR_CODES = {name: index + 1 for index, name in enumerate(COLOR_LABELS)}
CLASS_TO_TYPE = {1: 3, 2: 0, 5: 2, 7: 1}  # bicycle, car, bus, truck
TYPE_NAMES = ("car", "truck", "bus", "bicycle")

model = YOLO(MODEL_PATH).to("cuda")
color_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
color_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to("cuda").eval()
tracker = BYTETracker(IterableSimpleNamespace(**YAML.load(Path(__file__).with_name("vehicle_bytetrack.yaml"))))
app = FastAPI()
lock = Lock()
track_history: dict[int, list[int]] = defaultdict(list)
counted_ids: set[int] = set()
color_votes: dict[int, Counter[str]] = defaultdict(Counter)
color_frames: Counter[int] = Counter()
frozen_colors: dict[int, str] = {}
counts = {(vehicle_type, direction): 0 for vehicle_type in range(4) for direction in range(2)}
color_counts: Counter[str] = Counter()
output_frames: deque[tuple[int, bytes]] = deque(maxlen=180)
output_sequence = 0
metrics = {"cloud_ms": 0.0, "yolo_ms": 0.0, "clip_ms": 0.0, "process_fps": 0.0, "input_fps": 0.0}
last_input_at = 0.0
csv_day: date | None = None
csv_file = None
csv_writer = None
next_record_id = 1


def _open_csv() -> None:
    global csv_day, csv_file, csv_writer, next_record_id
    today = datetime.now().date()
    if csv_day == today:
        return
    if csv_file is not None:
        csv_file.close()
    directory = OUTPUT_ROOT / today.isoformat()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"count_{today:%Y%m%d}.csv"
    has_rows = path.exists() and path.stat().st_size > 0
    csv_file = path.open("a", newline="", encoding="utf-8", buffering=1)
    csv_writer = csv.DictWriter(csv_file, fieldnames=("id", "timestamp", "direction", "vehicle_type", "color", "time_of_day", "confidence"))
    if not has_rows:
        csv_writer.writeheader()
    next_record_id = 1 + max(
        (int(row.get("id", 0)) for csv_path in OUTPUT_ROOT.glob("**/count_*.csv") for row in csv.DictReader(csv_path.open("r", newline="", encoding="utf-8"))),
        default=0,
    )
    csv_day = today


def _upload_row(row: dict) -> None:
    if not UPLOAD_TO_SUPABASE:
        print("Supabase upload skipped: UPLOAD_TO_SUPABASE is false", flush=True)
        return
    url = os.environ.get("SUPABASE_URL", "").rstrip("/")
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
    if not url or not key:
        print("Supabase upload skipped: SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY is missing", flush=True)
        return
    try:
        response = requests.post(
            f"{url}/rest/v1/traffic_counts", params={"on_conflict": "record_id"},
            headers={"apikey": key, "Authorization": f"Bearer {key}", "Content-Type": "application/json", "Prefer": "resolution=merge-duplicates,return=minimal"},
            json=[row], timeout=10,
        )
        response.raise_for_status()
        print(f"Supabase upload complete: record {row['record_id']}", flush=True)
    except requests.RequestException as error:
        print(f"Supabase upload failed: {error}", flush=True)


def _record(track_id: int, confidence: float, vehicle_type: int, direction: int, color: str) -> None:
    global next_record_id
    _open_csv()
    counted_ids.add(track_id)
    counts[(vehicle_type, direction)] += 1
    if vehicle_type == 0 and color in COLOR_CODES:
        color_counts[color] += 1
    row = {"id": next_record_id, "timestamp": datetime.now().isoformat(timespec="milliseconds"), "direction": direction, "vehicle_type": vehicle_type, "color": COLOR_CODES.get(color, 0), "time_of_day": 0, "confidence": f"{confidence:.4f}"}
    csv_writer.writerow(row)
    csv_file.flush()
    # Local CSV IDs restart on a new cloud machine. Supabase IDs must not.
    supabase_record_id = time.time_ns()
    Thread(
        target=_upload_row,
        args=({"source_file": f"{csv_day.isoformat()}/count_{csv_day:%Y%m%d}.csv", "record_id": supabase_record_id, "timestamp": row["timestamp"], "direction": direction, "vehicle_type": vehicle_type, "color": row["color"], "time_of_day": 0, "confidence": confidence},),
        daemon=True,
    ).start()
    next_record_id += 1


def _track(result, image: np.ndarray) -> list[dict]:
    boxes = result.boxes
    tracks = tracker.update(boxes.cpu().numpy(), image) if boxes is not None else []
    return [{"xyxy": [round(float(value)) for value in track[:4]], "track_id": int(track[4]), "confidence": float(track[5]), "class_id": int(track[6])} for track in tracks]


def _classify_colours(images: list[np.ndarray], batches: list[list[dict]]) -> None:
    crops, sampled, cars = [], [], []
    for image, detections in zip(images, batches):
        height, width = image.shape[:2]
        for detection in detections:
            if detection["class_id"] != 2:
                continue
            cars.append(detection)
            track_id = detection["track_id"]
            if track_id in frozen_colors:
                detection["color"] = frozen_colors[track_id]
                continue
            color_frames[track_id] += 1
            if color_frames[track_id] != 1 and color_frames[track_id] % 4:
                continue
            x1, y1, x2, y2 = detection["xyxy"]
            x1, x2 = sorted((max(0, min(x1, width)), max(0, min(x2, width))))
            y1, y2 = sorted((max(0, min(y1, height)), max(0, min(y2, height))))
            if x2 - x1 >= 8 and y2 - y1 >= 8:
                crops.append(Image.fromarray(cv2.cvtColor(image[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)))
                sampled.append(detection)
    for start in range(0, len(crops), 32):
        inputs = color_processor(text=[f"a {color} vehicle" for color in COLOR_LABELS], images=crops[start:start + 32], return_tensors="pt", padding=True).to("cuda")
        with torch.inference_mode():
            probabilities = color_model(**inputs).logits_per_image.softmax(dim=1).cpu().numpy()
        for detection, scores in zip(sampled[start:start + 32], probabilities):
            color = COLOR_LABELS[int(scores.argmax())]
            track_id = detection["track_id"]
            color_votes[track_id][color] += 1
            leading, votes = color_votes[track_id].most_common(1)[0]
            if votes >= 6:
                frozen_colors[track_id] = leading
            detection["color"] = leading
    for detection in cars:
        track_id = detection["track_id"]
        if track_id in frozen_colors:
            detection["color"] = frozen_colors[track_id]
        elif color_votes[track_id]:
            detection["color"] = color_votes[track_id].most_common(1)[0][0]


def _count_and_annotate(image: np.ndarray, detections: list[dict]) -> None:
    height = image.shape[0]
    for detection in detections:
        vehicle_type = CLASS_TO_TYPE[detection["class_id"]]
        confidence = detection["confidence"]
        track_id = detection["track_id"]
        accepted = confidence >= (0.1 if vehicle_type == 3 else 0.4)
        x1, y1, x2, y2 = map(int, detection["xyxy"])
        color = detection.get("color", "unknown")
        if accepted:
            history = track_history[track_id]
            history.append((y1 + y2) // 2)
            if len(history) > 30:
                history.pop(0)
            distance = history[-1] - history[0]
            threshold = max(25, int(height * (0.03 if vehicle_type == 3 else 0.08)))
            direction = 1 if distance >= threshold else 0 if distance <= -threshold else None
            if direction is not None and track_id not in counted_ids:
                _record(track_id, confidence, vehicle_type, direction, color)
        box_color = (0, 255, 0) if accepted else (0, 165, 255)
        cv2.rectangle(image, (x1, y1), (x2, y2), box_color, 2)
        if vehicle_type == 0 and color != "unknown":
            cv2.putText(image, color, (x1, max(18, y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, box_color, 2, cv2.LINE_AA)


def _publish(image: np.ndarray) -> None:
    global output_sequence
    encoded, jpeg = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if encoded:
        output_sequence += 1
        output_frames.append((output_sequence, jpeg.tobytes()))


def _rectify_phone_images(images: list[np.ndarray], rotation: int, points_header: str) -> list[np.ndarray]:
    """Rotate phone frames upright, then rectify the app's four selected corners."""
    if rotation not in (0, 90, 180, 270) or not points_header:
        return images
    try:
        points = np.array([float(value) for value in points_header.split(",")], dtype=np.float32)
        if points.size != 8 or np.any(points < 0) or np.any(points > 1):
            raise ValueError
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid crop points")
    rotate_code = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}
    rotated = [cv2.rotate(image, rotate_code[rotation]) if rotation else image for image in images]
    height, width = rotated[0].shape[:2]
    source = points.reshape(4, 2) * np.array([width, height], dtype=np.float32)
    output_width = max(2, round((np.linalg.norm(source[1] - source[0]) + np.linalg.norm(source[2] - source[3])) / 2))
    output_height = max(2, round((np.linalg.norm(source[3] - source[0]) + np.linalg.norm(source[2] - source[1])) / 2))
    destination = np.float32([[0, 0], [output_width - 1, 0], [output_width - 1, output_height - 1], [0, output_height - 1]])
    projection = cv2.getPerspectiveTransform(source, destination)
    return [cv2.warpPerspective(image, projection, (output_width, output_height)) for image in rotated]


@app.post("/phone-crop")
def phone_crop(payload: dict, x_api_key: str = Header(default="")):
    """Store crop coordinates once for the RTMP bridge."""
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    try:
        points = [float(value) for value in payload["points"]]
        if len(points) != 8 or any(value < 0 or value > 1 for value in points):
            raise ValueError
    except (KeyError, TypeError, ValueError):
        raise HTTPException(status_code=400, detail="points must contain eight values from 0 to 1")
    PHONE_CROP_CONFIG.parent.mkdir(parents=True, exist_ok=True)
    temporary = PHONE_CROP_CONFIG.with_suffix(".tmp")
    temporary.write_text(json.dumps({"points": points}), encoding="utf-8")
    temporary.replace(PHONE_CROP_CONFIG)
    return {"ok": True}


@app.post("/rtmp-auth")
def rtmp_auth(payload: dict):
    """MediaMTX asks this local endpoint before accepting a publisher."""
    if payload.get("action") == "read" and payload.get("path") == "phone" and payload.get("ip") in ("127.0.0.1", "::1"):
        return {"ok": True}
    if payload.get("action") != "publish" or payload.get("path") != "phone" or payload.get("user") != "phone" or payload.get("password") != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid RTMP publisher")
    return {"ok": True}


@app.post("/ingest-batch")
async def ingest_batch(frames: list[UploadFile], x_api_key: str = Header(default=""), x_frame_rotation: int = Header(default=0), x_crop_points: str = Header(default="")):
    global last_input_at
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    if not 1 <= len(frames) <= 12:
        raise HTTPException(status_code=400, detail="Send 1 to 12 frames")
    images = [cv2.imdecode(np.frombuffer(await frame.read(), np.uint8), cv2.IMREAD_COLOR) for frame in frames]
    if any(image is None for image in images):
        raise HTTPException(status_code=400, detail="Invalid image")
    images = _rectify_phone_images(images, x_frame_rotation, x_crop_points)
    started = time.perf_counter()
    with lock:
        if last_input_at:
            instantaneous_input_fps = len(images) / max(0.001, started - last_input_at)
            metrics["input_fps"] = instantaneous_input_fps if metrics["input_fps"] == 0 else 0.85 * metrics["input_fps"] + 0.15 * instantaneous_input_fps
        last_input_at = started
        yolo_started = time.perf_counter()
        results = model(images, classes=[1, 2, 5, 7], conf=0.05, imgsz=640, max_det=15, batch=len(images), verbose=False)
        metrics["yolo_ms"] = (time.perf_counter() - yolo_started) * 1000
        batches = [_track(result, image) for result, image in zip(results, images)]
        clip_started = time.perf_counter()
        _classify_colours(images, batches)
        metrics["clip_ms"] = (time.perf_counter() - clip_started) * 1000
        for image, detections in zip(images, batches):
            _count_and_annotate(image, detections)
            _publish(image)
        metrics["cloud_ms"] = (time.perf_counter() - started) * 1000
        instantaneous_fps = len(images) / max(0.001, time.perf_counter() - started)
        metrics["process_fps"] = instantaneous_fps if metrics["process_fps"] == 0 else 0.85 * metrics["process_fps"] + 0.15 * instantaneous_fps
    return JSONResponse({"ok": True})


@app.get("/status")
def status(x_api_key: str = Header(default="")):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    into, out = (1, 0) if INTO_PASSAU_IS_DOWN else (0, 1)
    with lock:
        return {"metrics": {**{key: round(value, 1) for key, value in metrics.items()}, "other_cloud_ms": 0}, "counts": {TYPE_NAMES[vehicle_type]: {"into_passau": counts[(vehicle_type, into)], "out_of_passau": counts[(vehicle_type, out)]} for vehicle_type in range(4)}, "colors": dict(color_counts)}


@app.get("/stream.mjpeg")
async def stream(key: str = ""):
    if key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid stream key")
    async def frames():
        # New viewers start live; old frames are not replayed after a refresh.
        with lock:
            last_sequence = output_sequence
        while True:
            with lock:
                next_frame = next(((sequence, jpeg) for sequence, jpeg in output_frames if sequence > last_sequence), None)
            if next_frame is None:
                await asyncio.sleep(0.01)
                continue
            last_sequence, jpeg = next_frame
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
            await asyncio.sleep(1 / 30)
    return StreamingResponse(frames(), media_type="multipart/x-mixed-replace; boundary=frame")


if __name__ == "__main__":
    _open_csv()
    uvicorn.run(app, host="0.0.0.0", port=8000)
