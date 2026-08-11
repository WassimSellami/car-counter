"""Count detected cars once based on their horizontal travel direction."""

from collections import defaultdict
import csv
from datetime import datetime
from enum import IntEnum
import logging
from pathlib import Path
import time

import cv2
import numpy as np
import torch
from ultralytics import YOLO

logging.getLogger("ultralytics").disabled = True

CAMERA_INDEX = 0  # Set this to the DroidCam index printed by find_camera.py.
# Requested DroidCam capture resolution. DroidCam must also be configured to
# offer this resolution; otherwise its driver will use the nearest supported one.
CAMERA_WIDTH = 1920
CAMERA_HEIGHT = 1080
# Use the fast model to preserve a responsive camera preview.
MODEL_PATH = "yolo11s.pt"
BICYCLE_CLASS_ID = 1  # COCO class ID for bicycles.
CAR_CLASS_ID = 2  # COCO class ID for cars (including vans).
BUS_CLASS_ID = 5  # COCO class ID for buses.
TRUCK_CLASS_ID = 7  # COCO class ID for trucks.
VEHICLE_CLASS_IDS = [
    BICYCLE_CLASS_ID,
    CAR_CLASS_ID,
    BUS_CLASS_ID,
    TRUCK_CLASS_ID,
]
# Boxes at or above this score are accepted for counting.
CONFIDENCE = 0.40
# Two-wheeled vehicles are smaller and often receive lower YOLO confidence scores.
BICYCLE_CONFIDENCE = 0.10
# Ask YOLO to return low-score car candidates too, so the display can show
# whether each one was accepted or rejected by the threshold above.
# Keep low-score candidates visible so the per-type thresholds can accept them.
MODEL_CONFIDENCE = 0.05
IMAGE_SIZE = 960
# Maximum total detections retained per frame across every vehicle type.
MAX_DETECTIONS = 15
# Forget tracker IDs that have not appeared for this long. This prevents
# tracker bookkeeping from growing indefinitely during multi-hour runs.
TRACK_MEMORY_SECONDS = 120.0
# Enable this at night to detect moving bright headlights rather than cars.
NIGHT_MODE = False
LIGHT_BRIGHTNESS_THRESHOLD = 180
LIGHT_MIN_AREA = 8
# Merge nearby headlights/reflections from a single vehicle into one box.
LIGHT_MERGE_WIDTH = 55
LIGHT_MERGE_HEIGHT = 11
LIGHT_GROUP_X_DISTANCE = 60
LIGHT_GROUP_Y_DISTANCE = 25
LIGHT_TRACK_DISTANCE = 90
LIGHT_TRACK_MAX_MISSING = 45
# Draw the detection crop interactively at program startup. Select the road,
# then press Enter or Space to confirm.
SELECT_CROP_ON_START = True
# A vehicle must move at least this fraction of the frame width before it is
# counted. This prevents small tracker jitter from being counted as movement.
MIN_DIRECTION_DISTANCE_RATIO = 0.08
# Bicycles are smaller and usually remain visible for less of the road crop.
BICYCLE_DIRECTION_DISTANCE_RATIO = 0.03


class Direction(IntEnum):
    """Numeric direction values written to the CSV."""

    LEFT = 0
    RIGHT = 1


class VehicleType(IntEnum):
    """Vehicle categories written to the CSV."""

    CAR = 0
    TRUCK = 1
    BUS = 2
    BICYCLE = 3


class TimeOfDay(IntEnum):
    """Lighting modes written to the CSV."""

    DAY = 0
    NIGHT = 1


def _confidence_threshold(vehicle_type: VehicleType) -> float:
    """Return the score required to accept a detection for counting."""
    if vehicle_type is VehicleType.BICYCLE:
        return BICYCLE_CONFIDENCE
    return CONFIDENCE


def _direction_distance_ratio(vehicle_type: VehicleType) -> float:
    """Return the horizontal travel fraction required before counting."""
    if vehicle_type is VehicleType.BICYCLE:
        return BICYCLE_DIRECTION_DISTANCE_RATIO
    return MIN_DIRECTION_DISTANCE_RATIO


def _status_panel(
    frame_height: int,
    vehicle_counts: dict[tuple[VehicleType, Direction], int],
    fps: float,
) -> np.ndarray:
    """Create a sidebar that is separate from the camera image."""
    panel = np.full((frame_height, 430, 3), (28, 28, 28), dtype=np.uint8)
    text_color = (220, 220, 220)
    left_color = (0, 255, 255)
    right_color = (0, 255, 0)
    cv2.putText(panel, "OBJECT COUNTS", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.85, text_color, 2)
    cv2.putText(panel, "TYPE", (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
    cv2.putText(panel, "LEFT", (205, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, left_color, 2)
    cv2.putText(panel, "RIGHT", (315, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, right_color, 2)
    cv2.line(panel, (15, 100), (415, 100), (90, 90, 90), 1)

    rows = [
        ("Car / van", VehicleType.CAR),
        ("Truck", VehicleType.TRUCK),
        ("Bus", VehicleType.BUS),
        ("Bicycle", VehicleType.BICYCLE),
    ]
    for row_index, (label, vehicle_type) in enumerate(rows):
        y = 145 + row_index * 58
        cv2.putText(panel, label, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
        cv2.putText(
            panel,
            str(vehicle_counts[(vehicle_type, Direction.LEFT)]),
            (220, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            left_color,
            2,
        )
        cv2.putText(
            panel,
            str(vehicle_counts[(vehicle_type, Direction.RIGHT)]),
            (335, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            right_color,
            2,
        )
        cv2.line(panel, (15, y + 18), (415, y + 18), (55, 55, 55), 1)

    cv2.putText(panel, f"FPS  {fps:.1f}", (20, 430), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
    return panel


def _select_crop(frame: np.ndarray) -> tuple[int, int, int, int]:
    """Select a crop without OpenCV's terminal output."""
    window_name = "Select Road Crop"
    start: tuple[int, int] | None = None
    selection: tuple[int, int, int, int] | None = None

    def on_mouse(event: int, x: int, y: int, _flags: int, _params: object) -> None:
        nonlocal start, selection
        if event == cv2.EVENT_LBUTTONDOWN:
            start = (x, y)
            selection = None
        elif event == cv2.EVENT_MOUSEMOVE and start is not None:
            left, top = min(start[0], x), min(start[1], y)
            selection = (left, top, abs(x - start[0]), abs(y - start[1]))
        elif event == cv2.EVENT_LBUTTONUP and start is not None:
            left, top = min(start[0], x), min(start[1], y)
            selection = (left, top, abs(x - start[0]), abs(y - start[1]))
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


class MovingLightTracker:
    """Detect and assign IDs to bright regions that are moving."""

    def __init__(self) -> None:
        self.background = cv2.createBackgroundSubtractorMOG2(
            history=120, varThreshold=20, detectShadows=False
        )
        self.tracks: dict[int, tuple[tuple[int, int], int]] = {}
        self.next_id = 1
        self.close_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (LIGHT_MERGE_WIDTH, LIGHT_MERGE_HEIGHT)
        )
        self.open_kernel = np.ones((3, 3), np.uint8)

    def update(self, frame: np.ndarray) -> list[tuple[tuple[int, int, int, int], int]]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        motion = self.background.apply(gray)
        _, bright = cv2.threshold(
            gray, LIGHT_BRIGHTNESS_THRESHOLD, 255, cv2.THRESH_BINARY
        )
        mask = cv2.bitwise_and(motion, bright)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.close_kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.open_kernel)

        candidates: list[tuple[int, int, int, int]] = []
        for contour in cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
            if cv2.contourArea(contour) >= LIGHT_MIN_AREA:
                candidates.append(cv2.boundingRect(contour))
        candidates = self._merge_nearby_lights(candidates)

        matched_ids: set[int] = set()
        results: list[tuple[tuple[int, int, int, int], int]] = []
        for x, y, width, height in candidates:
            center = (x + width // 2, y + height // 2)
            best_id = None
            best_distance = LIGHT_TRACK_DISTANCE
            for track_id, (previous_center, _) in self.tracks.items():
                if track_id in matched_ids:
                    continue
                distance = np.hypot(center[0] - previous_center[0], center[1] - previous_center[1])
                if distance < best_distance:
                    best_id, best_distance = track_id, distance

            if best_id is None:
                best_id = self.next_id
                self.next_id += 1
            self.tracks[best_id] = (center, 0)
            matched_ids.add(best_id)
            results.append(((x, y, x + width, y + height), best_id))

        for track_id, (center, missing) in list(self.tracks.items()):
            if track_id not in matched_ids:
                if missing >= LIGHT_TRACK_MAX_MISSING:
                    del self.tracks[track_id]
                else:
                    self.tracks[track_id] = (center, missing + 1)
        return results

    @staticmethod
    def _merge_nearby_lights(
        boxes: list[tuple[int, int, int, int]]
    ) -> list[tuple[int, int, int, int]]:
        """Group headlights and reflections that are likely from one vehicle."""
        merged: list[list[int]] = []
        for x, y, width, height in boxes:
            candidate = [x, y, x + width, y + height]
            for group in merged:
                horizontal_gap = max(0, max(candidate[0], group[0]) - min(candidate[2], group[2]))
                candidate_center_y = (candidate[1] + candidate[3]) // 2
                group_center_y = (group[1] + group[3]) // 2
                if (
                    horizontal_gap <= LIGHT_GROUP_X_DISTANCE
                    and abs(candidate_center_y - group_center_y) <= LIGHT_GROUP_Y_DISTANCE
                ):
                    group[0] = min(group[0], candidate[0])
                    group[1] = min(group[1], candidate[1])
                    group[2] = max(group[2], candidate[2])
                    group[3] = max(group[3], candidate[3])
                    break
            else:
                merged.append(candidate)
        return [(x1, y1, x2 - x1, y2 - y1) for x1, y1, x2, y2 in merged]


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = None if NIGHT_MODE else YOLO(MODEL_PATH)
    if model is not None:
        model.to(device)
    light_tracker = MovingLightTracker() if NIGHT_MODE else None
    program_started_at = datetime.now()
    output_directory = Path("outputs") / program_started_at.strftime("%Y-%m-%d")
    output_directory.mkdir(parents=True, exist_ok=True)
    csv_path = output_directory / (
        f"car_counts_{program_started_at.strftime('%Y%m%d_%H%M%S')}.csv"
    )
    # Line buffering ensures each count is visible in the file immediately.
    csv_file = csv_path.open("w", newline="", encoding="utf-8", buffering=1)
    csv_writer = csv.DictWriter(
        csv_file,
        fieldnames=[
            "id",
            "timestamp",
            "direction",
            "vehicle_type",
            "time_of_day",
            "confidence",
        ],
    )
    csv_writer.writeheader()
    csv_file.flush()
    next_record_id = 1
    camera = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    if not camera.isOpened():
        csv_file.close()
        raise RuntimeError(f"Could not open camera index {CAMERA_INDEX}")

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if SELECT_CROP_ON_START:
        while True:
            success, first_frame = camera.read()
            if not success:
                camera.release()
                raise RuntimeError("Could not read a frame from the camera")

            cv2.imshow("Camera Preview", first_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("s"):
                break
            if key == ord("q"):
                camera.release()
                cv2.destroyAllWindows()
                csv_file.close()
                return

        cv2.destroyWindow("Camera Preview")
        crop_x, crop_y, crop_width, crop_height = _select_crop(first_frame)
    else:
        success, first_frame = camera.read()
        if not success:
            camera.release()
            raise RuntimeError("Could not read a frame from the camera")
        frame_height, frame_width = first_frame.shape[:2]
        crop_x, crop_y, crop_width, crop_height = 0, 0, frame_width, frame_height

    track_history: dict[int, list[int]] = defaultdict(list)
    counted_ids: set[int] = set()
    vehicle_counts = {
        (vehicle_type, direction): 0
        for vehicle_type in VehicleType
        for direction in Direction
    }
    minimum_distances = {
        vehicle_type: max(
            25, int(crop_width * _direction_distance_ratio(vehicle_type))
        )
        for vehicle_type in VehicleType
    }
    time_of_day_value = int(TimeOfDay.NIGHT if NIGHT_MODE else TimeOfDay.DAY)
    base_row = {
        "time_of_day": time_of_day_value,
    }
    class_to_vehicle_type = {
        BICYCLE_CLASS_ID: VehicleType.BICYCLE,
        CAR_CLASS_ID: VehicleType.CAR,
        BUS_CLASS_ID: VehicleType.BUS,
        TRUCK_CLASS_ID: VehicleType.TRUCK,
    }
    track_last_seen: dict[int, float] = {}
    fps = 0.0
    previous_frame_time = time.perf_counter()
    last_track_cleanup_time = previous_frame_time

    if NIGHT_MODE:
        assert light_tracker is not None
    else:
        assert model is not None

    try:
        while True:
            success, frame = camera.read()
            if not success:
                break
            current_time = time.perf_counter()

            frame = frame[
                crop_y : crop_y + crop_height,
                crop_x : crop_x + crop_width,
            ]
            if frame.size == 0:
                raise RuntimeError("Crop is empty; restart and select a valid crop")

            if NIGHT_MODE:
                detections = [
                    (box, track_id, 1.0, VehicleType.CAR)
                    for box, track_id in light_tracker.update(frame)
                ]
            else:
                result = model.track(
                    frame,
                    persist=True,
                    tracker="vehicle_bytetrack.yaml",
                    classes=VEHICLE_CLASS_IDS,
                    conf=MODEL_CONFIDENCE,
                    imgsz=IMAGE_SIZE,
                    max_det=MAX_DETECTIONS,
                    device=device,
                    verbose=False,
                )[0]
                detections = []
                if result.boxes is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    confidences = result.boxes.conf.cpu().tolist()
                    class_ids = result.boxes.cls.int().cpu().tolist()
                    track_ids = (
                        result.boxes.id.int().cpu().tolist()
                        if result.boxes.id is not None
                        else [None] * len(boxes)
                    )
                    detections = [
                        (
                            box,
                            track_id,
                            confidence,
                            class_to_vehicle_type[class_id],
                        )
                        for box, track_id, confidence, class_id in zip(
                            boxes, track_ids, confidences, class_ids
                        )
                    ]

            if detections:
                for box, track_id, confidence, vehicle_type in detections:
                    x1, y1, x2, y2 = map(int, box)
                    center_x = (x1 + x2) // 2
                    accepted = confidence >= _confidence_threshold(vehicle_type)
                    if track_id is not None and accepted:
                        track_last_seen[track_id] = current_time
                        history = track_history[track_id]
                        history.append(center_x)
                        if len(history) > 30:
                            history.pop(0)

                        initial_x = history[0]
                        horizontal_distance = center_x - initial_x
                        minimum_distance = minimum_distances[vehicle_type]
                        if horizontal_distance >= minimum_distance:
                            count_direction = Direction.RIGHT
                        elif horizontal_distance <= -minimum_distance:
                            count_direction = Direction.LEFT
                        else:
                            count_direction = None
                        if count_direction is not None and track_id not in counted_ids:
                            counted_ids.add(track_id)
                            vehicle_counts[(vehicle_type, count_direction)] += 1
                            csv_writer.writerow(
                                {
                                    "id": next_record_id,
                                    "timestamp": datetime.now().isoformat(timespec="milliseconds"),
                                    "direction": int(count_direction),
                                    "vehicle_type": int(vehicle_type),
                                    "confidence": f"{confidence:.4f}",
                                    **base_row,
                                }
                            )
                            csv_file.flush()
                            next_record_id += 1

                    color = (0, 255, 0) if accepted else (0, 165, 255)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            if current_time - last_track_cleanup_time >= 10.0:
                expired_track_ids = {
                    track_id
                    for track_id, last_seen in track_last_seen.items()
                    if current_time - last_seen >= TRACK_MEMORY_SECONDS
                }
                for track_id in expired_track_ids:
                    track_last_seen.pop(track_id, None)
                    track_history.pop(track_id, None)
                    counted_ids.discard(track_id)
                last_track_cleanup_time = current_time

            frame_duration = current_time - previous_frame_time
            if frame_duration > 0:
                instantaneous_fps = 1.0 / frame_duration
                fps = instantaneous_fps if fps == 0 else 0.9 * fps + 0.1 * instantaneous_fps
            previous_frame_time = current_time
            display_height = max(frame.shape[0], 460)
            video_area = np.zeros((display_height, frame.shape[1], 3), dtype=np.uint8)
            video_area[: frame.shape[0], :] = frame
            panel = _status_panel(display_height, vehicle_counts, fps)
            cv2.imshow("Car Counter", np.hstack((video_area, panel)))

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()
        csv_file.close()

if __name__ == "__main__":
    main()
