"""Count detected cars once based on their horizontal travel direction."""

from collections import defaultdict
import csv
from datetime import datetime, timedelta
import logging
from pathlib import Path
import time
from typing import TextIO

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from constants import (
    BICYCLE_CLASS_ID,
    BICYCLE_CONFIDENCE,
    BICYCLE_DIRECTION_DISTANCE_RATIO,
    BUS_CLASS_ID,
    CAMERA_HEIGHT,
    CAMERA_INDEX,
    CAMERA_WIDTH,
    CAR_CLASS_ID,
    CONFIDENCE,
    DETECTION_START_HOUR,
    DETECTION_START_MINUTE,
    IMAGE_SIZE,
    LIGHT_BRIGHTNESS_THRESHOLD,
    LIGHT_GROUP_X_DISTANCE,
    LIGHT_GROUP_Y_DISTANCE,
    LIGHT_MERGE_HEIGHT,
    LIGHT_MERGE_WIDTH,
    LIGHT_MIN_AREA,
    LIGHT_TRACK_DISTANCE,
    LIGHT_TRACK_MAX_MISSING,
    MAX_DETECTIONS,
    MIN_DIRECTION_DISTANCE_RATIO,
    MODEL_CONFIDENCE,
    MODEL_PATH,
    NIGHT_MODE,
    SELECT_CROP_ON_START,
    TRACK_MEMORY_SECONDS,
    TRUCK_CLASS_ID,
    VEHICLE_CLASS_IDS,
)
from camera_utils import open_camera
from enums import Direction, TimeOfDay, VehicleType

logging.getLogger("ultralytics").disabled = True


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


def _open_csv_writer(program_started_at: datetime) -> tuple[Path, TextIO, csv.DictWriter]:
    output_directory = Path("outputs") / program_started_at.strftime("%Y-%m-%d")
    output_directory.mkdir(parents=True, exist_ok=True)
    csv_path = output_directory / f"car_counts_{program_started_at.strftime('%Y%m%d_%H%M%S')}.csv"
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
    return csv_path, csv_file, csv_writer


def _wait_for_scheduled_start() -> None:
    """Wait until the next configured local start time without using the GPU."""
    now = datetime.now()
    start_at = now.replace(
        hour=DETECTION_START_HOUR,
        minute=DETECTION_START_MINUTE,
        second=0,
        microsecond=0,
    )
    if now > start_at:
        start_at += timedelta(days=1)

    remaining_seconds = (start_at - now).total_seconds()
    if remaining_seconds <= 0:
        return

    print(f"Waiting until {start_at:%Y-%m-%d %H:%M} local time to start detection.")
    while remaining_seconds > 0:
        time.sleep(min(remaining_seconds, 60.0))
        remaining_seconds = (start_at - datetime.now()).total_seconds()


def _select_startup_crop() -> tuple[int, int, int, int]:
    """Show the camera preview and choose the crop before scheduled detection."""
    camera = open_camera(CAMERA_INDEX, CAMERA_WIDTH, CAMERA_HEIGHT)
    if not camera.isOpened():
        raise RuntimeError(f"Could not open camera index {CAMERA_INDEX}")

    try:
        while True:
            success, frame = camera.read()
            if not success:
                raise RuntimeError("Could not read a frame from the camera")

            if not SELECT_CROP_ON_START:
                frame_height, frame_width = frame.shape[:2]
                return 0, 0, frame_width, frame_height

            cv2.imshow("Camera Preview", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("s"):
                cv2.destroyWindow("Camera Preview")
                return _select_crop(frame)
            if key == ord("q"):
                raise SystemExit(0)
    finally:
        camera.release()
        cv2.destroyAllWindows()


class VehicleCounter:
    def __init__(self, crop: tuple[int, int, int, int]) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None if NIGHT_MODE else YOLO(MODEL_PATH)
        if self.model is not None:
            self.model.to(self.device)
        self.light_tracker = MovingLightTracker() if NIGHT_MODE else None
        self.program_started_at = datetime.now()
        self.csv_path, self.csv_file, self.csv_writer = _open_csv_writer(self.program_started_at)
        self.next_record_id = 1
        self.camera = open_camera(CAMERA_INDEX, CAMERA_WIDTH, CAMERA_HEIGHT)
        if not self.camera.isOpened():
            self.csv_file.close()
            raise RuntimeError(f"Could not open camera index {CAMERA_INDEX}")

        self.crop_x, self.crop_y, self.crop_width, self.crop_height = crop
        self.track_history: dict[int, list[int]] = defaultdict(list)
        self.counted_ids: set[int] = set()
        self.vehicle_counts = {
            (vehicle_type, direction): 0
            for vehicle_type in VehicleType
            for direction in Direction
        }
        self.minimum_distances = {
            vehicle_type: max(25, int(self.crop_width * _direction_distance_ratio(vehicle_type)))
            for vehicle_type in VehicleType
        }
        self.time_of_day_value = int(TimeOfDay.NIGHT if NIGHT_MODE else TimeOfDay.DAY)
        self.base_row = {"time_of_day": self.time_of_day_value}
        self.class_to_vehicle_type = {
            BICYCLE_CLASS_ID: VehicleType.BICYCLE,
            CAR_CLASS_ID: VehicleType.CAR,
            BUS_CLASS_ID: VehicleType.BUS,
            TRUCK_CLASS_ID: VehicleType.TRUCK,
        }
        self.track_last_seen: dict[int, float] = {}
        self.fps = 0.0
        self.previous_frame_time = time.perf_counter()
        self.last_track_cleanup_time = self.previous_frame_time

        if NIGHT_MODE:
            assert self.light_tracker is not None
        else:
            assert self.model is not None

    def _read_detections(
        self, frame: np.ndarray
    ) -> list[tuple[tuple[float, float, float, float], int | None, float, VehicleType]]:
        if NIGHT_MODE:
            assert self.light_tracker is not None
            return [
                (
                    (float(box[0]), float(box[1]), float(box[2]), float(box[3])),
                    track_id,
                    1.0,
                    VehicleType.CAR,
                )
                for box, track_id in self.light_tracker.update(frame)
            ]

        assert self.model is not None
        result = self.model.track(
            frame,
            persist=True,
            tracker="vehicle_bytetrack.yaml",
            classes=VEHICLE_CLASS_IDS,
            conf=MODEL_CONFIDENCE,
            imgsz=IMAGE_SIZE,
            max_det=MAX_DETECTIONS,
            device=self.device,
            verbose=False,
        )[0]
        detections: list[tuple[tuple[float, float, float, float], int | None, float, VehicleType]] = []
        if result.boxes is not None:
            boxes: list[tuple[float, float, float, float]] = [
                (float(box[0]), float(box[1]), float(box[2]), float(box[3]))
                for box in result.boxes.xyxy.cpu().numpy()
            ]
            confidences = [float(value) for value in result.boxes.conf.cpu().numpy().tolist()]
            class_ids = [int(value) for value in result.boxes.cls.cpu().numpy().tolist()]
            track_ids = (
                [int(value) for value in result.boxes.id.cpu().numpy().tolist()]
                if result.boxes.id is not None
                else [None] * len(boxes)
            )
            detections = [
                (
                    box,
                    track_id,
                    confidence,
                    self.class_to_vehicle_type[class_id],
                )
                for box, track_id, confidence, class_id in zip(
                    boxes, track_ids, confidences, class_ids
                )
            ]
        return detections

    def _record_count(
        self,
        track_id: int,
        confidence: float,
        vehicle_type: VehicleType,
        count_direction: Direction,
    ) -> None:
        self.counted_ids.add(track_id)
        self.vehicle_counts[(vehicle_type, count_direction)] += 1
        self.csv_writer.writerow(
            {
                "id": self.next_record_id,
                "timestamp": datetime.now().isoformat(timespec="milliseconds"),
                "direction": int(count_direction),
                "vehicle_type": int(vehicle_type),
                "confidence": f"{confidence:.4f}",
                **self.base_row,
            }
        )
        self.csv_file.flush()
        self.next_record_id += 1

    def _process_detections(
        self,
        frame: np.ndarray,
        detections: list[tuple[tuple[float, float, float, float], int | None, float, VehicleType]],
        current_time: float,
    ) -> None:
        if not detections:
            return

        for box, track_id, confidence, vehicle_type in detections:
            x1, y1, x2, y2 = map(int, box)
            center_x = (x1 + x2) // 2
            accepted = confidence >= _confidence_threshold(vehicle_type)
            if track_id is not None and accepted:
                self.track_last_seen[track_id] = current_time
                history = self.track_history[track_id]
                history.append(center_x)
                if len(history) > 30:
                    history.pop(0)

                initial_x = history[0]
                horizontal_distance = center_x - initial_x
                minimum_distance = self.minimum_distances[vehicle_type]
                if horizontal_distance >= minimum_distance:
                    count_direction = Direction.RIGHT
                elif horizontal_distance <= -minimum_distance:
                    count_direction = Direction.LEFT
                else:
                    count_direction = None
                if count_direction is not None and track_id not in self.counted_ids:
                    self._record_count(
                        track_id, confidence, vehicle_type, count_direction
                    )

            color = (0, 255, 0) if accepted else (0, 165, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    def _cleanup_expired_tracks(self, current_time: float) -> None:
        if current_time - self.last_track_cleanup_time < 10.0:
            return

        expired_track_ids = {
            track_id
            for track_id, last_seen in self.track_last_seen.items()
            if current_time - last_seen >= TRACK_MEMORY_SECONDS
        }
        for track_id in expired_track_ids:
            self.track_last_seen.pop(track_id, None)
            self.track_history.pop(track_id, None)
            self.counted_ids.discard(track_id)
        self.last_track_cleanup_time = current_time

    def _update_fps(self, current_time: float) -> None:
        frame_duration = current_time - self.previous_frame_time
        if frame_duration > 0:
            instantaneous_fps = 1.0 / frame_duration
            self.fps = instantaneous_fps if self.fps == 0 else 0.9 * self.fps + 0.1 * instantaneous_fps
        self.previous_frame_time = current_time

    def _show_frame(self, frame: np.ndarray) -> None:
        display_height = max(frame.shape[0], 460)
        video_area = np.zeros((display_height, frame.shape[1], 3), dtype=np.uint8)
        video_area[: frame.shape[0], :] = frame
        panel = _status_panel(display_height, self.vehicle_counts, self.fps)
        cv2.imshow("Car Counter", np.hstack((video_area, panel)))

    def run(self) -> None:
        try:
            while True:
                success, frame = self.camera.read()
                if not success:
                    break

                current_time = time.perf_counter()
                frame = frame[
                    self.crop_y : self.crop_y + self.crop_height,
                    self.crop_x : self.crop_x + self.crop_width,
                ]
                if frame.size == 0:
                    raise RuntimeError("Crop is empty; restart and select a valid crop")

                detections = self._read_detections(frame)
                self._process_detections(frame, detections, current_time)
                self._cleanup_expired_tracks(current_time)
                self._update_fps(current_time)
                self._show_frame(frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        finally:
            self.camera.release()
            cv2.destroyAllWindows()
            self.csv_file.close()


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
                x, y, width, height = cv2.boundingRect(contour)
                candidates.append((x, y, width, height))
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
    crop = _select_startup_crop()
    _wait_for_scheduled_start()
    counter = VehicleCounter(crop)
    counter.run()

if __name__ == "__main__":
    main()
