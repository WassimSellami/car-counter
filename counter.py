"""Count detected cars once based on their horizontal travel direction."""

from collections import defaultdict, deque
import csv
from datetime import date, datetime, timedelta
import logging
import os
from pathlib import Path
from queue import Empty, Full, Queue
from threading import Event, Thread
import time
from typing import TextIO

import cv2
import numpy as np
import requests

from constants import (
    BICYCLE_CLASS_ID,
    BICYCLE_CONFIDENCE,
    BICYCLE_DIRECTION_DISTANCE_RATIO,
    BUS_CLASS_ID,
    CAMERA_HEIGHT,
    CAMERA_SOURCE,
    CAMERA_WIDTH,
    CAR_CLASS_ID,
    CLOUD_BATCH_SIZE,
    CONFIDENCE,
    DISPLAY_BUFFER_FRAMES,
    DISPLAY_FPS,
    LIGHT_BRIGHTNESS_THRESHOLD,
    LIGHT_GROUP_X_DISTANCE,
    LIGHT_GROUP_Y_DISTANCE,
    LIGHT_MERGE_HEIGHT,
    LIGHT_MERGE_WIDTH,
    LIGHT_MIN_AREA,
    LIGHT_TRACK_DISTANCE,
    LIGHT_TRACK_MAX_MISSING,
    MIN_DIRECTION_DISTANCE_RATIO,
    NIGHT_MODE,
    SELECT_CROP_ON_START,
    START_HOURS,
    START_IMMEDIATELY,
    START_MINUTES,
    TRACK_MEMORY_SECONDS,
    TRUCK_CLASS_ID,
    UPLOAD_TO_SUPABASE,
)
from camera_utils import open_camera
from cloud_detection_client import INFERENCE_WIDTH, JPEG_QUALITY, infer_frame_batch
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


CSV_FIELDNAMES = ["id", "timestamp", "direction", "vehicle_type", "time_of_day", "confidence"]


def _empty_vehicle_counts() -> dict[tuple[VehicleType, Direction], int]:
    return {
        (vehicle_type, direction): 0
        for vehicle_type in VehicleType
        for direction in Direction
    }


def _load_dotenv() -> None:
    """Load local Supabase credentials without committing them to the project."""
    env_file = Path(".env")
    if not env_file.is_file():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        if not line or line.lstrip().startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", maxsplit=1)
        os.environ.setdefault(name.strip(), value.strip().strip('"').strip("'"))


class SupabaseCountUploader:
    """Upload new count rows in the background without delaying detection."""

    def __init__(self) -> None:
        _load_dotenv()
        self.url = os.environ.get("SUPABASE_URL", "").rstrip("/")
        self.service_key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
        self.queue: Queue[dict | None] = Queue()
        self.worker: Thread | None = None
        if UPLOAD_TO_SUPABASE and self.url and self.service_key:
            self.worker = Thread(target=self._upload_loop, daemon=True)
            self.worker.start()

    def submit(self, row: dict) -> None:
        if self.worker is not None:
            self.queue.put(row)

    def close(self) -> None:
        if self.worker is not None:
            self.queue.put(None)
            self.worker.join(timeout=2)

    def _upload_loop(self) -> None:
        while True:
            row = self.queue.get()
            if row is None:
                return
            while True:
                try:
                    response = requests.post(
                        f"{self.url}/rest/v1/traffic_counts",
                        params={"on_conflict": "record_id"},
                        headers={
                            "apikey": self.service_key,
                            "Authorization": f"Bearer {self.service_key}",
                            "Content-Type": "application/json",
                            "Prefer": "resolution=merge-duplicates,return=minimal",
                        },
                        json=[row],
                        timeout=10,
                    )
                    response.raise_for_status()
                    break
                except requests.RequestException:
                    # The CSV remains the durable local record; retry without
                    # blocking the camera/detection loop.
                    time.sleep(5)


def _load_daily_counts(output_directory: Path) -> dict[tuple[VehicleType, Direction], int]:
    """Restore totals from the daily CSV, falling back to legacy files if needed."""
    counts = _empty_vehicle_counts()
    daily_files = list(output_directory.glob("count_*.csv"))
    # A count file may have been created by copying a legacy run file. Do
    # not add both, because they represent the same earlier detections.
    patterns = ("count_*.csv",) if daily_files else ("car_counts_*.csv",)
    for pattern in patterns:
        for csv_path in output_directory.glob(pattern):
            try:
                with csv_path.open("r", newline="", encoding="utf-8") as csv_file:
                    for row in csv.DictReader(csv_file):
                        try:
                            vehicle_type = VehicleType(int(row["vehicle_type"]))
                            direction = Direction(int(row["direction"]))
                        except (KeyError, TypeError, ValueError):
                            continue
                        counts[(vehicle_type, direction)] += 1
            except OSError:
                continue
    return counts


def _next_record_id() -> int:
    """Return the next globally unique ID across all counter CSV files."""
    highest_id = 0
    for csv_path in Path("outputs").glob("**/count_*.csv"):
        try:
            with csv_path.open("r", newline="", encoding="utf-8") as csv_file:
                for row in csv.DictReader(csv_file):
                    try:
                        highest_id = max(highest_id, int(row["id"]))
                    except (KeyError, TypeError, ValueError):
                        continue
        except OSError:
            continue
    return highest_id + 1


def _open_csv_writer(day: date) -> tuple[Path, TextIO, csv.DictWriter, dict[tuple[VehicleType, Direction], int], int]:
    """Append to the day's CSV and restore its accumulated totals."""
    output_directory = Path("outputs") / day.isoformat()
    output_directory.mkdir(parents=True, exist_ok=True)
    csv_path = output_directory / f"count_{day.strftime('%Y%m%d')}.csv"
    has_rows = csv_path.is_file() and csv_path.stat().st_size > 0
    counts = _load_daily_counts(output_directory)
    next_record_id = _next_record_id()
    csv_file = csv_path.open("a", newline="", encoding="utf-8", buffering=1)
    csv_writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDNAMES)
    if not has_rows:
        csv_writer.writeheader()
        csv_file.flush()
    return csv_path, csv_file, csv_writer, counts, next_record_id


def _select_startup_crop() -> tuple[int, int, int, int]:
    """Show the camera preview and choose the crop before scheduled detection."""
    camera = open_camera(CAMERA_SOURCE, CAMERA_WIDTH, CAMERA_HEIGHT)
    if not camera.isOpened():
        raise RuntimeError(f"Could not open camera source {CAMERA_SOURCE}")

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


def _wait_for_scheduled_start() -> None:
    """Wait for the next configured start time unless immediate start is enabled."""
    if START_IMMEDIATELY:
        return
    if not 0 <= START_HOURS <= 23 or not 0 <= START_MINUTES <= 59:
        raise ValueError("START_HOURS must be 0-23 and START_MINUTES must be 0-59")

    now = datetime.now()
    start_time = now.replace(hour=START_HOURS, minute=START_MINUTES, second=0, microsecond=0)
    if start_time <= now:
        start_time += timedelta(days=1)
    print(f"Crop saved. Counting will start at {start_time:%Y-%m-%d %H:%M}.")
    while (remaining := (start_time - datetime.now()).total_seconds()) > 0:
        time.sleep(min(remaining, 60))


class VehicleCounter:
    def __init__(self, crop: tuple[int, int, int, int]) -> None:
        _load_dotenv()
        self.cloud_url = os.environ.get("CLOUD_INFERENCE_URL", "")
        self.cloud_batch_url = self.cloud_url.rsplit("/", 1)[0] + "/infer-batch" if self.cloud_url else ""
        self.cloud_headers = {"X-API-Key": os.environ.get("CLOUD_INFERENCE_API_KEY", "")}
        self.cloud_session = requests.Session()
        if not 1 <= CLOUD_BATCH_SIZE <= 12:
            raise ValueError("CLOUD_BATCH_SIZE must be between 1 and 12")
        if not NIGHT_MODE and (not self.cloud_url or not self.cloud_headers["X-API-Key"]):
            raise RuntimeError("Set CLOUD_INFERENCE_URL and CLOUD_INFERENCE_API_KEY in .env")
        self.light_tracker = MovingLightTracker() if NIGHT_MODE else None
        self.program_started_at = datetime.now()
        self.csv_day = self.program_started_at.date()
        (
            self.csv_path,
            self.csv_file,
            self.csv_writer,
            self.vehicle_counts,
            self.next_record_id,
        ) = _open_csv_writer(self.csv_day)
        self.supabase_uploader = SupabaseCountUploader()
        self.camera = open_camera(CAMERA_SOURCE, CAMERA_WIDTH, CAMERA_HEIGHT)
        if not self.camera.isOpened():
            self.csv_file.close()
            self.supabase_uploader.close()
            raise RuntimeError(f"Could not open camera source {CAMERA_SOURCE}")

        self.crop_x, self.crop_y, self.crop_width, self.crop_height = crop
        self.track_history: dict[int, list[int]] = defaultdict(list)
        self.counted_ids: set[int] = set()
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
        self.render_fps = 0.0
        self.rendered_frame_times: deque[float] = deque()
        self.cloud_delay_ms = 0.0
        self.yolo_ms = 0.0
        self.local_ms = 0.0
        self.capture_queue: Queue[tuple[np.ndarray, float]] = Queue(maxsize=CLOUD_BATCH_SIZE * 3)
        self.display_queue: Queue[np.ndarray] = Queue(maxsize=CLOUD_BATCH_SIZE * 3)
        self.stop_event = Event()
        self.capture_finished = Event()
        self.inference_finished = Event()
        self.cloud_error: str | None = None
        self.last_track_cleanup_time = time.perf_counter()

        if NIGHT_MODE:
            assert self.light_tracker is not None

    def _read_detection_batches(
        self, frames: list[np.ndarray]
    ) -> list[list[tuple[tuple[float, float, float, float], int | None, float, VehicleType]]]:
        if NIGHT_MODE:
            assert self.light_tracker is not None
            return [
                [
                    (
                        (float(box[0]), float(box[1]), float(box[2]), float(box[3])),
                        track_id,
                        1.0,
                        VehicleType.CAR,
                    )
                    for box, track_id in self.light_tracker.update(frame)
                ]
                for frame in frames
            ]

        detection_batches, dimensions, cloud_delay, yolo_ms = infer_frame_batch(
            self.cloud_session,
            frames,
            self.cloud_batch_url,
            self.cloud_headers,
            INFERENCE_WIDTH,
            JPEG_QUALITY,
        )
        self.cloud_delay_ms = cloud_delay * 1000
        self.yolo_ms = yolo_ms
        if detection_batches and detection_batches[0] and "track_id" not in detection_batches[0][0]:
            raise RuntimeError(
                "Cloud server is outdated. Upload and restart vast_inference_server.py."
            )
        return [
            [
                (
                    (
                        detection["xyxy"][0] * frame.shape[1] / inference_width,
                        detection["xyxy"][1] * frame.shape[0] / inference_height,
                        detection["xyxy"][2] * frame.shape[1] / inference_width,
                        detection["xyxy"][3] * frame.shape[0] / inference_height,
                    ),
                    detection["track_id"],
                    detection["confidence"],
                    self.class_to_vehicle_type[detection["class_id"]],
                )
                for detection in detections
            ]
            for frame, detections, (inference_width, inference_height) in zip(frames, detection_batches, dimensions)
        ]

    def _record_count(
        self,
        track_id: int,
        confidence: float,
        vehicle_type: VehicleType,
        count_direction: Direction,
    ) -> None:
        self._rotate_csv_at_midnight()
        self.counted_ids.add(track_id)
        self.vehicle_counts[(vehicle_type, count_direction)] += 1
        row = {
            "id": self.next_record_id,
            "timestamp": datetime.now().isoformat(timespec="milliseconds"),
            "direction": int(count_direction),
            "vehicle_type": int(vehicle_type),
            "confidence": f"{confidence:.4f}",
            **self.base_row,
        }
        self.csv_writer.writerow(row)
        self.csv_file.flush()
        self.supabase_uploader.submit(
            {
                "source_file": self.csv_path.relative_to(Path("outputs")).as_posix(),
                "record_id": row["id"],
                "timestamp": row["timestamp"],
                "direction": row["direction"],
                "vehicle_type": row["vehicle_type"],
                "time_of_day": row["time_of_day"],
                "confidence": float(row["confidence"]),
            }
        )
        self.next_record_id += 1

    def _rotate_csv_at_midnight(self) -> None:
        """Switch to a new daily file if the counter continues past midnight."""
        current_day = datetime.now().date()
        if current_day == self.csv_day:
            return

        self.csv_file.close()
        (
            self.csv_path,
            self.csv_file,
            self.csv_writer,
            self.vehicle_counts,
            self.next_record_id,
        ) = _open_csv_writer(current_day)
        self.csv_day = current_day
        self.counted_ids.clear()
        self.track_history.clear()
        self.track_last_seen.clear()

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

    def _update_fps(self, frame_duration: float) -> None:
        if frame_duration > 0:
            instantaneous_fps = 1.0 / frame_duration
            self.fps = instantaneous_fps if self.fps == 0 else 0.9 * self.fps + 0.1 * instantaneous_fps

    def _update_render_fps(self) -> None:
        current_time = time.perf_counter()
        self.rendered_frame_times.append(current_time)
        while self.rendered_frame_times and self.rendered_frame_times[0] <= current_time - 1:
            self.rendered_frame_times.popleft()
        self.render_fps = float(len(self.rendered_frame_times))

    def _show_frame(self, frame: np.ndarray) -> None:
        display_height = max(frame.shape[0], 530)
        video_area = np.zeros((display_height, frame.shape[1], 3), dtype=np.uint8)
        video_area[: frame.shape[0], :] = frame
        panel = _status_panel(
            display_height,
            self.vehicle_counts,
            self.fps,
            self.render_fps,
            self.cloud_delay_ms,
            self.yolo_ms,
            self.local_ms,
        )
        cv2.imshow("Car Counter", np.hstack((video_area, panel)))

    def _queue_display_frame(self, frame: np.ndarray) -> None:
        while not self.stop_event.is_set():
            try:
                self.display_queue.put(frame, timeout=0.1)
                return
            except Full:
                pass

    def _capture_loop(self) -> None:
        try:
            while not self.stop_event.is_set():
                success, frame = self.camera.read()
                if not success:
                    return
                frame = frame[
                    self.crop_y : self.crop_y + self.crop_height,
                    self.crop_x : self.crop_x + self.crop_width,
                ]
                if frame.size == 0:
                    raise RuntimeError("Crop is empty; restart and select a valid crop")
                captured_frame = (frame, time.perf_counter())
                try:
                    self.capture_queue.put(captured_frame, timeout=0.01)
                except Full:
                    # Keep the newest camera frames when inference falls behind.
                    try:
                        self.capture_queue.get_nowait()
                    except Empty:
                        pass
                    self.capture_queue.put_nowait(captured_frame)
        except Exception as error:
            self.cloud_error = str(error)
            print(f"Camera capture stopped: {error}")
        finally:
            self.capture_finished.set()

    def _inference_loop(self) -> None:
        try:
            while not self.stop_event.is_set():
                batch: list[tuple[np.ndarray, float]] = []
                while len(batch) < CLOUD_BATCH_SIZE and not self.stop_event.is_set():
                    try:
                        batch.append(self.capture_queue.get(timeout=0.1))
                    except Empty:
                        if self.capture_finished.is_set():
                            break
                if not batch:
                    if self.capture_finished.is_set():
                        break
                    continue

                loop_started_at = time.perf_counter()
                try:
                    detection_batches = self._read_detection_batches([frame for frame, _ in batch])
                except requests.RequestException:
                    self.cloud_error = "Cloud unavailable - reconnecting..."
                    self._queue_display_frame(batch[-1][0])
                    time.sleep(0.25)
                    continue
                self.cloud_error = None
                for (frame, current_time), detections in zip(batch, detection_batches):
                    self._process_detections(frame, detections, current_time)
                    self._cleanup_expired_tracks(current_time)
                frame_duration = (time.perf_counter() - loop_started_at) / len(batch)
                self.local_ms = max(0.0, frame_duration * 1000 - self.cloud_delay_ms / len(batch))
                self._update_fps(frame_duration)
                for frame, _ in batch:
                    self._queue_display_frame(frame)
        except Exception as error:
            self.cloud_error = str(error)
            print(f"Cloud inference stopped: {error}")
        finally:
            self.inference_finished.set()

    def run(self) -> None:
        capture_worker = Thread(target=self._capture_loop, daemon=True)
        worker = Thread(target=self._inference_loop, daemon=True)
        capture_worker.start()
        worker.start()
        next_display_at = time.perf_counter()
        display_started = False
        try:
            while not (self.inference_finished.is_set() and self.display_queue.empty()):
                if not display_started and self.display_queue.qsize() < DISPLAY_BUFFER_FRAMES:
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                    continue
                display_started = True
                try:
                    frame = self.display_queue.get(timeout=0.05)
                except Empty:
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                    continue
                if self.cloud_error:
                    cv2.putText(frame, self.cloud_error, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                self._update_render_fps()
                self._show_frame(frame)
                next_display_at += 1 / DISPLAY_FPS
                remaining_ms = max(1, round((next_display_at - time.perf_counter()) * 1000))
                if remaining_ms == 1:
                    next_display_at = time.perf_counter()
                if cv2.waitKey(remaining_ms) & 0xFF == ord("q"):
                    break
        finally:
            self.stop_event.set()
            capture_worker.join(timeout=2)
            worker.join(timeout=16)
            self.camera.release()
            self.cloud_session.close()
            cv2.destroyAllWindows()
            self.csv_file.close()
            self.supabase_uploader.close()


def _status_panel(
    frame_height: int,
    vehicle_counts: dict[tuple[VehicleType, Direction], int],
    fps: float,
    render_fps: float,
    cloud_delay_ms: float,
    yolo_ms: float,
    local_ms: float,
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

    cv2.putText(panel, f"Cloud  {cloud_delay_ms:.0f} ms / {CLOUD_BATCH_SIZE}", (20, 365), cv2.FONT_HERSHEY_SIMPLEX, 0.65, text_color, 2)
    cv2.putText(panel, f"YOLO   {yolo_ms:.0f} ms / {CLOUD_BATCH_SIZE}", (20, 395), cv2.FONT_HERSHEY_SIMPLEX, 0.65, text_color, 2)
    cv2.putText(panel, f"Local  {local_ms:.0f} ms", (20, 425), cv2.FONT_HERSHEY_SIMPLEX, 0.65, text_color, 2)
    cv2.putText(panel, f"Process  {fps:.1f} FPS", (20, 465), cv2.FONT_HERSHEY_SIMPLEX, 0.65, text_color, 2)
    cv2.putText(panel, f"Render   {render_fps:.1f} FPS", (20, 500), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
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
