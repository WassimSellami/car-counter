"""Verify the DroidCam feed and measure its resolution and frame rate."""

import cv2
import time

from camera_utils import open_camera

CAMERA_INDEX = 0  # Set this to the DroidCam index printed by find_camera.py.
CAMERA_WIDTH = 1920
CAMERA_HEIGHT = 1080
MEASUREMENT_SECONDS = 5


def main() -> None:
    camera = open_camera(CAMERA_INDEX)
    if not camera.isOpened():
        raise RuntimeError(f"Could not open camera index {CAMERA_INDEX}")

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    try:
        print(f"Measuring the feed for {MEASUREMENT_SECONDS} seconds...")
        started_at = time.perf_counter()
        frames = 0
        first_frame = None
        while time.perf_counter() - started_at < MEASUREMENT_SECONDS:
            success, frame = camera.read()
            if not success:
                raise RuntimeError("Could not read a frame from the camera")
            first_frame = frame
            frames += 1

        assert first_frame is not None
        height, width = first_frame.shape[:2]
        elapsed = time.perf_counter() - started_at
        print(f"Actual feed: {width}x{height} at {frames / elapsed:.1f} FPS")
        print("Press q in the video window to close.")

        while True:
            success, frame = camera.read()
            if not success:
                raise RuntimeError("Could not read a frame from the camera")

            cv2.imshow("DroidCam Live Feed", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
