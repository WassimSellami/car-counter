START_IMMEDIATELY = True
START_HOURS = 5
START_MINUTES = 15
CAMERA_INDEX = 0  # Set this to the DroidCam index printed by find_camera.py.
# Requested DroidCam capture resolution. DroidCam must also be configured to
# offer this resolution; otherwise its driver will use the nearest supported one.
CAMERA_WIDTH = 1920
CAMERA_HEIGHT = 1080
# Default width for traffic-flow chart buckets.
CONTINUOUS_FLOW_INTERVAL_MINUTES = 20
# Fixed duration for the centred traffic-flow rolling average.
CONTINUOUS_FLOW_ROLLING_AVERAGE_MINUTES = 120
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
IMAGE_SIZE = 640  # YOLO input image size. Must be a multiple of 32.
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
