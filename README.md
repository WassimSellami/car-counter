# Directional Vehicle Counter

Counts vehicles independently by their direction of horizontal movement.

## Run

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python find_camera.py
```

Set `CAMERA_INDEX` in `view_camera.py` and `car_counter.py` to the DroidCam index.
`car_counter.py` requests a 1920×1080 feed by default; change `CAMERA_WIDTH` and
`CAMERA_HEIGHT` there if your DroidCam plan or phone supports a different size.

```powershell
python view_camera.py
python car_counter.py
```

`view_camera.py` first measures the source for five seconds and prints its actual
resolution and FPS. This measures the DroidCam feed without YOLO inference.

Press `q` to close either video window. The first run downloads `yolo11n.pt`.

There is no counting line. A tracked car is counted once after it moves horizontally by at least `MIN_DIRECTION_DISTANCE_RATIO` of the frame width. Movement right increments `Going right`; movement left increments `Going left`.

Each run creates a CSV file at `outputs/YYYY-MM-DD/car_counts_YYYYMMDD_HHMMSS.csv`, grouping results by the day the program started. A row is written and flushed immediately for every counted vehicle: `0.40` confidence for cars/vans, trucks, and buses; `0.10` for bicycles. Its columns are `id`, `timestamp`, `direction` (`0` = left, `1` = right), `vehicle_type` (`0` = car or van, `1` = truck, `2` = bus, `3` = bicycle), `time_of_day` (`0` = day, `1` = night), and `confidence`. The COCO model does not provide a separate van class, so vans are detected and recorded as cars.

## Traffic analysis

Generate direction and vehicle-type charts from every recorded CSV with:

```powershell
pip install -r requirements.txt
python vehicle_counter_analysis.py
```

The script deliberately ignores `confidence` and `time_of_day`. It writes these charts plus a `summary.csv` to `outputs/analysis/`:

- hourly stacked direction flow;
- hourly stacked vehicle types;
- continuous 30-minute flow with smoothed lines: category colour, solid right-bound line, and dashed left-bound line;
- direction × vehicle-type heatmap;
- vehicle-mix pie charts for left- and right-bound traffic.

To analyse only selected runs, pass their file paths, for example:

```powershell
python vehicle_counter_analysis.py outputs/2026-08-11/car_counts_20260811_195938.csv
```

Use a different bucket width when needed, for example `--interval-minutes 10`.

`vehicle_bytetrack.yaml` lowers ByteTrack's ID-creation threshold to match the bicycle threshold. A green box is accepted, but it is written to the CSV only after it receives a stable tracker ID and travels at least `MIN_DIRECTION_DISTANCE_RATIO` across the selected crop.

For long-running sessions, tracker histories and counted IDs that have been absent for two minutes are removed automatically to keep memory use bounded.

Bicycles use the shorter `BICYCLE_DIRECTION_DISTANCE_RATIO` (3% of the selected crop width); other vehicle types use `MIN_DIRECTION_DISTANCE_RATIO` (8%). The video displays boxes only.

`car_counter.py` uses the faster `yolo11n.pt`. The video part of the window shows only detection boxes; a separate sidebar shows the right and left totals.

## GPU acceleration

In day mode, the program automatically uses CUDA when PyTorch detects an NVIDIA GPU; otherwise it uses the CPU. Verify your active environment with `python -c "import torch; print(torch.cuda.is_available())"`. If it prints `False`, install a CUDA-enabled PyTorch build compatible with your NVIDIA driver, then update Ultralytics in that same environment.

No normal runtime details are printed to the terminal. Green boxes are accepted (`score >= 0.40`); orange boxes are rejected and never counted.

## Road Crop

At startup, a `Camera Preview` window opens. Wait until the DroidCam image is visible, then press `S`. Drag a rectangle around the road in `Select Road Crop` and press Enter or Space to confirm it. Press `C` to cancel and use the full camera frame. The preview and detector use only the selected crop.

## Night Mode

Set `NIGHT_MODE = True` to use moving bright lights instead of YOLO car detection. It is intended for visible headlights at night and can count reflections or other moving lights as vehicles, so select a tight road crop first.

`LIGHT_MERGE_WIDTH` joins nearby headlights into one vehicle box. Grouping and track retention further reduce duplicate IDs from separated lights and short detection gaps. Increase `LIGHT_GROUP_X_DISTANCE` if one car is still split; decrease it if neighbouring cars are merged together.
