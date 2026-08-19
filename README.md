# Cloud Vehicle Counter

The system is split between two machines:

| Where | Runs | Responsibility |
| --- | --- | --- |
| Local PC (camera access) | `cloud_camera_sender.py` | Captures DroidCam, lets you select the road, rectifies/crops it, resizes frames to 640 px wide, and sends batches to Vast. |
| Vast GPU machine | `vast_counter_server.py` | YOLO vehicle detection, ByteTrack tracking, CLIP car-colour classification, counting, annotated video, CSV writing, and optional Supabase upload. |
| Vast GPU machine | `vast_live_dashboard.py` | Streamlit dashboard: annotated live video from the counter and historical counts from Supabase. |

The local PC does **not** run YOLO in this architecture. It only needs camera access and sends cropped JPEG frames. The Vast machine is the source of new CSV files and Supabase rows.

## 1. Local PC setup

From the project folder:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-local.txt
```

Set the DroidCam camera index and requested capture resolution in [constants.py](constants.py). Usually `CAMERA_SOURCE = 0` works for DroidCam.

Create the ignored `.env` file beside this README:

```dotenv
# Vast public mapping for the instance's internal port 8000.
COUNTER_PUBLIC_URL=http://YOUR_VAST_IP:YOUR_PUBLIC_8000_PORT

# Choose a long random secret. It must exactly match the cloud value.
CLOUD_INFERENCE_API_KEY=YOUR_LONG_RANDOM_SECRET
```

Run the camera sender:

```powershell
python cloud_camera_sender.py
```

Press `S` when the camera preview is visible. Click the road corners in this order: top-left, top-right, bottom-right, bottom-left, then press Enter. The selected trapezoid is projected into a straight road rectangle before it leaves the PC. Press `Ctrl+C` to stop sending.

## 2. Create the Vast instance

Choose a GPU instance and ensure Vast exposes these **internal** ports:

| Internal port | Used by |
| --- | --- |
| `22` | SSH |
| `8000` | Counter API / camera sender |
| `8501` | Streamlit dashboard |

Vast assigns a different public port to each internal port. For example, if Vast displays:

```text
82.225.150.130:15473 -> 8000/tcp
82.225.150.130:15311 -> 8501/tcp
82.225.150.130:15498 -> 22/tcp
```

then:

```text
Counter URL:  http://82.225.150.130:15473
Dashboard:    http://82.225.150.130:15311
SSH:          ssh -p 15498 root@82.225.150.130
```

Use the mappings shown for **your own** instance; they change when you rent a new machine.

## 3. Deploy the cloud code from GitHub

Commit and push code changes from the local PC before deploying them:

```powershell
git add .
git commit -m "Describe the change"
git push
```

SSH into Vast:

```powershell
ssh -p SSH_PORT root@VAST_IP
```

Clone the repository once:

```bash
git clone https://github.com/WassimSellami/car-counter.git /workspace/car-counter
cd /workspace/car-counter
```

For later deployments, pull the committed code instead of uploading files:

```bash
cd /workspace/car-counter
git pull --ff-only
```

Install the cloud dependencies:

```bash
/venv/main/bin/python -m pip install -r requirements-cloud.txt
```

Create the cloud environment file:

```bash
nano /root/counter.env
```

Paste the following, replacing every placeholder. `COUNTER_PUBLIC_URL` is the public mapping to internal port `8000`.

```dotenv
CLOUD_INFERENCE_API_KEY='THE_SAME_SECRET_AS_THE_LOCAL_PC'
COUNTER_OUTPUT_DIR=/workspace/outputs
UPLOAD_TO_SUPABASE=true
COUNTER_INTERNAL_URL=http://127.0.0.1:8000
COUNTER_PUBLIC_URL=http://YOUR_VAST_IP:YOUR_PUBLIC_8000_PORT
SUPABASE_URL='https://YOUR-PROJECT.supabase.co'
SUPABASE_SERVICE_ROLE_KEY='YOUR_SERVICE_ROLE_KEY'
```

Save with `Ctrl+O`, Enter, then exit with `Ctrl+X`.

## 4. Start the cloud services

Load the environment and start one counter process:

```bash
set -a
source /root/counter.env
set +a

nohup /venv/main/bin/python -u /workspace/car-counter/vast_counter_server.py > /root/counter.log 2>&1 &
```

The first startup downloads model weights and can take a few minutes. Check it with:

```bash
tail -f /root/counter.log
```

Wait for:

```text
Uvicorn running on http://0.0.0.0:8000
```

Then start the dashboard:

```bash
nohup /venv/main/bin/python -m streamlit run /workspace/car-counter/vast_live_dashboard.py --server.address 0.0.0.0 --server.port 8501 > /root/dashboard.log 2>&1 &
```

Open the public Vast mapping for port `8501` in a browser. The dashboard's live video starts after `cloud_camera_sender.py` sends frames.

## Restart after changing configuration or code

On Vast, stop the old copies and start cleanly:

```bash
pkill -f '[v]ast_counter_server.py' || true
pkill -f '[s]treamlit run /workspace/car-counter/vast_live_dashboard.py' || true
sleep 3

cd /workspace/car-counter
git pull --ff-only

set -a
source /root/counter.env
set +a

nohup /venv/main/bin/python -u /workspace/car-counter/vast_counter_server.py > /root/counter.log 2>&1 &
nohup /venv/main/bin/python -m streamlit run /workspace/car-counter/vast_live_dashboard.py --server.address 0.0.0.0 --server.port 8501 > /root/dashboard.log 2>&1 &
```

Use these logs to diagnose the cloud services:

```bash
tail -f /root/counter.log
tail -f /root/dashboard.log
```

If you only change the local `.env`, stop and rerun `python cloud_camera_sender.py`; no cloud restart is needed.

## Supabase data

Create the `traffic_counts` table by running [supabase_setup.sql](supabase_setup.sql) in the Supabase SQL Editor. Keep the service-role key private and never commit `.env` or `/root/counter.env`.

With `UPLOAD_TO_SUPABASE=true`, each newly counted vehicle is saved in both places:

- Vast CSV: `/workspace/outputs/YYYY-MM-DD/count_YYYYMMDD.csv`
- Supabase: `traffic_counts`

The Vast dashboard always reads historical counts and car-colour charts from Supabase, so it includes archived data from previous Vast machines. Its live video, latency, and FPS values come from the currently running Vast counter.

### One-time archive repair / replacement

For a deliberate full rebuild of Supabase from the local `outputs` archive, first make a backup, then run locally:

```powershell
python sync_counts_to_supabase.py --repair-colors --replace
```

This adds a missing `color` column with value `0` (unknown) to old CSV files, **deletes every existing Supabase count**, and uploads the local archive. Do not use `--replace` during normal operation.

## Offline analysis

`vehicle_counter_analysis.py` creates an interactive HTML report from CSV files available on the machine where it is run:

```powershell
python vehicle_counter_analysis.py
```

For cloud-generated CSVs, download the required `outputs` files first, or use the Supabase-backed Vast dashboard instead.

## Notes

- `yolo11m.pt` is the default cloud detector. Change `MODEL_PATH` in `/root/counter.env`, for example `MODEL_PATH=yolo11s.pt`, then restart the counter to trade accuracy for speed.
- Car colours use CLIP and are classified only for cars/vans, not trucks, buses, or bicycles. `0` means unknown; colour codes `1` through `10` represent black, white, grey, silver, red, blue, green, yellow, orange, and brown.
- Vast public ports and IP addresses are instance-specific. Update both the local `.env` and cloud `/root/counter.env` whenever the instance changes.
- Previous local-counter and inference implementations are preserved in `legacy/`. They are not part of the cloud deployment.
