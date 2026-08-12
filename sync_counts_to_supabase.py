"""Continuously copy locally recorded traffic counts to Supabase.

Set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY, then run:
    python sync_counts_to_supabase.py

This is intentionally separate from car_counter.py. It only reads the CSVs, so
it can start or stop without interrupting vehicle detection.
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import time

import requests

from vehicle_counter_analysis import discover_count_csv_files


BATCH_SIZE = 500


def load_dotenv() -> None:
    """Load the two local Supabase settings without adding another dependency."""
    env_file = Path(".env")
    if not env_file.is_file():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        if not line or line.lstrip().startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", maxsplit=1)
        os.environ.setdefault(name.strip(), value.strip().strip('"').strip("'"))


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--interval-seconds",
        type=float,
        default=5.0,
        help="Seconds between scans for newly appended rows (default: 5).",
    )
    return parser.parse_args()


def supabase_settings() -> tuple[str, str]:
    url = os.environ.get("SUPABASE_URL")
    service_key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
    if not url or not service_key:
        raise RuntimeError(
            "Set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY before starting the sync program."
        )
    return url.rstrip("/"), service_key


def rows_from_csv(csv_path: Path) -> list[dict]:
    """Read valid rows. Re-uploading is safe because the database key is stable."""
    rows = []
    with csv_path.open("r", newline="", encoding="utf-8") as csv_file:
        for row in csv.DictReader(csv_file):
            try:
                rows.append(
                    {
                        "source_file": csv_path.relative_to(Path("outputs")).as_posix(),
                        "record_id": int(row["id"]),
                        "timestamp": row["timestamp"],
                        "direction": int(row["direction"]),
                        "vehicle_type": int(row["vehicle_type"]),
                        "time_of_day": int(row["time_of_day"]),
                        "confidence": float(row["confidence"]),
                    }
                )
            except (KeyError, TypeError, ValueError):
                # A row may be in the middle of being written; the next scan retries it.
                continue
    return rows


def upload_rows(url: str, service_key: str, rows: list[dict]) -> int:
    for start in range(0, len(rows), BATCH_SIZE):
        batch = rows[start : start + BATCH_SIZE]
        response = requests.post(
            f"{url}/rest/v1/traffic_counts",
            params={"on_conflict": "record_id"},
            headers={
                "apikey": service_key,
                "Authorization": f"Bearer {service_key}",
                "Content-Type": "application/json",
                "Prefer": "resolution=merge-duplicates,return=minimal",
            },
            json=batch,
            timeout=30,
        )
        response.raise_for_status()
    return len(rows)


def sync_once(url: str, service_key: str) -> int:
    uploaded = 0
    for csv_path in discover_count_csv_files():
        uploaded += upload_rows(url, service_key, rows_from_csv(csv_path))
    return uploaded


def main() -> None:
    arguments = parse_arguments()
    if arguments.interval_seconds <= 0:
        raise ValueError("--interval-seconds must be greater than zero.")

    load_dotenv()
    url, service_key = supabase_settings()
    print("Syncing counter CSVs to Supabase. Press Ctrl+C to stop this sync only.")
    while True:
        try:
            rows = sync_once(url, service_key)
            print(f"Synced {rows:,} row(s).", flush=True)
        except Exception as error:
            print(f"Sync failed; will retry: {error}", flush=True)
        time.sleep(arguments.interval_seconds)


if __name__ == "__main__":
    main()
