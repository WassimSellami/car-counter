"""One-time local CSV migration to make today's IDs globally sequential.

Run only after stopping counter.py and sync_counts_to_supabase.py.
"""

from __future__ import annotations

import argparse
import csv
from datetime import date
import os
from pathlib import Path
import shutil


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--day", type=date.fromisoformat, required=True, help="Day to migrate, YYYY-MM-DD.")
    return parser.parse_args()


def maximum_id(csv_paths: list[Path]) -> int:
    maximum = 0
    for csv_path in csv_paths:
        with csv_path.open("r", newline="", encoding="utf-8") as csv_file:
            for row in csv.DictReader(csv_file):
                try:
                    maximum = max(maximum, int(row["id"]))
                except (KeyError, TypeError, ValueError):
                    continue
    return maximum


def main() -> None:
    arguments = parse_arguments()
    target = Path("outputs") / arguments.day.isoformat() / f"count_{arguments.day:%Y%m%d}.csv"
    if not target.is_file():
        raise FileNotFoundError(f"Daily CSV not found: {target}")

    all_files = sorted(Path("outputs").glob("**/count_*.csv"))
    prior_files = [path for path in all_files if path != target]
    previous_maximum = maximum_id(prior_files)
    with target.open("r", newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        rows = list(reader)
        fieldnames = reader.fieldnames
    if not fieldnames:
        raise ValueError(f"CSV has no header: {target}")
    ids = [int(row["id"]) for row in rows]
    if not ids:
        print("No rows to migrate.")
        return
    if min(ids) > previous_maximum:
        print("Already globally sequential; no CSV changes made.")
        return

    backup = target.with_suffix(".csv.before_global_ids.bak")
    shutil.copy2(target, backup)
    temporary = target.with_suffix(".csv.migrating")
    with temporary.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            row["id"] = str(int(row["id"]) + previous_maximum)
            writer.writerow(row)
    os.replace(temporary, target)
    print(f"Migrated {len(rows):,} rows: IDs {min(ids)}-{max(ids)} became " f"{min(ids) + previous_maximum}-{max(ids) + previous_maximum}.")
    print(f"Backup kept at: {backup}")


if __name__ == "__main__":
    main()
