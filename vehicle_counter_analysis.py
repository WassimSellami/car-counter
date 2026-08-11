"""Create direction- and vehicle-type traffic charts from counter CSV files.

Examples:
    python vehicle_counter_analysis.py
    python vehicle_counter_analysis.py outputs/2026-08-11/car_counts_*.csv
    python vehicle_counter_analysis.py --output-dir outputs/analysis
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

# The script writes PNGs only; avoiding a GUI backend also makes it work on
# systems where Tk is not installed.
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


DIRECTION_LABELS = {0: "Left", 1: "Right"}
VEHICLE_TYPE_LABELS = {0: "Car / van", 1: "Truck", 2: "Bus", 3: "Bicycle"}
DIRECTION_COLORS = {"Left": "#4C78A8", "Right": "#F58518"}
VEHICLE_COLORS = {"Car / van": "#4C78A8", "Truck": "#F58518", "Bus": "#54A24B", "Bicycle": "#E45756"}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "csv_files",
        nargs="*",
        type=Path,
        help="CSV files to analyse. Defaults to every outputs/**/car_counts_*.csv file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs") / "analysis",
        help="Directory for PNG charts and summary.csv (default: outputs/analysis).",
    )
    parser.add_argument(
        "--interval-minutes",
        type=int,
        default=30,
        help="Width of each continuous time-series bucket in minutes (default: 30).",
    )
    return parser.parse_args()


def load_counts(csv_files: list[Path]) -> pd.DataFrame:
    frames = []
    for csv_file in csv_files:
        frame = pd.read_csv(csv_file, usecols=["timestamp", "direction", "vehicle_type"])
        frames.append(frame)

    if not frames:
        raise ValueError("No CSV files were supplied.")

    data = pd.concat(frames, ignore_index=True)
    data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce")
    data["direction"] = pd.to_numeric(data["direction"], errors="coerce").map(DIRECTION_LABELS)
    data["vehicle_type"] = pd.to_numeric(data["vehicle_type"], errors="coerce").map(VEHICLE_TYPE_LABELS)
    data = data.dropna(subset=["timestamp", "direction", "vehicle_type"]).copy()
    if data.empty:
        raise ValueError("The CSV files contain no usable timestamp, direction, and vehicle_type rows.")
    data["hour"] = data["timestamp"].dt.hour
    return data


def save_hourly_direction_chart(data: pd.DataFrame, output_dir: Path) -> None:
    hourly = (
        data.groupby(["hour", "direction"]).size().unstack(fill_value=0)
        .reindex(index=range(24), columns=DIRECTION_LABELS.values(), fill_value=0)
    )
    axis = hourly.plot.bar(
        stacked=True,
        color=[DIRECTION_COLORS[direction] for direction in hourly.columns],
        figsize=(12, 6),
        width=0.85,
    )
    axis.set(title="Hourly traffic flow by direction", xlabel="Hour of day", ylabel="Vehicles")
    axis.set_xticklabels(range(24), rotation=0)
    axis.legend(title="Direction")
    axis.figure.tight_layout()
    axis.figure.savefig(output_dir / "hourly_direction_flow.png", dpi=180)
    plt.close(axis.figure)


def save_continuous_flow_chart(data: pd.DataFrame, output_dir: Path, interval_minutes: int) -> None:
    """Plot one vehicle category colour with a distinct line per direction."""
    if interval_minutes < 1:
        raise ValueError("--interval-minutes must be at least 1.")

    frequency = f"{interval_minutes}min"
    buckets = pd.date_range(
        data["timestamp"].min().floor(frequency),
        data["timestamp"].max().ceil(frequency),
        freq=frequency,
    )
    grouped = data.groupby([pd.Grouper(key="timestamp", freq=frequency), "vehicle_type", "direction"]).size()

    figure, axis = plt.subplots(figsize=(15, 6))
    for vehicle_type in VEHICLE_TYPE_LABELS.values():
        for direction, line_style in (("Right", "-"), ("Left", "--")):
            series = grouped.reindex(
                pd.MultiIndex.from_product(
                    [buckets, [vehicle_type], [direction]],
                    names=["timestamp", "vehicle_type", "direction"],
                ),
                fill_value=0,
            )
            values = series.to_numpy(dtype=float)
            if len(values) > 2:
                smoothed = (
                    pd.Series(values)
                    .rolling(window=3, center=True, min_periods=1)
                    .mean()
                    .to_numpy()
                )
            else:
                smoothed = values

            axis.plot(
                buckets,
                smoothed,
                color=VEHICLE_COLORS[vehicle_type],
                linestyle=line_style,
                linewidth=2.0,
                alpha=0.95,
                label=f"{vehicle_type} — {direction}",
            )

    axis.set(
        title=f"Traffic flow every {interval_minutes} minutes",
        xlabel="Time",
        ylabel="Counted vehicles per interval",
    )
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%d %b\n%H:%M"))
    axis.legend(title="Category — direction", ncols=2, loc="upper left")
    figure.autofmt_xdate()
    figure.tight_layout()
    figure.savefig(output_dir / f"continuous_flow_{interval_minutes}_minutes.png", dpi=180)
    plt.close(figure)


def save_hourly_vehicle_chart(data: pd.DataFrame, output_dir: Path) -> None:
    hourly = (
        data.groupby(["hour", "vehicle_type"]).size().unstack(fill_value=0)
        .reindex(index=range(24), columns=VEHICLE_TYPE_LABELS.values(), fill_value=0)
    )
    axis = hourly.plot.bar(
        stacked=True,
        color=[VEHICLE_COLORS[vehicle] for vehicle in hourly.columns],
        figsize=(12, 6),
        width=0.85,
    )
    axis.set(title="Vehicle types by hour", xlabel="Hour of day", ylabel="Vehicles")
    axis.set_xticklabels(range(24), rotation=0)
    axis.legend(title="Vehicle type")
    axis.figure.tight_layout()
    axis.figure.savefig(output_dir / "hourly_vehicle_types.png", dpi=180)
    plt.close(axis.figure)


def save_direction_vehicle_heatmap(data: pd.DataFrame, output_dir: Path) -> None:
    matrix = (
        data.groupby(["direction", "vehicle_type"]).size().unstack(fill_value=0)
        .reindex(index=DIRECTION_LABELS.values(), columns=VEHICLE_TYPE_LABELS.values(), fill_value=0)
    )
    figure, axis = plt.subplots(figsize=(9, 4.5))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", linewidths=0.5, ax=axis)
    axis.set(title="Vehicle type by travel direction", xlabel="Vehicle type", ylabel="Direction")
    figure.tight_layout()
    figure.savefig(output_dir / "direction_vehicle_matrix.png", dpi=180)
    plt.close(figure)


def save_direction_pies(data: pd.DataFrame, output_dir: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11, 5))
    for axis, direction in zip(axes, DIRECTION_LABELS.values()):
        counts = data.loc[data["direction"] == direction, "vehicle_type"].value_counts()
        counts = counts.reindex(VEHICLE_TYPE_LABELS.values(), fill_value=0)
        nonzero = counts[counts > 0]
        if nonzero.empty:
            axis.text(0.5, 0.5, "No counts", ha="center", va="center")
        else:
            axis.pie(
                nonzero,
                labels=nonzero.index,
                autopct="%1.1f%%",
                startangle=90,
                colors=[VEHICLE_COLORS[vehicle] for vehicle in nonzero.index],
            )
        axis.set_title(f"{direction}-bound vehicles")
    figure.suptitle("Vehicle mix by direction")
    figure.tight_layout()
    figure.savefig(output_dir / "vehicle_mix_by_direction.png", dpi=180)
    plt.close(figure)


def save_summary(data: pd.DataFrame, output_dir: Path) -> None:
    summary = data.groupby(["direction", "vehicle_type"]).size().rename("vehicle_count").reset_index()
    summary.to_csv(output_dir / "summary.csv", index=False)


def main() -> None:
    arguments = parse_arguments()
    csv_files = arguments.csv_files or sorted(Path("outputs").glob("**/car_counts_*.csv"))
    missing = [path for path in csv_files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"CSV file not found: {missing[0]}")

    data = load_counts(csv_files)
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    save_continuous_flow_chart(data, arguments.output_dir, arguments.interval_minutes)
    save_hourly_direction_chart(data, arguments.output_dir)
    save_hourly_vehicle_chart(data, arguments.output_dir)
    save_direction_vehicle_heatmap(data, arguments.output_dir)
    save_direction_pies(data, arguments.output_dir)
    save_summary(data, arguments.output_dir)
    print(f"Analysed {len(data):,} counted vehicles from {len(csv_files)} file(s).")
    print(f"Charts saved to: {arguments.output_dir.resolve()}")


if __name__ == "__main__":
    main()
