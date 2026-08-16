"""Create an interactive traffic-flow report from counter CSV files.

Examples:
    python vehicle_counter_analysis.py
    python vehicle_counter_analysis.py outputs/2026-08-11/car_counts_*.csv
    python vehicle_counter_analysis.py --output-dir outputs/analysis
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from constants import CONTINUOUS_FLOW_INTERVAL_MINUTES, CONTINUOUS_FLOW_ROLLING_AVERAGE_MINUTES


DIRECTION_LABELS = {0: "Out of Passau", 1: "Into Passau"}
VEHICLE_TYPE_LABELS = {0: "Car / van", 1: "Truck", 2: "Bus", 3: "Bicycle"}
VEHICLE_COLORS = {"Car / van": "#4C78A8", "Truck": "#F58518", "Bus": "#54A24B", "Bicycle": "#E45756"}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "csv_files",
        nargs="*",
        type=Path,
        help="CSV files to analyse. Defaults to every current and legacy counter CSV under outputs/.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs") / "analysis",
        help="Directory for the standalone HTML report (default: outputs/analysis).",
    )
    parser.add_argument(
        "--interval-minutes",
        type=int,
        default=CONTINUOUS_FLOW_INTERVAL_MINUTES,
        help=(
            "Width of each continuous time-series bucket in minutes "
            f"(default: {CONTINUOUS_FLOW_INTERVAL_MINUTES})."
        ),
    )
    return parser.parse_args()


def discover_count_csv_files() -> list[Path]:
    """Find one source per day, preferring the new daily CSV over legacy runs."""
    output_root = Path("outputs")
    daily_files = sorted(output_root.glob("**/count_*.csv"))
    daily_directories = {csv_path.parent for csv_path in daily_files}
    legacy_files = sorted(
        csv_path
        for csv_path in output_root.glob("**/car_counts_*.csv")
        if csv_path.parent not in daily_directories
    )
    return daily_files + legacy_files


def load_counts(csv_files: list[Path]) -> pd.DataFrame:
    frames = []
    for csv_file in csv_files:
        frame = pd.read_csv(csv_file, usecols=["timestamp", "direction", "vehicle_type"])
        frames.append(frame)

    if not frames:
        raise ValueError("No CSV files were supplied.")

    return normalise_counts(pd.concat(frames, ignore_index=True))


def normalise_counts(data: pd.DataFrame) -> pd.DataFrame:
    """Validate and label count rows loaded from a CSV or Supabase."""
    data = data.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce")
    data["direction"] = pd.to_numeric(data["direction"], errors="coerce").map(DIRECTION_LABELS)
    data["vehicle_type"] = pd.to_numeric(data["vehicle_type"], errors="coerce").map(VEHICLE_TYPE_LABELS)
    data = data.dropna(subset=["timestamp", "direction", "vehicle_type"]).copy()
    if data.empty:
        raise ValueError("The CSV files contain no usable timestamp, direction, and vehicle_type rows.")
    return data


def build_continuous_flow_figure(
    data: pd.DataFrame,
    interval_minutes: int,
    vehicle_types: tuple[str, ...] | None = None,
):
    """Build a Plotly figure with one togglable trace per vehicle/direction."""
    if interval_minutes < 1:
        raise ValueError("--interval-minutes must be at least 1.")
    frequency = f"{interval_minutes}min"
    buckets = pd.date_range(
        data["timestamp"].min().floor(frequency),
        data["timestamp"].max().floor(frequency),
        freq=frequency,
    )
    grouped = data.groupby([pd.Grouper(key="timestamp", freq=frequency), "vehicle_type", "direction"]).size()

    import plotly.graph_objects as go

    figure = go.Figure()

    displayed_vehicle_types = vehicle_types or tuple(VEHICLE_TYPE_LABELS.values())
    for vehicle_type in displayed_vehicle_types:
        for direction, line_style in (("Into Passau", "-"), ("Out of Passau", "--")):
            series = grouped.reindex(
                pd.MultiIndex.from_product(
                    [buckets, [vehicle_type], [direction]],
                    names=["timestamp", "vehicle_type", "direction"],
                ),
                fill_value=0,
            )
            values = series.to_numpy(dtype=float)
            hourly_values = (
                pd.Series(values, index=buckets)
                .rolling(
                    window=f"{CONTINUOUS_FLOW_ROLLING_AVERAGE_MINUTES}min",
                    center=True,
                    min_periods=1,
                )
                .mean()
                .to_numpy()
                * 60
                / interval_minutes
            )
            trace_name = f"{vehicle_type} — {direction}"

            figure.add_trace(
                go.Scatter(
                    x=buckets,
                    y=hourly_values,
                    mode="lines",
                    name=trace_name,
                    legendgroup=trace_name,
                    line={
                        "color": VEHICLE_COLORS[vehicle_type],
                        "dash": "solid" if line_style == "-" else "dash",
                        "width": 3,
                        "shape": "spline",
                        "smoothing": 1.2,
                    },
                    hovertemplate=f"{trace_name}: %{{y:.1f}}/hour<extra></extra>",
                    customdata=[trace_name] * len(buckets),
                )
            )

    figure.update_layout(
        title=(
            f"Traffic flow every {interval_minutes} minutes "
            f"({CONTINUOUS_FLOW_ROLLING_AVERAGE_MINUTES}-minute rolling average)"
        ),
        template="plotly_white",
        hovermode="x unified",
        legend={"title": "Click a series to show or hide it"},
        xaxis={"title": "Time", "hoverformat": "%d %b %Y, %H:%M"},
        yaxis={"title": "Vehicles per hour", "rangemode": "tozero"},
        margin={"l": 70, "r": 30, "t": 80, "b": 70},
    )
    return figure


def save_continuous_flow_chart(data: pd.DataFrame, output_dir: Path, interval_minutes: int) -> Path:
    """Write a self-contained interactive Plotly report for one calendar day."""
    figure = build_continuous_flow_figure(data, interval_minutes)
    day_key = data["timestamp"].dt.strftime("%Y-%m-%d").iloc[0]
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"traffic_flow_{day_key}.html"
    figure.write_html(output_path, include_plotlyjs=True, full_html=True)
    return output_path


def main() -> None:
    arguments = parse_arguments()
    csv_files = arguments.csv_files or discover_count_csv_files()
    missing = [path for path in csv_files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"CSV file not found: {missing[0]}")

    data = load_counts(csv_files)
    saved_charts = [
        save_continuous_flow_chart(
            day_data,
            arguments.output_dir,
            arguments.interval_minutes,
        )
        for _, day_data in data.groupby(data["timestamp"].dt.date, sort=True)
    ]
    print(f"Analysed {len(data):,} counted vehicles from {len(csv_files)} file(s).")
    print(f"Saved {len(saved_charts)} standalone daily report(s) under: {arguments.output_dir.resolve()}")


if __name__ == "__main__":
    main()
