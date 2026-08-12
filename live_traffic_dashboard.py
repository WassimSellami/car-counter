"""Run a live, one-day-at-a-time traffic dashboard.

Start with: streamlit run live_traffic_dashboard.py
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import streamlit as st

from constants import CONTINUOUS_FLOW_INTERVAL_MINUTES
from vehicle_counter_analysis import (
    VEHICLE_TYPE_LABELS,
    build_continuous_flow_figure,
    load_counts,
)


REFRESH_INTERVAL_SECONDS = 5
TRACE_NAMES = [
    f"{vehicle_type} — {direction}"
    for vehicle_type in VEHICLE_TYPE_LABELS.values()
    for direction in ("Into Passau", "Out of Passau")
]


def csv_files() -> list[Path]:
    return sorted(Path("outputs").glob("**/car_counts_*.csv"))


def load_dashboard_data():
    files = csv_files()
    if not files:
        return None
    return load_counts(files)


st.set_page_config(page_title="Live traffic flow", layout="wide")
st.title("Live traffic flow")
st.caption(f"The selected day refreshes every {REFRESH_INTERVAL_SECONDS} seconds while this page is open.")

initial_data = load_dashboard_data()
latest_day = initial_data["timestamp"].dt.date.max() if initial_data is not None else date.today()
selected_day = st.date_input("Day", value=latest_day)
visible_traces = st.multiselect(
    "Shown lines",
    TRACE_NAMES,
    default=TRACE_NAMES,
    help="This choice is kept while the live chart refreshes.",
)


@st.fragment(run_every=f"{REFRESH_INTERVAL_SECONDS}s")
def render_live_chart() -> None:
    data = load_dashboard_data()
    if data is None:
        st.info("No counter CSV files found yet. Start car_counter.py to begin collecting data.")
        return

    day_data = data.loc[data["timestamp"].dt.date == selected_day].copy()
    if day_data.empty:
        st.info(f"No counted vehicles for {selected_day.isoformat()} yet.")
        return

    figure = build_continuous_flow_figure(day_data, CONTINUOUS_FLOW_INTERVAL_MINUTES)
    for trace in figure.data:
        trace.visible = trace.legendgroup in visible_traces

    figure.update_layout(
        title=(
            f"Traffic flow every {CONTINUOUS_FLOW_INTERVAL_MINUTES} minutes "
            f"(15-minute rolling average) — {selected_day:%d %b %Y}"
        ),
        # Streamlit reruns cannot receive Plotly legend clicks from the browser.
        # The persistent "Shown lines" selector above owns trace visibility instead.
        legend={"itemclick": False, "itemdoubleclick": False},
    )
    st.plotly_chart(
        figure,
        key=f"traffic-flow-{selected_day.isoformat()}",
        use_container_width=True,
        config={"responsive": True},
    )
    st.caption(
        f"{len(day_data):,} vehicles loaded; latest count: "
        f"{day_data['timestamp'].max():%d %b %Y, %H:%M:%S}. "
        "Bold lines are 15-minute rolling averages; faint lines are raw five-minute counts. "
        "Use the Shown lines selector above to show or hide both."
    )


render_live_chart()
