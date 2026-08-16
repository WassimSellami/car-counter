"""Run a live, one-day-at-a-time traffic dashboard.

Start with: streamlit run live_traffic_dashboard.py
"""

from __future__ import annotations

from datetime import date, datetime, timedelta

import pandas as pd
import requests
import streamlit as st

from constants import CONTINUOUS_FLOW_INTERVAL_MINUTES, CONTINUOUS_FLOW_ROLLING_AVERAGE_MINUTES
from vehicle_counter_analysis import (
    VEHICLE_TYPE_LABELS,
    build_continuous_flow_figure,
    discover_count_csv_files,
    load_counts,
    normalise_counts,
)


REFRESH_INTERVAL_SECONDS = 300
REFRESH_INTERVAL_LABEL = "5 minutes"
TRACE_NAMES = [
    f"{vehicle_type} — {direction}"
    for vehicle_type in VEHICLE_TYPE_LABELS.values()
    if vehicle_type != "Bicycle"
    for direction in ("Into Passau", "Out of Passau")
]


def csv_files():
    return discover_count_csv_files()


def supabase_credentials() -> tuple[str, str] | None:
    try:
        return st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_SERVICE_ROLE_KEY"]
    except (FileNotFoundError, KeyError):
        return None


@st.cache_resource
def supabase_headers(service_key: str) -> dict[str, str]:
    return {"apikey": service_key, "Authorization": f"Bearer {service_key}"}


def load_local_data():
    files = csv_files()
    if not files:
        return None
    return load_counts(files)


def fetch_supabase_rows(
    url: str,
    service_key: str,
    selected_day: date,
    since_timestamp: str | None = None,
) -> list[dict]:
    """Fetch a full day once, or only its newest rows on later refreshes."""
    next_day = selected_day + timedelta(days=1)
    rows = []
    page_size = 1_000
    offset = 0
    while True:
        timestamp_filter = (
            f"gte.{since_timestamp}"
            if since_timestamp is not None
            else f"gte.{selected_day.isoformat()}T00:00:00"
        )
        response = requests.get(
            f"{url.rstrip('/')}/rest/v1/traffic_counts",
            params=[
                ("select", "record_id,timestamp,direction,vehicle_type"),
                ("timestamp", timestamp_filter),
                ("timestamp", f"lt.{next_day.isoformat()}T00:00:00"),
                ("order", "timestamp.asc,record_id.asc"),
                ("limit", page_size),
                ("offset", offset),
            ],
            headers=supabase_headers(service_key),
            timeout=15,
        )
        response.raise_for_status()
        page = response.json()
        rows.extend(page)
        if len(page) < page_size:
            break
        offset += page_size
    return rows


def load_supabase_day(selected_day: date):
    credentials = supabase_credentials()
    if credentials is None:
        return None
    url, service_key = credentials
    cache_key = f"supabase_rows_{selected_day.isoformat()}"
    cached_rows = st.session_state.get(cache_key)
    if cached_rows is None:
        rows = fetch_supabase_rows(url, service_key, selected_day)
    else:
        last_timestamp = cached_rows["timestamp"].max()
        updates = fetch_supabase_rows(url, service_key, selected_day, last_timestamp)
        rows = pd.concat([cached_rows, pd.DataFrame(updates)], ignore_index=True)
        rows = rows.drop_duplicates(subset="record_id", keep="last")
    raw_data = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    if raw_data.empty:
        return None
    st.session_state[cache_key] = raw_data
    return normalise_counts(raw_data)


def latest_day() -> date:
    credentials = supabase_credentials()
    if credentials is not None:
        url, service_key = credentials
        response = requests.get(
            f"{url.rstrip('/')}/rest/v1/traffic_counts",
            params={"select": "timestamp", "order": "timestamp.desc", "limit": 1},
            headers=supabase_headers(service_key),
            timeout=15,
        )
        response.raise_for_status()
        rows = response.json()
        if rows:
            return pd.to_datetime(rows[0]["timestamp"]).date()

    local_data = load_local_data()
    return local_data["timestamp"].dt.date.max() if local_data is not None else date.today()


def data_for_count_timeframe(
    data: pd.DataFrame, timeframe: tuple[datetime, datetime]
) -> tuple[pd.DataFrame, str]:
    """Return count rows between the two handles on the selected day."""
    start_time, end_time = timeframe
    return (
        data.loc[data["timestamp"].between(start_time, end_time)].copy(),
        f"{start_time:%H:%M}–{end_time:%H:%M}",
    )


def render_object_counts(data: pd.DataFrame, timeframe: tuple[datetime, datetime]) -> None:
    """Show the same type-and-direction totals as the camera status panel."""
    timeframe_data, timeframe_label = data_for_count_timeframe(data, timeframe)
    counts = timeframe_data.groupby(["vehicle_type", "direction"]).size()

    st.subheader("Object counts")
    st.caption(f"Counts for {timeframe_label} on the selected day.")
    header = st.columns([1.4, 1, 1])
    header[0].markdown("**Type**")
    header[1].markdown("**Out of Passau**")
    header[2].markdown("**Into Passau**")

    for vehicle_type in VEHICLE_TYPE_LABELS.values():
        row = st.columns([1.4, 1, 1])
        row[0].markdown(f"**{vehicle_type}**")
        row[1].metric(
            "Out of Passau",
            counts.get((vehicle_type, "Out of Passau"), 0),
            label_visibility="collapsed",
        )
        row[2].metric(
            "Into Passau",
            counts.get((vehicle_type, "Into Passau"), 0),
            label_visibility="collapsed",
        )


st.set_page_config(page_title="Live traffic flow", layout="wide")
st.title("Live traffic flow")
st.caption(f"The selected day refreshes every {REFRESH_INTERVAL_LABEL} while this page is open.")

selected_day = st.date_input("Day", value=latest_day())
visible_traces = st.multiselect(
    "Shown lines",
    TRACE_NAMES,
    default=TRACE_NAMES,
    help="This choice is kept while the live chart refreshes.",
)


@st.fragment(run_every=f"{REFRESH_INTERVAL_SECONDS}s")
def render_live_chart() -> None:
    data = load_supabase_day(selected_day)
    if data is None:
        data = load_local_data()
    if data is None:
        st.info("No count data found yet. Start the counter and its Supabase sync program.")
        return

    day_data = data.loc[data["timestamp"].dt.date == selected_day].copy()
    if day_data.empty:
        st.info(f"No counted vehicles for {selected_day.isoformat()} yet.")
        return

    main_vehicle_types = tuple(
        vehicle_type
        for vehicle_type in VEHICLE_TYPE_LABELS.values()
        if vehicle_type != "Bicycle"
    )
    figure = build_continuous_flow_figure(
        day_data,
        CONTINUOUS_FLOW_INTERVAL_MINUTES,
        vehicle_types=main_vehicle_types,
    )
    for trace in figure.data:
        trace.visible = trace.legendgroup in visible_traces

    figure.update_layout(
        title=(
            f"Traffic flow every {CONTINUOUS_FLOW_INTERVAL_MINUTES} minutes "
            f"({CONTINUOUS_FLOW_ROLLING_AVERAGE_MINUTES}-minute rolling average) — {selected_day:%d %b %Y}"
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
    bicycle_figure = build_continuous_flow_figure(
        day_data,
        CONTINUOUS_FLOW_INTERVAL_MINUTES,
        vehicle_types=("Bicycle",),
    )
    bicycle_figure.update_layout(
        title=(
            f"Bicycle flow every {CONTINUOUS_FLOW_INTERVAL_MINUTES} minutes "
            f"({CONTINUOUS_FLOW_ROLLING_AVERAGE_MINUTES}-minute rolling average) — {selected_day:%d %b %Y}"
        )
    )
    st.plotly_chart(
        bicycle_figure,
        key=f"bicycle-flow-{selected_day.isoformat()}",
        use_container_width=True,
        config={"responsive": True},
    )
    st.caption(
        f"{len(day_data):,} vehicles loaded; latest count: "
        f"{day_data['timestamp'].max():%d %b %Y, %H:%M:%S}. "
        f"Lines use a {CONTINUOUS_FLOW_ROLLING_AVERAGE_MINUTES}-minute rolling average. "
        "Use the Shown lines selector above to show or hide them."
    )

    st.divider()
    count_panel, _ = st.columns([1, 2])
    with count_panel:
        available_start = day_data["timestamp"].min().to_pydatetime()
        available_end = day_data["timestamp"].max().to_pydatetime()
        if available_start == available_end:
            st.caption(f"Only one count is available at {available_start:%H:%M}.")
            count_timeframe = (available_start, available_end)
        else:
            slider_key = f"object_count_timeframe_{selected_day.isoformat()}"
            if slider_key not in st.session_state:
                st.session_state[slider_key] = (available_start, available_end)
            else:
                saved_start, saved_end = st.session_state[slider_key]
                bounded_start = max(available_start, min(saved_start, available_end))
                bounded_end = max(available_start, min(saved_end, available_end))
                st.session_state[slider_key] = (
                    (available_start, available_end)
                    if bounded_start > bounded_end
                    else (bounded_start, bounded_end)
                )
            count_timeframe = st.slider(
                "Object-count timeframe",
                min_value=available_start,
                max_value=available_end,
                step=timedelta(minutes=15),
                format="HH:mm",
                help="Drag the two handles to choose the start and end of the available count data.",
                key=slider_key,
            )
        render_object_counts(day_data, count_timeframe)


render_live_chart()
