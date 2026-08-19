"""Vast Streamlit dashboard: live stream plus Supabase-backed history."""

import os
from datetime import date, timedelta

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import streamlit.components.v1 as components

from constants import CONTINUOUS_FLOW_INTERVAL_MINUTES, CONTINUOUS_FLOW_ROLLING_AVERAGE_MINUTES
from vehicle_counter_analysis import VEHICLE_TYPE_LABELS, build_continuous_flow_figure, normalise_counts

COUNTER_URL = os.environ.get("COUNTER_INTERNAL_URL", "http://127.0.0.1:8000").rstrip("/")
PUBLIC_COUNTER_URL = os.environ.get("COUNTER_PUBLIC_URL", COUNTER_URL).rstrip("/")
API_KEY = os.environ["CLOUD_INFERENCE_API_KEY"]
SUPABASE_URL = os.environ["SUPABASE_URL"].rstrip("/")
SUPABASE_SERVICE_ROLE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
COLOR_NAMES = {1: "black", 2: "white", 3: "grey", 4: "silver", 5: "red", 6: "blue", 7: "green", 8: "yellow", 9: "orange", 10: "brown"}
COLOR_HEX = {"black": "#242424", "white": "#f3f3f3", "grey": "#808080", "silver": "#c0c0c0", "red": "#e53935", "blue": "#2d82d7", "green": "#41a85f", "yellow": "#f5dc28", "orange": "#ff8c00", "brown": "#874b2a"}
TRACE_NAMES = [f"{vehicle_type} — {direction}" for vehicle_type in VEHICLE_TYPE_LABELS.values() if vehicle_type != "Bicycle" for direction in ("Into Passau", "Out of Passau")]

def supabase_headers() -> dict[str, str]:
    return {"apikey": SUPABASE_SERVICE_ROLE_KEY, "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}"}


def status() -> dict | None:
    try:
        return requests.get(f"{COUNTER_URL}/status", headers={"X-API-Key": API_KEY}, timeout=2).json()
    except requests.RequestException:
        return None


@st.cache_data(ttl=10, show_spinner=False)
def load_cloud_counts(selected_day: date) -> pd.DataFrame | None:
    next_day = selected_day + timedelta(days=1)
    rows, offset = [], 0
    while True:
        response = requests.get(
            f"{SUPABASE_URL}/rest/v1/traffic_counts",
            params=[("select", "record_id,timestamp,direction,vehicle_type,color"), ("timestamp", f"gte.{selected_day.isoformat()}T00:00:00"), ("timestamp", f"lt.{next_day.isoformat()}T00:00:00"), ("order", "timestamp.asc,record_id.asc"), ("limit", 1000), ("offset", offset)],
            headers=supabase_headers(), timeout=15,
        )
        response.raise_for_status()
        page = response.json()
        rows.extend(page)
        if len(page) < 1000:
            break
        offset += 1000
    return normalise_counts(pd.DataFrame(rows)) if rows else None


@st.cache_data(ttl=30, show_spinner=False)
def latest_cloud_day() -> date:
    response = requests.get(f"{SUPABASE_URL}/rest/v1/traffic_counts", params={"select": "timestamp", "order": "timestamp.desc", "limit": 1}, headers=supabase_headers(), timeout=15)
    response.raise_for_status()
    rows = response.json()
    return pd.to_datetime(rows[0]["timestamp"]).date() if rows else date.today()


def render_counts(data: pd.DataFrame | None) -> None:
    count_panel, color_panel = st.columns([2, 1])
    with count_panel:
        st.subheader("Object counts")
        header = st.columns([1.4, 1, 1])
        header[0].markdown("**Type**")
        header[1].markdown("**Out of Passau**")
        header[2].markdown("**Into Passau**")
        counts = data.groupby(["vehicle_type", "direction"]).size() if data is not None else pd.Series(dtype=int)
        for vehicle_type in VEHICLE_TYPE_LABELS.values():
            row = st.columns([1.4, 1, 1])
            row[0].markdown(f"**{vehicle_type}**")
            row[1].metric("Out of Passau", counts.get((vehicle_type, "Out of Passau"), 0), label_visibility="collapsed")
            row[2].metric("Into Passau", counts.get((vehicle_type, "Into Passau"), 0), label_visibility="collapsed")
    with color_panel:
        st.subheader("Car colours")
        colors = {}
        if data is not None and "color" in data:
            colors = data.loc[data["vehicle_type"] == "Car / van", "color"].pipe(pd.to_numeric, errors="coerce").map(COLOR_NAMES).value_counts().to_dict()
        if not colors:
            st.caption("No classified cars yet.")
            return
        figure = go.Figure(go.Pie(labels=[name.title() for name in colors], values=list(colors.values()), marker={"colors": [COLOR_HEX.get(name, "#808080") for name in colors]}, hole=0.5, textinfo="none", hovertemplate="%{label}: %{percent}<extra></extra>"))
        figure.update_layout(showlegend=False, margin={"l": 0, "r": 0, "t": 0, "b": 0}, height=260)
        st.plotly_chart(figure, width="stretch", config={"responsive": True})


st.set_page_config(page_title="Cloud vehicle counter", layout="wide")
st.title("Live traffic flow")
st.caption("Live video on Vast. Counts, colours, and archive history from Supabase.")
selected_day = st.date_input("Day", value=latest_cloud_day())
visible_traces = st.multiselect("Shown lines", TRACE_NAMES, default=TRACE_NAMES)
st.subheader("Live camera")
components.html(f'<img src="{PUBLIC_COUNTER_URL}/stream.mjpeg?key={API_KEY}" style="width:100%;height:auto;display:block">', height=600)


@st.fragment(run_every="1s")
def render_status() -> None:
    data = status()
    if data is None:
        st.warning("Waiting for the Vast counter service.")
    else:
        metrics = data["metrics"]
        other = max(0, metrics["cloud_ms"] - metrics["yolo_ms"] - metrics["clip_ms"])
        cards = st.columns(6)
        for card, label, value in zip(cards, ("Cloud", "YOLO", "CLIP", "Other cloud", "Process", "Incoming"), (f"{metrics['cloud_ms']:.0f} ms", f"{metrics['yolo_ms']:.0f} ms", f"{metrics['clip_ms']:.0f} ms", f"{other:.0f} ms", f"{metrics['process_fps']:.1f} FPS", f"{metrics['input_fps']:.1f} FPS")):
            card.metric(label, value)
    render_counts(load_cloud_counts(selected_day))


render_status()
st.divider()


@st.fragment(run_every="5s")
def render_flow_charts() -> None:
    data = load_cloud_counts(selected_day)
    if data is None:
        st.info("No Supabase history is available for this day.")
        return
    main_types = tuple(vehicle_type for vehicle_type in VEHICLE_TYPE_LABELS.values() if vehicle_type != "Bicycle")
    figure = build_continuous_flow_figure(data, CONTINUOUS_FLOW_INTERVAL_MINUTES, vehicle_types=main_types)
    for trace in figure.data:
        trace.visible = trace.legendgroup in visible_traces
    figure.update_layout(title=f"Traffic flow every {CONTINUOUS_FLOW_INTERVAL_MINUTES} minutes ({CONTINUOUS_FLOW_ROLLING_AVERAGE_MINUTES}-minute rolling average) - {selected_day:%d %b %Y}", legend={"itemclick": False, "itemdoubleclick": False})
    st.plotly_chart(figure, width="stretch", config={"responsive": True})
    bicycle_figure = build_continuous_flow_figure(data, CONTINUOUS_FLOW_INTERVAL_MINUTES, vehicle_types=("Bicycle",))
    bicycle_figure.update_layout(title=f"Bicycle flow every {CONTINUOUS_FLOW_INTERVAL_MINUTES} minutes ({CONTINUOUS_FLOW_ROLLING_AVERAGE_MINUTES}-minute rolling average) - {selected_day:%d %b %Y}")
    st.plotly_chart(bicycle_figure, width="stretch", config={"responsive": True})


render_flow_charts()
