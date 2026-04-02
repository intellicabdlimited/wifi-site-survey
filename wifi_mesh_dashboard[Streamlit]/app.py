from __future__ import annotations

from pathlib import Path

import streamlit as st

from utils import charts, styles
from utils.data_loader import (
    METRIC_LABELS,
    available_filters,
    build_dashboard_payload,
    count_files,
    load_records,
    threshold_defaults,
)

DATA_ROOT = Path(__file__).parent / "data" / "compare_inputs"


def _fmt_metric(value: float, unit: str) -> str:
    if value != value:
        return "—"
    return f"{value:.1f} {unit}"


def _fmt_delta(value: float, unit: str) -> str | None:
    if value != value:
        return None
    return f"{value:+.1f} {unit} vs no mesh"


def _fmt_percent(value: float) -> str:
    if value != value:
        return "—"
    return f"{value:+.1f}%"


def _default_compare_routers(router_options: list[str], selected_router: str, limit: int = 4) -> list[str]:
    picked: list[str] = []
    if selected_router in router_options:
        picked.append(selected_router)

    for router in router_options:
        if router not in picked:
            picked.append(router)
        if len(picked) >= limit:
            break

    return picked


st.set_page_config(
    page_title="WiFi Mesh Analytics",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)
styles.inject()

records = load_records(str(DATA_ROOT))

if records.empty:
    styles.hero(
        "WiFi Mesh Performance Dashboard",
        "No CSV files were found yet. Put your pre-stored exports in data/compare_inputs before running the app.",
        [
            styles.badge("with_mesh / without_mesh", "purple"),
            styles.badge("throughput / signal_strength", "blue"),
            styles.badge("router folders", "green"),
        ],
    )
    st.markdown(
        """
        <div class="empty-card">
            Expected folder pattern:<br><br>
            <span class="code-note">data/compare_inputs/with_mesh/throughput/TMO-G5AR/*.csv</span><br>
            <span class="code-note">data/compare_inputs/without_mesh/throughput/TMO-G5AR/*.csv</span><br>
            <span class="code-note">data/compare_inputs/with_mesh/signal_strength/TMO-G5AR/*.csv</span><br>
            <span class="code-note">data/compare_inputs/without_mesh/signal_strength/TMO-G5AR/*.csv</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

filters = available_filters(records)
metric_options = filters.get("metrics", [])
if not metric_options:
    st.error("No metric folders were found in the loaded records.")
    st.stop()

default_metric = metric_options[0]

with st.sidebar:
    st.markdown("### 📡 Dashboard controls")
    st.caption("One page · pre-stored CSV inputs")

    selected_metric = st.selectbox(
        "Metric",
        metric_options,
        format_func=lambda x: METRIC_LABELS.get(x, x.replace("_", " ").title()),
        index=metric_options.index(default_metric) if default_metric in metric_options else 0,
        key="metric_select",
    )

    metric_records = records.loc[records["metric"] == selected_metric].copy()

    floor_options = sorted(
        metric_records["floor"].dropna().unique().tolist(),
        key=lambda x: {"Ground Floor": 0, "Lower Floor": 1, "Upper Floor": 2}.get(x, 99),
    )
    band_options = sorted(
        metric_records["band"].dropna().unique().tolist(),
        key=lambda x: {"2.4 GHz": 0, "5 GHz": 1}.get(x, 99),
    )
    router_options = sorted(
        metric_records["router"].dropna().unique().tolist(),
        key=str.lower,
    )

    if not floor_options or not band_options or not router_options:
        st.error("The selected metric does not have enough floor, band, or router data to render the dashboard.")
        st.stop()

    selected_floor = st.radio(
        "Floor",
        floor_options,
        horizontal=True,
        key="floor_radio",
    )

    selected_band = st.radio(
        "Band",
        band_options,
        horizontal=True,
        key="band_radio",
    )

    selected_routers = st.multiselect(
        "Routers",
        router_options,
        default=router_options[: min(3, len(router_options))],
        help="Choose 1 or more routers for the comparison charts.",
        key="routers_multiselect",
    )

    if not selected_routers:
        selected_routers = [router_options[0]]

    t_min, t_max, t_default, t_step, unit = threshold_defaults(selected_metric)
    threshold = st.slider(
        f"Coverage threshold ({unit})",
        min_value=float(t_min),
        max_value=float(t_max),
        value=float(t_default),
        step=float(t_step),
        key="threshold_slider",
    )

    st.markdown("---")
    st.caption(f"Loaded CSV files: {count_files(records)}")
    st.caption(f"Data root: {DATA_ROOT}")

metric_title = METRIC_LABELS.get(selected_metric, selected_metric.replace("_", " ").title())

data, kpis, summary_df = build_dashboard_payload(
    records,
    metric=selected_metric,
    floor=selected_floor,
    band=selected_band,
    threshold=threshold,
)

if not data:
    st.warning("No matching records were found for the selected metric, floor, and band.")
    st.stop()

data_view = {router: data[router] for router in selected_routers if router in data}
kpis_view = {router: kpis[router] for router in selected_routers if router in kpis}
summary_view = summary_df[summary_df["router"].isin(selected_routers)].copy()

if not data_view or not kpis_view or summary_view.empty:
    st.warning("No comparison data is available for the selected routers.")
    st.stop()

styles.hero(
    "WiFi Mesh Performance Dashboard",
    "Premium one-page comparison dashboard built from pre-stored mesh and non-mesh exports.",
    [
        styles.badge(metric_title, "purple"),
        styles.badge(selected_floor, "blue"),
        styles.badge(selected_band, "green"),
        styles.badge(f"Threshold {threshold:g} {unit}", "orange"),
        styles.badge(f"{len(selected_routers)} router(s) selected", "purple"),
    ],
)

avg_mesh_all = summary_view["avg_mesh"].mean()
avg_nomesh_all = summary_view["avg_nomesh"].mean()
peak_mesh_all = summary_view["peak_mesh"].max()
gain_abs_all = summary_view["gain_abs"].mean()
gain_pct_all = summary_view["gain_pct"].mean()
cov_mesh_all = summary_view["coverage_mesh"].mean()
cov_nomesh_all = summary_view["coverage_nomesh"].mean()

kpi_cols = st.columns(5)
kpi_cols[0].metric(
    "Selected avg with mesh",
    _fmt_metric(avg_mesh_all, unit),
    _fmt_delta(gain_abs_all, unit),
)
kpi_cols[1].metric(
    "Selected avg without mesh",
    _fmt_metric(avg_nomesh_all, unit),
)
kpi_cols[2].metric(
    "Best peak with mesh",
    _fmt_metric(peak_mesh_all, unit),
)
kpi_cols[3].metric(
    "Avg mesh gain %",
    _fmt_percent(gain_pct_all),
)
kpi_cols[4].metric(
    "Avg coverage bands",
    f"{cov_mesh_all:.1f}/8",
    f"No mesh: {cov_nomesh_all:.1f}/8",
)
styles.section(
    "Throughput analysis" if selected_metric == "throughput" else "Metric analysis",
    "Focus-router distance story plus all-router average comparison",
)
row1_col1, row1_col2 = st.columns([3, 2])

with row1_col1:
    st.markdown("#### Router comparison vs distance")
    st.plotly_chart(
        charts.line_chart(data_view, selected_routers, unit, threshold=threshold),
        width="stretch",
        config={"displayModeBar": False, "scrollZoom": False},
        key="line_chart",
    )

with row1_col2:
    st.markdown("#### Average comparison by router")
    st.plotly_chart(
        charts.grouped_bar(summary_view, unit),
        width="stretch",
        config={"displayModeBar": False, "scrollZoom": False},
        key="grouped_bar",
    )

styles.section(
    "Multi-router comparison",
    "Radar, coverage comparison, and drop-off comparison for the selected routers",
)
row2_col1, row2_col2, row2_col3 = st.columns(3)

with row2_col1:
    st.plotly_chart(
        charts.radar_chart(kpis_view),
        width="stretch",
        config={"displayModeBar": False, "scrollZoom": False},
        key="radar_chart",
    )

with row2_col2:
    st.plotly_chart(
        charts.coverage_compare_bar(summary_view, unit),
        width="stretch",
        config={"displayModeBar": False, "scrollZoom": False},
        key="coverage_compare",
    )

with row2_col3:
    st.plotly_chart(
        charts.drop_off_compare(summary_view, unit),
        width="stretch",
        config={"displayModeBar": False, "scrollZoom": False},
        key="dropoff_compare",
    )

styles.section("Coverage heatmap", "Distance-band view across the selected routers")
row3_col1, row3_col2 = st.columns(2)

with row3_col1:
    st.plotly_chart(
        charts.heatmap_chart(data_view, topology="mesh"),
        width="stretch",
        config={"displayModeBar": False, "scrollZoom": False},
        key="heatmap_mesh",
    )

with row3_col2:
    st.plotly_chart(
        charts.heatmap_chart(data_view, topology="nomesh"),
        width="stretch",
        config={"displayModeBar": False, "scrollZoom": False},
        key="heatmap_nomesh",
    )

styles.section(
    "Mesh improvement ranking",
    "Best to worst absolute gain for the selected metric, floor, and band",
)
st.plotly_chart(
    charts.horizontal_gain_bar(summary_view, unit),
    width="stretch",
    config={"displayModeBar": False, "scrollZoom": False},
    key="gain_bar",
)

with st.expander("Current filter summary table"):
    show_df = summary_view.copy()
    numeric_cols = [col for col in show_df.columns if col != "router"]
    for col in numeric_cols:
        show_df[col] = show_df[col].round(2)
    st.dataframe(show_df, width="stretch")

st.caption(
    "Pre-stored input flow: compare_inputs → topology folders → metric folders → router folders → dashboard charts"
)