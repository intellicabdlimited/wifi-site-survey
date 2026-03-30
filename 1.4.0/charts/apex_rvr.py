from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components


def _prepend_zero_point(xs: List[float], ys: List[float]) -> Tuple[List[float], List[float]]:
    if not xs or not ys or len(xs) != len(ys):
        return xs, ys

    paired = sorted((float(x), float(y)) for x, y in zip(xs, ys) if pd.notna(x) and pd.notna(y))
    if not paired:
        return [], []

    xs = [p[0] for p in paired]
    ys = [p[1] for p in paired]

    # Keep a single real point as-is.
    # Do not invent a fake point at x=0.
    if len(xs) < 2:
        return xs, ys

    if xs[0] <= 1e-9:
        return xs, ys

    if abs(xs[1] - xs[0]) > 1e-9:
        y0 = ys[0] + (0.0 - xs[0]) * (ys[1] - ys[0]) / (xs[1] - xs[0])
        y0 = max(min(y0, max(ys)), min(ys))
    else:
        y0 = ys[0]

    return [0.0] + xs, [float(y0)] + ys


def _safe_text(value: object) -> str:
    text = "" if value is None else str(value)
    return html.escape(text)

def _pretty_metric_name(metric_key: str) -> str:
    metric_map = {
        "signal_strength": "Signal Strength",
        "secondary_signal_strength": "Secondary Signal Strength",
        "tertiary_signal_strength": "Tertiary Signal Strength",
        "snr": "SNR",
        "noise": "Noise",
        "throughput": "Throughput",
        "data_rate": "Data Rate",
        "channel_utilization": "Channel Utilization",
        "channel_interference": "Channel Interference",
        "channel_width": "Channel Width",
        "spectrum_channel_power": "Spectrum Channel Power",
        "network_health": "Network Health",
        "network_issues": "Network Issues",
        "number_of_access_points": "Number of Access Points",
        "number_of_aps": "Number of APs",
    }
    return metric_map.get(metric_key, metric_key.replace("_", " ").title())


def _pretty_value_name(value_key: str) -> str:
    value_map = {
        "p50": "Median (p50)",
        "mean": "Average",
        "p10": "Lower Bound (p10)",
        "p90": "Upper Bound (p90)",
        "min": "Minimum",
        "max": "Maximum",
    }
    return value_map.get(value_key, value_key.upper())


def _pretty_band_name(band: str) -> str:
    if not band:
        return ""
    band = str(band).strip()
    band_map = {
        "2.4GHz": "2.4 GHz",
        "5GHz": "5 GHz",
        "6GHz": "6 GHz",
    }
    return band_map.get(band, band)


def _build_chart_title_and_subtitle(
    *,
    metric_key: str,
    compare_mode_label: str,
    selected_band: str,
    selected_value: str,
    selected_floor: str | None,
    selected_router: str | None,
    series_count: int,
) -> tuple[str, str]:
    metric_name = _pretty_metric_name(metric_key)
    band_name = _pretty_band_name(selected_band)
    value_name = _pretty_value_name(selected_value)

    if compare_mode_label == "Compare routers by floor":
        title = f"{metric_name} Comparison"
        subtitle_parts = [
            selected_floor or "Unknown Floor",
            band_name,
            value_name,
            f"{series_count} routers",
        ]
    else:
        title = f"{metric_name} by Floor"
        subtitle_parts = [
            selected_router or "Unknown Router",
            band_name,
            value_name,
            f"{series_count} floors",
        ]

    subtitle = " · ".join([part for part in subtitle_parts if part])
    return title, subtitle


def _load_curve_table(rvr_metric_dir: Path) -> pd.DataFrame:
    tables_dir = rvr_metric_dir / "tables"
    if not tables_dir.exists():
        raise FileNotFoundError(f"Missing tables directory: {tables_dir}")

    table_candidates = sorted(tables_dir.glob("*_curve_tables.csv"))
    if not table_candidates:
        raise FileNotFoundError(f"No *_curve_tables.csv found in: {tables_dir}")

    df = pd.read_csv(table_candidates[0])
    if df.empty:
        raise ValueError(f"Curve table is empty: {table_candidates[0]}")

    required = {"dist_ft_mid", "router_key", "floor_name", "band"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Curve table is missing required columns: {sorted(missing)}")

    if "router_display" not in df.columns:
        df["router_display"] = df["router_key"].astype(str)

    for col in ["dist_ft_mid", "p50", "mean", "p10", "p90"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["router_key"] = df["router_key"].astype(str)
    df["router_display"] = df["router_display"].fillna(df["router_key"]).astype(str)
    df["floor_name"] = df["floor_name"].astype(str)
    df["band"] = df["band"].astype(str)
    return df


def _display_name_from_group(df_group: pd.DataFrame, mode: str, key: str) -> str:
    if mode == "routers_by_floor":
        display = df_group["router_display"].dropna().astype(str)
        if not display.empty:
            return display.iloc[0]
    return key


def _build_series(
    df: pd.DataFrame,
    *,
    mode: str,
    selected_band: str,
    selected_value: str,
    selected_floor: str | None = None,
    selected_router: str | None = None,
) -> Tuple[List[Dict[str, object]], List[float], List[float]]:
    filtered = df[df["band"] == selected_band].copy()

    if mode == "routers_by_floor":
        if not selected_floor:
            return [], [], []
        filtered = filtered[filtered["floor_name"] == selected_floor]
        group_col = "router_key"
    else:
        if not selected_router:
            return [], [], []
        filtered = filtered[filtered["router_key"] == selected_router]
        group_col = "floor_name"

    filtered = filtered.dropna(subset=["dist_ft_mid", selected_value])
    if filtered.empty:
        return [], [], []

    series: List[Dict[str, object]] = []
    all_x: List[float] = []
    all_y: List[float] = []

    for key, grp in filtered.groupby(group_col, sort=True):
        grp_sorted = grp.sort_values("dist_ft_mid")
        xs = grp_sorted["dist_ft_mid"].astype(float).tolist()
        ys = grp_sorted[selected_value].astype(float).tolist()
        xs, ys = _prepend_zero_point(xs, ys)
        if not xs:
            continue

        display_name = _display_name_from_group(grp_sorted, mode, str(key))
        avg_value = float(pd.Series(ys).mean()) if ys else float("nan")
        data = [[round(float(x), 3), round(float(y), 3)] for x, y in zip(xs, ys)]

        series.append(
            {
                "name": display_name,
                "avg": None if pd.isna(avg_value) else round(avg_value, 1),
                "data": data,
            }
        )
        all_x.extend(xs)
        all_y.extend(ys)

    return series, all_x, all_y

def _build_mesh_series(
    df: pd.DataFrame,
    *,
    selected_band: str,
    selected_value: str,
    selected_floor: str | None,
    selected_router: str | None,
) -> Tuple[List[Dict[str, object]], List[float], List[float]]:
    filtered = df[df["band"] == selected_band].copy()

    if selected_router:
        filtered = filtered[filtered["router_key"] == selected_router]
    if selected_floor:
        filtered = filtered[filtered["floor_name"] == selected_floor]

    filtered = filtered.dropna(subset=["dist_ft_mid", selected_value])
    if filtered.empty:
        return [], [], []

    scenario_order = {"without_mesh": 0, "with_mesh": 1}
    series: List[Dict[str, object]] = []
    all_x: List[float] = []
    all_y: List[float] = []

    grouped_items = sorted(
        filtered.groupby("scenario", sort=False),
        key=lambda item: (scenario_order.get(str(item[0]), 99), str(item[0])),
    )

    for scenario_key, grp in grouped_items:
        grp_sorted = grp.sort_values("dist_ft_mid")
        xs = grp_sorted["dist_ft_mid"].astype(float).tolist()
        ys = grp_sorted[selected_value].astype(float).tolist()
        xs, ys = _prepend_zero_point(xs, ys)
        if not xs:
            continue

        scenario_name = grp_sorted["scenario_label"].dropna().astype(str)
        display_name = scenario_name.iloc[0] if not scenario_name.empty else str(scenario_key)
        avg_value = float(pd.Series(ys).mean()) if ys else float("nan")
        data = [[round(float(x), 3), round(float(y), 3)] for x, y in zip(xs, ys)]

        series.append(
            {
                "name": display_name,
                "avg": None if pd.isna(avg_value) else round(avg_value, 1),
                "data": data,
            }
        )
        all_x.extend(xs)
        all_y.extend(ys)

    return series, all_x, all_y

def _render_apex_chart(
    *,
    title: str,
    subtitle: str,
    y_label: str,
    series: List[Dict[str, object]],
    y_min: float | None,
    y_max: float | None,
    chip_texts: Iterable[str],
    height: int = 560,
) -> None:
    chip_html = "".join(f'<div class="chip">{_safe_text(chip)}</div>' for chip in chip_texts)
    stats_html = "".join(
        f"""
        <div class="stat-card">
          <div class="stat-label">{_safe_text(item["name"])} Avg</div>
          <div class="stat-value">{_safe_text(item["avg"] if item["avg"] is not None else "—")}</div>
        </div>
        """
        for item in series
    )

    html_doc = f"""
    <!DOCTYPE html>
    <html>
    <head>
      <meta charset="utf-8" />
      <meta name="viewport" content="width=device-width, initial-scale=1" />
      <script src="https://cdn.jsdelivr.net/npm/apexcharts"></script>
      <style>
        html, body {{
          margin: 0;
          padding: 0;
          background: transparent;
          font-family: Inter, Arial, sans-serif;
          color: #e5e7eb;
        }}
        .wrap {{
          background:
            radial-gradient(900px 300px at 15% 0%, rgba(56, 189, 248, 0.14), transparent 60%),
            radial-gradient(700px 260px at 85% 10%, rgba(251, 146, 60, 0.10), transparent 60%),
            linear-gradient(180deg, #111827 0%, #0b1220 100%);
          border: 1px solid rgba(255,255,255,0.08);
          border-radius: 24px;
          padding: 20px 20px 10px 20px;
          box-shadow: 0 18px 50px rgba(0,0,0,0.30);
        }}
        .top {{
          display: flex;
          justify-content: space-between;
          gap: 12px;
          align-items: flex-start;
          flex-wrap: wrap;
          margin-bottom: 10px;
        }}
        .title {{
          font-size: 26px;
          font-weight: 700;
          color: #f8fafc;
          line-height: 1.1;
        }}
        .subtitle {{
          margin-top: 6px;
          color: #94a3b8;
          font-size: 13px;
        }}
        .chips {{
          display: flex;
          gap: 8px;
          flex-wrap: wrap;
        }}
        .chip {{
          padding: 6px 12px;
          border-radius: 999px;
          background: rgba(255,255,255,0.06);
          border: 1px solid rgba(255,255,255,0.08);
          color: #dbeafe;
          font-size: 12px;
        }}
        .stats {{
          display: grid;
          grid-template-columns: repeat(4, minmax(180px, 1fr));
          gap: 12px;
          margin: 12px 0 8px 0;
        }}
        .stat-card {{
          background: rgba(255,255,255,0.04);
          border: 1px solid rgba(255,255,255,0.06);
          border-radius: 16px;
          padding: 10px 12px;
          min-height: 72px;
        }}
        .stat-label {{
          color: #94a3b8;
          font-size: 12px;
          white-space: nowrap;
          overflow: hidden;
          text-overflow: ellipsis;
        }}
        .stat-value {{
          margin-top: 2px;
          color: #f8fafc;
          font-size: 20px;
          font-weight: 700;
          line-height: 1.1;
        }}
        @media (max-width: 1200px) {{
            .stats {{
                grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
            }}
            }}

            @media (max-width: 960px) {{
            .stats {{
                grid-template-columns: repeat(2, minmax(140px, 1fr));
            }}
        }}
      </style>
    </head>
    <body>
      <div class="wrap">
        <div class="top">
          <div>
            <div class="title">{_safe_text(title)}</div>
            <div class="subtitle">{_safe_text(subtitle)}</div>
          </div>
          <div class="chips">{chip_html}</div>
        </div>
        <div class="stats">{stats_html}</div>
        <div id="chart"></div>
      </div>

      <script>
        const series = {json.dumps(series)};

        const allX = series
          .flatMap(item => item.data.map(point => Number(point[0])))
          .filter(v => Number.isFinite(v));

        const xMin = allX.length ? Math.min(...allX) : 0;
        const xMax = allX.length ? Math.max(...allX) : 100;

        function fmtDistance(value) {{
          if (!Number.isFinite(value)) return '';
          const rounded = Math.round(value);
          if (Math.abs(value - rounded) < 0.05) {{
            return String(rounded);
          }}
          return value.toFixed(1);
        }}

        const options = {{
          chart: {{
            type: 'line',
            height: {max(height - 170, 280)},
            background: 'transparent',
            toolbar: {{ show: true }},
            zoom: {{ enabled: true }}
          }},
          theme: {{ mode: 'dark' }},
          series: series.map(item => ({{ name: item.name, data: item.data }})),
          colors: ['#38bdf8', '#fb923c', '#a78bfa', '#34d399', '#f472b6', '#facc15'],
          stroke: {{ curve: 'straight', width: 3 }},
          markers: {{
            size: series.some(item => item.data.length <= 2) ? 4 : 0,
            hover: {{ sizeOffset: 3 }}
            }},
          dataLabels: {{ enabled: false }},
          legend: {{
            position: 'top',
            horizontalAlign: 'left',
            labels: {{ colors: '#cbd5e1' }}
          }},
          xaxis: {{
            type: 'numeric',
            min: Math.floor(xMin),
            max: Math.ceil(xMax),
            tickAmount: 7,
            title: {{ text: 'Distance from DUT (ft)' }},
            labels: {{
              style: {{ colors: '#94a3b8' }},
              formatter: function(value) {{
                return fmtDistance(Number(value));
              }}
            }},
            axisBorder: {{ color: 'rgba(255,255,255,0.08)' }},
            axisTicks: {{ color: 'rgba(255,255,255,0.08)' }}
          }},
          yaxis: {{
            min: {json.dumps(y_min)},
            max: {json.dumps(y_max)},
            forceNiceScale: true,
            decimalsInFloat: 1,
            title: {{ text: {json.dumps(y_label)} }},
            labels: {{ style: {{ colors: '#94a3b8' }} }}
          }},
          grid: {{
            borderColor: 'rgba(255,255,255,0.08)',
            strokeDashArray: 4
          }},
          tooltip: {{
            shared: true,
            intersect: false,
            x: {{
              formatter: function(value) {{
                return fmtDistance(Number(value)) + ' ft';
              }}
            }}
          }},
          noData: {{ text: 'No chart data found.' }}
        }};

        new ApexCharts(document.querySelector('#chart'), options).render();
      </script>
    </body>
    </html>
    """
    components.html(html_doc, height=height, scrolling=False)

def _metric_y_label(metric_key: str) -> str:
    unit_map = {
        "snr": "SNR (dB)",
        "signal_strength": "Signal Strength (dBm)",
        "signal_strength_main": "Signal Strength (dBm)",
        "throughput": "Throughput (Mbps)",
        "data_rate": "Data Rate (Mbps)",
        "channel_utilization": "Channel Utilization (%)",
        "spectrum_channel_power": "Spectrum Channel Power (dBm)",
    }
    return unit_map.get(metric_key, _pretty_metric_name(metric_key))


def _pad_y_axis(all_y: List[float]) -> Tuple[float | None, float | None]:
    if not all_y:
        return None, None

    y_min = float(min(all_y))
    y_max = float(max(all_y))
    if y_min == y_max:
        pad = 1.0 if y_min == 0 else abs(y_min) * 0.08
        return y_min - pad, y_max + pad

    pad = (y_max - y_min) * 0.08
    return y_min - pad, y_max + pad


def _render_chart_preview(series: List[Dict[str, object]], selected_value: str) -> None:
    with st.expander("Chart data preview", expanded=False):
        preview_rows = []
        for item in series:
            for x, y in item["data"][:12]:
                preview_rows.append({"series": item["name"], "dist_ft": x, selected_value: y})
        if preview_rows:
            st.dataframe(pd.DataFrame(preview_rows), width="stretch")



# def _metric_higher_is_better(metric_key: str) -> bool:
#     lower_is_better = {
#         "channel_utilization",
#         "channel_interference",
#         "noise",
#         "spectrum_channel_power",
#     }
#     return metric_key not in lower_is_better


# def _normalize_score_series(values: pd.Series, higher_is_better: bool) -> pd.Series:
#     values = pd.to_numeric(values, errors="coerce")
#     valid = values.dropna()

#     if valid.empty:
#         return pd.Series([float("nan")] * len(values), index=values.index, dtype="float64")

#     vmin = float(valid.min())
#     vmax = float(valid.max())

#     if abs(vmax - vmin) < 1e-9:
#         out = pd.Series(100.0, index=values.index, dtype="float64")
#         out[values.isna()] = float("nan")
#         return out.round(1)

#     scaled = (values - vmin) / (vmax - vmin) * 100.0
#     if not higher_is_better:
#         scaled = 100.0 - scaled

#     scaled[values.isna()] = float("nan")
#     return scaled.round(1)


# def _build_summary_score_table(
#     filtered: pd.DataFrame,
#     *,
#     group_col: str,
#     label_col: str,
#     metric_key: str,
#     selected_value: str,
# ) -> pd.DataFrame:
#     working = filtered.copy()
#     working["dist_ft_mid"] = pd.to_numeric(working["dist_ft_mid"], errors="coerce")
#     working[selected_value] = pd.to_numeric(working[selected_value], errors="coerce")

#     working = working.dropna(subset=[group_col, label_col, "dist_ft_mid", selected_value])
#     if working.empty:
#         return pd.DataFrame()

#     working["zone"] = pd.cut(
#         working["dist_ft_mid"],
#         bins=[-1e9, 20.0, 40.0, 1e9],
#         labels=ZONE_ORDER,
#         include_lowest=True,
#         right=True,
#     )

#     grouped = (
#         working.groupby([group_col, label_col, "zone"], observed=True)[selected_value]
#         .mean()
#         .unstack("zone")
#         .reset_index()
#     )

#     if grouped.empty:
#         return pd.DataFrame()

#     for zone in ZONE_ORDER:
#         if zone not in grouped.columns:
#             grouped[zone] = float("nan")

#     higher_is_better = _metric_higher_is_better(metric_key)

#     for zone in ZONE_ORDER:
#         grouped[f"{zone}_score"] = _normalize_score_series(grouped[zone], higher_is_better)

#     score_cols = [f"{zone}_score" for zone in ZONE_ORDER]
#     grouped["overall_score"] = grouped[score_cols].mean(axis=1, skipna=True).round(1)
#     grouped["raw_average"] = grouped[ZONE_ORDER].mean(axis=1, skipna=True).round(1)

#     grouped = grouped.rename(columns={group_col: "key", label_col: "label"})
#     grouped = grouped.sort_values(["overall_score", "label"], ascending=[False, True]).reset_index(drop=True)
#     return grouped


# def _render_score_stacked_bar(
#     summary_df: pd.DataFrame,
#     *,
#     title: str,
#     subtitle: str,
#     height: int = 360,
# ) -> None:
#     if summary_df.empty:
#         st.info("No ranking summary available for the current selection.")
#         return

#     categories = summary_df["label"].astype(str).tolist()
#     series = [
#         {
#             "name": "Near",
#             "data": [0.0 if pd.isna(v) else round(float(v), 1) for v in summary_df["Near_score"]],
#         },
#         {
#             "name": "Mid",
#             "data": [0.0 if pd.isna(v) else round(float(v), 1) for v in summary_df["Mid_score"]],
#         },
#         {
#             "name": "Far",
#             "data": [0.0 if pd.isna(v) else round(float(v), 1) for v in summary_df["Far_score"]],
#         },
#     ]

#     total_max = summary_df[["Near_score", "Mid_score", "Far_score"]].sum(axis=1, min_count=1).max()
#     y_max = 100.0 if pd.isna(total_max) else max(100.0, float(total_max) + 10.0)

#     html_doc = f"""
#     <!DOCTYPE html>
#     <html>
#     <head>
#       <meta charset="utf-8" />
#       <meta name="viewport" content="width=device-width, initial-scale=1" />
#       <script src="https://cdn.jsdelivr.net/npm/apexcharts"></script>
#       <style>
#         html, body {{
#           margin: 0;
#           padding: 0;
#           background: transparent;
#           font-family: Inter, Arial, sans-serif;
#           color: #e5e7eb;
#         }}
#         .wrap {{
#           background: linear-gradient(180deg, #1f2437 0%, #1a2030 100%);
#           border: 1px solid rgba(255,255,255,0.07);
#           border-radius: 20px;
#           padding: 16px 16px 8px 16px;
#           box-shadow: 0 14px 36px rgba(0,0,0,0.24);
#         }}
#         .title {{
#           color: #f8fafc;
#           font-size: 20px;
#           font-weight: 700;
#         }}
#         .subtitle {{
#           margin-top: 4px;
#           color: #94a3b8;
#           font-size: 12px;
#         }}
#         #chart {{
#           margin-top: 10px;
#         }}
#       </style>
#     </head>
#     <body>
#       <div class="wrap">
#         <div class="title">{_safe_text(title)}</div>
#         <div class="subtitle">{_safe_text(subtitle)}</div>
#         <div id="chart"></div>
#       </div>
#       <script>
#         const options = {{
#           chart: {{
#             type: 'bar',
#             height: {max(height - 70, 220)},
#             stacked: true,
#             background: 'transparent',
#             toolbar: {{ show: false }}
#           }},
#           theme: {{ mode: 'dark' }},
#           series: {json.dumps(series)},
#           colors: ['#1d9bf0', '#16e0a0', '#f6b01a'],
#           plotOptions: {{
#             bar: {{
#               horizontal: false,
#               columnWidth: '45%',
#               borderRadius: 3
#             }}
#           }},
#           dataLabels: {{ enabled: false }},
#           xaxis: {{
#             categories: {json.dumps(categories)},
#             labels: {{
#               rotate: -20,
#               style: {{ colors: '#cbd5e1' }}
#             }},
#             axisBorder: {{ color: 'rgba(255,255,255,0.08)' }},
#             axisTicks: {{ color: 'rgba(255,255,255,0.08)' }}
#           }},
#           yaxis: {{
#             min: 0,
#             max: {json.dumps(y_max)},
#             tickAmount: 5,
#             title: {{ text: 'Score' }},
#             labels: {{ style: {{ colors: '#94a3b8' }} }}
#           }},
#           legend: {{
#             position: 'bottom',
#             labels: {{ colors: '#cbd5e1' }}
#           }},
#           grid: {{
#             borderColor: 'rgba(255,255,255,0.08)',
#             strokeDashArray: 3
#           }},
#           tooltip: {{
#             shared: true,
#             intersect: false
#           }}
#         }};
#         new ApexCharts(document.querySelector('#chart'), options).render();
#       </script>
#     </body>
#     </html>
#     """
#     components.html(html_doc, height=height, scrolling=False)

def _render_avg_value_bar(
    series: List[Dict[str, object]],
    *,
    title: str,
    subtitle: str,
    y_label: str,
    height: int = 360,
) -> None:
    categories = [str(item["name"]) for item in series]
    values = [
        None if item.get("avg") is None else round(float(item["avg"]), 1)
        for item in series
    ]

    html_doc = f"""
    <!DOCTYPE html>
    <html>
    <head>
      <meta charset="utf-8" />
      <meta name="viewport" content="width=device-width, initial-scale=1" />
      <script src="https://cdn.jsdelivr.net/npm/apexcharts"></script>
      <style>
        html, body {{
          margin: 0;
          padding: 0;
          background: transparent;
          font-family: Inter, Arial, sans-serif;
          color: #e5e7eb;
        }}
        .wrap {{
          background:
            radial-gradient(900px 300px at 15% 0%, rgba(56, 189, 248, 0.10), transparent 60%),
            radial-gradient(700px 260px at 85% 10%, rgba(251, 146, 60, 0.08), transparent 60%),
            linear-gradient(180deg, #111827 0%, #0b1220 100%);
          border: 1px solid rgba(255,255,255,0.08);
          border-radius: 24px;
          padding: 18px 18px 10px 18px;
          box-shadow: 0 18px 50px rgba(0,0,0,0.30);
        }}
        .title {{
          font-size: 24px;
          font-weight: 700;
          color: #f8fafc;
          line-height: 1.1;
        }}
        .subtitle {{
          margin-top: 6px;
          color: #94a3b8;
          font-size: 13px;
        }}
        #chart {{
          margin-top: 12px;
        }}
      </style>
    </head>
    <body>
      <div class="wrap">
        <div class="title">{_safe_text(title)}</div>
        <div class="subtitle">{_safe_text(subtitle)}</div>
        <div id="chart"></div>
      </div>

      <script>
        const options = {{
          chart: {{
            type: 'bar',
            height: {max(height - 70, 240)},
            background: 'transparent',
            toolbar: {{ show: false }}
          }},
          theme: {{ mode: 'dark' }},
          series: [{{
            name: 'Avg',
            data: {json.dumps(values)}
          }}],
          colors: ['#38bdf8'],
          plotOptions: {{
            bar: {{
              horizontal: false,
              columnWidth: '42%',
              borderRadius: 4
            }}
          }},
          dataLabels: {{
            enabled: true,
            formatter: function(val) {{
              return val == null ? '' : Number(val).toFixed(1);
            }},
            style: {{
              colors: ['#e5e7eb']
            }}
          }},
          xaxis: {{
            categories: {json.dumps(categories)},
            labels: {{
              style: {{ colors: '#cbd5e1' }},
              rotate: -15
            }},
            axisBorder: {{ color: 'rgba(255,255,255,0.08)' }},
            axisTicks: {{ color: 'rgba(255,255,255,0.08)' }}
          }},
          yaxis: {{
            title: {{ text: {json.dumps(y_label)} }},
            labels: {{ style: {{ colors: '#94a3b8' }} }}
          }},
          grid: {{
            borderColor: 'rgba(255,255,255,0.08)',
            strokeDashArray: 4
          }},
          legend: {{
            show: false
          }},
          tooltip: {{
            y: {{
              formatter: function(val) {{
                return val == null ? '' : Number(val).toFixed(1);
              }}
            }}
          }}
        }};

        new ApexCharts(document.querySelector('#chart'), options).render();
      </script>
    </body>
    </html>
    """
    components.html(html_doc, height=height, scrolling=False)


def _render_avg_radial_chart(
    series: List[Dict[str, object]],
    *,
    title: str,
    subtitle: str,
    height: int = 360,
    top_n: int = 4,
) -> None:
    avg_items = []
    for item in series:
        avg = item.get("avg")
        if avg is None:
            continue
        try:
            avg_num = float(avg)
        except Exception:
            continue
        if pd.isna(avg_num):
            continue

        avg_items.append(
            {
                "name": str(item["name"]),
                "avg": round(avg_num, 1),
            }
        )

    if not avg_items:
        st.info("No average summary available for the current selection.")
        return

    radial_items = avg_items[: min(top_n, len(avg_items))]
    labels = [item["name"] for item in radial_items]
    real_values = [item["avg"] for item in radial_items]

    vmax = max(real_values) if real_values else 0.0
    normalized = [0.0 if vmax <= 0 else round((v / vmax) * 100.0, 1) for v in real_values]

    html_doc = f"""
    <!DOCTYPE html>
    <html>
    <head>
      <meta charset="utf-8" />
      <meta name="viewport" content="width=device-width, initial-scale=1" />
      <script src="https://cdn.jsdelivr.net/npm/apexcharts"></script>
      <style>
        html, body {{
          margin: 0;
          padding: 0;
          background: transparent;
          font-family: Inter, Arial, sans-serif;
          color: #e5e7eb;
        }}
        .wrap {{
          background: linear-gradient(180deg, #1f2437 0%, #1a2030 100%);
          border: 1px solid rgba(255,255,255,0.07);
          border-radius: 20px;
          padding: 16px 16px 8px 16px;
          box-shadow: 0 14px 36px rgba(0,0,0,0.24);
        }}
        .title {{
          color: #f8fafc;
          font-size: 20px;
          font-weight: 700;
        }}
        .subtitle {{
          margin-top: 4px;
          color: #94a3b8;
          font-size: 12px;
        }}
        #chart {{
          margin-top: 8px;
        }}
      </style>
    </head>
    <body>
      <div class="wrap">
        <div class="title">{_safe_text(title)}</div>
        <div class="subtitle">{_safe_text(subtitle)}</div>
        <div id="chart"></div>
      </div>
      <script>
        const realValues = {json.dumps(real_values)};
        const normalizedValues = {json.dumps(normalized)};
        const labels = {json.dumps(labels)};

        function safeRealValue(index) {{
          const v = realValues[index];
          return Number.isFinite(Number(v)) ? Number(v) : 0;
        }}

        const options = {{
          chart: {{
            type: 'radialBar',
            height: {max(height - 70, 220)},
            background: 'transparent',
            toolbar: {{ show: false }}
          }},
          theme: {{ mode: 'dark' }},
          series: normalizedValues,
          labels: labels,
          colors: ['#1d9bf0', '#16e0a0', '#f6b01a', '#a78bfa', '#fb7185'],
          plotOptions: {{
            radialBar: {{
              startAngle: -135,
              endAngle: 225,
              hollow: {{
                size: '18%'
              }},
              track: {{
                background: 'rgba(255,255,255,0.08)'
              }},
              dataLabels: {{
                name: {{
                  show: true,
                  fontSize: '13px',
                  color: '#cbd5e1'
                }},
                value: {{
                  show: true,
                  fontSize: '15px',
                  color: '#f8fafc',
                  formatter: function(val) {{
                    return Number(val).toFixed(1);
                  }}
                }}
              }}
            }}
          }},
          legend: {{
            show: true,
            position: 'bottom',
            labels: {{ colors: '#cbd5e1' }}
          }},
          tooltip: {{
            y: {{
              formatter: function(val, opts) {{
                const idx = opts.seriesIndex ?? 0;
                return safeRealValue(idx).toFixed(1);
              }}
            }}
          }},
          stroke: {{
            lineCap: 'round'
          }}
        }};

        new ApexCharts(document.querySelector('#chart'), options).render();
      </script>
    </body>
    </html>
    """
    components.html(html_doc, height=height, scrolling=False)


def render_rvr_apex_dashboard(
    router_dir: Path,
    rvr_outputs_root: Path,
    *,
    step_label: str = "Step 4",
    title: str = "Interactive ApexCharts",
    subtle: str | None = None,
) -> None:
    subtle = subtle or (
        "Reads the curve table already generated by <code>parameter_vs_range.py</code> "
        "and renders an interactive line chart."
    )

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="card-title">
        <h2><span class="step">{step_label}</span> {title}</h2>
        </div>
        <div class="subtle">
        {subtle}
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not rvr_outputs_root.exists():
        st.info("Run Step 3 first. No rvr_outputs folder exists yet for this router.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    metric_dirs = []
    for path in sorted([p for p in rvr_outputs_root.iterdir() if p.is_dir()]):
        if list((path / "tables").glob("*_curve_tables.csv")):
            metric_dirs.append(path.name)

    if not metric_dirs:
        st.info("No curve-table outputs were found yet. Run Step 3 successfully for at least one metric.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    metric_name = st.selectbox(
        "Select RvR output metric",
        metric_dirs,
        key="apex_metric_dir",
    )

    metric_dir = rvr_outputs_root / metric_name

    try:
        df = _load_curve_table(metric_dir)
    except Exception as exc:
        st.error(f"Could not load curve table: {exc}")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    numeric_options = [col for col in ["p50", "mean", "p10", "p90"] if col in df.columns]
    if not numeric_options:
        st.error("The curve table does not have any of the expected numeric columns: p50, mean, p10, p90.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    ui_left, ui_mid, ui_right = st.columns([1.2, 1.2, 1.2])

    with ui_left:
        compare_mode_label = st.radio(
            "Chart mode",
            ["Compare routers by floor", "Compare floors by router"],
            horizontal=False,
            key="apex_compare_mode",
        )

    with ui_mid:
        band_values = sorted(df["band"].dropna().astype(str).unique().tolist())
        selected_band = st.selectbox("Band", band_values, key="apex_band")

    with ui_right:
        selected_value = st.selectbox(
            "Value column",
            numeric_options,
            index=numeric_options.index("p50") if "p50" in numeric_options else 0,
            key="apex_value_column",
        )

    mode = "routers_by_floor" if compare_mode_label == "Compare routers by floor" else "floors_by_router"
    filtered_band = df[df["band"] == selected_band].copy()

    selected_floor = None
    selected_router = None

    if mode == "routers_by_floor":
        floor_values = sorted(filtered_band["floor_name"].dropna().astype(str).unique().tolist())
        selected_floor = st.selectbox("Floor", floor_values, key="apex_floor")
    else:
        router_values = sorted(filtered_band["router_key"].dropna().astype(str).unique().tolist())
        selected_router = st.selectbox("Router", router_values, key="apex_router")

    series, all_x, all_y = _build_series(
        df,
        mode=mode,
        selected_band=selected_band,
        selected_value=selected_value,
        selected_floor=selected_floor,
        selected_router=selected_router,
    )

    if not series:
        st.warning("No chartable rows matched the current filter combination.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    title_text, subtitle_text = _build_chart_title_and_subtitle(
        metric_key=metric_dir.name,
        compare_mode_label=compare_mode_label,
        selected_band=selected_band,
        selected_value=selected_value,
        selected_floor=selected_floor,
        selected_router=selected_router,
        series_count=len(series),
    )

    y_min, y_max = _pad_y_axis(all_y)
    y_label = _metric_y_label(metric_dir.name)

    chip_texts = [
        metric_dir.name,
        compare_mode_label,
        f"Series: {len(series)}",
        f"Points: {sum(len(item['data']) for item in series)}",
    ]

    _render_apex_chart(
        title=title_text,
        subtitle=subtitle_text,
        y_label=y_label,
        series=series,
        y_min=y_min,
        y_max=y_max,
        chip_texts=chip_texts,
        height=620,
    )


    avg_bar_title = "Router Average Comparison" if mode == "routers_by_floor" else "Floor Average Comparison"
    avg_radial_title = "Router Average Summary" if mode == "routers_by_floor" else "Floor Average Summary"

    summary_left, summary_right = st.columns([1.35, 1.0])

    with summary_left:
        _render_avg_value_bar(
            series,
            title=avg_bar_title,
            subtitle=subtitle_text,
            y_label=y_label,
            height=360,
        )

    with summary_right:
        _render_avg_radial_chart(
            series,
            title=avg_radial_title,
            subtitle=subtitle_text,
            height=360,
            top_n=4,
        )

    _render_chart_preview(series, selected_value)
    st.markdown("</div>", unsafe_allow_html=True)

def render_mesh_compare_apex_dashboard(
    compare_outputs_root: Path,
    *,
    step_label: str = "Step 5",
    title: str = "Interactive Graph — Mesh vs No Mesh",
    subtle: str | None = None,
) -> None:
    subtle = subtle or (
        "Reads the comparison curve table generated by <code>comparison.py</code> "
        "and renders an interactive mesh-vs-no-mesh line chart."
    )

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="card-title">
          <h2><span class="step">{step_label}</span> {title}</h2>
        </div>
        <div class="subtle">
          {subtle}
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not compare_outputs_root.exists():
        st.info("Run Step 4 first. No compare_outputs folder exists yet for this router.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    metric_dirs = []
    for path in sorted([p for p in compare_outputs_root.iterdir() if p.is_dir()]):
        if list((path / "tables").glob("*_mesh_curve_tables.csv")):
            metric_dirs.append(path.name)

    if not metric_dirs:
        st.info("No mesh comparison curve tables were found yet. Run Step 4 successfully for at least one metric.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    metric_name = st.selectbox(
        "Select mesh comparison metric",
        metric_dirs,
        key="mesh_apex_metric_dir",
    )
    metric_dir = compare_outputs_root / metric_name

    tables_dir = metric_dir / "tables"
    table_candidates = sorted(tables_dir.glob("*_mesh_curve_tables.csv"))
    if not table_candidates:
        st.error("No *_mesh_curve_tables.csv found.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    df = pd.read_csv(table_candidates[0])

    numeric_options = [col for col in ["p50", "mean", "p10", "p90"] if col in df.columns]
    if not numeric_options:
        st.error("The comparison curve table does not have any of the expected numeric columns: p50, mean, p10, p90.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    ui_left, ui_mid, ui_right, ui_far = st.columns([1.0, 1.0, 1.0, 1.0])

    with ui_left:
        band_values = sorted(df["band"].dropna().astype(str).unique().tolist())
        selected_band = st.selectbox("Band", band_values, key="mesh_apex_band")

    filtered_band = df[df["band"] == selected_band].copy()

    with ui_mid:
        router_values = sorted(filtered_band["router_key"].dropna().astype(str).unique().tolist())
        if not router_values:
            st.warning("No routers were found for the selected band.")
            st.markdown("</div>", unsafe_allow_html=True)
            return
        selected_router = st.selectbox("Router", router_values, key="mesh_apex_router")

    filtered_router = filtered_band[filtered_band["router_key"] == selected_router].copy()

    with ui_right:
        floor_values = sorted(filtered_router["floor_name"].dropna().astype(str).unique().tolist())
        if not floor_values:
            st.warning("No floors were found for the selected router and band.")
            st.markdown("</div>", unsafe_allow_html=True)
            return
        selected_floor = st.selectbox("Floor", floor_values, key="mesh_apex_floor")

    with ui_far:
        selected_value = st.selectbox(
            "Value column",
            numeric_options,
            index=numeric_options.index("p50") if "p50" in numeric_options else 0,
            key="mesh_apex_value_column",
        )

    series, _, all_y = _build_mesh_series(
        df,
        selected_band=selected_band,
        selected_value=selected_value,
        selected_floor=selected_floor,
        selected_router=selected_router,
    )

    if not series:
        st.warning("No chartable rows matched the current filter combination.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    y_min, y_max = _pad_y_axis(all_y)
    subtitle_parts = [
        selected_router,
        selected_floor,
        _pretty_band_name(selected_band),
        _pretty_value_name(selected_value),
        f"{len(series)} scenarios",
    ]

    chip_texts = [
        _pretty_metric_name(metric_dir.name),
        "Mesh vs No Mesh",
        f"Series: {len(series)}",
        f"Points: {sum(len(item['data']) for item in series)}",
    ]

    _render_apex_chart(
        title=f"{_pretty_metric_name(metric_dir.name)} Mesh Comparison",
        subtitle=" · ".join([part for part in subtitle_parts if part]),
        y_label=_metric_y_label(metric_dir.name),
        series=series,
        y_min=y_min,
        y_max=y_max,
        chip_texts=chip_texts,
        height=620,
    )

    mesh_subtitle = " · ".join([part for part in subtitle_parts if part])

    summary_left, summary_right = st.columns([1.35, 1.0])

    with summary_left:
        _render_avg_value_bar(
            series,
            title="Mesh Average Comparison",
            subtitle=mesh_subtitle,
            y_label=_metric_y_label(metric_dir.name),
            height=360,
        )

    with summary_right:
        _render_avg_radial_chart(
            series,
            title="Mesh Average Summary",
            subtitle=mesh_subtitle,
            height=360,
            top_n=2,
        )

    _render_chart_preview(series, selected_value)
    st.markdown("</div>", unsafe_allow_html=True)
