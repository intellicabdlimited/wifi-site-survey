from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

DISTANCE_POINTS = [0, 10, 20, 30, 40, 50, 60, 70]
DISTANCE_BANDS = ["0-10", "10-20", "20-30", "30-40", "40-50", "50-60", "60-70", "70+"]
TOPOLOGY_LABELS = {"with_mesh": "mesh", "without_mesh": "nomesh"}
METRIC_LABELS = {"throughput": "Throughput", "signal_strength": "Signal strength"}
FILE_RE = re.compile(
    r"(?P<router>.+?)_(?P<label>Throughput|Signal Strength) for (?P<floor>Ground Floor|Lower Floor|Upper Floor) on (?P<band>2\.4 GHz|5 GHz) band_output\.csv",
    re.IGNORECASE,
)


@st.cache_data(show_spinner=False)
def load_records(base_path: str) -> pd.DataFrame:
    root = Path(base_path)
    rows: list[dict[str, Any]] = []
    if not root.exists():
        return pd.DataFrame()

    for topology_dir in root.iterdir():
        if not topology_dir.is_dir() or topology_dir.name not in TOPOLOGY_LABELS:
            continue
        for metric_dir in topology_dir.iterdir():
            if not metric_dir.is_dir() or metric_dir.name not in METRIC_LABELS:
                continue
            for router_dir in metric_dir.iterdir():
                if not router_dir.is_dir():
                    continue
                for csv_path in sorted(router_dir.glob("*.csv")):
                    parsed = _parse_metadata(csv_path.name, router_dir.name, metric_dir.name)
                    if not parsed:
                        continue
                    summary = _summarize_csv(csv_path)
                    rows.append({
                        "path": str(csv_path),
                        "topology": topology_dir.name,
                        "topology_label": TOPOLOGY_LABELS[topology_dir.name],
                        "metric": metric_dir.name,
                        "router": parsed["router"],
                        "floor": parsed["floor"],
                        "band": parsed["band"],
                        **summary,
                    })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["router_sort"] = df["router"].str.lower()
    return df.sort_values(["metric", "router_sort", "floor", "band", "topology"]).reset_index(drop=True)


def available_filters(records: pd.DataFrame) -> dict[str, list[str]]:
    if records.empty:
        return {"metrics": [], "floors": [], "bands": [], "routers": []}
    return {
        "metrics": sorted(records["metric"].dropna().unique().tolist()),
        "floors": sorted(records["floor"].dropna().unique().tolist(), key=_floor_order),
        "bands": sorted(records["band"].dropna().unique().tolist(), key=_band_order),
        "routers": sorted(records["router"].dropna().unique().tolist(), key=str.lower),
    }


def threshold_defaults(metric: str) -> tuple[float, float, float, float, str]:
    if metric == "signal_strength":
        return -85.0, -35.0, -67.0, 1.0, "dBm"
    return 0.0, 1500.0, 500.0, 25.0, "Mbps"


def build_dashboard_payload(
    records: pd.DataFrame,
    *,
    metric: str,
    floor: str,
    band: str,
    threshold: float,
) -> tuple[dict[str, dict[str, list[float]]], dict[str, dict[str, Any]], pd.DataFrame]:
    if records.empty:
        return {}, {}, pd.DataFrame()

    filtered = records[
        (records["metric"] == metric)
        & (records["floor"] == floor)
        & (records["band"] == band)
    ].copy()

    routers = sorted(records.loc[records["metric"] == metric, "router"].dropna().unique().tolist(), key=str.lower)
    data: dict[str, dict[str, list[float]]] = {}
    kpis: dict[str, dict[str, Any]] = {}
    table_rows: list[dict[str, Any]] = []

    for router in routers:
        router_rows = filtered[filtered["router"] == router]
        mesh_row = router_rows[router_rows["topology_label"] == "mesh"]
        nomesh_row = router_rows[router_rows["topology_label"] == "nomesh"]

        mesh_series = _series_or_nan(mesh_row)
        nomesh_series = _series_or_nan(nomesh_row)

        mesh_mean = _safe_mean(mesh_series)
        nomesh_mean = _safe_mean(nomesh_series)
        mesh_peak = _safe_max(mesh_series)
        nomesh_peak = _safe_max(nomesh_series)
        gain_abs = mesh_mean - nomesh_mean if not math.isnan(mesh_mean) and not math.isnan(nomesh_mean) else float("nan")
        gain_pct = _safe_gain_pct(mesh_mean, nomesh_mean)

        coverage_mesh = _coverage_count(mesh_series, threshold)
        coverage_nomesh = _coverage_count(nomesh_series, threshold)
        drop_mesh = _drops(mesh_series)
        drop_nomesh = _drops(nomesh_series)
        consistency_mesh = _series_consistency(mesh_series)
        consistency_nomesh = _series_consistency(nomesh_series)

        data[router] = {"mesh": mesh_series, "nomesh": nomesh_series}
        kpis[router] = {
            "avg_mesh": round(mesh_mean, 1) if not math.isnan(mesh_mean) else float("nan"),
            "avg_nomesh": round(nomesh_mean, 1) if not math.isnan(nomesh_mean) else float("nan"),
            "peak_mesh": round(mesh_peak, 1) if not math.isnan(mesh_peak) else float("nan"),
            "peak_nomesh": round(nomesh_peak, 1) if not math.isnan(nomesh_peak) else float("nan"),
            "gain_abs": round(gain_abs, 1) if not math.isnan(gain_abs) else float("nan"),
            "gain_pct": round(gain_pct, 1) if not math.isnan(gain_pct) else float("nan"),
            "coverage_mesh": coverage_mesh,
            "coverage_nomesh": coverage_nomesh,
            "drops_mesh": drop_mesh,
            "drops_nomesh": drop_nomesh,
            "consistency_mesh": consistency_mesh,
            "consistency_nomesh": consistency_nomesh,
        }

        table_rows.append({
            "router": router,
            "avg_mesh": mesh_mean,
            "avg_nomesh": nomesh_mean,
            "peak_mesh": mesh_peak,
            "peak_nomesh": nomesh_peak,
            "coverage_mesh": coverage_mesh,
            "coverage_nomesh": coverage_nomesh,
            "gain_abs": gain_abs,
            "gain_pct": gain_pct,
            "drop_mesh_total": float(np.nansum(drop_mesh)) if drop_mesh else np.nan,
            "drop_nomesh_total": float(np.nansum(drop_nomesh)) if drop_nomesh else np.nan,
            "consistency_mesh": consistency_mesh,
            "consistency_nomesh": consistency_nomesh,
        })

    summary_df = pd.DataFrame(table_rows).sort_values("gain_abs", ascending=False, na_position="last")
    return data, kpis, summary_df


def count_files(records: pd.DataFrame) -> int:
    return 0 if records.empty else len(records)


def _parse_metadata(filename: str, fallback_router: str, metric_name: str) -> dict[str, str] | None:
    match = FILE_RE.match(filename)
    if not match:
        return None
    router = match.group("router") or fallback_router
    return {
        "router": router,
        "metric": metric_name,
        "floor": match.group("floor"),
        "band": match.group("band"),
    }


def _summarize_csv(path: Path) -> dict[str, Any]:
    df = pd.read_csv(path)
    if "value" not in df.columns:
        raise ValueError(f"Missing 'value' column in {path}")

    values = pd.to_numeric(df["value"], errors="coerce")
    valid = df.loc[values.notna()].copy()
    valid["value"] = pd.to_numeric(valid["value"], errors="coerce")

    if valid.empty:
        return {
            "series": [float("nan")] * len(DISTANCE_POINTS),
            "mean_value": float("nan"),
            "peak_value": float("nan"),
            "value_std": float("nan"),
            "valid_points": 0,
        }

    if "col" in valid.columns:
        axis = pd.to_numeric(valid["col"], errors="coerce")
    elif "cx" in valid.columns:
        axis = pd.to_numeric(valid["cx"], errors="coerce")
    else:
        axis = pd.Series(np.arange(len(valid)), index=valid.index, dtype=float)

    axis = axis.fillna(axis.median() if not axis.dropna().empty else 0.0)
    if axis.nunique() == 1:
        band_idx = pd.Series([0] * len(valid), index=valid.index, dtype=int)
    else:
        edges = np.linspace(axis.min(), axis.max() + 1e-9, len(DISTANCE_POINTS) + 1)
        band_idx = pd.cut(axis, bins=edges, labels=False, include_lowest=True)
        band_idx = band_idx.fillna(0).astype(int).clip(0, len(DISTANCE_POINTS) - 1)

    valid["band_idx"] = band_idx
    grouped = valid.groupby("band_idx")["value"].mean()

    series = [float(grouped.get(i, np.nan)) for i in range(len(DISTANCE_POINTS))]
    series = _fill_series(series)

    return {
        "series": [round(v, 2) if not math.isnan(v) else float("nan") for v in series],
        "mean_value": float(valid["value"].mean()),
        "peak_value": float(valid["value"].max()),
        "value_std": float(valid["value"].std(ddof=0)),
        "valid_points": int(valid["value"].notna().sum()),
    }


def _series_or_nan(rows: pd.DataFrame) -> list[float]:
    if rows.empty:
        return [float("nan")] * len(DISTANCE_POINTS)
    return list(rows.iloc[0]["series"])


def _fill_series(series: list[float]) -> list[float]:
    ser = pd.Series(series, dtype=float)
    if ser.notna().sum() == 0:
        return [float("nan")] * len(series)

    # only fill gaps inside the series
    ser = ser.interpolate(limit_area="inside")
    return ser.round(2).tolist()


def _safe_mean(series: list[float]) -> float:
    arr = np.asarray(series, dtype=float)
    return float(np.nanmean(arr)) if np.isfinite(arr).any() else float("nan")


def _safe_max(series: list[float]) -> float:
    arr = np.asarray(series, dtype=float)
    return float(np.nanmax(arr)) if np.isfinite(arr).any() else float("nan")


def _coverage_count(series: list[float], threshold: float) -> int:
    arr = np.asarray(series, dtype=float)
    return int(np.nansum(arr >= threshold)) if np.isfinite(arr).any() else 0


def _drops(series: list[float]) -> list[float]:
    arr = np.asarray(series, dtype=float)
    if len(arr) < 2 or not np.isfinite(arr).any():
        return [0.0] * (len(DISTANCE_POINTS) - 1)
    drops = []
    for prev, curr in zip(arr[:-1], arr[1:]):
        if np.isnan(prev) or np.isnan(curr):
            drops.append(0.0)
        else:
            drops.append(round(max(prev - curr, 0.0), 2))
    return drops


def _series_consistency(series: list[float]) -> float:
    arr = np.asarray(series, dtype=float)
    if not np.isfinite(arr).any():
        return float("nan")
    return float(np.nanstd(arr, ddof=0))


def _safe_gain_pct(mesh_mean: float, nomesh_mean: float) -> float:
    if math.isnan(mesh_mean) or math.isnan(nomesh_mean):
        return float("nan")
    denom = abs(nomesh_mean) if nomesh_mean != 0 else 1.0
    return ((mesh_mean - nomesh_mean) / denom) * 100.0


def _floor_order(value: str) -> tuple[int, str]:
    order = {"Ground Floor": 0, "Lower Floor": 1, "Upper Floor": 2}
    return order.get(value, 99), value


def _band_order(value: str) -> tuple[int, str]:
    order = {"2.4 GHz": 0, "5 GHz": 1}
    return order.get(value, 99), value
