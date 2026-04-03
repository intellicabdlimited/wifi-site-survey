from __future__ import annotations

import math
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

DISTANCE_POINTS: list[int] = [0, 10, 20, 30, 40, 50, 60, 70]
DISTANCE_BANDS: list[str] = ["0-10", "10-20", "20-30", "30-40", "40-50", "50-60", "60-70", "70+"]

METRIC_FOLDERS = {
    "throughput": "throughput",
    "signal_strength": "signal_strength",
}
METRIC_UNITS = {
    "throughput": "Mbps",
    "signal_strength": "dBm",
}
METRIC_THRESHOLD_DEFAULTS = {
    "throughput": 500.0,
    "signal_strength": -70.0,
}


FLOOR_ORDER = {"Ground Floor": 0, "Lower Floor": 1, "Upper Floor": 2}
BAND_ORDER = {"2.4 GHz": 0, "5 GHz": 1}

_FILE_RE = re.compile(
    r"(?P<router>.+?)_(?:Throughput|Signal Strength) for "
    r"(?P<floor>Ground Floor|Lower Floor|Upper Floor) on "
    r"(?P<band>2\.4 GHz|5 GHz) band_output\.csv",
    re.IGNORECASE,
)

def router_display_name(name: str) -> str:
    """
    Convert any router folder/file name into a readable display label.
    This is fully dynamic, so changing folder names automatically updates the UI.
    """
    if not name:
        return ""

    label = str(name).strip()

    label = re.sub(r"\.[A-Za-z0-9]+$", "", label)
    label = label.replace("_", " ").replace("-", " ")
    label = re.sub(r"\s+", " ", label).strip()

    return label

def _parse_filename(fname: str) -> dict[str, str] | None:
    match = _FILE_RE.match(fname)
    if not match:
        return None
    return {
        "router": match.group("router"),
        "floor": match.group("floor"),
        "band": match.group("band"),
    }


def _normalize_topology(name: str) -> str | None:
    value = name.strip().lower()
    if value == "with_mesh":
        return "with_mesh"
    if value == "without_mesh":
        return "without_mesh"
    return None


def _csv_to_series(path: Path) -> list[float]:
    """
    Read one CSV and reduce it to 8 distance-band mean values.
    Uses 'col' if present, else 'cx', else row order.
    """
    df = pd.read_csv(path)
    if "value" not in df.columns:
        return [float("nan")] * len(DISTANCE_POINTS)

    values = pd.to_numeric(df["value"], errors="coerce")
    valid = df.loc[values.notna()].copy()
    valid["value"] = values.loc[values.notna()]

    if valid.empty:
        return [float("nan")] * len(DISTANCE_POINTS)

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
    series = [float(grouped.get(i, float("nan"))) for i in range(len(DISTANCE_POINTS))]

    interpolated = pd.Series(series, dtype=float).interpolate(limit_area="inside")
    return [round(float(v), 2) if not math.isnan(v) else float("nan") for v in interpolated]


def _safe_mean(series: list[float]) -> float:
    arr = np.asarray(series, dtype=float)
    return float(np.nanmean(arr)) if np.isfinite(arr).any() else float("nan")


def _safe_max(series: list[float]) -> float:
    arr = np.asarray(series, dtype=float)
    return float(np.nanmax(arr)) if np.isfinite(arr).any() else float("nan")


def _coverage(series: list[float], threshold: float) -> int:
    arr = np.asarray(series, dtype=float)
    return int(np.nansum(arr >= threshold)) if np.isfinite(arr).any() else 0


def _consistency(series: list[float]) -> float:
    arr = np.asarray(series, dtype=float)
    return float(np.nanstd(arr, ddof=0)) if np.isfinite(arr).any() else float("nan")


def _drops(series: list[float]) -> list[float]:
    arr = np.asarray(series, dtype=float)
    result: list[float] = []
    for prev, curr in zip(arr[:-1], arr[1:]):
        if np.isnan(prev) or np.isnan(curr):
            result.append(0.0)
        else:
            result.append(round(max(prev - curr, 0.0), 2))
    return result


def _gain_series(mesh: list[float], nomesh: list[float]) -> list[float]:
    out: list[float] = []
    for m, n in zip(mesh, nomesh):
        if math.isnan(m) or math.isnan(n):
            out.append(float("nan"))
        else:
            out.append(round(m - n, 2))
    return out


def _primary_series(mesh: list[float], nomesh: list[float]) -> list[float]:
    mesh_ok = np.isfinite(np.asarray(mesh, dtype=float)).any()
    nomesh_ok = np.isfinite(np.asarray(nomesh, dtype=float)).any()

    if mesh_ok:
        return mesh
    if nomesh_ok:
        return nomesh
    return [float("nan")] * len(DISTANCE_POINTS)


def _primary_value(mesh_value: float, nomesh_value: float) -> float:
    if not math.isnan(mesh_value):
        return mesh_value
    return nomesh_value


@lru_cache(maxsize=4)
def load_all(data_root: str) -> dict[str, list[dict[str, Any]]]:
    """
    Scan nested metric/topology/router folders and load all CSV files.

    Expected layout:
        data_root/
            throughput/
                with_mesh/<router>/*.csv
                without_mesh/<router>/*.csv
            signal_strength/
                with_mesh/<router>/*.csv
                without_mesh/<router>/*.csv
    """
    root = Path(data_root)
    result: dict[str, list[dict[str, Any]]] = {
        "throughput": [],
        "signal_strength": [],
    }

    for metric_key, folder_name in METRIC_FOLDERS.items():
        metric_root = root / folder_name
        if not metric_root.exists():
            continue

        for csv_path in sorted(metric_root.rglob("*.csv")):
            topology = None
            router_from_path = None

            try:
                rel = csv_path.relative_to(metric_root)
                parts = rel.parts
                if len(parts) >= 3:
                    topology = _normalize_topology(parts[0])
                    router_from_path = parts[1]
            except Exception:
                pass

            if topology is None:
                continue

            meta = _parse_filename(csv_path.name)
            if meta is None:
                continue

            router = meta["router"]
            if router_from_path and router_from_path != router:
                router = router_from_path

            series = _csv_to_series(csv_path)

            result[metric_key].append(
                {
                    "router": router,
                    "floor": meta["floor"],
                    "band": meta["band"],
                    "topology": topology,
                    "series": series,
                    "mean": round(_safe_mean(series), 2),
                    "peak": round(_safe_max(series), 2),
                    "path": str(csv_path),
                }
            )

    return result


def get_options(data_root: str) -> dict[str, list[str]]:
    all_data = load_all(data_root)
    routers, floors, bands, metrics = set(), set(), set(), set()

    for metric, records in all_data.items():
        if records:
            metrics.add(metric)
        for row in records:
            routers.add(row["router"])
            floors.add(row["floor"])
            bands.add(row["band"])

    return {
        "routers": sorted(routers, key=str.lower),
        "floors": sorted(floors, key=lambda x: FLOOR_ORDER.get(x, 99)),
        "bands": sorted(bands, key=lambda x: BAND_ORDER.get(x, 99)),
        "metrics": [m for m in ["throughput", "signal_strength"] if m in metrics] or ["throughput", "signal_strength"],
    }


def build_payload(
    data_root: str,
    *,
    metric: str,
    floor: str,
    band: str,
    threshold: float | None = None,
) -> dict[str, Any]:
    """
    Build one dashboard payload for a specific metric / floor / band.
    Keeps backward-compatible keys for the current frontend and also adds
    richer mesh-vs-no-mesh comparison data.
    """
    all_data = load_all(data_root)
    metric_rows = [
        row for row in all_data.get(metric, [])
        if row["floor"] == floor and row["band"] == band
    ]

    if threshold is None:
        threshold = METRIC_THRESHOLD_DEFAULTS.get(metric, 500.0)

    unit = METRIC_UNITS.get(metric, "")
    routers = sorted({row["router"] for row in metric_rows}, key=str.lower)

    by_router: dict[str, dict[str, dict[str, Any]]] = {router: {} for router in routers}
    for row in metric_rows:
        by_router.setdefault(row["router"], {})[row["topology"]] = row

    series_primary: dict[str, list[float]] = {}
    series_compare: dict[str, dict[str, list[float]]] = {}
    kpis: dict[str, dict[str, Any]] = {}
    summary: list[dict[str, Any]] = []

    for router in routers:
        pair = by_router.get(router, {})
        mesh_row = pair.get("with_mesh")
        nomesh_row = pair.get("without_mesh")

        mesh_series = mesh_row["series"] if mesh_row else [float("nan")] * len(DISTANCE_POINTS)
        nomesh_series = nomesh_row["series"] if nomesh_row else [float("nan")] * len(DISTANCE_POINTS)

        primary_series = _primary_series(mesh_series, nomesh_series)
        gain_series = _gain_series(mesh_series, nomesh_series)

        avg_mesh = _safe_mean(mesh_series)
        avg_nomesh = _safe_mean(nomesh_series)
        peak_mesh = _safe_max(mesh_series)
        peak_nomesh = _safe_max(nomesh_series)

        coverage_mesh = _coverage(mesh_series, threshold)
        coverage_nomesh = _coverage(nomesh_series, threshold)

        drops_mesh = _drops(mesh_series)
        drops_nomesh = _drops(nomesh_series)
        drop_mesh_total = round(float(np.nansum(drops_mesh)), 2)
        drop_nomesh_total = round(float(np.nansum(drops_nomesh)), 2)

        consistency_mesh = _consistency(mesh_series)
        consistency_nomesh = _consistency(nomesh_series)

        gain_abs = (
            round(avg_mesh - avg_nomesh, 2)
            if not math.isnan(avg_mesh) and not math.isnan(avg_nomesh)
            else float("nan")
        )
        gain_pct = (
            round(((avg_mesh - avg_nomesh) / avg_nomesh) * 100.0, 2)
            if not math.isnan(avg_mesh) and not math.isnan(avg_nomesh) and avg_nomesh != 0
            else float("nan")
        )

        primary_avg = _primary_value(avg_mesh, avg_nomesh)
        primary_peak = _primary_value(peak_mesh, peak_nomesh)
        primary_coverage = coverage_mesh if mesh_row else coverage_nomesh
        primary_drops = drops_mesh if mesh_row else drops_nomesh
        primary_drop_total = drop_mesh_total if mesh_row else drop_nomesh_total
        primary_consistency = consistency_mesh if mesh_row else consistency_nomesh

        series_primary[router] = primary_series
        series_compare[router] = {
            "with_mesh": mesh_series,
            "without_mesh": nomesh_series,
            "gain": gain_series,
        }

        kpis[router] = {
            # Backward-compatible fields used by current dashboard.html
            "avg": round(primary_avg, 2) if not math.isnan(primary_avg) else None,
            "peak": round(primary_peak, 2) if not math.isnan(primary_peak) else None,
            "coverage": int(primary_coverage),
            "drops": primary_drops,
            "drop_total": primary_drop_total,
            "consistency": round(primary_consistency, 2) if not math.isnan(primary_consistency) else None,
            # Rich comparison fields for future mesh/no-mesh charts
            "avg_mesh": round(avg_mesh, 2) if not math.isnan(avg_mesh) else None,
            "avg_nomesh": round(avg_nomesh, 2) if not math.isnan(avg_nomesh) else None,
            "peak_mesh": round(peak_mesh, 2) if not math.isnan(peak_mesh) else None,
            "peak_nomesh": round(peak_nomesh, 2) if not math.isnan(peak_nomesh) else None,
            "coverage_mesh": int(coverage_mesh),
            "coverage_nomesh": int(coverage_nomesh),
            "drops_mesh": drops_mesh,
            "drops_nomesh": drops_nomesh,
            "drop_mesh_total": drop_mesh_total,
            "drop_nomesh_total": drop_nomesh_total,
            "consistency_mesh": round(consistency_mesh, 2) if not math.isnan(consistency_mesh) else None,
            "consistency_nomesh": round(consistency_nomesh, 2) if not math.isnan(consistency_nomesh) else None,
            "gain_abs": gain_abs if not math.isnan(gain_abs) else None,
            "gain_pct": gain_pct if not math.isnan(gain_pct) else None,
            "has_mesh": mesh_row is not None,
            "has_nomesh": nomesh_row is not None,
        }

        summary.append(
            {
                "router": router,
                "label": router_display_name(router),
                # Backward-compatible summary fields
                "avg": round(primary_avg, 2) if not math.isnan(primary_avg) else None,
                "peak": round(primary_peak, 2) if not math.isnan(primary_peak) else None,
                "coverage": int(primary_coverage),
                "drop_total": primary_drop_total,
                "consistency": round(primary_consistency, 2) if not math.isnan(primary_consistency) else None,
                # Rich comparison fields
                "avg_mesh": round(avg_mesh, 2) if not math.isnan(avg_mesh) else None,
                "avg_nomesh": round(avg_nomesh, 2) if not math.isnan(avg_nomesh) else None,
                "peak_mesh": round(peak_mesh, 2) if not math.isnan(peak_mesh) else None,
                "peak_nomesh": round(peak_nomesh, 2) if not math.isnan(peak_nomesh) else None,
                "coverage_mesh": int(coverage_mesh),
                "coverage_nomesh": int(coverage_nomesh),
                "drop_mesh_total": drop_mesh_total,
                "drop_nomesh_total": drop_nomesh_total,
                "consistency_mesh": round(consistency_mesh, 2) if not math.isnan(consistency_mesh) else None,
                "consistency_nomesh": round(consistency_nomesh, 2) if not math.isnan(consistency_nomesh) else None,
                "gain_abs": gain_abs if not math.isnan(gain_abs) else None,
                "gain_pct": gain_pct if not math.isnan(gain_pct) else None,
            }
        )

    summary.sort(key=lambda row: (row["label"] or row["router"]).lower())

    return {
        "meta": {
            "metric": metric,
            "floor": floor,
            "band": band,
            "unit": unit,
            "threshold": threshold,
            "distances": DISTANCE_POINTS,
            "bands": DISTANCE_BANDS,
        },
        "routers": routers,
        # Current frontend uses this
        "series": series_primary,
        # Future frontend can use this for real mesh comparison
        "series_compare": series_compare,
        "kpis": kpis,
        "summary": summary,
    }