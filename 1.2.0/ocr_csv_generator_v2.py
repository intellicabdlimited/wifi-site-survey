from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Set

import pandas as pd

import ocr_csv_generator as core
from metadata_utils import (
    PARAM_PRETTY,
    canonical_metric_key,
    clean_router_name,
    normalize_band_value,
    normalize_floor_name,
    to_core_metric_key,
)

IMG_EXTS = core.IMG_EXTS
ALLOWED_METRIC_KEYS = core.ALLOWED_METRIC_KEYS
METRIC_CFG_BY_KEY = core.METRIC_CFG_BY_KEY


def _normalize_selected_parameters(selected_parameters: Optional[List[str]] = None) -> Optional[Set[str]]:
    if not selected_parameters:
        return None
    keys: Set[str] = set()
    for value in selected_parameters:
        key = canonical_metric_key(value or "")
        if key:
            keys.add(key)
    return keys or None



def _filter_jobs_by_parameter(jobs: List[Dict], selected_parameters: Optional[List[str]] = None) -> List[Dict]:
    selected_keys = _normalize_selected_parameters(selected_parameters)
    if not selected_keys:
        return jobs
    filtered: List[Dict] = []
    for job in jobs:
        key = canonical_metric_key(job.get("parameter_key") or job.get("parameter_display", ""))
        if key in selected_keys:
            filtered.append(job)
    return filtered



def _load_manifest_pairs(extracted_root: str) -> List[Dict]:
    root = Path(extracted_root)
    rows: List[Dict] = []
    manifest_files = sorted(root.rglob("_extract_manifest.csv"))
    if not manifest_files:
        return rows

    for manifest in manifest_files:
        with manifest.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            grouped: Dict[str, List[dict]] = {}
            for row in reader:
                grouped.setdefault(row.get("group_id", ""), []).append(row)

        for group_id, items in grouped.items():
            heatmap = next((r for r in items if r.get("role") == "heatmap"), None)
            scale = next((r for r in items if r.get("role") == "scale"), None)
            if not heatmap:
                continue
            project_key = canonical_metric_key(heatmap.get("parameter_key") or heatmap.get("parameter_display", ""))
            if not project_key:
                project_key = canonical_metric_key(core.metric_key_from_filename(Path(heatmap.get("path", "")).name) or "")
            core_key = to_core_metric_key(project_key or "")
            if not project_key or not core_key or core_key not in ALLOWED_METRIC_KEYS:
                continue
            heatmap_path = Path(heatmap.get("path", ""))
            scale_path = Path(scale.get("path", "")) if scale else None
            if not heatmap_path.exists():
                continue
            rows.append(
                {
                    "group_id": group_id,
                    "router_key": clean_router_name(heatmap.get("router_key", "")),
                    "parameter_key": project_key,
                    "core_metric_key": core_key,
                    "parameter_display": heatmap.get("parameter_display") or PARAM_PRETTY.get(project_key, project_key.replace("_", " ").title()),
                    "floor_name": normalize_floor_name(heatmap.get("floor_name", "")),
                    "band": normalize_band_value(heatmap.get("band", "")),
                    "heatmap_path": str(heatmap_path),
                    "scale_path": str(scale_path) if scale_path and scale_path.exists() else "",
                    "caption_text": heatmap.get("caption_text", ""),
                    "source_docx": heatmap.get("source_docx", ""),
                }
            )
    return rows


def _fallback_heatmap_rows(extracted_root: str) -> List[Dict]:
    heatmaps = core.scan_heatmaps_extracted(extracted_root)
    rows: List[Dict] = []
    for hm_path in heatmaps:
        hm_name = Path(hm_path).name
        project_key = canonical_metric_key(core.metric_key_from_filename(hm_name) or "")
        core_key = to_core_metric_key(project_key or "")
        if not project_key or not core_key or core_key not in ALLOWED_METRIC_KEYS:
            continue
        scale_path = core.find_matching_scale(hm_path)
        rows.append(
            {
                "group_id": Path(hm_path).stem,
                "router_key": clean_router_name(core.device_from_filename(hm_path)),
                "parameter_key": project_key,
                "core_metric_key": core_key,
                "parameter_display": PARAM_PRETTY.get(project_key, project_key.replace("_", " ").title()),
                "floor_name": "",
                "band": normalize_band_value(core.detect_band_from_name(hm_name) or ""),
                "heatmap_path": hm_path,
                "scale_path": scale_path,
                "caption_text": Path(hm_path).stem,
                "source_docx": "",
            }
        )
    return rows


def _iter_heatmap_jobs(extracted_root: str) -> List[Dict]:
    rows = _load_manifest_pairs(extracted_root)
    return rows if rows else _fallback_heatmap_rows(extracted_root)


def run_ocr_generate_csv(
    extracted_root: str,
    csv_out_root: str,
    *,
    debug: bool = False,
    max_heatmaps: Optional[int] = None,
    selected_parameters: Optional[List[str]] = None,
) -> dict:
    core.DEBUG = bool(debug)

    out_dir = Path(csv_out_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    jobs = _filter_jobs_by_parameter(_iter_heatmap_jobs(extracted_root), selected_parameters)
    if max_heatmaps is not None:
        jobs = jobs[: int(max_heatmaps)]

    meta_rows: List[dict] = []
    failed: List[dict] = []

    debug_root = out_dir / "_debug"
    if core.DEBUG:
        debug_root.mkdir(parents=True, exist_ok=True)

    for job in jobs:
        hm_path = job["heatmap_path"]
        hm_name = Path(hm_path).name
        hm_stem = Path(hm_name).stem
        tag = f"{core.slug(hm_stem)}__"
        project_key = job["parameter_key"]
        core_key = job["core_metric_key"]
        metric_cfg = METRIC_CFG_BY_KEY.get(core_key)
        if not metric_cfg:
            continue
        scale_path = job.get("scale_path") or ""

        try:
            if metric_cfg.get("scale_kind") != "categorical" and not scale_path:
                raise FileNotFoundError(f"No scale image available for {hm_name}")

            hm_debug_dir = None
            if core.DEBUG:
                hm_debug_dir = debug_root / core.slug(hm_stem)
                hm_debug_dir.mkdir(parents=True, exist_ok=True)
                with open(hm_debug_dir / "pairing.json", "w", encoding="utf-8") as f:
                    json.dump(job, f, indent=2)

            hm_rgb = core.read_image_rgb(hm_path)
            scale_rgb = core.read_image_rgb(scale_path) if scale_path else None

            if core_key == "snr":
                df_hex, _ = core.extract_hexagons(hm_rgb, roi_sat_thresh=6, roi_val_thresh=254)
            else:
                df_hex, _ = core.extract_hexagons(hm_rgb)

            df_hex = core.assign_row_major_ids(df_hex)
            df_hex, snapped_uniform = core.snap_uniform_hex_colors(df_hex, de_thresh=3.0, min_frac=0.95)

            if core.DEBUG and hm_debug_dir:
                with open(hm_debug_dir / f"{tag}uniform_snap.json", "w", encoding="utf-8") as f:
                    json.dump({"snapped_uniform": bool(snapped_uniform)}, f, indent=2)

            out_csv = out_dir / f"{hm_stem}_output.csv"

            if metric_cfg.get("scale_kind") == "categorical":
                cat_model = core.build_categorical_scale_model(
                    scale_rgb,
                    reverse_lr=False,
                    debug_dir=str(hm_debug_dir) if hm_debug_dir else None,
                    tag=tag,
                )
                df_vals = core.map_hex_to_categorical(
                    df_hex,
                    cat_model,
                    debug_dir=str(hm_debug_dir) if hm_debug_dir else None,
                    tag=tag,
                )
                core.write_clean_output_csv(df_vals, str(out_csv), is_categorical=True)
                meta = {"scale_kind": "categorical", "n_bins": int(len(cat_model.steps))}
            else:
                mn, mx = core.ocr_minmax_from_scale_path(
                    scale_path,
                    metric_cfg,
                    debug_dir=str(hm_debug_dir) if hm_debug_dir else None,
                    tag=tag,
                )
                num_model = core.build_numeric_scale_model(
                    scale_rgb,
                    metric_cfg,
                    reverse_lr=False,
                    mn=mn,
                    mx=mx,
                    debug_dir=str(hm_debug_dir) if hm_debug_dir else None,
                    tag=tag,
                )
                df_vals = core.map_hex_to_numeric(
                    df_hex,
                    num_model,
                    debug_dir=str(hm_debug_dir) if hm_debug_dir else None,
                    tag=tag,
                )
                core.write_clean_output_csv(df_vals, str(out_csv), is_categorical=False)
                meta = {
                    "scale_kind": "numeric",
                    "numeric_kind": num_model.kind,
                    "mn": float(num_model.mn),
                    "mx": float(num_model.mx),
                    "n_bins": int(len(num_model.steps)),
                }

            meta_rows.append(
                {
                    "group_id": job.get("group_id", ""),
                    "router_key": job.get("router_key", ""),
                    "parameter_key": project_key,
                    "parameter_display": job.get("parameter_display", ""),
                    "floor_name": job.get("floor_name", ""),
                    "band": job.get("band", ""),
                    "heatmap": hm_name,
                    "heatmap_path": hm_path,
                    "scale_path": scale_path,
                    "caption_text": job.get("caption_text", ""),
                    "source_docx": job.get("source_docx", ""),
                    "csv": str(out_csv),
                    **meta,
                }
            )
        except Exception as exc:
            failed.append(
                {
                    "group_id": job.get("group_id", ""),
                    "heatmap": hm_name,
                    "floor_name": job.get("floor_name", ""),
                    "band": job.get("band", ""),
                    "parameter_key": project_key,
                    "error": str(exc),
                }
            )

    index_csv = out_dir / "_index.csv"
    pd.DataFrame(meta_rows).to_csv(index_csv, index=False)

    failed_csv = out_dir / "_failed.csv"
    if failed:
        pd.DataFrame(failed).to_csv(failed_csv, index=False)

    return {
        "csv_out_root": str(out_dir),
        "index_csv": str(index_csv),
        "failed_csv": str(failed_csv) if failed else "",
        "processed": len(meta_rows),
        "failed_count": len(failed),
        "used_manifest": bool(_load_manifest_pairs(extracted_root)),
        "selected_parameters": sorted(_normalize_selected_parameters(selected_parameters) or []),
    }
