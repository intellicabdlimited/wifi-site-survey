from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from metadata_utils import (
    AssetMetadata,
    canonical_band,
    canonical_metric_key,
    clean_router_name,
    extract_floor_name,
    normalize_band_value,
    normalize_floor_name,
    parse_filename_metadata,
)


@dataclass
class AssetRecord:
    metadata: AssetMetadata
    path: Path


def _is_summary_asset(path: Path) -> bool:
    lower = path.name.lower()
    return any(keyword in lower for keyword in ["summary", "overview", "report", "total", "aggregate"])


class AssetRegistry:
    def __init__(self, records: Iterable[AssetRecord]):
        self.records: List[AssetRecord] = list(records)
        self._index: Dict[Tuple[str, str, str, str], Dict[str, Path]] = {}
        for record in self.records:
            md = record.metadata
            pkey = canonical_metric_key(md.parameter_key)
            if not pkey:
                continue
            key = (
                clean_router_name(md.router_key),
                pkey,
                normalize_floor_name(md.floor_name),
                normalize_band_value(md.band),
            )
            entry = self._index.setdefault(key, {})
            existing = entry.get(md.role)
            if existing is None:
                entry[md.role] = record.path
            else:
                # Prefer non-summary assets over summary/overview pages.
                if _is_summary_asset(existing) and not _is_summary_asset(record.path):
                    entry[md.role] = record.path

    @classmethod
    def from_roots(
        cls,
        extracted_root: Path | str | None = None,
        csv_outputs_root: Path | str | None = None,
    ) -> "AssetRegistry":
        records: List[AssetRecord] = []

        if csv_outputs_root:
            csv_root = Path(csv_outputs_root)
            if csv_root.exists():
                for index_csv in sorted(csv_root.rglob("_index.csv")):
                    try:
                        with index_csv.open("r", encoding="utf-8", newline="") as handle:
                            reader = csv.DictReader(handle)
                            for row in reader:
                                pkey = canonical_metric_key(row.get("parameter_key") or row.get("parameter_display", ""))
                                if not pkey:
                                    continue

                                floor = normalize_floor_name(row.get("floor_name", ""))
                                band = normalize_band_value(row.get("band", ""))
                                if (not floor or not band) and row.get("caption_text"):
                                    floor = floor or extract_floor_name(row.get("caption_text", ""))
                                    band = band or canonical_band(row.get("caption_text", ""))

                                if (not floor or not band) and row.get("heatmap"):
                                    try:
                                        heatmap_name = str(row.get("heatmap", "")).replace("\n", " ").strip()
                                        parsed = parse_filename_metadata(Path(heatmap_name).name)
                                        if parsed:
                                            floor = floor or parsed.floor_name
                                            band = band or parsed.band
                                    except Exception:
                                        parsed = None

                                if not floor or not band:
                                    # If we can't reliably deduce, keep the row for audit but don't index it.
                                    continue

                                base_md = AssetMetadata(
                                    router_key=clean_router_name(row.get("router_key", "")),
                                    parameter_key=pkey,
                                    parameter_display=row.get("parameter_display", ""),
                                    floor_name=floor,
                                    band=band,
                                    role="heatmap",
                                    caption_text=row.get("caption_text", ""),
                                    source_docx=row.get("source_docx", ""),
                                    group_id=row.get("group_id", ""),
                                    path=row.get("heatmap_path", ""),
                                )
                                heatmap_path = Path(row.get("heatmap_path", ""))
                                if heatmap_path.exists():
                                    records.append(AssetRecord(metadata=base_md, path=heatmap_path))
                                scale_path = Path(row.get("scale_path", "")) if row.get("scale_path") else None
                                if scale_path and scale_path.exists():
                                    scale_md = AssetMetadata(
                                        router_key=base_md.router_key,
                                        parameter_key=base_md.parameter_key,
                                        parameter_display=base_md.parameter_display,
                                        floor_name=base_md.floor_name,
                                        band=base_md.band,
                                        role="scale",
                                        caption_text=base_md.caption_text,
                                        source_docx=base_md.source_docx,
                                        group_id=base_md.group_id,
                                        path=str(scale_path),
                                    )
                                    records.append(AssetRecord(metadata=scale_md, path=scale_path))
                    except Exception:
                        continue

        if not records and extracted_root:
            root = Path(extracted_root)
            if root.exists():
                manifest_files = sorted(root.rglob("_extract_manifest.csv"))
                if manifest_files:
                    for manifest in manifest_files:
                        with manifest.open("r", encoding="utf-8", newline="") as handle:
                            reader = csv.DictReader(handle)
                            for row in reader:
                                path = Path(row.get("path", ""))
                                if not path.is_absolute():
                                    path = (manifest.parent / path).resolve()
                                pkey = canonical_metric_key(row.get("parameter_key") or row.get("parameter_display", ""))
                                md = AssetMetadata(
                                    router_key=row.get("router_key", ""),
                                    parameter_key=pkey or "",
                                    parameter_display=row.get("parameter_display", ""),
                                    floor_name=normalize_floor_name(row.get("floor_name", "")),
                                    band=normalize_band_value(row.get("band", "")),
                                    role=row.get("role", "heatmap"),
                                    caption_text=row.get("caption_text", ""),
                                    source_docx=row.get("source_docx", ""),
                                    group_id=row.get("group_id", ""),
                                    path=str(path),
                                )
                                if path.exists() and md.parameter_key and md.floor_name and md.band:
                                    records.append(AssetRecord(metadata=md, path=path))
                else:
                    for file_path in sorted(root.rglob("*")):
                        if not file_path.is_file() or file_path.name.startswith("_"):
                            continue
                        md = parse_filename_metadata(file_path.name)
                        if md:
                            records.append(AssetRecord(metadata=md, path=file_path))
        return cls(records)

    def get_pair(
        self,
        router_key: str,
        parameter_key: str,
        floor_name: str,
        band: str,
    ) -> Tuple[Optional[Path], Optional[Path]]:
        pkey = canonical_metric_key(parameter_key)
        key = (
            clean_router_name(router_key),
            pkey or "",
            normalize_floor_name(floor_name),
            normalize_band_value(band),
        )
        entry = self._index.get(key, {})
        return entry.get("heatmap"), entry.get("scale")

    def list_router_keys(self) -> List[str]:
        return sorted({k[0] for k in self._index})
