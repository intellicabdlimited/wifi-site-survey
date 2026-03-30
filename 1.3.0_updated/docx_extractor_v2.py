from __future__ import annotations

import csv
import os
import re
import shutil
from io import BytesIO
from pathlib import Path
from typing import List, Optional, Set

from docx import Document
from docx.oxml.ns import qn
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P
from docx.table import Table
from docx.text.paragraph import Paragraph
from PIL import Image

from metadata_utils import canonical_metric_key, clean_router_name, normalize_spaces, parse_caption_metadata, parse_filename_metadata
from docx_extractor import extract_maps_best_side_save_one as legacy_extract_maps_best_side_save_one

NO_BAND_TYPES = {
    "survey routes and access points",
    "associated access point",
    "bluetooth coverage",
}

BAND_RE = re.compile(r"\bon\s+(?:the\s+)?(2\.4|5|6)\s*ghz\s+band\b", re.IGNORECASE)
TYPE_START_RE = re.compile(
    r"""
^(?P<type>
    signal\s*strength|
    (?:signal\s*(?:-|/)?\s*to\s*(?:-|/)?\s*noise\s*ratio)(?:\s*\(snr\))?|
    snr|
    noise|
    data\s*rate|
    throughput|
    channel\s*utilization|
    spectrum\s*channel\s*power
)\b
""",
    re.IGNORECASE | re.VERBOSE,
)

REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
OFFICE_NS = "urn:schemas-microsoft-com:office:office"


def norm_text(text: str) -> str:
    text = normalize_spaces(text)
    if text.endswith("."):
        text = text[:-1]
    return text



def canonical_type(type_text: str) -> str:
    s = (type_text or "").lower().replace("–", "-").replace("—", "-")
    s = re.sub(r"\s+", " ", s).strip()
    if s == "snr" or ("noise ratio" in s and "signal" in s):
        return "snr"
    return s



def is_caption_text(text: str) -> bool:
    s = norm_text(text)
    if not s:
        return False
    s2 = re.sub(r"^(figure|fig)\s*\d+\s*[:\-]\s*", "", s, flags=re.IGNORECASE).strip()
    m = TYPE_START_RE.match(s2)
    if not m:
        return False
    if re.search(r"\bfor\b", s2, re.IGNORECASE) is None:
        return False
    t_key = canonical_type(m.group("type"))
    if t_key not in NO_BAND_TYPES and BAND_RE.search(s2) is None:
        return False
    return True



def device_from_docx(docx_path: str) -> str:
    return clean_router_name(Path(docx_path).stem)



def safe_filename(text: str, max_len: int = 170) -> str:
    s = norm_text(text)
    s = re.sub(r"[\\/:*?\"<>|]+", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s[:max_len].rstrip() if len(s) > max_len else s



def unique_path(path: str) -> str:
    if not os.path.exists(path):
        return path
    root, ext = os.path.splitext(path)
    k = 2
    while True:
        p2 = f"{root}_{k}{ext}"
        if not os.path.exists(p2):
            return p2
        k += 1



def iter_block_items(parent):
    parent_elm = parent.element.body if hasattr(parent, "element") and hasattr(parent.element, "body") else parent._tc
    for child in parent_elm.iterchildren():
        if isinstance(child, CT_P):
            yield Paragraph(child, parent)
        elif isinstance(child, CT_Tbl):
            yield Table(child, parent)



def iter_all_paragraphs(doc: Document):
    def walk(parent):
        for item in iter_block_items(parent):
            if isinstance(item, Paragraph):
                yield item
            elif isinstance(item, Table):
                for row in item.rows:
                    for cell in row.cells:
                        yield from walk(cell)
    yield from walk(doc)



def paragraph_image_rids(p: Paragraph):
    el = p._p
    for blip in el.xpath(".//a:blip"):
        rid = blip.get(qn("r:embed"))
        if rid:
            yield rid
    for imdata in el.xpath(".//*[local-name()='imagedata']"):
        rid = imdata.get(f"{{{REL_NS}}}id") or imdata.get(f"{{{OFFICE_NS}}}relid")
        if rid:
            yield rid



def ext_from_part(part) -> str:
    ext = Path(str(part.partname)).suffix.lower().lstrip(".")
    return ext if ext else "png"



def get_dims(blob: bytes):
    try:
        im = Image.open(BytesIO(blob))
        return im.size[0], im.size[1]
    except Exception:
        return 0, 0



def looks_like_scale(w: int, h: int) -> bool:
    if w <= 0 or h <= 0:
        return False
    aspect = w / float(h)
    return (aspect >= 2.0 and h <= 260 and w >= 150)



def score(blob: bytes) -> int:
    w, h = get_dims(blob)
    return (w * h) if (w and h) else len(blob)



def _normalize_selected_parameters(selected_parameters: Optional[List[str]] = None) -> Optional[Set[str]]:
    if not selected_parameters:
        return None
    keys: Set[str] = set()
    for value in selected_parameters:
        key = canonical_metric_key(value or "")
        if key:
            keys.add(key)
    return keys or None



def _row_parameter_key(row: dict) -> str:
    for candidate in (row.get("parameter_key", ""), row.get("parameter_display", ""), Path(row.get("path", "")).name):
        key = canonical_metric_key(candidate or "")
        if key:
            return key
    return ""



def extract_maps_best_side_save_one(
    docx_path: str,
    out_dir: str,
    min_icon_area: int = 250,
    selected_parameters: Optional[List[str]] = None,
):
    """
    Legacy-compatible extraction wrapper.

    The upgraded pipeline keeps using the original extractor for the actual
    DOCX image harvesting because that path is known to recover the full image
    set from Ekahau Word exports. After extraction, this wrapper adds the
    metadata manifest used by the upgraded OCR/reporting pipeline.
    """
    os.makedirs(out_dir, exist_ok=True)

    legacy_result = legacy_extract_maps_best_side_save_one(
        docx_path,
        out_dir,
        MIN_ICON_AREA=min_icon_area,
    )

    saved_paths = list(legacy_result.get("saved_paths", []))
    selected_param_keys = _normalize_selected_parameters(selected_parameters)
    manifest_rows: List[dict] = []
    group_map = {}

    for path_str in saved_paths:
        path = Path(path_str)
        metadata = parse_filename_metadata(path.name)
        stem = path.stem
        base_stem = stem[:-6] if stem.lower().endswith('_scale') else stem
        if base_stem not in group_map:
            group_map[base_stem] = f"{clean_router_name(Path(docx_path).stem)}_{len(group_map) + 1:03d}"
        group_id = group_map[base_stem]

        role = 'scale' if stem.lower().endswith('_scale') else 'heatmap'
        caption_text = ''
        parameter_key = ''
        parameter_display = ''
        floor_name = ''
        band = ''
        router_key = clean_router_name(Path(docx_path).stem)

        if metadata is not None:
            caption_text = metadata.caption_text
            parameter_key = metadata.parameter_key
            parameter_display = metadata.parameter_display
            floor_name = metadata.floor_name
            band = metadata.band
            router_key = metadata.router_key or router_key

        manifest_rows.append({
            'group_id': group_id,
            'router_key': router_key,
            'parameter_key': parameter_key,
            'parameter_display': parameter_display,
            'floor_name': floor_name,
            'band': band,
            'role': role,
            'caption_text': caption_text,
            'source_docx': str(Path(docx_path).name),
            'path': str(path.resolve()),
        })

    if selected_param_keys is not None:
        selected_group_ids = {
            row['group_id']
            for row in manifest_rows
            if _row_parameter_key(row) in selected_param_keys
        }
        removed_rows = [row for row in manifest_rows if row['group_id'] not in selected_group_ids]
        for row in removed_rows:
            row_path = Path(row.get('path', ''))
            if row_path.exists():
                row_path.unlink()
        manifest_rows = [row for row in manifest_rows if row['group_id'] in selected_group_ids]
        saved_paths = [str(Path(row['path'])) for row in manifest_rows if Path(row['path']).exists()]
        legacy_result['selected_parameters'] = sorted(selected_param_keys)
        legacy_result['filtered_out_files'] = len(removed_rows)

    if manifest_rows:
        manifest_path = Path(out_dir) / '_extract_manifest.csv'
        with manifest_path.open('w', encoding='utf-8', newline='') as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    'group_id',
                    'router_key',
                    'parameter_key',
                    'parameter_display',
                    'floor_name',
                    'band',
                    'role',
                    'caption_text',
                    'source_docx',
                    'path',
                ],
            )
            writer.writeheader()
            writer.writerows(manifest_rows)

    legacy_result['saved_paths'] = saved_paths
    legacy_result['manifest_rows'] = len(manifest_rows)
    return legacy_result


def process_many_docx_local(
    docx_paths,
    out_root="out",
    download_per_docx_zip=False,
    also_make_master_zip=True,
    selected_parameters: Optional[List[str]] = None,
):
    os.makedirs(out_root, exist_ok=True)
    zip_dir = os.path.join(out_root, "_zips")
    os.makedirs(zip_dir, exist_ok=True)
    zip_files = []

    for docx_path in docx_paths:
        if not os.path.exists(docx_path):
            print(f"SKIP (not found): {docx_path}")
            continue

        stem = Path(docx_path).stem
        device = device_from_docx(docx_path)
        out_dir = os.path.join(out_root, f"{device}_{stem}")
        if os.path.exists(out_dir):
            shutil.rmtree(out_dir)
        os.makedirs(out_dir, exist_ok=True)

        print("\n====================================")
        print("DOCX:", docx_path)
        print("OUT :", out_dir)

        res = extract_maps_best_side_save_one(
            docx_path,
            out_dir,
            min_icon_area=250,
            selected_parameters=selected_parameters,
        )

        print("Matched captions:", res["matched_captions"])
        print("Saved files    :", len(res["saved_paths"]))
        print("Zero-image caps:", res["skipped_zero_image"])
        print("Manifest rows  :", res.get("manifest_rows", 0))

        zip_base = os.path.join(zip_dir, f"{device}_{stem}_extracted")
        zip_file = shutil.make_archive(zip_base, "zip", out_dir)
        print("ZIP created:", zip_file)
        zip_files.append(zip_file)

    if also_make_master_zip and zip_files:
        master_base = os.path.join(zip_dir, "ALL_EXTRACTED")
        master_zip = shutil.make_archive(master_base, "zip", out_root)
        print("MASTER ZIP created:", master_zip)
        zip_files.append(master_zip)

    return zip_files
