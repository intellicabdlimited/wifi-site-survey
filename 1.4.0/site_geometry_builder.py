from __future__ import annotations

import json
import os
import re
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np

from metadata_utils import clean_router_name


def esx_has_member(esx_path: Path, member_suffix: str) -> bool:
    try:
        with zipfile.ZipFile(esx_path, 'r') as zf:
            return any(name.lower().endswith(member_suffix.lower()) for name in zf.namelist())
    except Exception:
        return False


def find_master_esx(esx_paths: Iterable[Path]) -> Path:
    candidates = [Path(p) for p in esx_paths]
    if not candidates:
        raise FileNotFoundError('No ESX files were provided.')
    with_floorplans = [p for p in candidates if esx_has_member(p, 'floorPlans.json')]
    if not with_floorplans:
        raise FileNotFoundError('No uploaded ESX contains floorPlans.json.')
    return with_floorplans[0]


def read_esx_json_member(esx_path: Path, member_suffix: str) -> dict:
    with zipfile.ZipFile(esx_path, 'r') as zf:
        target = next((n for n in zf.namelist() if n.lower().endswith(member_suffix.lower())), None)
        if not target:
            raise FileNotFoundError(f'{member_suffix} not found in {esx_path.name}')
        return json.loads(zf.read(target).decode('utf-8'))


def sha256_file(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def extract_floor_images(master_esx: Path, out_dir: Path) -> Dict[str, Dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    fps = read_esx_json_member(master_esx, 'floorPlans.json').get('floorPlans', [])
    floors_meta: Dict[str, Dict] = {}
    with zipfile.ZipFile(master_esx, 'r') as zf:
        names = set(zf.namelist())
        for fp in fps:
            name = str(fp.get('name', '')).strip()
            image_id = fp.get('imageId')
            member = f'image-{image_id}'
            if not name or member not in names:
                continue
            safe = re.sub(r'[^A-Za-z0-9_-]+', '_', name).strip('_') or 'floor'
            out_path = out_dir / f'{safe}.png'
            if out_path.exists():
                img = cv2.imread(str(out_path))
            else:
                img = cv2.imdecode(np.frombuffer(zf.read(member), np.uint8), cv2.IMREAD_COLOR)
                if img is None:
                    continue
                cv2.imwrite(str(out_path), img)
            if img is None:
                continue
            h, w = img.shape[:2]
            floors_meta[name] = {
                'floorPlanId': fp.get('id'),
                'metersPerUnit': float(fp.get('metersPerUnit', 1.0) or 1.0),
                'img_path': str(out_path),
                'w': int(w),
                'h': int(h),
            }
    if not floors_meta:
        raise FileNotFoundError(f'No floor images could be extracted from {master_esx.name}.')
    return floors_meta


def pick_points_local(image_path: str, n_points: int = 1, title: str = 'Click points') -> np.ndarray:
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(image_path)

    pts: List[Tuple[float, float]] = []
    disp = img.copy()
    win = f'{title} :: {os.path.basename(image_path)}'
    try:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    except cv2.error as exc:
        raise RuntimeError(
            'Interactive site geometry creation needs desktop OpenCV window support. '
            'Install opencv-python and run the app in a desktop session, or upload an existing site_geometry.json.'
        ) from exc

    instruction = f'Click {n_points} point(s). ESC cancels.'

    def _draw_overlay() -> None:
        nonlocal disp
        disp = img.copy()
        cv2.putText(disp, instruction, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
        for idx, (x, y) in enumerate(pts):
            xi = int(round(x))
            yi = int(round(y))
            cv2.circle(disp, (xi, yi), 7, (0, 0, 255), -1)
            cv2.putText(disp, str(idx + 1), (xi + 10, yi - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

    def _cb(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            pts.append((float(x), float(y)))
            _draw_overlay()
            cv2.imshow(win, disp)

    _draw_overlay()
    cv2.setMouseCallback(win, _cb)
    cv2.imshow(win, disp)

    while True:
        key = cv2.waitKey(20) & 0xFF
        if key == 27:
            cv2.destroyWindow(win)
            raise KeyboardInterrupt('Point picking cancelled (ESC).')
        if len(pts) >= int(n_points):
            cv2.destroyWindow(win)
            break

    return np.array(pts, dtype=float)


def infer_base_floor(floors_meta: Dict[str, Dict]) -> str:
    if 'Kitchen' in floors_meta:
        return 'Kitchen'
    return sorted(floors_meta.keys())[0]


def _infer_router_key_from_filename(path: Path) -> str:
    stem = Path(path).stem
    match = re.match(r'^([A-Za-z0-9_-]+)\s+', stem)
    if match:
        return clean_router_name(match.group(1))
    return clean_router_name(stem)


def _find_ap_placement_files(esx_path: Path) -> List[str]:
    with zipfile.ZipFile(esx_path, 'r') as zf:
        names = zf.namelist()
    patt = re.compile(r'access.*(placement|location|position).*\.json$', re.IGNORECASE)
    return [n for n in names if patt.search(n)]


def _extract_xy_floor_from_obj(obj) -> List[Tuple[float, float, Optional[str]]]:
    out: List[Tuple[float, float, Optional[str]]] = []

    def rec(o, inherited_floor=None):
        if isinstance(o, dict):
            floor = o.get('floorPlanId') or o.get('floorplanId') or o.get('floor_id') or inherited_floor
            if ('x' in o) and ('y' in o) and isinstance(o['x'], (int, float)) and isinstance(o['y'], (int, float)):
                out.append((float(o['x']), float(o['y']), floor))
            for v in o.values():
                rec(v, floor)
        elif isinstance(o, list):
            for v in o:
                rec(v, inherited_floor)

    rec(obj, None)
    return out


def extract_mine_ap_candidates(esx_path: Path, base_floor_name: str) -> List[Dict]:
    try:
        fp = read_esx_json_member(esx_path, 'floorPlans.json').get('floorPlans', [])
        fp_by_id = {f.get('id'): f for f in fp}
    except Exception:
        fp_by_id = {}

    try:
        aps = read_esx_json_member(esx_path, 'accessPoints.json').get('accessPoints', [])
    except Exception:
        return []

    mine_aps = [ap for ap in aps if ap.get('mine') is True]
    if not mine_aps:
        return []

    placement_files = _find_ap_placement_files(esx_path)
    candidates: List[Dict] = []
    for ap in mine_aps:
        ap_id = ap.get('id')
        ap_name = ap.get('name', 'AP')
        xys = _extract_xy_floor_from_obj(ap)
        if not xys and placement_files:
            for member in placement_files:
                try:
                    data = read_esx_json_member(esx_path, member)
                except Exception:
                    continue
                hits = []

                def rec(o):
                    if isinstance(o, dict):
                        if o.get('accessPointId') == ap_id or o.get('access_point_id') == ap_id:
                            hits.append(o)
                        for v in o.values():
                            rec(v)
                    elif isinstance(o, list):
                        for v in o:
                            rec(v)

                rec(data)
                for hit in hits:
                    xys.extend(_extract_xy_floor_from_obj(hit))

        best = None
        for x, y, fpid in xys:
            if fpid is not None:
                best = (x, y, fpid)
                break
        if best is None and xys:
            best = xys[0]
        if best is None:
            continue

        x, y, fpid = best
        fpinfo = fp_by_id.get(fpid, {})
        floor_name = fpinfo.get('name', str(fpid))
        mpu = float(fpinfo.get('metersPerUnit', 1.0) or 1.0)
        candidates.append(
            {
                'ap_id': ap_id,
                'ap_name': ap_name,
                'floorPlanId': fpid,
                'floorPlanName': floor_name,
                'x': float(x),
                'y': float(y),
                'metersPerUnit': mpu,
            }
        )

    base = [c for c in candidates if str(c.get('floorPlanName', '')).strip().lower() == str(base_floor_name).strip().lower()]
    return base if base else candidates


def _dist_m(c1: Dict, c2: Dict) -> float:
    x1m = float(c1['x']) * float(c1.get('metersPerUnit', 1.0) or 1.0)
    y1m = float(c1['y']) * float(c1.get('metersPerUnit', 1.0) or 1.0)
    x2m = float(c2['x']) * float(c2.get('metersPerUnit', 1.0) or 1.0)
    y2m = float(c2['y']) * float(c2.get('metersPerUnit', 1.0) or 1.0)
    return float(np.hypot(x1m - x2m, y1m - y2m))


def choose_consistent_dut_per_router(router_to_candidates: Dict[str, List[Dict]], base_floor_name: str, max_iters: int = 8) -> Dict[str, Dict]:
    all_pts: List[Tuple[float, float]] = []
    for candidates in router_to_candidates.values():
        for c in candidates:
            if str(c.get('floorPlanName', '')).strip().lower() == str(base_floor_name).strip().lower():
                all_pts.append((float(c['x']) * float(c.get('metersPerUnit', 1.0)), float(c['y']) * float(c.get('metersPerUnit', 1.0))))
    if not all_pts:
        for candidates in router_to_candidates.values():
            for c in candidates:
                all_pts.append((float(c['x']) * float(c.get('metersPerUnit', 1.0)), float(c['y']) * float(c.get('metersPerUnit', 1.0))))

    cx = float(np.mean([p[0] for p in all_pts])) if all_pts else 0.0
    cy = float(np.mean([p[1] for p in all_pts])) if all_pts else 0.0

    selected: Dict[str, Dict] = {}
    for rk, candidates in router_to_candidates.items():
        if not candidates:
            continue
        selected[rk] = min(
            candidates,
            key=lambda c: float(np.hypot(float(c['x']) * float(c.get('metersPerUnit', 1.0)) - cx, float(c['y']) * float(c.get('metersPerUnit', 1.0)) - cy)),
        )

    for _ in range(max_iters):
        changed = False
        for rk, candidates in router_to_candidates.items():
            if not candidates:
                continue
            others = [selected[k] for k in selected if k != rk]
            if not others:
                continue

            def mean_dist(c: Dict) -> float:
                dists = [
                    _dist_m(c, o)
                    for o in others
                    if str(c.get('floorPlanName', '')).strip().lower() == str(base_floor_name).strip().lower()
                    and str(o.get('floorPlanName', '')).strip().lower() == str(base_floor_name).strip().lower()
                ]
                if not dists:
                    dists = [_dist_m(c, o) for o in others]
                return float(np.mean(dists)) if dists else 1e18

            best = min(candidates, key=mean_dist)
            if best.get('ap_id') != selected[rk].get('ap_id'):
                selected[rk] = best
                changed = True
        if not changed:
            break
    return selected


def _build_router_map(router_esx_paths: Iterable[Path]) -> Dict[str, Path]:
    router_map: Dict[str, Path] = {}
    for path in router_esx_paths:
        rk = _infer_router_key_from_filename(Path(path))
        if rk:
            router_map[rk] = Path(path)
    return router_map


def create_site_geometry(
    *,
    master_esx: Path,
    router_esx_paths: Iterable[Path],
    output_path: Path,
    base_floor: Optional[str] = None,
    align_to_map: Optional[Dict[str, str]] = None,
    force_recreate: bool = True,
) -> Path:
    output_path = Path(output_path)
    if output_path.exists() and not force_recreate:
        return output_path

    router_esx_paths = [Path(p) for p in router_esx_paths]
    floor_image_dir = output_path.parent / '_site_geometry_floorplans'
    floors_meta = extract_floor_images(Path(master_esx), floor_image_dir)
    if not floors_meta:
        raise ValueError('No floor plans were available in the selected master ESX.')

    align_to_map = dict(align_to_map or {})
    base_floor = base_floor or infer_base_floor(floors_meta)
    if base_floor not in floors_meta:
        raise ValueError(f'Base floor {base_floor!r} was not found in master ESX floor plans.')

    site = {
        'version': 3,
        'master_esx_filename': Path(master_esx).name,
        'master_esx_sha256': sha256_file(Path(master_esx)),
        'base_floor': base_floor,
        'links': {},
        'dut_px_by_floor': {},
        'dut_px_by_router': {},
        'floor_elevation_m': {},
        'notes': 'anchors are in floor image pixel coords; transforms built in meters; links can chain.',
    }

    floors_to_use = list(floors_meta.keys())
    for floor_name in floors_to_use:
        if floor_name == base_floor:
            continue
        parent = align_to_map.get(floor_name, base_floor)
        if parent not in floors_meta:
            raise ValueError(f'Alignment parent {parent!r} for floor {floor_name!r} is missing from the master ESX.')
        child_pts = pick_points_local(
            floors_meta[floor_name]['img_path'],
            2,
            f'[{floor_name}] Click 2 anchors on THIS floor: (1) stair, (2) reference/corner',
        )
        parent_pts = pick_points_local(
            floors_meta[parent]['img_path'],
            2,
            f'[{parent}] Click the SAME 2 physical points matching {floor_name}: (1) stair, (2) reference/corner',
        )
        site['links'][floor_name] = {
            'align_to': parent,
            'labels': ['stair', 'ref'],
            'child_anchors_px': {
                'stair_px': [float(child_pts[0][0]), float(child_pts[0][1])],
                'ref_px': [float(child_pts[1][0]), float(child_pts[1][1])],
            },
            'parent_anchors_px': {
                'stair_px': [float(parent_pts[0][0]), float(parent_pts[0][1])],
                'ref_px': [float(parent_pts[1][0]), float(parent_pts[1][1])],
            },
        }

    router_map = _build_router_map(router_esx_paths)
    router_to_candidates: Dict[str, List[Dict]] = {}
    for router_key, esx_path in router_map.items():
        router_to_candidates[router_key] = extract_mine_ap_candidates(esx_path, base_floor)

    selected = choose_consistent_dut_per_router(router_to_candidates, base_floor)
    for router_key, candidate in selected.items():
        site['dut_px_by_router'][router_key] = {
            'base_floor': base_floor,
            'floorPlanId': candidate.get('floorPlanId'),
            'floorPlanName': candidate.get('floorPlanName'),
            'dut_px': [float(candidate['x']), float(candidate['y'])],
            'source': {
                'esx_filename': router_map[router_key].name,
                'ap_id': candidate.get('ap_id'),
                'ap_name': candidate.get('ap_name'),
                'method': 'mine_true_ap_position',
            },
        }

    if base_floor not in site['dut_px_by_floor']:
        preferred = next(iter(site['dut_px_by_router'].keys()), None)
        if preferred:
            site['dut_px_by_floor'][base_floor] = list(site['dut_px_by_router'][preferred]['dut_px'])

    if base_floor not in site['dut_px_by_floor']:
        dut = pick_points_local(
            floors_meta[base_floor]['img_path'],
            1,
            f'[{base_floor}] Click DUT position (fallback)',
        )[0]
        site['dut_px_by_floor'][base_floor] = [float(dut[0]), float(dut[1])]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(site, indent=2), encoding='utf-8')
    return output_path


__all__ = [
    'create_site_geometry',
    'find_master_esx',
]
