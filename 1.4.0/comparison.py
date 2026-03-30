# from pathlib import Path
# import os
# import re
# import json
# import zipfile
# import shutil
# import sys
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt

# # Ensure project root is importable from wherever script is executed.
# script_dir = Path(__file__).resolve().parent
# project_root = script_dir
# while project_root != project_root.parent and not (project_root / "metadata_utils.py").exists():
#     project_root = project_root.parent
# if (project_root / "metadata_utils.py").exists():
#     sys.path.insert(0, str(project_root))

# from metadata_utils import clean_router_name

# SCRIPT_DIR = Path(__file__).resolve().parent

# WITH_DIR = Path(os.environ.get("COMPARE_WITH_DIR", str(SCRIPT_DIR / "inputs_compare" / "with_mesh")))
# WO_DIR   = Path(os.environ.get("COMPARE_WITHOUT_DIR", str(SCRIPT_DIR / "inputs_compare" / "without_mesh")))
# OUT_DIR  = Path(os.environ.get("COMPARE_OUT_DIR", str(SCRIPT_DIR / "outputs_compare_rvr_style")))
# ZIP_PATH = Path(os.environ.get("COMPARE_ZIP_PATH", str(SCRIPT_DIR / "compare_overlay_outputs.zip")))

# SITE_GEOM_PATH = os.environ.get("COMPARE_SITE_GEOM", "").strip()
# MASTER_ESX_PATH = os.environ.get("COMPARE_MASTER_ESX", "").strip()
# ESX_DIR = Path(os.environ.get("COMPARE_ESX_DIR", str(SCRIPT_DIR)))

# WITH_DOCX_DIR = Path(os.environ.get("COMPARE_WITH_DOCX_DIR", str(SCRIPT_DIR)))
# WITHOUT_DOCX_DIR = Path(os.environ.get("COMPARE_WITHOUT_DOCX_DIR", str(SCRIPT_DIR)))

# BIN_FT = 2.0
# Y_STAT = "p50"
# ZERO_INSERT_METHOD = "linear"
# CLAMP_ZERO_TO_DATA_RANGE = True
# FEET_PER_METER = 3.280839895

# PARAM_PRESETS = {
#     "SNR": {"PARAM_KEY": "snr", "PARAM_PRETTY": "SNR", "PARAM_UNIT": "dB"},
#     "Throughput": {"PARAM_KEY": "throughput", "PARAM_PRETTY": "Throughput", "PARAM_UNIT": "Mbps"},
#     "Data Rate": {"PARAM_KEY": "data_rate", "PARAM_PRETTY": "Data Rate", "PARAM_UNIT": "Mbps"},
#     "Signal Strength": {"PARAM_KEY": "signal_strength", "PARAM_PRETTY": "Signal Strength", "PARAM_UNIT": "dBm"},
#     "Secondary Signal Strength": {"PARAM_KEY": "secondary_signal_strength", "PARAM_PRETTY": "Secondary Signal Strength", "PARAM_UNIT": "dBm"},
#     "Tertiary Signal Strength": {"PARAM_KEY": "tertiary_signal_strength", "PARAM_PRETTY": "Tertiary Signal Strength", "PARAM_UNIT": "dBm"},
#     "Noise": {"PARAM_KEY": "noise", "PARAM_PRETTY": "Noise", "PARAM_UNIT": "dBm"},
#     "Channel Utilization": {"PARAM_KEY": "channel_utilization", "PARAM_PRETTY": "Channel Utilization", "PARAM_UNIT": "%"},
#     "Channel Interference": {"PARAM_KEY": "channel_interference", "PARAM_PRETTY": "Channel Interference", "PARAM_UNIT": "dB"},
#     "Channel Width": {"PARAM_KEY": "channel_width", "PARAM_PRETTY": "Channel Width", "PARAM_UNIT": "MHz"},
#     "Spectrum Channel Power": {"PARAM_KEY": "spectrum_channel_power", "PARAM_PRETTY": "Spectrum Channel Power", "PARAM_UNIT": "dBm"},
#     "Network Health": {"PARAM_KEY": "network_health", "PARAM_PRETTY": "Network Health", "PARAM_UNIT": "score"},
#     "Network Issues": {"PARAM_KEY": "network_issues", "PARAM_PRETTY": "Network Issues", "PARAM_UNIT": "count"},
#     "Number of Access Points": {"PARAM_KEY": "number_of_access_points", "PARAM_PRETTY": "Number of Access Points", "PARAM_UNIT": "count"},
# }

# PARAM_NAME = os.environ.get("COMPARE_PARAM_NAME", "SNR").strip()

# PARAM_KEY    = PARAM_PRESETS[PARAM_NAME]["PARAM_KEY"]
# PARAM_PRETTY = PARAM_PRESETS[PARAM_NAME]["PARAM_PRETTY"]
# PARAM_UNIT   = PARAM_PRESETS[PARAM_NAME]["PARAM_UNIT"]


# for d in [WITH_DIR, WO_DIR, OUT_DIR]:
#     d.mkdir(parents=True, exist_ok=True)

# print("[INFO] PARAM_NAME:", PARAM_NAME, "| PARAM_KEY:", PARAM_KEY)
# print("[INFO] WITH_DIR:", WITH_DIR)
# print("[INFO] WO_DIR:", WO_DIR)
# print("[INFO] OUT_DIR:", OUT_DIR)

# def extract_any_zips(root: Path):
#     for z in root.rglob("*.zip"):
#         if not zipfile.is_zipfile(z):
#             continue
#         out = z.parent / z.stem
#         out.mkdir(parents=True, exist_ok=True)
#         # only extract if folder is empty
#         if any(out.rglob("*")):
#             continue
#         with zipfile.ZipFile(z, "r") as zf:
#             zf.extractall(out)
#         print("[OK] extracted:", z.name, "->", out)

# extract_any_zips(WITH_DIR)
# extract_any_zips(WO_DIR)

# def count_csvs(root: Path):
#     return sum(1 for p in root.rglob("*.csv") if p.is_file())

# print("WITH CSV count   :", count_csvs(WITH_DIR))
# print("WITHOUT CSV count:", count_csvs(WO_DIR))

# # --- Load site_geometry.json ---
# if SITE_GEOM_PATH:
#     site_path = Path(SITE_GEOM_PATH)
# else:
#     site_path = next(SCRIPT_DIR.rglob("site_geometry.json"), None)

# if site_path is None or not Path(site_path).exists():
#     raise FileNotFoundError("site_geometry.json not found.")

# SITE = json.load(open(site_path, "r", encoding="utf-8"))
# BASE_FLOOR = SITE.get("base_floor")
# if not BASE_FLOOR:
#     raise ValueError("site_geometry.json missing 'base_floor'")

# print("[OK] site_geometry:", site_path)
# print("[OK] BASE_FLOOR:", BASE_FLOOR)

# def esx_has_member(esx_path: Path, member: str) -> bool:
#     try:
#         with zipfile.ZipFile(esx_path, "r") as zf:
#             return any(name.lower().endswith(member.lower()) for name in zf.namelist())
#     except Exception:
#         return False


# def find_master_esx(esx_dir: Path) -> Path:
#     candidates = sorted(esx_dir.rglob("*.esx"))
#     if not candidates:
#         raise FileNotFoundError(f"No .esx files found in {esx_dir}")

#     with_floorplans = [p for p in candidates if esx_has_member(p, "floorPlans.json")]
#     if not with_floorplans:
#         raise FileNotFoundError(f"No uploaded ESX contains floorPlans.json in {esx_dir}")

#     return with_floorplans[0]

# # --- Find master ESX from site_geometry (for metersPerUnit) ---
# if MASTER_ESX_PATH:
#     master_esx = Path(MASTER_ESX_PATH)
# else:
#     master_esx = find_master_esx(ESX_DIR)

# if master_esx is None or not Path(master_esx).exists():
#     raise FileNotFoundError("No valid master ESX found.")

# def read_esx_json_member(esx_path: Path, member: str) -> dict:
#     with zipfile.ZipFile(esx_path, "r") as zf:
#         target = next((n for n in zf.namelist() if n.lower().endswith(member.lower())), None)
#         if not target:
#             raise FileNotFoundError(f"{member} not found in {esx_path.name}")
#         return json.loads(zf.read(target).decode("utf-8"))

# fps = read_esx_json_member(master_esx, "floorPlans.json")["floorPlans"]
# FLOORS_META = {fp["name"].strip(): {"metersPerUnit": float(fp.get("metersPerUnit", 1.0))} for fp in fps}
# print("[OK] Floors:", list(FLOORS_META.keys()))

# # --- RvR transform builders (meters) ---
# def build_similarity_from_2pts(A1, A2, B1, B2):
#     A1 = np.array(A1, float); A2=np.array(A2,float)
#     B1 = np.array(B1, float); B2=np.array(B2,float)
#     vA = A2 - A1
#     vB = B2 - B1
#     nA = np.linalg.norm(vA)
#     nB = np.linalg.norm(vB)
#     if nA < 1e-9 or nB < 1e-9:
#         raise ValueError("Anchors too close.")
#     s = nB / nA
#     angA = np.arctan2(vA[1], vA[0])
#     angB = np.arctan2(vB[1], vB[0])
#     th = angB - angA
#     c, si = np.cos(th), np.sin(th)
#     R = np.array([[c, -si],[si, c]], float)
#     t = B1 - (s * (R @ A1))
#     return s, R, t

# def get_full_transform(site_obj, target_floor, current_floor, current_s=1.0, current_R=np.eye(2), current_t=np.zeros(2)):
#     if current_floor == target_floor:
#         return {"s": current_s, "R": current_R, "t": current_t}

#     link = site_obj["links"].get(current_floor)
#     if not link:
#         raise ValueError(f"No link for floor '{current_floor}' in site_geometry.json")

#     parent = link["align_to"]

#     mpu_child = float(FLOORS_META[current_floor]["metersPerUnit"])
#     child_stair_m = np.array(link["child_anchors_px"]["stair_px"], float) * mpu_child
#     child_ref_m   = np.array(link["child_anchors_px"]["ref_px"],   float) * mpu_child

#     mpu_parent = float(FLOORS_META[parent]["metersPerUnit"])
#     parent_stair_m = np.array(link["parent_anchors_px"]["stair_px"], float) * mpu_parent
#     parent_ref_m   = np.array(link["parent_anchors_px"]["ref_px"],   float) * mpu_parent

#     s_c2p, R_c2p, t_c2p = build_similarity_from_2pts(child_stair_m, child_ref_m, parent_stair_m, parent_ref_m)

#     new_s = current_s * s_c2p
#     new_R = current_R @ R_c2p
#     new_t = current_s * (current_R @ t_c2p) + current_t

#     return get_full_transform(site_obj, target_floor, parent, new_s, new_R, new_t)

# # Build TRANSFORMS floor->BASE (like RvR)
# TRANSFORMS = {}
# for f in FLOORS_META.keys():
#     if f == BASE_FLOOR:
#         TRANSFORMS[f] = {"s": 1.0, "R": np.eye(2), "t": np.zeros(2)}
#     else:
#         if f in SITE.get("links", {}):
#             TRANSFORMS[f] = get_full_transform(SITE, BASE_FLOOR, f)

# # DUT_GLOBAL_M from dut_px_by_floor (RvR behavior)
# if "dut_px_by_floor" not in SITE or BASE_FLOOR not in SITE["dut_px_by_floor"]:
#     # fallback: use first router dut
#     any_rk = next(iter(SITE.get("dut_px_by_router", {}).keys()), None)
#     if any_rk is None:
#         raise ValueError("site_geometry.json missing dut_px_by_floor and dut_px_by_router. Cannot compute DUT.")
#     SITE.setdefault("dut_px_by_floor", {})
#     SITE["dut_px_by_floor"][BASE_FLOOR] = SITE["dut_px_by_router"][any_rk]["dut_px"]
#     print("[WARN] dut_px_by_floor missing; using dut from router:", any_rk)

# dut_base_px = np.array(SITE["dut_px_by_floor"][BASE_FLOOR], float)
# mpu_base = float(FLOORS_META[BASE_FLOOR]["metersPerUnit"])
# DUT_GLOBAL_M = dut_base_px * mpu_base
# print("[OK] DUT_GLOBAL_M:", DUT_GLOBAL_M)

# def compute_global_distance_ft(floor_name: str, cx_px: np.ndarray, cy_px: np.ndarray) -> np.ndarray:
#     floor_name = str(floor_name).strip()
#     mpu = float(FLOORS_META[floor_name]["metersPerUnit"])
#     pts_local_m = np.stack([cx_px*mpu, cy_px*mpu], axis=1)
#     tf = TRANSFORMS.get(floor_name)
#     if tf is None:
#         # if no transform, assume it's base floor
#         tf = {"s":1.0, "R":np.eye(2), "t":np.zeros(2)}
#     pts_global_m = (tf["s"] * (pts_local_m @ tf["R"].T)) + tf["t"]
#     dxy = pts_global_m - DUT_GLOBAL_M.reshape(1,2)
#     return np.sqrt(dxy[:,0]**2 + dxy[:,1]**2) * FEET_PER_METER

# CSV_RE = re.compile(
#     r"^(?P<router>.+?)_(?P<param>.+?)\s+for\s+(?P<floor>.+?)\s+on\s+(?P<band>2\.4|5|6)\s*GHz\s+band_output\.csv$",
#     re.I
# )

# def canonical_param_from_text(text: str):
#     t = re.sub(r"\s+", " ", str(text).strip()).lower()
#     if "secondary signal strength" in t: return "secondary_signal_strength"
#     if "tertiary signal strength" in t: return "tertiary_signal_strength"
#     if "signal strength" in t or t == "rssi": return "signal_strength"
#     if "snr" in t or "signal to noise" in t: return "snr"
#     if t == "noise" or t.endswith(" noise"): return "noise"
#     if "data rate" in t or "datarate" in t: return "data_rate"
#     if "throughput" in t: return "throughput"
#     if "channel utilization" in t: return "channel_utilization"
#     if "channel interference" in t or "channel interferecne" in t: return "channel_interference"
#     if "channel width" in t: return "channel_width"
#     if "network health" in t: return "network_health"
#     if "network issues" in t: return "network_issues"
#     if "number of aps" in t or "number of access points" in t: return "number_of_access_points"
#     if "spectrum channel power" in t: return "spectrum_channel_power"
#     return None

# def scan_meta(root: Path) -> pd.DataFrame:
#     rows = []
#     for p in root.rglob("*band_output.csv"):
#         m = CSV_RE.match(p.name)
#         if not m:
#             continue
#         router_key = clean_router_name(m.group("router").strip())
#         pkey = canonical_param_from_text(m.group("param"))
#         if pkey != PARAM_KEY:
#             continue
#         floor_name = m.group("floor").strip()
#         band = f"{m.group('band')}GHz"
#         rows.append({"router_key": router_key, "floor_name": floor_name, "band": band, "csv_path": str(p)})
#     return pd.DataFrame(rows)

# META_WITH = scan_meta(WITH_DIR)
# META_WO   = scan_meta(WO_DIR)

# print("WITH matched CSVs:", len(META_WITH))
# print("WO   matched CSVs:", len(META_WO))

# if META_WITH.empty or META_WO.empty:
#     raise FileNotFoundError(f"No CSVs matched PARAM_KEY={PARAM_KEY} in one/both folders. Check filenames end with 'band_output.csv'.")

# try:
#     import matplotlib.patheffects as pe
#     _HAS_PE = True
# except Exception:
#     _HAS_PE = False

# LINEWIDTH = 2.2
# ALPHA = 0.92
# OUTLINE = True
# OUTLINE_WIDTH = 4.0
# OUTLINE_COLOR = "white"
# LEGEND_FONTSIZE = 8

# def pad_limits(vmin, vmax, frac=0.06):
#     if not np.isfinite(vmin) or not np.isfinite(vmax):
#         return (0.0, 1.0)
#     if vmax == vmin:
#         return (vmin - 1.0, vmax + 1.0)
#     span = vmax - vmin
#     return (vmin - frac*span, vmax + frac*span)

# def _prepend_zero_point(x, y, method="linear", clamp_to_data=True):
#     x = np.asarray(x, float)
#     y = np.asarray(y, float)
#     if len(x) == 0:
#         return x, y
#     order = np.argsort(x)
#     x = x[order]; y = y[order]
#     if x[0] <= 1e-9:
#         return x, y
#     if method == "linear" and len(x) >= 2 and (x[1] - x[0]) != 0:
#         y0 = y[0] + (0.0 - x[0]) * (y[1] - y[0]) / (x[1] - x[0])
#     else:
#         y0 = y[0]
#     if clamp_to_data:
#         finite = np.isfinite(y)
#         if np.any(finite):
#             ymin = float(np.nanmin(y[finite]))
#             ymax = float(np.nanmax(y[finite]))
#             y0 = float(np.clip(y0, ymin, ymax))
#     return np.concatenate([[0.0], x]), np.concatenate([[y0], y])

# def aggregate_by_distance_with_edges(distance_ft: np.ndarray, value: np.ndarray, edges: np.ndarray) -> pd.DataFrame:
#     df = pd.DataFrame({"distance_ft": distance_ft, "value": value})
#     df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["distance_ft","value"])
#     if df.empty:
#         return pd.DataFrame()

#     df["bin"] = pd.cut(df["distance_ft"], bins=edges, include_lowest=True, right=False)
#     g = df.groupby("bin", observed=True)["value"]

#     def q(arr, p):
#         return float(np.nanquantile(arr, p)) if len(arr) else np.nan

#     mids = [float(iv.left + (iv.right - iv.left)/2.0) for iv in g.groups.keys()]

#     out = pd.DataFrame({
#         "dist_ft_mid": mids,
#         "n": g.size().values,
#         "mean": g.mean().values,
#         "p50": g.median().values,
#         "p10": g.apply(lambda s: q(s.values, 0.10)).values,
#         "p90": g.apply(lambda s: q(s.values, 0.90)).values,
#         "min": g.min().values,
#         "max": g.max().values,
#     }).sort_values("dist_ft_mid").reset_index(drop=True)
#     return out

# def plot_overlay_actual(router_key, band, floor_name, curve_with, curve_wo, out_png: Path):
#     x_w = curve_with["dist_ft_mid"].to_numpy(float)
#     y_w = curve_with[Y_STAT].to_numpy(float)

#     x_n = curve_wo["dist_ft_mid"].to_numpy(float)
#     y_n = curve_wo[Y_STAT].to_numpy(float)

#     x_w, y_w = _prepend_zero_point(x_w, y_w, method=ZERO_INSERT_METHOD, clamp_to_data=CLAMP_ZERO_TO_DATA_RANGE)
#     x_n, y_n = _prepend_zero_point(x_n, y_n, method=ZERO_INSERT_METHOD, clamp_to_data=CLAMP_ZERO_TO_DATA_RANGE)

#     avg_w  = float(np.nanmean(curve_with[Y_STAT].to_numpy(float)))
#     avg_wo = float(np.nanmean(curve_wo[Y_STAT].to_numpy(float)))

#     fig, ax = plt.subplots(figsize=(10, 6))

#     # RvR-like: two clean lines (no steps-post)
#     l1, = ax.plot(x_n, y_n, linewidth=LINEWIDTH, alpha=ALPHA, label=f"Without mesh | avg: {avg_wo:.1f} {PARAM_UNIT}")
#     l2, = ax.plot(x_w, y_w, linewidth=LINEWIDTH, alpha=ALPHA, label=f"With mesh | avg: {avg_w:.1f} {PARAM_UNIT}")

#     if _HAS_PE and OUTLINE:
#         l1.set_path_effects([pe.Stroke(linewidth=OUTLINE_WIDTH, foreground=OUTLINE_COLOR), pe.Normal()])
#         l2.set_path_effects([pe.Stroke(linewidth=OUTLINE_WIDTH, foreground=OUTLINE_COLOR), pe.Normal()])

#     ax.set_title(f"{PARAM_PRETTY} (Actual) — {band} — Floor: {floor_name} | Router: {router_key}")
#     ax.set_xlabel("Distance from DUT (ft)")
#     ax.set_ylabel(f"{PARAM_PRETTY} ({PARAM_UNIT})")
#     ax.grid(True, alpha=0.22)

#     # RvR-like axis padding
#     all_x = np.concatenate([x_n, x_w])
#     all_y = np.concatenate([y_n, y_w])
#     _, xmax = pad_limits(0.0, float(np.nanmax(all_x)), 0.04)
#     ymin, ymax = pad_limits(float(np.nanmin(all_y)), float(np.nanmax(all_y)), 0.08)
#     ax.set_xlim(left=0.0, right=xmax)
#     ax.set_ylim(bottom=ymin, top=ymax)

#     # --- Legend: show bigger avg on TOP ---
#     legend_items = [
#         (avg_wo, l1, f"Without mesh | avg: {avg_wo:.1f} {PARAM_UNIT}"),
#         (avg_w,  l2, f"With mesh | avg: {avg_w:.1f} {PARAM_UNIT}"),
#     ]
#     legend_items.sort(key=lambda t: t[0], reverse=True)  # biggest first

#     handles = [t[1] for t in legend_items]
#     labels  = [t[2] for t in legend_items]
#     ax.legend(handles, labels, loc="best", fontsize=LEGEND_FONTSIZE, frameon=True, framealpha=0.85)
#     fig.tight_layout()

#     out_png.parent.mkdir(parents=True, exist_ok=True)
#     fig.savefig(out_png, dpi=200, bbox_inches="tight")
#     plt.close(fig)

# def _pick_cols(df):
#     # x/y columns
#     if {"cx","cy"}.issubset(df.columns):
#         xcol, ycol = "cx", "cy"
#     elif {"cx_px","cy_px"}.issubset(df.columns):
#         xcol, ycol = "cx_px", "cy_px"
#     elif {"center_x_px","center_y_px"}.issubset(df.columns):
#         xcol, ycol = "center_x_px", "center_y_px"
#     else:
#         raise ValueError(f"Missing coordinate columns. Found: {list(df.columns)}")

#     # value column
#     if "value" in df.columns:
#         vcol = "value"
#     else:
#         # fallback: pick first numeric column not x/y
#         candidates = [c for c in df.columns if c not in (xcol, ycol)]
#         vcol = None
#         for c in candidates:
#             s = pd.to_numeric(df[c], errors="coerce")
#             if np.isfinite(s).sum() > 0:
#                 vcol = c
#                 break
#         if vcol is None:
#             raise ValueError(f"Missing value column. Found: {list(df.columns)}")

#     return xcol, ycol, vcol


# # Pair WITH vs WITHOUT by router/floor/band
# pairs = pd.merge(
#     META_WITH, META_WO,
#     on=["router_key", "floor_name", "band"],
#     suffixes=("_with", "_wo"),
#     how="inner",
# )

# print("[INFO] Matched pairs:", len(pairs))
# if pairs.empty:
#     raise ValueError("No matching (router,floor,band) between WITH and WITHOUT.")

# plots_made = 0
# curve_table_rows = []

# for _, r in pairs.iterrows():
#     router_key = r["router_key"]
#     floor_name = r["floor_name"]
#     band       = r["band"]

#     df_w = pd.read_csv(r["csv_path_with"])
#     df_n = pd.read_csv(r["csv_path_wo"])

#     xw, yw, vw = _pick_cols(df_w)
#     xn, yn, vn = _pick_cols(df_n)

#     df_w[vw] = pd.to_numeric(df_w[vw], errors="coerce")
#     df_n[vn] = pd.to_numeric(df_n[vn], errors="coerce")

#     dist_w = compute_global_distance_ft(
#         floor_name,
#         df_w[xw].astype(float).to_numpy(),
#         df_w[yw].astype(float).to_numpy(),
#     )
#     dist_n = compute_global_distance_ft(
#         floor_name,
#         df_n[xn].astype(float).to_numpy(),
#         df_n[yn].astype(float).to_numpy(),
#     )

#     val_w = df_w[vw].to_numpy(float)
#     val_n = df_n[vn].to_numpy(float)

#     max_d = float(np.nanmax(np.concatenate([dist_w, dist_n])))
#     if not np.isfinite(max_d) or max_d <= 0:
#         print("[WARN] Skipping (bad distance):", router_key, floor_name, band)
#         continue

#     edges = np.arange(0.0, max_d + BIN_FT * 2, BIN_FT)

#     curve_w = aggregate_by_distance_with_edges(dist_w, val_w, edges)
#     curve_n = aggregate_by_distance_with_edges(dist_n, val_n, edges)
#     if curve_w.empty or curve_n.empty:
#         print("[WARN] Skipping (empty curve):", router_key, floor_name, band)
#         continue

#     out_png = OUT_DIR / router_key / floor_name / f"{band}_{PARAM_KEY}.png"
#     plot_overlay_actual(router_key, band, floor_name, curve_w, curve_n, out_png)
#     plots_made += 1

#     curve_table_rows.append(
#         curve_n.assign(
#             router_key=router_key,
#             router_display=router_key,
#             floor_name=floor_name,
#             band=band,
#             scenario="without_mesh",
#             scenario_label="Without mesh",
#         )
#     )
#     curve_table_rows.append(
#         curve_w.assign(
#             router_key=router_key,
#             router_display=router_key,
#             floor_name=floor_name,
#             band=band,
#             scenario="with_mesh",
#             scenario_label="With mesh",
#         )
#     )
#     # if curve_w.empty or curve_n.empty:
#     #     print("[WARN] Skipping (empty curve):", router_key, floor_name, band)
#     #     continue

#     # out_png = OUT_DIR / router_key / floor_name / f"{band}_{PARAM_KEY}.png"
#     # plot_overlay_actual(router_key, band, floor_name, curve_w, curve_n, out_png)
#     # plots_made += 1

# print("[DONE] plots_made =", plots_made)
# if plots_made == 0:
#     raise RuntimeError("No plots were generated. Check CSV columns (cx/cy/value) and filename pattern.")

# if curve_table_rows:
#     tables_dir = OUT_DIR / "tables"
#     tables_dir.mkdir(parents=True, exist_ok=True)
#     curve_table_df = pd.concat(curve_table_rows, ignore_index=True)
#     curve_table_path = tables_dir / f"{PARAM_KEY}_mesh_curve_tables.csv"
#     curve_table_df.to_csv(curve_table_path, index=False)
#     print("[DONE] Mesh curve table saved:", curve_table_path)
# else:
#     raise RuntimeError("No comparison curve tables were generated.")

# # --- Generate DOCX Report ---
# try:
#     from docx import Document
#     from ai_report_generator import generate_report
#     import csv

#     def extract_docx_assets(docx_path: Path, output_dir: Path):
#         """Extract images from DOCX and create asset registry compatible structure."""
#         doc = Document(str(docx_path))
#         images_dir = output_dir / "images"
#         images_dir.mkdir(parents=True, exist_ok=True)
        
#         asset_rows = []
#         image_count = 0
        
#         for rel in doc.part.rels.values():
#             if "image" in rel.reltype:
#                 image_count += 1
#                 image_data = rel.target_part.blob
#                 image_ext = rel.target_part.content_type.split('/')[-1]
#                 if image_ext == 'jpeg':
#                     image_ext = 'jpg'
                
#                 image_filename = f"{docx_path.stem}_image_{image_count}.{image_ext}"
#                 image_path = images_dir / image_filename
#                 image_path.write_bytes(image_data)
                
#                 # Try to determine if this is a heatmap or scale based on filename or size
#                 # For now, assume all images are heatmaps (this is a simplification)
#                 asset_rows.append({
#                     'router_key': docx_path.stem.split('_')[0] if '_' in docx_path.stem else 'unknown',
#                     'parameter_key': PARAM_KEY,
#                     'parameter_display': PARAM_PRETTY,
#                     'floor_name': 'unknown',  # Would need OCR or manual mapping
#                     'band': 'unknown',  # Would need OCR or manual mapping
#                     'role': 'heatmap',
#                     'path': str(image_path.relative_to(output_dir)),
#                     'source_docx': str(docx_path)
#                 })
        
#         # Create _index.csv for asset registry
#         if asset_rows:
#             index_path = output_dir / "_index.csv"
#             with index_path.open('w', newline='', encoding='utf-8') as f:
#                 writer = csv.DictWriter(f, fieldnames=asset_rows[0].keys())
#                 writer.writeheader()
#                 writer.writerows(asset_rows)

#     # Extract assets from uploaded DOCX reports
#     with_extracted_dir = OUT_DIR / "with_mesh_extracted"
#     without_extracted_dir = OUT_DIR / "without_mesh_extracted"

#     with_extracted_dir.mkdir(parents=True, exist_ok=True)
#     without_extracted_dir.mkdir(parents=True, exist_ok=True)

#     # Extract from with mesh DOCX
#     for docx_file in WITH_DOCX_DIR.glob("*.docx"):
#         print(f"[INFO] Extracting assets from {docx_file.name}")
#         extract_docx_assets(docx_file, with_extracted_dir)

#     # Extract from without mesh DOCX
#     for docx_file in WITHOUT_DOCX_DIR.glob("*.docx"):
#         print(f"[INFO] Extracting assets from {docx_file.name}")
#         extract_docx_assets(docx_file, without_extracted_dir)

#     # Generate the DOCX report
#     docx_output_path = OUT_DIR / f"{PARAM_KEY}_mesh_comparison_report.docx"
#     extracted_roots_by_scenario = {
#         "with_mesh": with_extracted_dir,
#         "without_mesh": without_extracted_dir
#     }

#     generate_report(
#         rvr_outputs_root=OUT_DIR,  # Not used in mesh_compare mode
#         extracted_root=with_extracted_dir,  # Fallback
#         output_path=docx_output_path,
#         metric_folders=[PARAM_KEY],
#         config_label="Mesh vs No Mesh",
#         mode="mesh_compare",
#         compare_outputs_root=OUT_DIR,
#         extracted_roots_by_scenario=extracted_roots_by_scenario,
#         use_ai=False  # Disable AI for now to avoid complexity
#     )

#     print("[DONE] DOCX report generated:", docx_output_path)

# except ImportError as e:
#     print(f"[WARN] Could not generate DOCX report: {e}")
# except Exception as e:
#     print(f"[WARN] Error generating DOCX report: {e}")


# from pathlib import Path
# import zipfile

# def zip_folder_overwrite(folder_path, zip_path):
#     folder_path = Path(folder_path)
#     zip_path = Path(zip_path)

#     if zip_path.exists():
#         zip_path.unlink()

#     with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
#         for file in folder_path.rglob("*"):
#             if file.is_file():
#                 zf.write(file, arcname=file.relative_to(folder_path))

#     return str(zip_path)

# pngs = list(OUT_DIR.rglob("*.png"))
# docxs = list(OUT_DIR.rglob("*.docx"))
# if len(pngs) == 0 and len(docxs) == 0:
#     raise RuntimeError(f"No plots or reports found in {OUT_DIR}.")

# zip_path = zip_folder_overwrite(OUT_DIR, ZIP_PATH)
# print("[DONE] Output zipped:", zip_path)

from pathlib import Path
import os
import re
import json
import zipfile
import shutil
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Ensure project root is importable from wherever script is executed.
script_dir = Path(__file__).resolve().parent
project_root = script_dir
while project_root != project_root.parent and not (project_root / "metadata_utils.py").exists():
    project_root = project_root.parent
if (project_root / "metadata_utils.py").exists():
    sys.path.insert(0, str(project_root))

from metadata_utils import clean_router_name
from docx_extractor_v2 import process_many_docx_local

SCRIPT_DIR = Path(__file__).resolve().parent

WITH_DIR = Path(os.environ.get("COMPARE_WITH_DIR", str(SCRIPT_DIR / "inputs_compare" / "with_mesh")))
WO_DIR   = Path(os.environ.get("COMPARE_WITHOUT_DIR", str(SCRIPT_DIR / "inputs_compare" / "without_mesh")))
OUT_DIR  = Path(os.environ.get("COMPARE_OUT_DIR", str(SCRIPT_DIR / "outputs_compare_rvr_style")))
ZIP_PATH = Path(os.environ.get("COMPARE_ZIP_PATH", str(SCRIPT_DIR / "compare_overlay_outputs.zip")))

SITE_GEOM_PATH = os.environ.get("COMPARE_SITE_GEOM", "").strip()
MASTER_ESX_PATH = os.environ.get("COMPARE_MASTER_ESX", "").strip()
ESX_DIR = Path(os.environ.get("COMPARE_ESX_DIR", str(SCRIPT_DIR)))

WITH_DOCX_DIR = Path(os.environ.get("COMPARE_WITH_DOCX_DIR", str(SCRIPT_DIR)))
WITHOUT_DOCX_DIR = Path(os.environ.get("COMPARE_WITHOUT_DOCX_DIR", str(SCRIPT_DIR)))

BIN_FT = 2.0
Y_STAT = "p50"
ZERO_INSERT_METHOD = "linear"
CLAMP_ZERO_TO_DATA_RANGE = True
FEET_PER_METER = 3.280839895

PARAM_PRESETS = {
    "SNR": {"PARAM_KEY": "snr", "PARAM_PRETTY": "SNR", "PARAM_UNIT": "dB"},
    "Throughput": {"PARAM_KEY": "throughput", "PARAM_PRETTY": "Throughput", "PARAM_UNIT": "Mbps"},
    "Data Rate": {"PARAM_KEY": "data_rate", "PARAM_PRETTY": "Data Rate", "PARAM_UNIT": "Mbps"},
    "Signal Strength": {"PARAM_KEY": "signal_strength", "PARAM_PRETTY": "Signal Strength", "PARAM_UNIT": "dBm"},
    "Secondary Signal Strength": {"PARAM_KEY": "secondary_signal_strength", "PARAM_PRETTY": "Secondary Signal Strength", "PARAM_UNIT": "dBm"},
    "Tertiary Signal Strength": {"PARAM_KEY": "tertiary_signal_strength", "PARAM_PRETTY": "Tertiary Signal Strength", "PARAM_UNIT": "dBm"},
    "Noise": {"PARAM_KEY": "noise", "PARAM_PRETTY": "Noise", "PARAM_UNIT": "dBm"},
    "Channel Utilization": {"PARAM_KEY": "channel_utilization", "PARAM_PRETTY": "Channel Utilization", "PARAM_UNIT": "%"},
    "Channel Interference": {"PARAM_KEY": "channel_interference", "PARAM_PRETTY": "Channel Interference", "PARAM_UNIT": "dB"},
    "Channel Width": {"PARAM_KEY": "channel_width", "PARAM_PRETTY": "Channel Width", "PARAM_UNIT": "MHz"},
    "Spectrum Channel Power": {"PARAM_KEY": "spectrum_channel_power", "PARAM_PRETTY": "Spectrum Channel Power", "PARAM_UNIT": "dBm"},
    "Network Health": {"PARAM_KEY": "network_health", "PARAM_PRETTY": "Network Health", "PARAM_UNIT": "score"},
    "Network Issues": {"PARAM_KEY": "network_issues", "PARAM_PRETTY": "Network Issues", "PARAM_UNIT": "count"},
    "Number of Access Points": {"PARAM_KEY": "number_of_access_points", "PARAM_PRETTY": "Number of Access Points", "PARAM_UNIT": "count"},
}

PARAM_NAME = os.environ.get("COMPARE_PARAM_NAME", "SNR").strip()

PARAM_KEY    = PARAM_PRESETS[PARAM_NAME]["PARAM_KEY"]
PARAM_PRETTY = PARAM_PRESETS[PARAM_NAME]["PARAM_PRETTY"]
PARAM_UNIT   = PARAM_PRESETS[PARAM_NAME]["PARAM_UNIT"]


for d in [WITH_DIR, WO_DIR, OUT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

print("[INFO] PARAM_NAME:", PARAM_NAME, "| PARAM_KEY:", PARAM_KEY)
print("[INFO] WITH_DIR:", WITH_DIR)
print("[INFO] WO_DIR:", WO_DIR)
print("[INFO] OUT_DIR:", OUT_DIR)

def extract_any_zips(root: Path):
    for z in root.rglob("*.zip"):
        if not zipfile.is_zipfile(z):
            continue
        out = z.parent / z.stem
        out.mkdir(parents=True, exist_ok=True)
        # only extract if folder is empty
        if any(out.rglob("*")):
            continue
        with zipfile.ZipFile(z, "r") as zf:
            zf.extractall(out)
        print("[OK] extracted:", z.name, "->", out)

extract_any_zips(WITH_DIR)
extract_any_zips(WO_DIR)

def count_csvs(root: Path):
    return sum(1 for p in root.rglob("*.csv") if p.is_file())

print("WITH CSV count   :", count_csvs(WITH_DIR))
print("WITHOUT CSV count:", count_csvs(WO_DIR))

# --- Load site_geometry.json ---
if SITE_GEOM_PATH:
    site_path = Path(SITE_GEOM_PATH)
else:
    site_path = next(SCRIPT_DIR.rglob("site_geometry.json"), None)

if site_path is None or not Path(site_path).exists():
    raise FileNotFoundError("site_geometry.json not found.")

SITE = json.load(open(site_path, "r", encoding="utf-8"))
BASE_FLOOR = SITE.get("base_floor")
if not BASE_FLOOR:
    raise ValueError("site_geometry.json missing 'base_floor'")

print("[OK] site_geometry:", site_path)
print("[OK] BASE_FLOOR:", BASE_FLOOR)

def esx_has_member(esx_path: Path, member: str) -> bool:
    try:
        with zipfile.ZipFile(esx_path, "r") as zf:
            return any(name.lower().endswith(member.lower()) for name in zf.namelist())
    except Exception:
        return False


def find_master_esx(esx_dir: Path) -> Path:
    candidates = sorted(esx_dir.rglob("*.esx"))
    if not candidates:
        raise FileNotFoundError(f"No .esx files found in {esx_dir}")

    with_floorplans = [p for p in candidates if esx_has_member(p, "floorPlans.json")]
    if not with_floorplans:
        raise FileNotFoundError(f"No uploaded ESX contains floorPlans.json in {esx_dir}")

    return with_floorplans[0]

# --- Find master ESX from site_geometry (for metersPerUnit) ---
if MASTER_ESX_PATH:
    master_esx = Path(MASTER_ESX_PATH)
else:
    master_esx = find_master_esx(ESX_DIR)

if master_esx is None or not Path(master_esx).exists():
    raise FileNotFoundError("No valid master ESX found.")

def read_esx_json_member(esx_path: Path, member: str) -> dict:
    with zipfile.ZipFile(esx_path, "r") as zf:
        target = next((n for n in zf.namelist() if n.lower().endswith(member.lower())), None)
        if not target:
            raise FileNotFoundError(f"{member} not found in {esx_path.name}")
        return json.loads(zf.read(target).decode("utf-8"))

fps = read_esx_json_member(master_esx, "floorPlans.json")["floorPlans"]
FLOORS_META = {fp["name"].strip(): {"metersPerUnit": float(fp.get("metersPerUnit", 1.0))} for fp in fps}
print("[OK] Floors:", list(FLOORS_META.keys()))

# --- RvR transform builders (meters) ---
def build_similarity_from_2pts(A1, A2, B1, B2):
    A1 = np.array(A1, float); A2=np.array(A2,float)
    B1 = np.array(B1, float); B2=np.array(B2,float)
    vA = A2 - A1
    vB = B2 - B1
    nA = np.linalg.norm(vA)
    nB = np.linalg.norm(vB)
    if nA < 1e-9 or nB < 1e-9:
        raise ValueError("Anchors too close.")
    s = nB / nA
    angA = np.arctan2(vA[1], vA[0])
    angB = np.arctan2(vB[1], vB[0])
    th = angB - angA
    c, si = np.cos(th), np.sin(th)
    R = np.array([[c, -si],[si, c]], float)
    t = B1 - (s * (R @ A1))
    return s, R, t

def get_full_transform(site_obj, target_floor, current_floor, current_s=1.0, current_R=np.eye(2), current_t=np.zeros(2)):
    if current_floor == target_floor:
        return {"s": current_s, "R": current_R, "t": current_t}

    link = site_obj["links"].get(current_floor)
    if not link:
        raise ValueError(f"No link for floor '{current_floor}' in site_geometry.json")

    parent = link["align_to"]

    mpu_child = float(FLOORS_META[current_floor]["metersPerUnit"])
    child_stair_m = np.array(link["child_anchors_px"]["stair_px"], float) * mpu_child
    child_ref_m   = np.array(link["child_anchors_px"]["ref_px"],   float) * mpu_child

    mpu_parent = float(FLOORS_META[parent]["metersPerUnit"])
    parent_stair_m = np.array(link["parent_anchors_px"]["stair_px"], float) * mpu_parent
    parent_ref_m   = np.array(link["parent_anchors_px"]["ref_px"],   float) * mpu_parent

    s_c2p, R_c2p, t_c2p = build_similarity_from_2pts(child_stair_m, child_ref_m, parent_stair_m, parent_ref_m)

    new_s = current_s * s_c2p
    new_R = current_R @ R_c2p
    new_t = current_s * (current_R @ t_c2p) + current_t

    return get_full_transform(site_obj, target_floor, parent, new_s, new_R, new_t)

# Build TRANSFORMS floor->BASE (like RvR)
TRANSFORMS = {}
for f in FLOORS_META.keys():
    if f == BASE_FLOOR:
        TRANSFORMS[f] = {"s": 1.0, "R": np.eye(2), "t": np.zeros(2)}
    else:
        if f in SITE.get("links", {}):
            TRANSFORMS[f] = get_full_transform(SITE, BASE_FLOOR, f)

# DUT_GLOBAL_M from dut_px_by_floor (RvR behavior)
if "dut_px_by_floor" not in SITE or BASE_FLOOR not in SITE["dut_px_by_floor"]:
    # fallback: use first router dut
    any_rk = next(iter(SITE.get("dut_px_by_router", {}).keys()), None)
    if any_rk is None:
        raise ValueError("site_geometry.json missing dut_px_by_floor and dut_px_by_router. Cannot compute DUT.")
    SITE.setdefault("dut_px_by_floor", {})
    SITE["dut_px_by_floor"][BASE_FLOOR] = SITE["dut_px_by_router"][any_rk]["dut_px"]
    print("[WARN] dut_px_by_floor missing; using dut from router:", any_rk)

dut_base_px = np.array(SITE["dut_px_by_floor"][BASE_FLOOR], float)
mpu_base = float(FLOORS_META[BASE_FLOOR]["metersPerUnit"])
DUT_GLOBAL_M = dut_base_px * mpu_base
print("[OK] DUT_GLOBAL_M:", DUT_GLOBAL_M)

def compute_global_distance_ft(floor_name: str, cx_px: np.ndarray, cy_px: np.ndarray) -> np.ndarray:
    floor_name = str(floor_name).strip()
    mpu = float(FLOORS_META[floor_name]["metersPerUnit"])
    pts_local_m = np.stack([cx_px*mpu, cy_px*mpu], axis=1)
    tf = TRANSFORMS.get(floor_name)
    if tf is None:
        # if no transform, assume it's base floor
        tf = {"s":1.0, "R":np.eye(2), "t":np.zeros(2)}
    pts_global_m = (tf["s"] * (pts_local_m @ tf["R"].T)) + tf["t"]
    dxy = pts_global_m - DUT_GLOBAL_M.reshape(1,2)
    return np.sqrt(dxy[:,0]**2 + dxy[:,1]**2) * FEET_PER_METER

CSV_RE = re.compile(
    r"^(?P<router>.+?)_(?P<param>.+?)\s+for\s+(?P<floor>.+?)\s+on\s+(?P<band>2\.4|5|6)\s*GHz\s+band_output\.csv$",
    re.I
)

def canonical_param_from_text(text: str):
    t = re.sub(r"\s+", " ", str(text).strip()).lower()
    if "secondary signal strength" in t: return "secondary_signal_strength"
    if "tertiary signal strength" in t: return "tertiary_signal_strength"
    if "signal strength" in t or t == "rssi": return "signal_strength"
    if "snr" in t or "signal to noise" in t: return "snr"
    if t == "noise" or t.endswith(" noise"): return "noise"
    if "data rate" in t or "datarate" in t: return "data_rate"
    if "throughput" in t: return "throughput"
    if "channel utilization" in t: return "channel_utilization"
    if "channel interference" in t or "channel interferecne" in t: return "channel_interference"
    if "channel width" in t: return "channel_width"
    if "network health" in t: return "network_health"
    if "network issues" in t: return "network_issues"
    if "number of aps" in t or "number of access points" in t: return "number_of_access_points"
    if "spectrum channel power" in t: return "spectrum_channel_power"
    return None

def scan_meta(root: Path) -> pd.DataFrame:
    rows = []
    for p in root.rglob("*band_output.csv"):
        m = CSV_RE.match(p.name)
        if not m:
            continue
        router_key = clean_router_name(m.group("router").strip())
        pkey = canonical_param_from_text(m.group("param"))
        if pkey != PARAM_KEY:
            continue
        floor_name = m.group("floor").strip()
        band = f"{m.group('band')}GHz"
        rows.append({"router_key": router_key, "floor_name": floor_name, "band": band, "csv_path": str(p)})
    return pd.DataFrame(rows)

META_WITH = scan_meta(WITH_DIR)
META_WO   = scan_meta(WO_DIR)

print("WITH matched CSVs:", len(META_WITH))
print("WO   matched CSVs:", len(META_WO))

if META_WITH.empty or META_WO.empty:
    raise FileNotFoundError(f"No CSVs matched PARAM_KEY={PARAM_KEY} in one/both folders. Check filenames end with 'band_output.csv'.")

try:
    import matplotlib.patheffects as pe
    _HAS_PE = True
except Exception:
    _HAS_PE = False

LINEWIDTH = 2.2
ALPHA = 0.92
OUTLINE = True
OUTLINE_WIDTH = 4.0
OUTLINE_COLOR = "white"
LEGEND_FONTSIZE = 8

def pad_limits(vmin, vmax, frac=0.06):
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        return (0.0, 1.0)
    if vmax == vmin:
        return (vmin - 1.0, vmax + 1.0)
    span = vmax - vmin
    return (vmin - frac*span, vmax + frac*span)

def _prepend_zero_point(x, y, method="linear", clamp_to_data=True):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if len(x) == 0:
        return x, y
    order = np.argsort(x)
    x = x[order]; y = y[order]
    if x[0] <= 1e-9:
        return x, y
    if method == "linear" and len(x) >= 2 and (x[1] - x[0]) != 0:
        y0 = y[0] + (0.0 - x[0]) * (y[1] - y[0]) / (x[1] - x[0])
    else:
        y0 = y[0]
    if clamp_to_data:
        finite = np.isfinite(y)
        if np.any(finite):
            ymin = float(np.nanmin(y[finite]))
            ymax = float(np.nanmax(y[finite]))
            y0 = float(np.clip(y0, ymin, ymax))
    return np.concatenate([[0.0], x]), np.concatenate([[y0], y])

def aggregate_by_distance_with_edges(distance_ft: np.ndarray, value: np.ndarray, edges: np.ndarray) -> pd.DataFrame:
    df = pd.DataFrame({"distance_ft": distance_ft, "value": value})
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["distance_ft","value"])
    if df.empty:
        return pd.DataFrame()

    df["bin"] = pd.cut(df["distance_ft"], bins=edges, include_lowest=True, right=False)
    g = df.groupby("bin", observed=True)["value"]

    def q(arr, p):
        return float(np.nanquantile(arr, p)) if len(arr) else np.nan

    mids = [float(iv.left + (iv.right - iv.left)/2.0) for iv in g.groups.keys()]

    out = pd.DataFrame({
        "dist_ft_mid": mids,
        "n": g.size().values,
        "mean": g.mean().values,
        "p50": g.median().values,
        "p10": g.apply(lambda s: q(s.values, 0.10)).values,
        "p90": g.apply(lambda s: q(s.values, 0.90)).values,
        "min": g.min().values,
        "max": g.max().values,
    }).sort_values("dist_ft_mid").reset_index(drop=True)
    return out

def plot_overlay_actual(router_key, band, floor_name, curve_with, curve_wo, out_png: Path):
    x_w = curve_with["dist_ft_mid"].to_numpy(float)
    y_w = curve_with[Y_STAT].to_numpy(float)

    x_n = curve_wo["dist_ft_mid"].to_numpy(float)
    y_n = curve_wo[Y_STAT].to_numpy(float)

    x_w, y_w = _prepend_zero_point(x_w, y_w, method=ZERO_INSERT_METHOD, clamp_to_data=CLAMP_ZERO_TO_DATA_RANGE)
    x_n, y_n = _prepend_zero_point(x_n, y_n, method=ZERO_INSERT_METHOD, clamp_to_data=CLAMP_ZERO_TO_DATA_RANGE)

    avg_w  = float(np.nanmean(curve_with[Y_STAT].to_numpy(float)))
    avg_wo = float(np.nanmean(curve_wo[Y_STAT].to_numpy(float)))

    fig, ax = plt.subplots(figsize=(10, 6))

    # RvR-like: two clean lines (no steps-post)
    l1, = ax.plot(x_n, y_n, linewidth=LINEWIDTH, alpha=ALPHA, label=f"Without mesh | avg: {avg_wo:.1f} {PARAM_UNIT}")
    l2, = ax.plot(x_w, y_w, linewidth=LINEWIDTH, alpha=ALPHA, label=f"With mesh | avg: {avg_w:.1f} {PARAM_UNIT}")

    if _HAS_PE and OUTLINE:
        l1.set_path_effects([pe.Stroke(linewidth=OUTLINE_WIDTH, foreground=OUTLINE_COLOR), pe.Normal()])
        l2.set_path_effects([pe.Stroke(linewidth=OUTLINE_WIDTH, foreground=OUTLINE_COLOR), pe.Normal()])

    ax.set_title(f"{PARAM_PRETTY} (Actual) — {band} — Floor: {floor_name} | Router: {router_key}")
    ax.set_xlabel("Distance from DUT (ft)")
    ax.set_ylabel(f"{PARAM_PRETTY} ({PARAM_UNIT})")
    ax.grid(True, alpha=0.22)

    # RvR-like axis padding
    all_x = np.concatenate([x_n, x_w])
    all_y = np.concatenate([y_n, y_w])
    _, xmax = pad_limits(0.0, float(np.nanmax(all_x)), 0.04)
    ymin, ymax = pad_limits(float(np.nanmin(all_y)), float(np.nanmax(all_y)), 0.08)
    ax.set_xlim(left=0.0, right=xmax)
    ax.set_ylim(bottom=ymin, top=ymax)

    # --- Legend: show bigger avg on TOP ---
    legend_items = [
        (avg_wo, l1, f"Without mesh | avg: {avg_wo:.1f} {PARAM_UNIT}"),
        (avg_w,  l2, f"With mesh | avg: {avg_w:.1f} {PARAM_UNIT}"),
    ]
    legend_items.sort(key=lambda t: t[0], reverse=True)  # biggest first

    handles = [t[1] for t in legend_items]
    labels  = [t[2] for t in legend_items]
    ax.legend(handles, labels, loc="best", fontsize=LEGEND_FONTSIZE, frameon=True, framealpha=0.85)
    fig.tight_layout()

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)

def _pick_cols(df):
    # x/y columns
    if {"cx","cy"}.issubset(df.columns):
        xcol, ycol = "cx", "cy"
    elif {"cx_px","cy_px"}.issubset(df.columns):
        xcol, ycol = "cx_px", "cy_px"
    elif {"center_x_px","center_y_px"}.issubset(df.columns):
        xcol, ycol = "center_x_px", "center_y_px"
    else:
        raise ValueError(f"Missing coordinate columns. Found: {list(df.columns)}")

    # value column
    if "value" in df.columns:
        vcol = "value"
    else:
        # fallback: pick first numeric column not x/y
        candidates = [c for c in df.columns if c not in (xcol, ycol)]
        vcol = None
        for c in candidates:
            s = pd.to_numeric(df[c], errors="coerce")
            if np.isfinite(s).sum() > 0:
                vcol = c
                break
        if vcol is None:
            raise ValueError(f"Missing value column. Found: {list(df.columns)}")

    return xcol, ycol, vcol


# Pair WITH vs WITHOUT by router/floor/band
pairs = pd.merge(
    META_WITH, META_WO,
    on=["router_key", "floor_name", "band"],
    suffixes=("_with", "_wo"),
    how="inner",
)

print("[INFO] Matched pairs:", len(pairs))
if pairs.empty:
    raise ValueError("No matching (router,floor,band) between WITH and WITHOUT.")

plots_made = 0
curve_table_rows = []

for _, r in pairs.iterrows():
    router_key = r["router_key"]
    floor_name = r["floor_name"]
    band       = r["band"]

    df_w = pd.read_csv(r["csv_path_with"])
    df_n = pd.read_csv(r["csv_path_wo"])

    xw, yw, vw = _pick_cols(df_w)
    xn, yn, vn = _pick_cols(df_n)

    df_w[vw] = pd.to_numeric(df_w[vw], errors="coerce")
    df_n[vn] = pd.to_numeric(df_n[vn], errors="coerce")

    dist_w = compute_global_distance_ft(
        floor_name,
        df_w[xw].astype(float).to_numpy(),
        df_w[yw].astype(float).to_numpy(),
    )
    dist_n = compute_global_distance_ft(
        floor_name,
        df_n[xn].astype(float).to_numpy(),
        df_n[yn].astype(float).to_numpy(),
    )

    val_w = df_w[vw].to_numpy(float)
    val_n = df_n[vn].to_numpy(float)

    max_d = float(np.nanmax(np.concatenate([dist_w, dist_n])))
    if not np.isfinite(max_d) or max_d <= 0:
        print("[WARN] Skipping (bad distance):", router_key, floor_name, band)
        continue

    edges = np.arange(0.0, max_d + BIN_FT * 2, BIN_FT)

    curve_w = aggregate_by_distance_with_edges(dist_w, val_w, edges)
    curve_n = aggregate_by_distance_with_edges(dist_n, val_n, edges)
    if curve_w.empty or curve_n.empty:
        print("[WARN] Skipping (empty curve):", router_key, floor_name, band)
        continue

    out_png = OUT_DIR / router_key / floor_name / f"{band}_{PARAM_KEY}.png"
    plot_overlay_actual(router_key, band, floor_name, curve_w, curve_n, out_png)
    plots_made += 1

    curve_table_rows.append(
        curve_n.assign(
            router_key=router_key,
            router_display=router_key,
            floor_name=floor_name,
            band=band,
            scenario="without_mesh",
            scenario_label="Without mesh",
        )
    )
    curve_table_rows.append(
        curve_w.assign(
            router_key=router_key,
            router_display=router_key,
            floor_name=floor_name,
            band=band,
            scenario="with_mesh",
            scenario_label="With mesh",
        )
    )
    # if curve_w.empty or curve_n.empty:
    #     print("[WARN] Skipping (empty curve):", router_key, floor_name, band)
    #     continue

    # out_png = OUT_DIR / router_key / floor_name / f"{band}_{PARAM_KEY}.png"
    # plot_overlay_actual(router_key, band, floor_name, curve_w, curve_n, out_png)
    # plots_made += 1

print("[DONE] plots_made =", plots_made)
if plots_made == 0:
    raise RuntimeError("No plots were generated. Check CSV columns (cx/cy/value) and filename pattern.")

if curve_table_rows:
    tables_dir = OUT_DIR / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    curve_table_df = pd.concat(curve_table_rows, ignore_index=True)
    curve_table_path = tables_dir / f"{PARAM_KEY}_mesh_curve_tables.csv"
    curve_table_df.to_csv(curve_table_path, index=False)
    print("[DONE] Mesh curve table saved:", curve_table_path)
else:
    raise RuntimeError("No comparison curve tables were generated.")

# --- Generate DOCX Report ---
try:
    from ai_report_generator import generate_report

    # Extract assets from uploaded DOCX reports using the same manifest-aware
    # extractor as the main app so report asset pairing stays exact.
    with_extracted_dir = OUT_DIR / "with_mesh_extracted"
    without_extracted_dir = OUT_DIR / "without_mesh_extracted"

    with_docx_paths = [str(p) for p in sorted(WITH_DOCX_DIR.glob("*.docx"))]
    without_docx_paths = [str(p) for p in sorted(WITHOUT_DOCX_DIR.glob("*.docx"))]

    if with_docx_paths:
        print(f"[INFO] Extracting with-mesh report assets for {PARAM_KEY}")
        process_many_docx_local(
            with_docx_paths,
            out_root=str(with_extracted_dir),
            download_per_docx_zip=False,
            also_make_master_zip=False,
            selected_parameters=[PARAM_KEY],
        )
    if without_docx_paths:
        print(f"[INFO] Extracting without-mesh report assets for {PARAM_KEY}")
        process_many_docx_local(
            without_docx_paths,
            out_root=str(without_extracted_dir),
            download_per_docx_zip=False,
            also_make_master_zip=False,
            selected_parameters=[PARAM_KEY],
        )

    # Generate the DOCX report
    docx_output_path = OUT_DIR / f"{PARAM_KEY}_mesh_comparison_report.docx"
    extracted_roots_by_scenario = {
        "with_mesh": with_extracted_dir,
        "without_mesh": without_extracted_dir,
    }

    generate_report(
        rvr_outputs_root=OUT_DIR,  # Not used in mesh_compare mode
        extracted_root=with_extracted_dir,  # Fallback
        output_path=docx_output_path,
        metric_folders=[PARAM_KEY],
        config_label="Mesh vs No Mesh",
        mode="mesh_compare",
        compare_outputs_root=OUT_DIR,
        extracted_roots_by_scenario=extracted_roots_by_scenario,
        use_ai=False,
    )

    print("[DONE] DOCX report generated:", docx_output_path)

except ImportError as e:
    print(f"[WARN] Could not generate DOCX report: {e}")
except Exception as e:
    print(f"[WARN] Error generating DOCX report: {e}")


from pathlib import Path
import zipfile

def zip_folder_overwrite(folder_path, zip_path):
    folder_path = Path(folder_path)
    zip_path = Path(zip_path)

    if zip_path.exists():
        zip_path.unlink()

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for file in folder_path.rglob("*"):
            if file.is_file():
                zf.write(file, arcname=file.relative_to(folder_path))

    return str(zip_path)

pngs = list(OUT_DIR.rglob("*.png"))
docxs = list(OUT_DIR.rglob("*.docx"))
if len(pngs) == 0 and len(docxs) == 0:
    raise RuntimeError(f"No plots or reports found in {OUT_DIR}.")

zip_path = zip_folder_overwrite(OUT_DIR, ZIP_PATH)
print("[DONE] Output zipped:", zip_path)

