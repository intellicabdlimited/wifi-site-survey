import shutil

# shutil.rmtree('/content/out')

# ============================================================
# INSTALLS (Colab)
# ============================================================
# !apt-get -qq update
# !apt-get -qq install -y tesseract-ocr
# !pip -q install pytesseract opencv-python-headless scipy pandas matplotlib seaborn


# ============================================================
# IMPORTS
# ============================================================
# from __future__ import annotations

import os, re, glob, zipfile, json, math
from pathlib import Path
from dataclasses import dataclass

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
#from google.colab import files
import seaborn as sns
import pytesseract
sns.set_theme()
from pytesseract import Output



# ============================================================
# PATHS
# ============================================================
INPUT_ROOT  = "out"
RESULT_ROOT = "ss_results"

# ============================================================
# GLOBAL DEBUG SETTINGS (EDIT HERE)
# ============================================================
DEBUG = False                 # saves extra debug images + JSON per heatmap
DEBUG_MAX_HEATMAPS = None    # set to an int to process only first N picked heatmaps
DELTAE_REJECT = 25.0         # reject hex if too far from legend color/sample
SAT_MIN_REJECT = 20          # reject low-saturation (background-ish) hexes
N_SAMPLES_CONTINUOUS = 256   # legend samples for continuous scales
N_BINS_CONTINUOUS = 12       # pseudo-bins for continuous legend (for histogram coloring)

IMG_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff")

# ============================================================
# Band parsing (STRICT: avoids Nokia5G21 false matches)
# Must contain: "on X GHz band"
# ============================================================
BAND_RX = re.compile(r"\bon\s*(2\.?\s*4|5)\s*ghz\s*band\b", re.IGNORECASE)

# ============================================================
# Metric Catalog (serial order = menu order)
# scale_kind: "numeric" | "categorical"
# has_band:   True for almost all Ekahau heatmaps
# range_hint: clamp user input if it goes wild (optional)
# ============================================================
DBM_EDGES_FALLBACK = [-np.inf, -90, -85, -80, -75, -70, -67, -65, -60, -55, -50, -45, -40, -35, -30, np.inf]

METRICS = [
    dict(key="signal_strength_main", name="Signal Strength", preset_edges=DBM_EDGES_FALLBACK,
         axis_label="dBm (Higher is Better)", expected_sign="negative", fixed_style="dbm",
         scale_kind="numeric", has_band=True),

    dict(key="snr", name="SNR",
        axis_label="SNR (Higher is Better)",
        expected_sign="positive",
        range_hint=(0.0, 60.0),
        fixed_style=None,
        scale_kind="numeric", has_band=True),

    dict(key="noise", name="Noise", preset_edges=None,
         axis_label="Noise", expected_sign="negative", fixed_style=None,
         scale_kind="numeric", has_band=True),

    dict(key="throughput", name="Throughput", preset_edges=None,
         axis_label="Throughput", expected_sign="positive", fixed_style=None,
         scale_kind="numeric", has_band=True),

    dict(key="data_rate", name="Data Rate", preset_edges=None,
         axis_label="Data Rate", expected_sign="positive", fixed_style=None,
         scale_kind="numeric", has_band=True),

    dict(key="channel_utilization", name="Channel Utilization", preset_edges=None,
         axis_label="Channel Utilization (%)", expected_sign="positive", fixed_style=None,
         range_hint=(0.0, 100.0), scale_kind="numeric", has_band=True),

    dict(key="channel_interference", name="Channel Interference", preset_edges=None,
         axis_label="Channel Interference", expected_sign="positive", fixed_style=None,
         scale_kind="numeric", has_band=True),

    dict(key="channel_width", name="Channel Width", preset_edges=None,
         axis_label="Channel Width (MHz)", expected_sign="positive", fixed_style=None,
         scale_kind="numeric", has_band=True),

    dict(key="network_health", name="Network Health", preset_edges=None,
         axis_label="Network Health", expected_sign="any", fixed_style=None,
         scale_kind="categorical", has_band=True),

    dict(key="network_issues", name="Network Issues", preset_edges=None,
         axis_label="Network Issues", expected_sign="any", fixed_style=None,
         scale_kind="categorical", has_band=True),

    dict(key="number_of_aps", name="Number of APs", preset_edges=None,
         axis_label="AP Count", expected_sign="positive", fixed_style=None,
         scale_kind="numeric", has_band=True),

    dict(
        key="spectrum_channel_power",
        name="Spectrum Channel Power",
        preset_edges=None,
        axis_label="Spectrum Channel Power (dBm)",
        expected_sign="negative",     # ✅ force negative
        fixed_style="dbm",            # ✅ use your dbm outlier handling
        range_hint=(-200.0, 0.0),     # ✅ keep OCR sane (optional but recommended)
        scale_kind="numeric",
        has_band=True
    ),
]

METRIC_CFG_BY_KEY = {m["key"]: m for m in METRICS}
ALLOWED_METRIC_KEYS = {
    "data_rate",
    "throughput",
    "signal_strength_main",
    "snr",
    "channel_utilization",
    "spectrum_channel_power",
}

# ============================================================
# EXCLUDE UNWANTED METRICS ENTIRELY (won't show in menu/inventory)
# ============================================================
EXCLUDE_METRIC_TEXT_RX = [
    r"\bassociated\s+access\s+point\b",
    r"\bbluetooth\s+coverage\b",
    r"\bsurvey\s+routes\s+and\s+access\s+points\b",
    r"\binterferers\b",
]

# ============================================================
# LOCATIONS + BANDS (NO ALL / NO UNKNOWN)
# ============================================================
LOCATIONS = [
    ("Upper Floor",  [r"upper\s*floor", r"upper\s*room", r"upstairs"]),
    ("Lower Floor",  [r"lower\s*floor", r"living\s*room", r"livingroom"]),
    ("Ground Floor", [r"ground\s*floor", r"groundfloor", r"kitchen", r"kithchen"]),
]

BANDS = [("2.4 GHz", []), ("5 GHz", [])]

# ============================================================
# SMALL UTILS
# ============================================================
def ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p

def unique_dir(base: str) -> str:
    if not os.path.exists(base):
        return base
    k = 2
    while True:
        p = f"{base}_{k}"
        if not os.path.exists(p):
            return p
        k += 1

def slug(s: str) -> str:
    s = (s or "").lower().strip()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_")

def norm_for_match(s: str) -> str:
    s = Path(s).name
    s = os.path.splitext(s)[0]
    s = (s or "").lower().replace("\u00A0", " ")
    s = re.sub(r"[_\-]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def log(msg: str):
    print(msg)

def dlog(msg: str):
    if DEBUG:
        print("[DEBUG]", msg)

def is_scale_file(path: str) -> bool:
    return "_scale" in Path(path).name.lower()

def device_from_filename(path: str) -> str:
    stem = Path(path).stem
    stem = re.sub(r"_scale(_\d+)?$", "", stem, flags=re.IGNORECASE)
    if "_" in stem:
        return stem.split("_", 1)[0]
    return stem.split(" ", 1)[0]

def parse_band_from_text(path_or_name: str) -> str | None:
    s = norm_for_match(path_or_name)
    m = BAND_RX.search(s)
    if not m:
        return None
    g = m.group(1).replace(" ", "")
    return "2.4 GHz" if g.startswith("2.4") or g.startswith("24") else "5 GHz"

def detect_band_from_name(filename: str) -> str:
    b = parse_band_from_text(filename)
    return b if b else "Unknown"

def parse_caption_part(filename: str) -> str:
    stem = Path(filename).stem
    stem = re.sub(r"_scale(_\d+)?$", "", stem, flags=re.IGNORECASE)
    if "_" in stem:
        return stem.split("_", 1)[1]
    return stem

def parse_metric_text(filename: str) -> str | None:
    cap = parse_caption_part(filename)
    m = re.search(r"^\s*(?P<metric>.+?)\s+for\s+", cap, flags=re.IGNORECASE)
    if not m:
        return None
    return m.group("metric").strip()

def metric_key_from_filename(filename: str) -> str | None:
    """
    - Collapses 'g4ar <metric>' => same metric
    - Drops excluded metrics
    - IMPORTANT: excludes Signal Strength Secondary/Tertiary when you pick Signal Strength
    """
    t = (parse_metric_text(filename) or "").strip().lower()
    if not t:
        return None

    t = re.sub(r"[_\-]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()

    # normalize prefix noise
    t = re.sub(r"^g4ar\s+", "", t)

    # drop unwanted metrics completely
    for rx in EXCLUDE_METRIC_TEXT_RX:
        if re.search(rx, t):
            return None

    # --- Signal Strength: main only (exclude secondary/tertiary)
    if re.search(r"\bsignal\s+strength\b", t):
        if re.search(r"\bsecondary\b|\btertiary\b", t):
            return None
        return "signal_strength_main"

    if re.search(r"\bsnr\b", t) or re.search(r"signal\s+to\s+noise", t):
        return "snr"
    if re.search(r"\bnoise\b", t):
        return "noise"
    if re.search(r"\bthroughput\b", t):
        return "throughput"
    if re.search(r"\bdata\s+rate\b", t):
        return "data_rate"
    if re.search(r"\bchannel\s+utilization\b", t):
        return "channel_utilization"
    if re.search(r"\bchannel\s+interference\b", t):
        return "channel_interference"
    if re.search(r"\bchannel\s+width\b", t):
        return "channel_width"
    if re.search(r"\bnetwork\s+health\b", t):
        return "network_health"
    if re.search(r"\bnetwork\s+issues\b", t):
        return "network_issues"
    if re.search(r"\bnumber\s+of\s+aps\b", t):
        return "number_of_aps"
    if re.search(r"\bspectrum\s+channel\s+power\b", t):
        return "spectrum_channel_power"

    return None

# ============================================================
# MATCHERS
# ============================================================
def matches_location(filename: str, loc_key: str) -> bool:
    s = norm_for_match(filename)
    for name, patterns in LOCATIONS:
        if name == loc_key:
            return any(re.search(pat, s) for pat in patterns)
    return False

def matches_band(filename: str, band_key: str) -> bool:
    band = parse_band_from_text(filename)
    if band is None:
        return False
    return band == band_key

# ============================================================
# SCALE IMAGE MATCHING (STRICT + BAND-SAFE)
# ============================================================
def find_matching_scale(main_path: str) -> str:
    """
    Best-effort strict pairing:
      1) exact stem + _scale*
      2) stem-start + _scale
    Band-safe:
      - If heatmap band exists, prefer same-band scale filename.
      - If only bandless scales exist, only allow if exactly ONE candidate.
      - Otherwise: error (forces correct naming instead of silently wrong pairing)
    """
    p = Path(main_path)
    folder = p.parent
    stem = p.stem
    main_band = parse_band_from_text(p.name)

    candidates = []
    for ext in IMG_EXTS:
        candidates.extend(folder.glob(f"{stem}_scale*{ext}"))

    if not candidates:
        pref = (stem + "_scale").lower()
        for q in folder.iterdir():
            if q.is_file() and q.suffix.lower() in IMG_EXTS and q.stem.lower().startswith(pref):
                candidates.append(q)

    if not candidates:
        raise FileNotFoundError(f"Scale image not found for heatmap:\n{main_path}")

    if main_band is not None:
        exact = []
        bandless = []
        for c in candidates:
            c_band = parse_band_from_text(c.name)
            if c_band == main_band:
                exact.append(c)
            elif c_band is None:
                bandless.append(c)

        if exact:
            candidates = exact
        else:
            if len(bandless) == 1:
                candidates = bandless
            else:
                raise FileNotFoundError(
                    "Scale band mismatch/ambiguous.\n"
                    f"Heatmap: {p.name}\n"
                    f"Heatmap band: {main_band}\n"
                    f"Candidates:\n" + "\n".join([c.name for c in candidates])
                )

    candidates = sorted(candidates, key=lambda x: len(x.name))
    return str(candidates[0])

# ============================================================
# IMAGE IO
# ============================================================
def read_image_rgb(path: str) -> np.ndarray:
    im = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if im is None:
        raise FileNotFoundError(f"Could not read: {path}")
    if im.ndim == 2:
        return cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
    if im.shape[2] == 4:
        return cv2.cvtColor(im, cv2.COLOR_BGRA2RGB)
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

def save_gray(mask_u8: np.ndarray, out_file: str):
    cv2.imwrite(out_file, mask_u8)
    return out_file

def save_rgb(rgb: np.ndarray, out_file: str):
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(out_file, bgr)
    return out_file


# ---------------- OCR (robust) ----------------
OCR_MINMAX_CACHE = {}  # scale_path -> (mn, mx)

TESS_TIMEOUT_SEC = 2.0   # prevents Tesseract hanging forever
OCR_UPSCALE = 8          # resolution boost for small text

def _auto_upscale(h: int, target_px: int = 110, lo: int = 2, hi: int = 14) -> int:
    """
    Choose an upscale factor so small legend text becomes readable.
    target_px ~ target crop height after scaling.
    """
    return int(np.clip(target_px / max(h, 1), lo, hi))


def _tess_data(img, config: str):
    """Safe pytesseract.image_to_data with timeout."""
    try:
        return pytesseract.image_to_data(
            img, config=config, output_type=Output.DICT, timeout=TESS_TIMEOUT_SEC
        )
    except Exception:
        return {"text": [], "conf": []}


def _normalize_signs(s: str) -> str:
    return (s.replace("−", "-").replace("–", "-").replace("—", "-")
              .replace("_", "-").replace("＋", "+"))


def _normalize_ops(s: str) -> str | None:
    s = (s or "").replace("≤", "<=").replace("≥", ">=")
    s = s.replace("=<", "<=").replace("=>", ">=")
    s = re.sub(r"[^<>=]", "", s.strip())
    if "<" in s and "=" in s: return "<="
    if ">" in s and "=" in s: return ">="
    if "<" in s: return "<"
    if ">" in s: return ">"
    return None


def _parse_signed_floats(txt: str) -> list[float]:
    txt = _normalize_signs(txt)
    matches = re.findall(r"([+\-]{0,4})\s*(\d+(?:\.\d+)?)", txt)
    vals = []
    for signs, digits in matches:
        v = float(digits)
        if "-" in signs:
            v = -v
        vals.append(v)
    if not vals:
        vals = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", txt)]
    return vals

def _parse_unsigned_numbers(txt: str) -> list[float]:
    """
    Digits-only extraction.
    Also adds a merged candidate for split blocks like '2 500' -> 2500.
    """
    parts = re.findall(r"\d+(?:\.\d+)?", txt or "")
    vals = [float(p) for p in parts]

    # merge split integer chunks: 2 + 500 => 2500, 5 + 85 => 585
    if len(parts) >= 2 and all("." not in p for p in parts):
        try:
            vals.append(float(int("".join(parts))))
        except Exception:
            pass
    return vals


def _fix_db_outliers(val: float | None, lo=-200, hi=200) -> float | None:
    if val is None:
        return None
    if lo <= val <= hi:
        return float(val)
    sign = -1 if val < 0 else 1
    a = abs(val)
    for div in (10, 100):
        cand = sign * (math.floor(a / div))
        if lo <= cand <= hi:
            return float(cand)
    return float(val)


def _mask_color_bar_keep_text(bgr: np.ndarray) -> np.ndarray:
    """
    Remove saturated colored pixels (legend bar) but DO NOT erase dark text/edges.
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    _, S, V = cv2.split(hsv)

    colored = (S > 35) & (V > 60)
    keep_dark = (V < 170)
    kill = colored & (~keep_dark)

    out = bgr.copy()
    out[kill] = (255, 255, 255)

    near_white = (V > 240) & (S < 25)
    out[near_white] = (255, 255, 255)
    return out


def _crop_digits_only(bgr_crop: np.ndarray) -> np.ndarray:
    """
    Best-effort crop that tries to keep only digit cluster (drops Mb/s).
    If it fails, returns original crop.
    """
    bgr = _mask_color_bar_keep_text(bgr_crop)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    up = cv2.resize(gray, None, fx=4, fy=4, interpolation=cv2.INTER_LANCZOS4)
    thr = cv2.threshold(up, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    thr = cv2.morphologyEx(thr, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)

    n, _, stats, _ = cv2.connectedComponentsWithStats(thr, connectivity=8)
    H, W = thr.shape[:2]

    boxes = []
    for i in range(1, n):
        x, y, w, h, a = stats[i].tolist()
        if a < 120 or h < 0.20 * H:
            continue
        boxes.append((x, y, w, h))

    if not boxes:
        return bgr_crop

    boxes = sorted(boxes, key=lambda t: t[0])
    gap = int(0.15 * W)

    groups = [[boxes[0]]]
    last_end = boxes[0][0] + boxes[0][2]
    for (x, y, w, h) in boxes[1:]:
        if x - last_end > gap:
            groups.append([(x, y, w, h)])
        else:
            groups[-1].append((x, y, w, h))
        last_end = max(last_end, x + w)

    def gscore(g):
        hs = [b[3] for b in g]
        areas = [b[2] * b[3] for b in g]
        return (float(np.median(hs)), float(np.sum(areas)))

    digit_group = max(groups, key=gscore)

    x1 = min(b[0] for b in digit_group)
    y1 = min(b[1] for b in digit_group)
    x2 = max(b[0] + b[2] for b in digit_group)
    y2 = max(b[1] + b[3] for b in digit_group)

    pad = 18
    x1 = max(0, x1 - pad); y1 = max(0, y1 - pad)
    x2 = min(W, x2 + pad); y2 = min(H, y2 + pad)

    crop_up = up[y1:y2, x1:x2]
    crop = cv2.resize(crop_up, (0, 0), fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)


def _preprocess_variants_for_ocr(bgr: np.ndarray, upscale: int = OCR_UPSCALE) -> list[np.ndarray]:
    """
    OCR variants (ensemble):
      - dark-text segmentation (thin + thick)
      - blackhat text isolation
      - CLAHE+sharpen Otsu/adaptive
      - NEW: background flattening (divide by blurred background) + Otsu
    """
    bgr = _mask_color_bar_keep_text(bgr)

    bgr_u = cv2.resize(bgr, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_LANCZOS4)
    gray = cv2.cvtColor(bgr_u, cv2.COLOR_BGR2GRAY)

    # add border (helps Tesseract not clip edge digits)
    bgr_u = cv2.copyMakeBorder(bgr_u, 18, 18, 18, 18, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    gray  = cv2.copyMakeBorder(gray,  18, 18, 18, 18, cv2.BORDER_CONSTANT, value=255)

    outs = []

    k = np.ones((3, 3), np.uint8)

    # ---------------- NEW) BACKGROUND FLATTENING ----------------
    # Great for dark digits sitting on a gradient legend background
    bg = cv2.GaussianBlur(gray, (0, 0), 25)
    flat = cv2.divide(gray, bg, scale=255)
    flat = cv2.normalize(flat, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    flat_otsu = cv2.threshold(flat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    if flat_otsu.mean() < 127:
        flat_otsu = 255 - flat_otsu
    flat_otsu = cv2.morphologyEx(flat_otsu, cv2.MORPH_CLOSE, k, iterations=1)

    outs.append(flat)        # grayscale flattened
    outs.append(flat_otsu)   # binarized flattened

    # ---------------- A) DARK TEXT SEGMENTATION (THIN + THICK) ----------------
    hsv_u = cv2.cvtColor(bgr_u, cv2.COLOR_BGR2HSV)
    V = hsv_u[:, :, 2]
    V = cv2.medianBlur(V, 3)

    dark_inv = cv2.threshold(V, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    thin = cv2.morphologyEx(dark_inv, cv2.MORPH_OPEN, k, iterations=1)
    thin = cv2.morphologyEx(thin, cv2.MORPH_CLOSE, k, iterations=1)
    thin_text = np.full_like(thin, 255); thin_text[thin > 0] = 0
    outs.append(thin_text)

    thick = cv2.dilate(thin, k, iterations=1)
    thick_text = np.full_like(thick, 255); thick_text[thick > 0] = 0
    outs.append(thick_text)

    # ---------------- B) BLACKHAT (TEXT ISOLATION) ----------------
    bh_k = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 9))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, bh_k)
    blackhat = cv2.normalize(blackhat, None, 0, 255, cv2.NORM_MINMAX)

    bh_bin = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    bh_bin = cv2.morphologyEx(bh_bin, cv2.MORPH_CLOSE, k, iterations=1)
    if bh_bin.mean() < 127:
        bh_bin = 255 - bh_bin
    outs.append(bh_bin)

    # ---------------- C) CLAHE + SHARPEN ----------------
    try:
        gray_dn = cv2.fastNlMeansDenoising(gray, None, h=15, templateWindowSize=7, searchWindowSize=21)
    except Exception:
        gray_dn = gray

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    g = clahe.apply(gray_dn)

    blur = cv2.GaussianBlur(g, (0, 0), 1.6)
    sharp = cv2.addWeighted(g, 1.8, blur, -0.8, 0)

    t1 = cv2.threshold(sharp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    t2 = cv2.threshold(sharp, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    outs.extend([t1, t2])

    a1 = cv2.adaptiveThreshold(sharp, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 41, 11)
    outs.append(a1)

    # reconnect small gaps + ensure black text on white
    fixed = []
    for img in outs:
        img2 = cv2.morphologyEx(img, cv2.MORPH_CLOSE, k, iterations=1)
        if img2.mean() < 127:
            img2 = 255 - img2
        fixed.append(img2)

    # include grayscale too (sometimes best)
    fixed.append(sharp)

    return fixed

def _tight_text_bbox_from_gray(gray: np.ndarray) -> tuple[int, int, int, int] | None:
    thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    thr = cv2.morphologyEx(thr, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8), iterations=1)
    thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)

    n, _, stats, _ = cv2.connectedComponentsWithStats(thr, connectivity=8)
    H, W = gray.shape[:2]

    boxes = []
    for i in range(1, n):
        x, y, ww, hh, a = stats[i].tolist()
        if a < 40:
            continue

        # NEW: drop skinny full-height vertical bars (legend border)
        if ww <= max(2, int(0.02 * W)) and hh >= int(0.80 * H):
            continue

        boxes.append((x, y, x + ww, y + hh))

    if not boxes:
        return None

    x1 = min(b[0] for b in boxes); y1 = min(b[1] for b in boxes)
    x2 = max(b[2] for b in boxes); y2 = max(b[3] for b in boxes)

    pad = 10
    x1 = max(0, x1 - pad); y1 = max(0, y1 - pad)
    x2 = min(W, x2 + pad); y2 = min(H, y2 + pad)
    return (x1, y1, x2, y2)

def _crop_numeric_focus_remove_operator(bgr_crop: np.ndarray, expected_sign: str = "any") -> np.ndarray:
    """
    Removes operator-like junk (≥, ≤) by cropping to the left edge of the 'main' text cluster.
    Keeps a little left padding to preserve a minus sign if expected_sign is negative.
    """
    if bgr_crop is None or bgr_crop.size == 0:
        return bgr_crop

    gray = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2GRAY)

    thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    thr = cv2.morphologyEx(thr, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8), iterations=1)
    thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)

    n, _, stats, _ = cv2.connectedComponentsWithStats(thr, connectivity=8)
    H, W = gray.shape[:2]
    if n <= 1:
        return bgr_crop

    # Only consider components that are reasonably "text-like"
    comps = []
    for i in range(1, n):
        x, y, w, h, a = stats[i].tolist()
        if a < 12:
            continue
        if h < int(0.20 * H):  # ignore tiny specks
            continue
        comps.append((i, x, y, w, h, a))

    if not comps:
        return bgr_crop

    max_area = max(c[-1] for c in comps)

    # Keep "main" components (digits/letters), operator is usually much smaller
    keep = []
    for (_, x, y, w, h, a) in comps:
        if a >= 0.25 * max_area:
            keep.append((x, x + w))

    if not keep:
        return bgr_crop

    x_min = min(k[0] for k in keep)

    # Preserve minus sign if needed (minus is small area, near digits)
    expand = int(0.10 * W) if expected_sign == "negative" else int(0.04 * W)
    x_start = max(0, x_min - expand)

    return bgr_crop[:, x_start:, :]

def detect_minus_sign(bgr_crop: np.ndarray) -> bool:
    """
    Detects a minus sign "-" by finding a thin horizontal stroke
    near the left side of the numeric text.
    Works on tiny legend labels.
    """
    if bgr_crop is None or bgr_crop.size == 0:
        return False

    gray = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2GRAY)

    # Binarize text (dark) vs background (bright)
    thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    H, W = thr.shape[:2]

    # Only search in the left half-ish (minus is before digits)
    thr = thr[:, :max(10, int(0.65 * W))]

    # Emphasize horizontal strokes
    k_w = max(8, W // 18)  # tune 16-24 if needed
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_w, 1))
    horiz = cv2.morphologyEx(thr, cv2.MORPH_OPEN, kernel, iterations=1)

    # Clean small noise
    horiz = cv2.morphologyEx(horiz, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8), iterations=1)

    n, _, stats, _ = cv2.connectedComponentsWithStats(horiz, connectivity=8)
    for i in range(1, n):
        x, y, w, h, a = stats[i].tolist()

        # minus-like: thin and wide
        if a < 10:
            continue
        if h <= max(2, H // 10) and w >= max(6, W // 25) and (w / max(1, h)) >= 3.0:
            # also avoid detecting random underline near bottom
            if y < 0.75 * H:
                return True

    return False

# ============================================================
# Segmentation-based signed OCR fallback (from ocr.py, simplified)
# Paste BELOW _crop_numeric_focus_remove_operator() and ABOVE _ocr_best_numeric()
# ============================================================

def _seg_preprocess_binary_for_segmentation(bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # normalize gradient background (good for legend bars)
    bg = cv2.GaussianBlur(gray, (0, 0), 25)
    norm = cv2.divide(gray, bg, scale=255)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    norm = clahe.apply(norm)

    thr = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    thr = cv2.dilate(thr, np.ones((2, 2), np.uint8), iterations=1)
    thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8), iterations=1)
    return thr

def _seg_contour_boxes(thr: np.ndarray) -> list[tuple[int,int,int,int]]:
    contours, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w*h < 20:
            continue
        boxes.append((x, y, w, h))
    return boxes

def _seg_group_boxes_into_blocks(boxes, W, H, gap_frac=0.12, pad=6):
    if not boxes:
        return []
    boxes = sorted(boxes, key=lambda b: b[0])
    gap_thresh = int(gap_frac * W)

    groups = []
    cur = [boxes[0]]
    last_end = boxes[0][0] + boxes[0][2]

    for (x,y,w,h) in boxes[1:]:
        if x - last_end > gap_thresh:
            groups.append(cur)
            cur = [(x,y,w,h)]
        else:
            cur.append((x,y,w,h))
        last_end = max(last_end, x+w)
    groups.append(cur)

    out = []
    for g in groups:
        xs = [b[0] for b in g]
        ys = [b[1] for b in g]
        x2 = [b[0]+b[2] for b in g]
        y2 = [b[1]+b[3] for b in g]
        x1,y1,x22,y22 = min(xs), min(ys), max(x2), max(y2)
        x1 = max(0, x1-pad); y1 = max(0, y1-pad)
        x22 = min(W, x22+pad); y22 = min(H, y22+pad)
        out.append((x1,y1,x22,y22))
    return out

def _seg_normalize_signs(s: str) -> str:
    return (s.replace("−","-").replace("–","-").replace("—","-")
              .replace("_","-").replace("＋","+"))

def _seg_normalize_ops(s: str) -> str | None:
    s = (s or "").replace("≤", "<=").replace("≥", ">=")
    s = s.replace("=<", "<=").replace("=>", ">=")
    s = re.sub(r"[^<>=]", "", s.strip())
    if "<" in s and "=" in s: return "<="
    if ">" in s and "=" in s: return ">="
    if "<" in s: return "<"
    if ">" in s: return ">"
    return None

def _seg_parse_signed_candidates(txt: str) -> list[float]:
    txt = _seg_normalize_signs(txt)
    # allow decimals too
    matches = re.findall(r"([+\-]{0,3})\s*(\d+(?:\.\d+)?)", txt)
    vals = []
    for signs, digits in matches:
        v = float(digits)
        if "-" in signs:
            v = -v
        vals.append(v)
    if not vals:
        vals = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", txt)]
    return vals

def _seg_ocr_operator(bgr_crop: np.ndarray) -> str | None:
    gray = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2GRAY)
    thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    if thr.mean() < 127:
        thr = 255 - thr

    cfgs = [
        "--oem 1 --psm 7 -c tessedit_char_whitelist=<>=",
        "--oem 1 --psm 8 -c tessedit_char_whitelist=<>=",
        "--oem 1 --psm 13 -c tessedit_char_whitelist=<>=",
    ]
    for cfg in cfgs:
        t = pytesseract.image_to_string(thr, config=cfg).strip()
        op = _seg_normalize_ops(t)
        if op:
            return op
    return None

def _seg_ocr_signed_number(bgr_crop: np.ndarray) -> float | None:
    gray = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

    thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    if thr.mean() < 127:
        thr = 255 - thr

    blur = cv2.GaussianBlur(gray, (0,0), 1.0)
    sharp = cv2.addWeighted(gray, 1.6, blur, -0.6, 0)

    variants = [thr, sharp]
    cfgs = [
        "--oem 1 --psm 7  -c tessedit_char_whitelist=+-0123456789.",
        "--oem 1 --psm 8  -c tessedit_char_whitelist=+-0123456789.",
        "--oem 1 --psm 13 -c tessedit_char_whitelist=+-0123456789.",
    ]

    best = None
    best_len = -1
    for img in variants:
        for cfg in cfgs:
            t = pytesseract.image_to_string(img, config=cfg).strip()
            vals = _seg_parse_signed_candidates(t)
            if not vals:
                continue
            # choose candidate with most digits
            cand = max(vals, key=lambda n: len(str(abs(n)).replace(".","")))
            L = len(str(abs(cand)).replace(".",""))
            if L > best_len:
                best = cand
                best_len = L
    return None if best is None else float(best)

def seg_extract_op_and_value_from_crop(bgr_crop: np.ndarray, side: str) -> tuple[str | None, float | None]:
    """
    Takes your already-cropped label zone (left or right) and:
      - finds the edge text block
      - splits operator block vs number block (if present)
      - OCR signed number from number block
    """
    thr = _seg_preprocess_binary_for_segmentation(bgr_crop)
    H, W = thr.shape[:2]

    boxes = _seg_contour_boxes(thr)
    blocks = _seg_group_boxes_into_blocks(boxes, W, H, gap_frac=0.10, pad=6)
    if not blocks:
        return None, None

    # edge block: leftmost or rightmost
    x1,y1,x2,y2 = blocks[0] if side == "left" else blocks[-1]
    block_crop = bgr_crop[y1:y2, x1:x2]

    thr2 = _seg_preprocess_binary_for_segmentation(block_crop)
    H2, W2 = thr2.shape[:2]
    sub_boxes = _seg_contour_boxes(thr2)
    subs = _seg_group_boxes_into_blocks(sub_boxes, W2, H2, gap_frac=0.12, pad=6)

    op = None
    number_crop = block_crop

    if len(subs) >= 2:
        ox1,oy1,ox2,oy2 = subs[0]
        nx1,ny1,nx2,ny2 = subs[-1]
        op_crop = block_crop[oy1:oy2, ox1:ox2]
        number_crop = block_crop[ny1:ny2, nx1:nx2]
        op = _seg_ocr_operator(op_crop)

    val = _seg_ocr_signed_number(number_crop)
    return op, val

from collections import defaultdict

def _ocr_best_numeric(bgr_crop: np.ndarray, side: str, metric_cfg: dict) -> tuple[float | None, dict]:
    cfgs = [
        "--oem 1 --psm 7 -c tessedit_char_whitelist=+-0123456789. "
        "-c classify_bln_numeric_mode=1 -c load_system_dawg=0 -c load_freq_dawg=0",
        "--oem 1 --psm 8 -c tessedit_char_whitelist=+-0123456789. "
        "-c classify_bln_numeric_mode=1 -c load_system_dawg=0 -c load_freq_dawg=0",
        "--oem 1 --psm 6 -c tessedit_char_whitelist=+-0123456789. "
        "-c classify_bln_numeric_mode=1 -c load_system_dawg=0 -c load_freq_dawg=0",
    ]

    expected_sign = metric_cfg.get("expected_sign", "any")
    range_hint = metric_cfg.get("range_hint", None)

    def plausible(v: float) -> bool:
        if abs(v) > 100000:
            return False
        if expected_sign == "positive" and v < 0:
            return False
        # STRICT: negative metrics must be < 0 (reject 0)
        if expected_sign == "negative" and v >= 0:
            return False
        if range_hint is not None:
            lo, hi = range_hint
            if v < lo - 1e-6 or v > hi + 1e-6:
                return False
        return True

    # operator removal focus crop first
    focus = _crop_numeric_focus_remove_operator(bgr_crop, expected_sign=expected_sign)
    minus_present = detect_minus_sign(focus)
    seg_op, seg_val = seg_extract_op_and_value_from_crop(focus, side=side)
    crops = [("focus", focus), ("orig", bgr_crop)]

    # digits-only crop based on focus
    try:
        c2 = _crop_digits_only(focus)
        if c2 is not None and c2.size > 0:
            crops.append(("digits", c2))
    except Exception:
        pass

    votes = defaultdict(lambda: {"count": 0, "w": 0.0, "best_conf": -1.0, "best_txt": "", "src": ""})

    for src, crop in crops:
        up = _auto_upscale(crop.shape[0])
        for img in _preprocess_variants_for_ocr(crop, upscale=up):
            for cfg in cfgs:
                d = _tess_data(img, cfg)

                tokens, confs = [], []
                for t, c in zip(d.get("text", []), d.get("conf", [])):
                    t = (t or "").strip()
                    if not t:
                        continue
                    try:
                        cf = float(c)
                    except Exception:
                        continue
                    if cf < 35:
                        continue
                    tokens.append(t)
                    confs.append(cf)

                txt = " ".join(tokens).strip()
                if not txt:
                    continue

                # ---- THIS MUST BE HERE (txt exists here) ----
                raw = _parse_signed_floats(txt)
                vals = []
                for v in raw:
                    if plausible(v):
                        vals.append(v)
                    # recovery if '-' got dropped
                    if expected_sign == "negative" and v > 0 and plausible(-v):
                        vals.append(-v)

                if not vals:
                    continue
                # --------------------------------------------

                cand = min(vals) if side == "left" else max(vals)

                # ✅ PLACE IT HERE (right after cand is chosen)
                cand = float(cand)
                if expected_sign == "negative":
                    # dBm/noise should always be negative
                    cand = -abs(cand)
                elif expected_sign == "positive":
                    # only flip for positive metrics if we really saw a minus sign
                    if minus_present:
                        cand = -abs(cand)
                else:
                    # expected_sign == "any"
                    if minus_present:
                        cand = -abs(cand)

                conf = float(np.mean(confs)) if confs else 0.0

                v = float(cand)
                votes[v]["count"] += 1
                votes[v]["w"] += max(0.0, conf)
                if conf > votes[v]["best_conf"]:
                    votes[v]["best_conf"] = conf
                    votes[v]["best_txt"] = txt
                    votes[v]["src"] = src

    if not votes:
        return None, {"conf": -1.0, "txt": "", "src": ""}
    # ✅ segmentation fallback vote (helps signed negatives like -60, -90)
    if seg_val is not None and plausible(float(seg_val)):
        sv = float(seg_val)
        votes[sv]["count"] += 3
        votes[sv]["w"] += 500.0
        if votes[sv]["best_conf"] < 99:
            votes[sv]["best_conf"] = 99
            votes[sv]["best_txt"] = f"SEG:{seg_op or ''}{seg_val}"
            votes[sv]["src"] = "segmented"

    # keep your positive left guard
    if side == "left" and expected_sign == "positive":
        if 0.0 in votes or 1.0 in votes:
            pick = 0.0 if 0.0 in votes else 1.0
            meta = votes[pick]
            return pick, {"conf": meta["best_conf"], "txt": meta["best_txt"], "src": meta["src"], "vote": meta}

    best_v, meta = max(votes.items(), key=lambda kv: (kv[1]["count"], kv[1]["w"], kv[1]["best_conf"]))
    return best_v, {"conf": meta["best_conf"], "txt": meta["best_txt"], "src": meta["src"], "vote": meta}


def _ocr_best_operator(bgr_crop: np.ndarray) -> tuple[str | None, dict]:
    cfgs = [
        "--oem 1 --psm 10 -c tessedit_char_whitelist=<>=",
        "--oem 1 --psm 7 -c tessedit_char_whitelist=<>=",
        "--oem 1 --psm 8 -c tessedit_char_whitelist=<>=",
    ]

    best_op = None
    best_conf = -1.0
    best_txt = ""

    up = _auto_upscale(bgr_crop.shape[0])
    for img in _preprocess_variants_for_ocr(bgr_crop, upscale=up):
        for cfg in cfgs:
            d = _tess_data(img, cfg)

            tokens, confs = [], []
            for t, c in zip(d.get("text", []), d.get("conf", [])):
                t = (t or "").strip()
                if not t:
                    continue
                try:
                    cf = float(c)
                except Exception:
                    continue
                if cf < 0:
                    continue
                tokens.append(t)
                confs.append(cf)

            txt = " ".join(tokens).strip()
            op = _normalize_ops(txt)
            conf = float(np.mean(confs)) if confs else -1.0

            if op and conf > best_conf:
                best_conf = conf
                best_op = op
                best_txt = txt

    return best_op, {"conf": best_conf, "txt": best_txt}


# ============================================================
# OCR (Scale min/max) — preprocess + data extraction (Ekahau legends)
# Drop this whole block into your notebook and call:
#   mn, mx = ocr_minmax_from_scale_path(scale_path, metric_cfg, debug_dir=..., tag=...)
# ============================================================

import os, re, json, math
import cv2
import numpy as np
import pytesseract

# ----------------------------
# SETTINGS
# ----------------------------
OCR_MINMAX_CACHE = {}     # scale_path -> (mn, mx)
TESS_TIMEOUT_SEC = 2.5    # one OCR call is ~1–2s (process startup dominates)
DET_UPSCALE = 8           # scale text is tiny (~30px). Upscale for block detection.
BLOCK_GAP_FRAC = 0.05     # split [op][number][unit] blocks inside a label crop


# ============================================================
# Small text helpers
# ============================================================
def _tess_string(img, config: str) -> str:
    """Safe pytesseract OCR with timeout."""
    try:
        return pytesseract.image_to_string(img, config=config, timeout=TESS_TIMEOUT_SEC) or ""
    except Exception:
        return ""

def _normalize_signs(s: str) -> str:
    return (s.replace("−", "-").replace("–", "-").replace("—", "-")
              .replace("_", "-").replace("＋", "+"))

def _parse_signed_numbers(txt: str) -> list[float]:
    """
    Extract signed numbers robustly.
    Handles cases like "2-60" -> [2, -60]
    """
    txt = _normalize_signs(txt)
    matches = re.findall(r"([+\-]{0,4})\s*(\d+(?:\.\d+)?)", txt)
    out = []
    for signs, digits in matches:
        v = float(digits)
        if "-" in signs:
            v = -v
        out.append(v)
    if not out:
        out = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", txt)]
    return out

def _candidate_digit_len(v: float) -> int:
    """Prefer -90 over -0, 2500 over 250, etc."""
    v = float(v)
    a = abs(v)
    if abs(a - round(a)) < 1e-6:
        return len(str(int(round(a))))
    s = f"{a:.6f}".rstrip("0").rstrip(".")
    return len(re.sub(r"\D", "", s))

def _fix_db_outliers(val: float | None, lo=-200, hi=200) -> float | None:
    """
    If OCR does something like -900 or -600, try to recover (-90, -60).
    Safe-ish for dB/dBm-like metrics.
    """
    if val is None:
        return None
    try:
        val = float(val)
    except Exception:
        return None
    if lo <= val <= hi:
        return float(val)

    sign = -1 if val < 0 else 1
    a = abs(val)
    for div in (10, 100):
        cand = sign * (math.floor(a / div))
        if lo <= cand <= hi:
            return float(cand)
    return float(val)


# ============================================================
# Legend masking (remove colored bar, keep dark text)
# ============================================================
def _mask_color_bar_keep_text(bgr: np.ndarray) -> np.ndarray:
    """
    Whiten saturated bar pixels while keeping dark text/edges.
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    S = hsv[:, :, 1]
    V = hsv[:, :, 2]

    colored = (S > 35) & (V > 60)
    keep_dark = (V < 170)
    kill = colored & (~keep_dark)

    out = bgr.copy()
    out[kill] = (255, 255, 255)

    near_white = (V > 240) & (S < 25)
    out[near_white] = (255, 255, 255)
    return out


# ============================================================
# Segmentation utilities (split operator / number / unit)
# ============================================================
def _seg_preprocess_binary(bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # flatten gradient background (legend bars often have gradients)
    bg = cv2.GaussianBlur(gray, (0, 0), 25)
    norm = cv2.divide(gray, bg, scale=255)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    norm = clahe.apply(norm)

    thr = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    thr = cv2.dilate(thr, np.ones((2, 2), np.uint8), iterations=1)
    thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8), iterations=1)
    return thr

def _seg_contour_boxes(thr: np.ndarray) -> list[tuple[int, int, int, int]]:
    contours, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w * h < 20:
            continue
        boxes.append((x, y, w, h))
    return boxes

def _seg_group_boxes_into_blocks(
    boxes: list[tuple[int, int, int, int]],
    W: int,
    gap_frac: float = BLOCK_GAP_FRAC,
    pad: int = 6
) -> list[tuple[int, int, int, int]]:
    """
    Group nearby component boxes into horizontal "blocks".
    Returns list of (x1,y1,x2,y2) in block coordinates.
    """
    if not boxes:
        return []
    boxes = sorted(boxes, key=lambda b: b[0])
    gap_thresh = int(gap_frac * W)

    groups = []
    cur = [boxes[0]]
    last_end = boxes[0][0] + boxes[0][2]

    for (x, y, w, h) in boxes[1:]:
        if x - last_end > gap_thresh:
            groups.append(cur)
            cur = [(x, y, w, h)]
        else:
            cur.append((x, y, w, h))
        last_end = max(last_end, x + w)
    groups.append(cur)

    out = []
    for g in groups:
        xs = [b[0] for b in g]
        ys = [b[1] for b in g]
        x2 = [b[0] + b[2] for b in g]
        y2 = [b[1] + b[3] for b in g]
        x1, y1, x22, y22 = min(xs), min(ys), max(x2), max(y2)
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x22 = min(W, x22 + pad)
        y22 = y22 + pad
        out.append((x1, y1, x22, y22))
    return out


# ============================================================
# 1) Detect the LEFT/RIGHT endpoint label blocks on the scale image
# ============================================================
def detect_left_right_label_blocks(scale_bgr: np.ndarray, det_upscale: int = DET_UPSCALE):
    """
    Returns:
      left_label_bgr, right_label_bgr, dbg
    where each label crop is already upscaled.
    """
    bgr = _mask_color_bar_keep_text(scale_bgr)
    bgr_u = cv2.resize(bgr, None, fx=det_upscale, fy=det_upscale, interpolation=cv2.INTER_LANCZOS4)
    gray = cv2.cvtColor(bgr_u, cv2.COLOR_BGR2GRAY)

    # flatten + enhance
    bg = cv2.GaussianBlur(gray, (0, 0), 25)
    flat = cv2.divide(gray, bg, scale=255)
    flat = cv2.normalize(flat, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    flat = clahe.apply(flat)

    thr = cv2.threshold(flat, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    thr = cv2.morphologyEx(thr, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=2)

    H, W = thr.shape[:2]
    n, _, stats, _ = cv2.connectedComponentsWithStats(thr, connectivity=8)

    boxes = []
    for i in range(1, n):
        x, y, w, h, a = stats[i].tolist()
        if a < 50:
            continue
        if h < int(0.30 * H):
            continue
        # drop skinny vertical legend borders
        if w <= max(2, int(0.02 * W)) and h >= int(0.80 * H):
            continue
        boxes.append((x, y, w, h))

    if not boxes:
        return None, None, {"thr": thr, "blocks": []}

    blocks = _seg_group_boxes_into_blocks(boxes, W, gap_frac=0.10, pad=int(0.02 * W))
    blocks = sorted(blocks, key=lambda b: b[0])

    left = blocks[0]
    right = blocks[-1]

    # extend a bit so we don't clip operator signs
    pad_extra = int(0.05 * W)

    lx1, ly1, lx2, ly2 = left
    rx1, ry1, rx2, ry2 = right

    lx1 = 0
    lx2 = min(W, lx2 + pad_extra)

    rx1 = max(0, rx1 - pad_extra)
    rx2 = W

    left_crop = bgr_u[ly1:ly2, lx1:lx2].copy()
    right_crop = bgr_u[ry1:ry2, rx1:rx2].copy()

    return left_crop, right_crop, {"thr": thr, "blocks": blocks, "bgr_u": bgr_u}


# ============================================================
# 2) Split a label into operator + number (+ unit)
# ============================================================
def extract_op_and_number_crops(label_bgr: np.ndarray):
    """
    Robust split:
      - Detect blocks
      - Identify operator blocks by OCRing ONLY <>=
      - Identify numeric blocks by OCRing ONLY digits/sign
      - Merge consecutive numeric-ish blocks (fixes 585->85, 2500->2000)
      - Ignore trailing unit blocks (Mb/s, dBm, %, dB)
    Returns: (op_crop_or_None, number_crop_bgr)
    """
    if label_bgr is None or label_bgr.size == 0:
        return None, label_bgr

    thr = _seg_preprocess_binary(label_bgr)
    H, W = thr.shape[:2]

    boxes = _seg_contour_boxes(thr)
    blocks = _seg_group_boxes_into_blocks(boxes, W, gap_frac=BLOCK_GAP_FRAC, pad=10)
    blocks = sorted(blocks, key=lambda b: b[0])

    if len(blocks) <= 1:
        return None, label_bgr

    def crop_block(b):
        x1,y1,x2,y2 = b
        return label_bgr[y1:y2, x1:x2].copy()

    def ocr_block_op(bgr):
        g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        t = pytesseract.image_to_string(
            g, config="--oem 1 --psm 10 -c tessedit_char_whitelist=<>=", timeout=TESS_TIMEOUT_SEC
        ).strip()
        return _normalize_ops(t)

    def ocr_block_has_digit(bgr):
        g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        t = pytesseract.image_to_string(
            g, config="--oem 1 --psm 10 -c tessedit_char_whitelist=+-0123456789.", timeout=TESS_TIMEOUT_SEC
        )
        return bool(re.search(r"\d", t or ""))

    op_crop = None
    i = 0

    # 1) Optional leading operator block (≤/≥ sometimes becomes <=/>=)
    first = crop_block(blocks[0])
    op = ocr_block_op(first)
    if op is not None:
        op_crop = first
        i = 1

    # 2) Merge consecutive numeric blocks (this fixes 5+85 => 585, 2+500 => 2500)
    num_blocks = []
    while i < len(blocks):
        b = crop_block(blocks[i])

        if ocr_block_has_digit(b):
            num_blocks.append(blocks[i])
            i += 1
            continue

        # if we already started collecting digits, stop when we hit a non-digit block (unit)
        if num_blocks:
            break

        # if we haven't started digits yet, skip junk
        i += 1

    if not num_blocks:
        return op_crop, label_bgr

    x1 = min(b[0] for b in num_blocks)
    y1 = min(b[1] for b in num_blocks)
    x2 = max(b[2] for b in num_blocks)
    y2 = max(b[3] for b in num_blocks)

    # pad so we don’t clip a leading minus
    pad = int(0.06 * W)
    x1 = max(0, x1 - pad)
    x2 = min(W, x2 + pad)

    num_crop = label_bgr[y1:y2, x1:x2].copy()
    return op_crop, num_crop


# ============================================================
# 3) Numeric OCR preprocess + single-pass OCR
# ============================================================
def preprocess_for_numeric_ocr(bgr: np.ndarray, target_h_px: int = 180) -> np.ndarray:
    """
    Returns a clean binary image (black text on white) ready for OCR.
    """
    if bgr is None or bgr.size == 0:
        return None

    h = bgr.shape[0]
    upscale = int(np.clip(target_h_px / max(h, 1), 2, 14))
    bgr_u = cv2.resize(bgr, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_LANCZOS4)

    bgr_u = _mask_color_bar_keep_text(bgr_u)
    gray = cv2.cvtColor(bgr_u, cv2.COLOR_BGR2GRAY)
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

    bg = cv2.GaussianBlur(gray, (0, 0), 25)
    flat = cv2.divide(gray, bg, scale=255)
    flat = cv2.normalize(flat, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    flat = clahe.apply(flat)

    thr = cv2.threshold(flat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    # ensure black text on white
    if thr.mean() < 127:
        thr = 255 - thr

    thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)
    thr = cv2.copyMakeBorder(thr, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255)
    return thr

def ocr_number_from_crop(num_crop_bgr: np.ndarray, expected_sign: str = "any", range_hint=None):
    """
    One label => one number.
    Uses minimal tesseract calls (psm7, fallback psm8).
    """
    if num_crop_bgr is None or num_crop_bgr.size == 0:
        return None, {"txt": ""}

    thr = preprocess_for_numeric_ocr(num_crop_bgr, target_h_px=180)
    if thr is None:
        return None, {"txt": ""}

    cfg7 = "--oem 1 --psm 7 -c tessedit_char_whitelist=+-0123456789. -c classify_bln_numeric_mode=1 -c load_system_dawg=0 -c load_freq_dawg=0"
    cfg8 = "--oem 1 --psm 8 -c tessedit_char_whitelist=+-0123456789. -c classify_bln_numeric_mode=1 -c load_system_dawg=0 -c load_freq_dawg=0"

    txt = _tess_string(thr, cfg7).strip()
    vals = _parse_signed_numbers(txt)

    if not vals:
        txt2 = _tess_string(thr, cfg8).strip()
        vals = _parse_signed_numbers(txt2)
        txt = (txt + " | " + txt2).strip()

    if not vals:
        return None, {"txt": txt}

    # normalize sign expectations
    cands = []
    for v in vals:
        v = float(v)
        if expected_sign == "negative":
            v = -abs(v)
        elif expected_sign == "positive":
            v = abs(v)

        # plausibility guards
        if abs(v) > 100000:
            continue
        if expected_sign == "negative":
            if v >= 0:
                continue
            if abs(v) > 200:   # avoids ≥40 -> 240 and similar
                continue
        if expected_sign == "positive" and v < 0:
            continue

        if range_hint is not None:
            lo, hi = range_hint
            if v < lo - 1e-6 or v > hi + 1e-6:
                continue

        cands.append(v)

    if not cands:
        return None, {"txt": txt, "raw": vals}

    # pick the candidate with most digits (avoid "-0")
    cands.sort(key=lambda x: (_candidate_digit_len(x), abs(x)), reverse=True)
    best = float(cands[0])
    return best, {"txt": txt, "raw": vals, "cands": cands}


# ============================================================
# 4) Public function: OCR min/max from a scale image
# ============================================================
def ocr_minmax_from_scale_path(scale_path: str, metric_cfg: dict, debug_dir: str | None = None, tag: str = ""):
    """
    Returns (mn, mx) as floats.
    - Uses block detection to find left/right endpoint label areas.
    - Splits operator/number/unit blocks to OCR only the numeric block.
    - Uses caching per scale_path.
    """
    if scale_path in OCR_MINMAX_CACHE:
        return OCR_MINMAX_CACHE[scale_path]

    bgr = cv2.imread(scale_path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Could not read scale image: {scale_path}")

    expected_sign = metric_cfg.get("expected_sign", "any")
    range_hint = metric_cfg.get("range_hint", None)
    fixed_style = metric_cfg.get("fixed_style", None)

    L, R, det_dbg = detect_left_right_label_blocks(bgr, det_upscale=DET_UPSCALE)
    if L is None or R is None:
        raise ValueError(f"OCR label detection failed (left/right blocks not found): {scale_path}")

    # isolate numeric crops (avoid operator => '2' issues)
    _, Lnum = extract_op_and_number_crops(L)
    _, Rnum = extract_op_and_number_crops(R)

    def pick_best(a, a_meta, b, b_meta):
        """Pick value with more digits; if tie, bigger abs. Keeps meta of chosen."""
        if a is None and b is None:
            return None, {"txt": "", "picked": "none", "num": a_meta, "full": b_meta}
        if a is None:
            return b, {"picked": "full", "num": a_meta, "full": b_meta}
        if b is None:
            return a, {"picked": "num", "num": a_meta, "full": b_meta}

        da = _candidate_digit_len(a)
        db = _candidate_digit_len(b)

        if db > da:
            return b, {"picked": "full", "num": a_meta, "full": b_meta}
        if da > db:
            return a, {"picked": "num", "num": a_meta, "full": b_meta}

        # tie-breaker: prefer larger magnitude (helps 2500 vs 2000, 90 vs 80)
        if abs(b) > abs(a):
            return b, {"picked": "full", "num": a_meta, "full": b_meta}
        return a, {"picked": "num", "num": a_meta, "full": b_meta}


    # --- OCR both ways for each side ---
    mn_num, mn_num_meta = ocr_number_from_crop(Lnum, expected_sign=expected_sign, range_hint=range_hint)
    mx_num, mx_num_meta = ocr_number_from_crop(Rnum, expected_sign=expected_sign, range_hint=range_hint)

    mn_full, mn_full_meta = ocr_number_from_crop(L, expected_sign=expected_sign, range_hint=range_hint)
    mx_full, mx_full_meta = ocr_number_from_crop(R, expected_sign=expected_sign, range_hint=range_hint)

    # --- pick best ---
    mn, mn_meta = pick_best(mn_num, mn_num_meta, mn_full, mn_full_meta)
    mx, mx_meta = pick_best(mx_num, mx_num_meta, mx_full, mx_full_meta)

    # optional outlier fix for dBm-ish metrics
    if fixed_style == "dbm" or expected_sign == "negative":
        mn = _fix_db_outliers(mn, lo=-200, hi=200)
        mx = _fix_db_outliers(mx, lo=-200, hi=200)

    if mn is None or mx is None:
        raise ValueError(
            f"OCR failed for endpoints on scale: {scale_path}\n"
            f"left_meta={mn_meta}\n"
            f"right_meta={mx_meta}"
        )

    mn = float(mn)
    mx = float(mx)
    if mn > mx:
        mn, mx = mx, mn

    OCR_MINMAX_CACHE[scale_path] = (mn, mx)

    # optional debugging artifacts
    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)
        try:
            cv2.imwrite(os.path.join(debug_dir, f"{tag}ocr_label_left.png"), L)
            cv2.imwrite(os.path.join(debug_dir, f"{tag}ocr_label_right.png"), R)
            cv2.imwrite(os.path.join(debug_dir, f"{tag}ocr_num_left.png"), Lnum)
            cv2.imwrite(os.path.join(debug_dir, f"{tag}ocr_num_right.png"), Rnum)
            with open(os.path.join(debug_dir, f"{tag}ocr_minmax.json"), "w") as f:
                json.dump(
                    {
                        "scale": os.path.basename(scale_path),
                        "mn": mn,
                        "mx": mx,
                        "expected_sign": expected_sign,
                        "range_hint": range_hint,
                        "left": mn_meta,
                        "right": mx_meta,
                        "det_blocks": det_dbg.get("blocks", []),
                    },
                    f,
                    indent=2,
                )
        except Exception:
            pass

    return mn, mx
    # ---------- end NEW ----------

    # detect bar x0/x1
    try:
        _, x0, x1, _ = detect_bar_and_colors(
            scale_rgb,
            strip_y_frac=(0.0, 1.0) if tiny else (0.75, 0.98),
            debug_dir=None,
            tag=tag
        )
    except Exception:
        x0, x1 = int(w * 0.20), int(w * 0.80)

    fixed_style = metric_cfg.get("fixed_style", None)
    expected_sign = metric_cfg.get("expected_sign", "any")

    strip_fracs = [(0.0, 1.0)] if tiny else [(0.60, 0.95), (0.65, 0.98), (0.55, 0.90)]
    pad_fracs   = [0.22, 0.28, 0.35] if tiny else [0.14, 0.18, 0.24, 0.30]

    best = {
        "score": -1,
        "conf_sum": -1.0,
        "mn": None,
        "mx": None,
        "meta": {}
    }

    for (ya, yb) in strip_fracs:
        yA = int(h * ya)
        yB = int(h * yb)
        strip = bgr[yA:yB, :, :]

        for pf in pad_fracs:
            pad = max(22, int(pf * w), int(2.0 * h) if tiny else 0)

            # --- NEW: widen zones on tiny legend strips so labels never get clipped ---
            metric_key = metric_cfg.get("key", "")
            if tiny:
                # throughput/data_rate labels are long (e.g., "980 Mb/s"), give them more space
                min_zone_frac = 0.45 if metric_key in ("throughput", "data_rate") else 0.35
            else:
                min_zone_frac = 0.25

            min_zone_w = int(min_zone_frac * w)

            # initial bar-anchored windows
            lx1 = max(0, int(x0 - 1.8 * pad)); lx2 = min(w, int(x0 + 0.8 * pad))
            rx1 = max(0, int(x1 - 0.8 * pad)); rx2 = min(w, int(x1 + 1.8 * pad))

            # force edges on tiny strips
            if tiny:
                lx1 = 0
                rx2 = w

            # enforce minimum widths (this is the key fix)
            if (lx2 - lx1) < min_zone_w:
                lx1 = max(0, lx2 - min_zone_w)
            if (rx2 - rx1) < min_zone_w:
                rx1 = max(0, rx2 - min_zone_w)

            Lz = strip[:, lx1:lx2, :]
            Rz = strip[:, rx1:rx2, :]

            def tighten(zone_bgr: np.ndarray):
                if tiny:
                    return zone_bgr  # don't over-tighten tiny (30px) legends
                g = cv2.cvtColor(zone_bgr, cv2.COLOR_BGR2GRAY)
                bb = _tight_text_bbox_from_gray(g)
                if bb is None:
                    return zone_bgr
                x1b, y1b, x2b, y2b = bb
                return zone_bgr[y1b:y2b, x1b:x2b]

            L = tighten(Lz)
            R = tighten(Rz)

            def op_and_val(crop_bgr: np.ndarray, side: str):
                if crop_bgr.size == 0:
                    return None, None, {"op": {}, "val": {}}

                Wc = crop_bgr.shape[1]
                op_crop = crop_bgr[:, :max(1, int(0.30 * Wc))]
                op, op_dbg = _ocr_best_operator(op_crop)

                # DO NOT cut here; minus lives near the start for negative numbers
                val, val_dbg = _ocr_best_numeric(crop_bgr, side=side, metric_cfg=metric_cfg)
                return op, val, {"op": op_dbg, "val": val_dbg}

            lop, mn, dbgL = op_and_val(L, "left")
            rop, mx, dbgR = op_and_val(R, "right")

            if fixed_style == "dbm" or expected_sign == "negative":
                mn = _fix_db_outliers(mn, lo=-200, hi=200)
                mx = _fix_db_outliers(mx, lo=-200, hi=200)

            score = int(mn is not None) + int(mx is not None)
            conf_sum = float(dbgL["val"].get("conf", -1.0)) + float(dbgR["val"].get("conf", -1.0))

            if (score > best["score"]) or (score == best["score"] and conf_sum > best["conf_sum"]):
                best = {
                    "score": score,
                    "conf_sum": conf_sum,
                    "mn": mn,
                    "mx": mx,
                    "meta": {
                        "strip_y0": yA, "strip_y1": yB,
                        "pad": pad,
                        "bar_x0": int(x0), "bar_x1": int(x1),
                        "left_op": lop, "right_op": rop,
                        "dbgL": dbgL, "dbgR": dbgR,
                        "lx1": int(lx1), "lx2": int(lx2),
                        "rx1": int(rx1), "rx2": int(rx2),
                    }
                }

            if best["score"] == 2:
                break
        if best["score"] == 2:
            break

    mn = best["mn"]
    mx = best["mx"]
    if mn is None or mx is None:
        if DEBUG and debug_dir:
            ensure_dir(debug_dir)
            with open(os.path.join(debug_dir, f"{tag}ocr_failed_meta.json"), "w") as f:
                json.dump(best["meta"], f, indent=2)
        raise ValueError(f"OCR failed for min/max on scale: {scale_path} (mn={mn}, mx={mx})")

    mn = float(mn); mx = float(mx)
    if mn > mx:
        mn, mx = mx, mn

    if DEBUG and debug_dir:
        ensure_dir(debug_dir)
        yA = best["meta"]["strip_y0"]; yB = best["meta"]["strip_y1"]
        strip = bgr[yA:yB, :, :]
        cv2.imwrite(os.path.join(debug_dir, f"{tag}ocr_strip.png"), strip)

        lx1 = best["meta"]["lx1"]; lx2 = best["meta"]["lx2"]
        rx1 = best["meta"]["rx1"]; rx2 = best["meta"]["rx2"]
        cv2.imwrite(os.path.join(debug_dir, f"{tag}ocr_left_zone.png"),  strip[:, lx1:lx2, :])
        cv2.imwrite(os.path.join(debug_dir, f"{tag}ocr_right_zone.png"), strip[:, rx1:rx2, :])

        with open(os.path.join(debug_dir, f"{tag}ocr_minmax.json"), "w") as f:
            json.dump({**best["meta"], "mn": mn, "mx": mx}, f, indent=2)

    OCR_MINMAX_CACHE[scale_path] = (mn, mx)
    dlog(f"OCR mn/mx: mn={mn}, mx={mx} | scale={os.path.basename(scale_path)}")
    return mn, mx
# ---------------- end OCR ----------------

# ============================================================
# LEGEND BAR COLOR EXTRACTION (DISCRETE + CONTINUOUS)
# ============================================================
def extract_colorbar_bbox(scale_rgb: np.ndarray, strip_y_frac=(0.70, 0.98)):
    """
    Find x-range where the legend bar exists by scanning a horizontal strip near the bottom.
    Returns (x0, x1, y0, y1) where y0/y1 are strip bounds in the ORIGINAL image.
    """
    h, w, _ = scale_rgb.shape
    y0 = int(h * strip_y_frac[0])
    y1 = int(h * strip_y_frac[1])
    strip = scale_rgb[y0:y1, :, :]

    hsv = cv2.cvtColor(strip, cv2.COLOR_RGB2HSV)
    S = hsv[:, :, 1].astype(np.float32)
    V = hsv[:, :, 2].astype(np.float32)

    col_score = np.median(((V < 245) | (S > 8)).astype(np.uint8), axis=0)
    bar_mask = col_score > 0.05

    xs = np.where(bar_mask)[0]
    if len(xs) == 0:
        raise ValueError("Could not detect legend bar area.")
    x0, x1 = int(xs[0]), int(xs[-1])
    return x0, x1, y0, y1

def _bar_strip(scale_rgb: np.ndarray, strip_y_frac=(0.75, 0.98)):
    h, w, _ = scale_rgb.shape
    y0 = int(h * strip_y_frac[0])
    y1 = int(h * strip_y_frac[1])
    strip = scale_rgb[y0:y1, :, :]
    return strip, y0, y1

def detect_bar_and_colors(scale_rgb: np.ndarray, strip_y_frac=(0.75, 0.98),
                          major_min_frac=0.008, debug_dir: str | None = None, tag: str = ""):
    """
    Returns:
      - bar_rgb: 1D array (width x 3) median color across strip
      - x0,x1: bar region
      - major_colors_df: detected major colors with their x_med positions
    """
    strip, y0, y1 = _bar_strip(scale_rgb, strip_y_frac=strip_y_frac)
    bar_rgb = np.median(strip, axis=0).astype(np.uint8)  # shape: (W,3)

    hsv = cv2.cvtColor(strip, cv2.COLOR_RGB2HSV)
    medS = np.median(hsv[:, :, 1], axis=0)
    medV = np.median(hsv[:, :, 2], axis=0)
    bar_mask = (medV < 250) | (medS > 5)
    xs = np.where(bar_mask)[0]
    if len(xs) == 0:
        raise ValueError("Could not detect legend bar area.")
    x0, x1 = int(xs[0]), int(xs[-1])

    bar_rgb_cut = bar_rgb[x0:x1+1]
    bar_w = bar_rgb_cut.shape[0]

    uniq, counts = np.unique(bar_rgb_cut, axis=0, return_counts=True)
    min_count = max(5, int(bar_w * major_min_frac))
    major = counts >= min_count

    records = []
    for c_rgb, cnt in zip(uniq[major], counts[major]):
        idxs = np.where((bar_rgb_cut == c_rgb).all(axis=1))[0]
        records.append({
            "step_rgb": tuple(int(v) for v in c_rgb),
            "count": int(cnt),
            "x_med": float(np.median(idxs)),   # within bar_rgb_cut coordinates
        })
    if records:
        steps = pd.DataFrame(records).sort_values("x_med").reset_index(drop=True)
    else:
        steps = pd.DataFrame(columns=["step_rgb", "count", "x_med"])

    if DEBUG and debug_dir:
        mask_img = (bar_mask.astype(np.uint8) * 255)[None, :]
        mask_img = np.repeat(mask_img, 60, axis=0)
        save_gray(mask_img, os.path.join(debug_dir, f"{tag}legend_bar_mask.png"))

        viz = np.zeros((40, bar_w, 3), dtype=np.uint8)
        viz[:, :, :] = bar_rgb_cut[None, :, :]
        save_rgb(viz, os.path.join(debug_dir, f"{tag}legend_bar_rgb.png"))

    return bar_rgb_cut, x0, x1, steps

# ============================================================
# NUMERIC SCALE MODEL (DISCRETE or CONTINUOUS fallback)
# ============================================================
def rgb_to_lab(rgb_arr_u8: np.ndarray) -> np.ndarray:
    """rgb_arr_u8 shape (N,3) -> LAB float32 shape (N,3)"""
    lab = cv2.cvtColor(rgb_arr_u8[None, :, :], cv2.COLOR_RGB2LAB)[0].astype(np.float32)
    return lab

def deltaE_lab(lab1: np.ndarray, lab2: np.ndarray) -> np.ndarray:
    """Euclidean distance in LAB (approx deltaE)"""
    return np.linalg.norm(lab1 - lab2, axis=1)

@dataclass
class NumericScaleModel:
    kind: str  # "discrete" | "continuous"
    steps: pd.DataFrame
    tree: cKDTree
    sample_lab: np.ndarray
    sample_pos: np.ndarray
    mn: float
    mx: float
    reverse_lr: bool

def build_numeric_scale_model(
    scale_rgb: np.ndarray,
    metric_cfg: dict,
    reverse_lr: bool,
    mn: float,
    mx: float,
    debug_dir: str | None = None,
    tag: str = ""
) -> NumericScaleModel:
    """
    Same legend logic as before, but NO OCR:
    - Uses user-supplied mn/mx for numeric value scaling.
    """
    # detect bar + major colors
    bar_rgb_cut, x0, x1, major_steps = detect_bar_and_colors(
        scale_rgb,
        debug_dir=debug_dir,
        tag=tag
    )
    n_major = len(major_steps)

    # # ------------------------------------------------------------
    # # FORCE PRESET BINS (for fixed Ekahau scales like SNR)
    # # ------------------------------------------------------------
    # preset = metric_cfg.get("preset_edges", None)
    # if preset is not None and metric_cfg.get("force_preset_bins", False):
    #     edges = [float(x) for x in preset]
    #     n_bins = len(edges) - 1

    #     bar_w = bar_rgb_cut.shape[0]

    #     xs = ((np.arange(n_bins) + 0.5) / n_bins * (bar_w - 1)).astype(int)

    #     step_rgb = []
    #     for x in xs:
    #         a = max(0, x - 2)
    #         b = min(bar_w, x + 3)
    #         rgb = np.median(bar_rgb_cut[a:b], axis=0).astype(np.uint8)
    #         step_rgb.append(tuple(int(v) for v in rgb))

    #     steps = pd.DataFrame({
    #         "step_rgb": step_rgb,
    #         "value_min": edges[:-1],
    #         "value_max": edges[1:],
    #     })

    #     if reverse_lr:
    #         steps = steps.iloc[::-1].reset_index(drop=True)

    #     mids = []
    #     for a, b in zip(steps["value_min"], steps["value_max"]):
    #         if np.isneginf(a) and np.isfinite(b):
    #             mids.append(float(b))
    #         elif np.isposinf(b) and np.isfinite(a):
    #             mids.append(float(a))
    #         else:
    #             mids.append(float((a + b) / 2))
    #     steps["value_mid"] = mids

    #     step_rgb_arr = np.array(steps["step_rgb"].tolist(), dtype=np.uint8)
    #     lab = rgb_to_lab(step_rgb_arr)
    #     tree = cKDTree(lab)

    #     finite = [x for x in edges if np.isfinite(x)]
    #     mn_rep = float(min(finite)) if finite else 0.0
    #     mx_rep = float(max(finite)) if finite else 40.0

    #     return NumericScaleModel(
    #         kind="discrete",
    #         steps=steps,
    #         tree=tree,
    #         sample_lab=lab,
    #         sample_pos=np.linspace(0, 1, len(steps)).astype(np.float32),
    #         mn=mn_rep,
    #         mx=mx_rep,
    #         reverse_lr=reverse_lr
    #     )


    # --- DISCRETE path ---
    if n_major >= 2:
        n_bins = n_major

        preset = metric_cfg.get("preset_edges", None)
        if preset is not None and (len(preset) - 1) == n_bins:
            edges = [float(x) for x in preset]
            edge_mode = "preset"
        else:
            if metric_cfg.get("expected_sign") == "negative":
                finite = np.linspace(mn, mx, n_bins - 1).astype(float).tolist()
                edges = [-np.inf] + finite + [np.inf]
            else:
                edges = np.linspace(mn, mx, n_bins + 1).astype(float).tolist()
            edge_mode = "user"

        steps = major_steps.copy().reset_index(drop=True)
        if reverse_lr:
            steps = steps.iloc[::-1].reset_index(drop=True)

        expected = len(edges) - 1
        if len(steps) != expected:
            dlog(f"{tag} discrete mismatch: steps={len(steps)} edges bins={expected} -> switching to continuous")
            n_major = 0  # force continuous
        else:
            steps["value_min"] = edges[:-1]
            steps["value_max"] = edges[1:]
            mids = []
            for a, b in zip(steps["value_min"], steps["value_max"]):
                if np.isneginf(a) and np.isfinite(b):
                    mids.append(float(b))
                elif np.isposinf(b) and np.isfinite(a):
                    mids.append(float(a))
                else:
                    mids.append(float((a + b) / 2))
            steps["value_mid"] = mids

            step_rgb_arr = np.array(steps["step_rgb"].tolist(), dtype=np.uint8)
            lab = rgb_to_lab(step_rgb_arr)
            tree = cKDTree(lab)

            if DEBUG and debug_dir:
                with open(os.path.join(debug_dir, f"{tag}numeric_discrete_meta.json"), "w") as f:
                    json.dump(
                        dict(kind="discrete", mn=float(mn), mx=float(mx), n_bins=n_bins, edge_mode=edge_mode,
                             edges=[float(x) if np.isfinite(x) else (float("inf") if np.isposinf(x) else -float("inf")) for x in edges]),
                        f, indent=2
                    )

            sample_pos = np.linspace(0, 1, len(steps)).astype(np.float32)
            return NumericScaleModel(
                kind="discrete",
                steps=steps,
                tree=tree,
                sample_lab=lab,
                sample_pos=sample_pos,
                mn=float(mn),
                mx=float(mx),
                reverse_lr=reverse_lr
            )

    # --- CONTINUOUS fallback ---
    bar_w = bar_rgb_cut.shape[0]
    xs = np.linspace(0, bar_w - 1, N_SAMPLES_CONTINUOUS).astype(int)
    sample_rgb = bar_rgb_cut[xs].astype(np.uint8)

    uniq_rgb = []
    uniq_pos = []
    last = None
    for i, rgb in enumerate(sample_rgb):
        tup = tuple(int(v) for v in rgb)
        if last is None or tup != last:
            uniq_rgb.append(rgb)
            pos = float(xs[i] / max(1, (bar_w - 1)))
            uniq_pos.append(pos)
            last = tup

    sample_rgb_u8 = np.array(uniq_rgb, dtype=np.uint8)
    sample_pos = np.array(uniq_pos, dtype=np.float32)

    if reverse_lr:
        sample_pos = 1.0 - sample_pos

    sample_lab = rgb_to_lab(sample_rgb_u8)
    tree = cKDTree(sample_lab)

    edges = np.linspace(mn, mx, N_BINS_CONTINUOUS + 1).astype(float).tolist()
    mids = [float((a + b) / 2) for a, b in zip(edges[:-1], edges[1:])]

    mid_pos = np.linspace(0.5 / N_BINS_CONTINUOUS, 1 - 0.5 / N_BINS_CONTINUOUS, N_BINS_CONTINUOUS)
    if reverse_lr:
        mid_pos = 1.0 - mid_pos
    mid_x = (mid_pos * (bar_w - 1)).astype(int)
    step_rgb = [tuple(int(v) for v in bar_rgb_cut[x]) for x in mid_x]

    steps = pd.DataFrame({
        "step_rgb": step_rgb,
        "value_min": edges[:-1],
        "value_max": edges[1:],
        "value_mid": mids
    })

    if DEBUG and debug_dir:
        with open(os.path.join(debug_dir, f"{tag}numeric_continuous_meta.json"), "w") as f:
            json.dump(
                dict(kind="continuous", mn=float(mn), mx=float(mx),
                     n_samples=int(len(sample_pos)), n_bins=int(N_BINS_CONTINUOUS)),
                f, indent=2
            )
        viz = np.zeros((40, bar_w, 3), dtype=np.uint8)
        viz[:, :, :] = bar_rgb_cut[None, :, :]
        save_rgb(viz, os.path.join(debug_dir, f"{tag}legend_bar_rgb_full.png"))

    return NumericScaleModel(
        kind="continuous",
        steps=steps,
        tree=tree,
        sample_lab=sample_lab,
        sample_pos=sample_pos,
        mn=float(mn),
        mx=float(mx),
        reverse_lr=reverse_lr
    )

# ============================================================
# CATEGORICAL SCALE MODEL (NO OCR: labels are Cat1..CatN)
# ============================================================
def _clean_cat_label(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", " ", s)
    s = s.replace("=>", ">=").replace("≥", ">=")
    return s.strip(" |[](){}")

@dataclass
class CategoricalScaleModel:
    steps: pd.DataFrame
    tree: cKDTree
    step_lab: np.ndarray

def build_categorical_scale_model(scale_rgb: np.ndarray, reverse_lr: bool,
                                  debug_dir: str | None = None, tag: str = "") -> CategoricalScaleModel:
    bar_rgb_cut, x0, x1, major_steps = detect_bar_and_colors(scale_rgb, debug_dir=debug_dir, tag=tag)
    if len(major_steps) < 2:
        raise ValueError("Categorical legend detection failed (too few colors).")

    steps = major_steps.copy().reset_index(drop=True)
    if reverse_lr:
        steps = steps.iloc[::-1].reset_index(drop=True)

    # NO OCR: assign stable placeholder labels
    labels = [f"Cat{i+1}" for i in range(len(steps))]
    steps["label"] = labels

    step_rgb_arr = np.array(steps["step_rgb"].tolist(), dtype=np.uint8)
    step_lab = rgb_to_lab(step_rgb_arr)
    tree = cKDTree(step_lab)

    if DEBUG and debug_dir:
        with open(os.path.join(debug_dir, f"{tag}categorical_meta.json"), "w") as f:
            json.dump(dict(labels=labels), f, indent=2)

    return CategoricalScaleModel(steps=steps, tree=tree, step_lab=step_lab)

# ============================================================
# HEX EXTRACTION + MAPPING
# ============================================================
def extract_hexagons(heatmap_rgb: np.ndarray,
                     canny1=30, canny2=100,
                     edge_dilate=1,
                     roi_sat_thresh=15,
                     roi_val_thresh=250,
                     area_min_factor=0.75,
                     area_max_factor=1.35):
    bgr = cv2.cvtColor(heatmap_rgb, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (3, 3), 0)

    edges = cv2.Canny(gray_blur, canny1, canny2)
    kernel = np.ones((3, 3), np.uint8)
    if edge_dilate > 0:
        edges = cv2.dilate(edges, kernel, iterations=edge_dilate)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)

    hsv = cv2.cvtColor(heatmap_rgb, cv2.COLOR_RGB2HSV)
    S = hsv[:, :, 1]
    V = hsv[:, :, 2]

    bg = (V >= 245) & (S <= 20)
    dark = (V <= 25)
    roi = ((~bg) & (~dark)).astype(np.uint8) * 255

    interior = cv2.bitwise_and(roi, cv2.bitwise_not(edges))

    num, labels, stats, cents = cv2.connectedComponentsWithStats(interior, connectivity=8)
    areas = stats[1:, cv2.CC_STAT_AREA].astype(np.int32)

    big = areas[areas > 200]
    if len(big) == 0:
        raise ValueError("No components found. Try adjusting Canny/ROI thresholds.")

    med_area = float(np.median(big))
    a_min = int(med_area * area_min_factor)
    a_max = int(med_area * area_max_factor)
    keep_ids = np.where((areas >= a_min) & (areas <= a_max))[0] + 1

    records = []
    for lbl_id in keep_ids:
        x, y, w, h, area = stats[lbl_id].tolist()
        cx, cy = cents[lbl_id].tolist()

        sub = labels[y:y+h, x:x+w]

        m = (sub == lbl_id).astype(np.uint8)
        m_er = cv2.erode(m, np.ones((3, 3), np.uint8), iterations=2).astype(bool)

        patch = heatmap_rgb[y:y+h, x:x+w]

        pixels = patch[m_er]
        if pixels.size == 0:
            pixels = patch[m.astype(bool)]

        med_rgb = np.median(pixels, axis=0)

        records.append({
            "label_id": int(lbl_id),
            "cx": float(cx), "cy": float(cy),
            "r": float(med_rgb[0]), "g": float(med_rgb[1]), "b": float(med_rgb[2]),
            "area": int(area),
        })

    df = pd.DataFrame(records)
    debug = {"edges": edges, "roi": roi, "interior": interior}
    return df, debug

def assign_row_major_ids(df: pd.DataFrame, y_gap_min=2.0) -> pd.DataFrame:
    ys = np.sort(df["cy"].to_numpy())
    diffs = np.diff(ys)
    row_gaps = diffs[diffs > y_gap_min]
    if len(row_gaps) == 0:
        tmp = df.sort_values(["cy","cx"]).reset_index(drop=True)
        tmp["row"] = 0
        tmp["col"] = np.arange(len(tmp))
        tmp.insert(0, "hex_id", np.arange(1, len(tmp)+1))
        return tmp

    row_step = float(np.median(row_gaps))
    y0 = float(df["cy"].min())

    out = df.copy()
    out["row"] = np.round((out["cy"] - y0) / row_step).astype(int)
    out = out.sort_values(["row","cx"]).reset_index(drop=True)
    out["col"] = out.groupby("row").cumcount()
    out.insert(0, "hex_id", np.arange(1, len(out)+1))
    return out

def _hex_sat(rgb_arr: np.ndarray) -> np.ndarray:
    rgb_u8 = np.clip(rgb_arr, 0, 255).astype(np.uint8)
    hsv = cv2.cvtColor(rgb_u8[None, :, :], cv2.COLOR_RGB2HSV)[0]
    return hsv[:, 1].astype(np.int32)

def map_hex_to_numeric(df_hex: pd.DataFrame, model: NumericScaleModel, debug_dir: str | None = None, tag: str = "") -> pd.DataFrame:
    rgb = df_hex[["r","g","b"]].to_numpy().astype(np.uint8)
    lab = rgb_to_lab(rgb)

    dist, idx = model.tree.query(lab, k=1)
    dist = dist.astype(float)
    idx = idx.astype(int)

    out = df_hex.copy()
    out["deltaE"] = dist

    hsv = cv2.cvtColor(rgb[None, :, :], cv2.COLOR_RGB2HSV)[0]
    S = hsv[:, 1].astype(np.int32)
    V = hsv[:, 2].astype(np.int32)
    is_bg = (V > 245) & (S < 10)

    step_rgb = np.array(model.steps["step_rgb"].tolist(), dtype=np.uint8) if "step_rgb" in model.steps.columns else None
    legend_sat_min = int(np.min(_hex_sat(step_rgb))) if step_rgb is not None and len(step_rgb) else 255
    allow_low_sat = legend_sat_min < SAT_MIN_REJECT

    sat = _hex_sat(rgb)
    out["sat"] = sat

    sat_reject = (sat < SAT_MIN_REJECT) & (not allow_low_sat)
    out["rejected"] = is_bg | (out["deltaE"] > DELTAE_REJECT) | sat_reject

    if model.kind == "discrete":
        out["scale_idx"] = idx
        steps = model.steps.reset_index(drop=True)
        out = out.reset_index(drop=True).join(
            steps.loc[out["scale_idx"], ["step_rgb","value_min","value_max","value_mid"]].reset_index(drop=True)
        )
        out.loc[out["rejected"], ["value_min","value_max","value_mid"]] = np.nan

    else:
        pos = model.sample_pos[idx]
        val = model.mn + pos * (model.mx - model.mn)

        steps = model.steps.reset_index(drop=True)
        edges = np.array([float(x) for x in steps["value_min"].tolist()] + [float(steps["value_max"].iloc[-1])], dtype=float)
        bin_idx = np.digitize(val, edges, right=False) - 1
        bin_idx = np.clip(bin_idx, 0, len(steps)-1)

        out["scale_idx"] = bin_idx.astype(int)
        out["value_mid"] = val

        out = out.reset_index(drop=True).join(
            steps.loc[out["scale_idx"], ["step_rgb","value_min","value_max"]].reset_index(drop=True)
        )
        out.loc[out["rejected"], ["value_min","value_max","value_mid"]] = np.nan

    if DEBUG and debug_dir:
        rej_rate = float(out["rejected"].mean()) if len(out) else 0.0
        with open(os.path.join(debug_dir, f"{tag}mapping_numeric_stats.json"), "w") as f:
            json.dump(dict(
                n_hex=int(len(out)),
                rejected_rate=rej_rate,
                deltaE_median=float(np.nanmedian(out["deltaE"])) if len(out) else None,
                sat_median=float(np.nanmedian(out["sat"])) if len(out) else None,
                kind=model.kind
            ), f, indent=2)

    return out

def map_hex_to_categorical(df_hex: pd.DataFrame, model: CategoricalScaleModel, debug_dir: str | None = None, tag: str = "") -> pd.DataFrame:
    rgb = df_hex[["r","g","b"]].to_numpy().astype(np.uint8)
    lab = rgb_to_lab(rgb)

    dist, idx = model.tree.query(lab, k=1)
    out = df_hex.copy()
    out["deltaE"] = dist.astype(float)
    out["scale_idx"] = idx.astype(int)

    hsv = cv2.cvtColor(rgb[None, :, :], cv2.COLOR_RGB2BGR)[0]  # keep original behavior? (was HSV from RGB)
    hsv = cv2.cvtColor(rgb[None, :, :], cv2.COLOR_RGB2HSV)[0]
    S = hsv[:, 1].astype(np.int32)
    V = hsv[:, 2].astype(np.int32)
    is_bg = (V > 245) & (S < 10)

    step_rgb = np.array(model.steps["step_rgb"].tolist(), dtype=np.uint8)
    legend_sat_min = int(np.min(_hex_sat(step_rgb))) if len(step_rgb) else 255
    allow_low_sat = legend_sat_min < SAT_MIN_REJECT

    sat = _hex_sat(rgb)
    out["sat"] = sat
    sat_reject = (sat < SAT_MIN_REJECT) & (not allow_low_sat)

    out["rejected"] = is_bg | (out["deltaE"] > DELTAE_REJECT) | sat_reject

    steps = model.steps.reset_index(drop=True)
    out = out.reset_index(drop=True).join(
        steps.loc[out["scale_idx"], ["step_rgb","label"]].reset_index(drop=True)
    )
    out.loc[out["rejected"], ["label"]] = ""

    if DEBUG and debug_dir:
        with open(os.path.join(debug_dir, f"{tag}mapping_cat_stats.json"), "w") as f:
            json.dump(dict(
                n_hex=int(len(out)),
                rejected_rate=float(out["rejected"].mean()) if len(out) else 0.0,
                deltaE_median=float(np.nanmedian(out["deltaE"])) if len(out) else None,
            ), f, indent=2)

    return out

# ============================================================
# ANNOTATION
# ============================================================
def annotate_hexes_id_value_clear(img_rgb: np.ndarray,
                                  df: pd.DataFrame,
                                  out_file: str,
                                  value_col="value_mid",
                                  font_scale=0.32,
                                  thickness=1,
                                  pad_x=3, pad_y=2,
                                  box_alpha=0.65,
                                  border_px=1):
    bgr = cv2.cvtColor(img_rgb.copy(), cv2.COLOR_RGB2BGR)
    font = cv2.FONT_HERSHEY_SIMPLEX
    h, w = bgr.shape[:2]

    for _, r in df.iterrows():
        cx = int(round(r["cx"]))
        cy = int(round(r["cy"]))

        v = r.get(value_col, np.nan)
        if pd.isna(v):
            val_str = "NA"
        else:
            val_str = f"{int(round(float(v)))}" if abs(float(v) - round(float(v))) < 1e-6 else f"{float(v):.2f}"

        text = f"({int(r['hex_id'])}:{val_str})"

        lum = 0.2126*r["r"] + 0.7152*r["g"] + 0.0722*r["b"]
        if lum >= 128:
            text_color, outline_color = (0, 0, 0), (255, 255, 255)
            box_color, border_color = (255, 255, 255), (0, 0, 0)
        else:
            text_color, outline_color = (255, 255, 255), (0, 0, 0)
            box_color, border_color = (0, 0, 0), (255, 255, 255)

        (tw, th), base = cv2.getTextSize(text, font, font_scale, thickness)
        x = cx - tw // 2
        y = cy + th // 2

        x1 = max(0, x - pad_x)
        y1 = max(0, y - th - pad_y)
        x2 = min(w - 1, x + tw + pad_x)
        y2 = min(h - 1, y + base + pad_y)

        roi = bgr[y1:y2, x1:x2].astype(np.float32)
        box = np.full_like(roi, box_color, dtype=np.float32)
        bgr[y1:y2, x1:x2] = ((1 - box_alpha) * roi + box_alpha * box).astype(np.uint8)

        if border_px > 0:
            cv2.rectangle(bgr, (x1, y1), (x2, y2), border_color, border_px)

        cv2.putText(bgr, text, (x, y), font, font_scale, outline_color, thickness+2, cv2.LINE_AA)
        cv2.putText(bgr, text, (x, y), font, font_scale, text_color, thickness, cv2.LINE_AA)

    cv2.imwrite(out_file, bgr)
    return out_file

# ============================================================
# HISTOGRAMS
# ============================================================
def _fmt_num(x: float) -> str:
    if np.isneginf(x): return "-inf"
    if np.isposinf(x): return "inf"
    if abs(x - round(x)) < 1e-6:
        return str(int(round(x)))
    return f"{x:.2f}".rstrip("0").rstrip(".")

def save_histogram_colored(df_vals: pd.DataFrame, steps: pd.DataFrame, out_file: str, axis_label: str):
    df_use = df_vals.dropna(subset=["value_mid"]).copy()
    if df_use.empty:
        plt.figure(figsize=(12, 4))
        plt.title("Value Distribution by Range (no valid values)")
        plt.tight_layout()
        plt.savefig(out_file, dpi=220, bbox_inches="tight")
        plt.close()
        return out_file

    counts = df_use.groupby(["value_min","value_max"]).size().reset_index(name="count")
    dist = steps[["value_min","value_max","step_rgb"]].merge(counts, on=["value_min","value_max"], how="left")
    dist["count"] = dist["count"].fillna(0).astype(int)
    dist["pct"] = dist["count"] / max(1, len(df_use)) * 100.0

    labels = []
    for a, b in zip(dist["value_min"], dist["value_max"]):
        if np.isneginf(a) and np.isfinite(b):
            labels.append(f"≤ {_fmt_num(b)}")
        elif np.isposinf(b) and np.isfinite(a):
            labels.append(f"≥ {_fmt_num(a)}")
        else:
            labels.append(f"{_fmt_num(a)} to {_fmt_num(b)}")

    colors = [tuple(np.array(rgb)/255.0) for rgb in dist["step_rgb"]]

    plt.figure(figsize=(12, 4))
    x = np.arange(len(dist))
    plt.bar(x, dist["pct"], color=colors, edgecolor="white", linewidth=1.0)
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylabel("Percentage (%)")
    plt.xlabel(axis_label)
    plt.title("Value Distribution by Range")
    plt.tight_layout()
    plt.savefig(out_file, dpi=220, bbox_inches="tight")
    plt.close()
    return out_file

def save_histogram_categorical(df_vals: pd.DataFrame, steps: pd.DataFrame, out_file: str, axis_label: str):
    df_use = df_vals[df_vals["label"].astype(str).str.len() > 0].copy()
    if df_use.empty:
        plt.figure(figsize=(12, 4))
        plt.title("Value Distribution by Category (no valid values)")
        plt.tight_layout()
        plt.savefig(out_file, dpi=220, bbox_inches="tight")
        plt.close()
        return out_file

    counts = df_use.groupby(["scale_idx","label"]).size().reset_index(name="count")
    counts["pct"] = counts["count"] / max(1, len(df_use)) * 100.0
    counts = counts.sort_values("scale_idx").reset_index(drop=True)

    colors = []
    for i in counts["scale_idx"].tolist():
        rgb = steps.loc[int(i), "step_rgb"]
        colors.append(tuple(np.array(rgb)/255.0))

    plt.figure(figsize=(12, 4))
    x = np.arange(len(counts))
    plt.bar(x, counts["pct"], color=colors, edgecolor="white", linewidth=1.0)
    plt.xticks(x, counts["label"].tolist(), rotation=45, ha="right")
    plt.ylabel("Percentage (%)")
    plt.xlabel(axis_label)
    plt.title("Value Distribution by Category")
    plt.tight_layout()
    plt.savefig(out_file, dpi=220, bbox_inches="tight")
    plt.close()
    return out_file

# ============================================================
# CSV OUTPUT
# ============================================================
CSV_COLS = ["hex_id","label_id","cx","cy","r","g","b","area","deltaE","scale_idx","value","row","col"]

def write_clean_output_csv(df_vals: pd.DataFrame, out_path: str, is_categorical: bool):
    df = df_vals.copy()

    if is_categorical:
        df["value"] = df.get("label", "").astype(str)
    else:
        df["value"] = pd.to_numeric(df.get("value_mid", np.nan), errors="coerce")

    for c in CSV_COLS:
        if c not in df.columns:
            if c == "value" and is_categorical:
                df[c] = ""
            else:
                df[c] = np.nan

    df = df[CSV_COLS]
    df.to_csv(out_path, index=False)
    return out_path

# ============================================================
# COMPARISON OUTPUTS (NUMERIC ONLY)
# ============================================================
def comparison_outputs(df_band: pd.DataFrame, out_dir: str, metric_cfg: dict, band_name: str):
    ensure_dir(out_dir)
    outs = []
    axis_label = metric_cfg["axis_label"]
    metric_name = metric_cfg["name"]

    summary = df_band.groupby("device")["value"].agg(["count","mean","median","std","min","max"]).reset_index()
    p_summary = os.path.join(out_dir, f"summary_{slug(metric_cfg['key'])}_{slug(band_name)}.csv")
    summary.to_csv(p_summary, index=False)
    outs.append(p_summary)

    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df_band, x="device", y="value", hue="device", dodge=False, palette="Set2")
    plt.legend([], [], frameon=False)
    plt.title(f"{metric_name} Distribution by Device - {band_name}")
    plt.xlabel("Device")
    plt.ylabel(axis_label)
    plt.xticks(rotation=45, ha="right")
    plt.grid(True, axis="y")
    if metric_cfg.get("fixed_style") == "dbm":
        plt.ylim(-100, -20)
        plt.yticks(np.arange(-100, -19, 10))
    plt.tight_layout()
    p_box = os.path.join(out_dir, f"boxplot_{slug(metric_cfg['key'])}_{slug(band_name)}.png")
    plt.savefig(p_box, dpi=220, bbox_inches="tight")
    plt.close()
    outs.append(p_box)

    plt.figure(figsize=(10, 6))
    for dev in df_band["device"].unique():
        sns.kdeplot(data=df_band[df_band["device"] == dev], x="value", fill=True, alpha=0.3, label=dev)
    plt.title(f"Density of {metric_name} by Device - {band_name}")
    plt.xlabel(axis_label)
    plt.ylabel("Density")
    plt.legend(title="Device")
    plt.grid(True)
    if metric_cfg.get("fixed_style") == "dbm":
        plt.xlim(-100, -20)
        plt.xticks(np.arange(-100, -19, 10))
        plt.ylim(0.0, 0.09)
    plt.tight_layout()
    p_den = os.path.join(out_dir, f"density_{slug(metric_cfg['key'])}_{slug(band_name)}.png")
    plt.savefig(p_den, dpi=220, bbox_inches="tight")
    plt.close()
    outs.append(p_den)

    return outs

# ============================================================
# UI HELPERS (NO COUNTS / NO ALL)
# ============================================================
def pick(prompt, options):
    print(prompt)
    for i, opt in enumerate(options, 1):
        print(f"{i}. {opt}")
    while True:
        s = input("Enter choice number: ").strip()
        if s.isdigit() and 1 <= int(s) <= len(options):
            return int(s) - 1
        print("Invalid choice, try again.\n")

def scan_heatmaps(input_root: str) -> list[str]:
    all_imgs = []
    for ext in IMG_EXTS:
        all_imgs.extend(glob.glob(os.path.join(input_root, "**", f"*{ext}"), recursive=True))
    heatmaps = [p for p in all_imgs if not is_scale_file(p)]
    return sorted(heatmaps)

def build_inventory(heatmaps: list[str]) -> list[str]:
    found = set()
    for p in heatmaps:
        fn = os.path.basename(p)
        k = metric_key_from_filename(fn)
        if k and (k in METRIC_CFG_BY_KEY):
            found.add(k)
    ordered = [m["key"] for m in METRICS if m["key"] in found]
    return ordered

import shutil

# shutil.rmtree('/content/out')

# ============================================================
# INSTALLS (Colab)
# ============================================================
# !apt-get -qq update
# !apt-get -qq install -y tesseract-ocr
# !pip -q install pytesseract opencv-python-headless scipy pandas matplotlib seaborn


# ============================================================
# IMPORTS
# ============================================================
# from __future__ import annotations

import os, re, glob, zipfile, json, math
from pathlib import Path
from dataclasses import dataclass

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
#from google.colab import files
import seaborn as sns
import pytesseract
sns.set_theme()
from pytesseract import Output



# ============================================================
# PATHS
# ============================================================
INPUT_ROOT  = "out"
RESULT_ROOT = "ss_results"

# ============================================================
# GLOBAL DEBUG SETTINGS (EDIT HERE)
# ============================================================
DEBUG = False                 # saves extra debug images + JSON per heatmap
DEBUG_MAX_HEATMAPS = None    # set to an int to process only first N picked heatmaps
DELTAE_REJECT = 25.0         # reject hex if too far from legend color/sample
SAT_MIN_REJECT = 20          # reject low-saturation (background-ish) hexes
N_SAMPLES_CONTINUOUS = 256   # legend samples for continuous scales
N_BINS_CONTINUOUS = 12       # pseudo-bins for continuous legend (for histogram coloring)

IMG_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff")

# ============================================================
# Band parsing (STRICT: avoids Nokia5G21 false matches)
# Must contain: "on X GHz band"
# ============================================================
BAND_RX = re.compile(r"\bon\s*(2\.?\s*4|5)\s*ghz\s*band\b", re.IGNORECASE)

# ============================================================
# Metric Catalog (serial order = menu order)
# scale_kind: "numeric" | "categorical"
# has_band:   True for almost all Ekahau heatmaps
# range_hint: clamp user input if it goes wild (optional)
# ============================================================
DBM_EDGES_FALLBACK = [-np.inf, -90, -85, -80, -75, -70, -67, -65, -60, -55, -50, -45, -40, -35, -30, np.inf]

METRICS = [
    dict(key="signal_strength_main", name="Signal Strength", preset_edges=DBM_EDGES_FALLBACK,
         axis_label="dBm (Higher is Better)", expected_sign="negative", fixed_style="dbm",
         scale_kind="numeric", has_band=True),

    dict(key="snr", name="SNR",
        axis_label="SNR (Higher is Better)",
        expected_sign="positive",
        range_hint=(0.0, 60.0),
        fixed_style=None,
        scale_kind="numeric", has_band=True),

    dict(key="noise", name="Noise", preset_edges=None,
         axis_label="Noise", expected_sign="negative", fixed_style=None,
         scale_kind="numeric", has_band=True),

    dict(key="throughput", name="Throughput", preset_edges=None,
         axis_label="Throughput", expected_sign="positive", fixed_style=None,
         scale_kind="numeric", has_band=True),

    dict(key="data_rate", name="Data Rate", preset_edges=None,
         axis_label="Data Rate", expected_sign="positive", fixed_style=None,
         scale_kind="numeric", has_band=True),

    dict(key="channel_utilization", name="Channel Utilization", preset_edges=None,
         axis_label="Channel Utilization (%)", expected_sign="positive", fixed_style=None,
         range_hint=(0.0, 100.0), scale_kind="numeric", has_band=True),

    dict(key="channel_interference", name="Channel Interference", preset_edges=None,
         axis_label="Channel Interference", expected_sign="positive", fixed_style=None,
         scale_kind="numeric", has_band=True),

    dict(key="channel_width", name="Channel Width", preset_edges=None,
         axis_label="Channel Width (MHz)", expected_sign="positive", fixed_style=None,
         scale_kind="numeric", has_band=True),

    dict(key="network_health", name="Network Health", preset_edges=None,
         axis_label="Network Health", expected_sign="any", fixed_style=None,
         scale_kind="categorical", has_band=True),

    dict(key="network_issues", name="Network Issues", preset_edges=None,
         axis_label="Network Issues", expected_sign="any", fixed_style=None,
         scale_kind="categorical", has_band=True),

    dict(key="number_of_aps", name="Number of APs", preset_edges=None,
         axis_label="AP Count", expected_sign="positive", fixed_style=None,
         scale_kind="numeric", has_band=True),

    dict(
        key="spectrum_channel_power",
        name="Spectrum Channel Power",
        preset_edges=None,
        axis_label="Spectrum Channel Power (dBm)",
        expected_sign="negative",     # ✅ force negative
        fixed_style="dbm",            # ✅ use your dbm outlier handling
        range_hint=(-200.0, 0.0),     # ✅ keep OCR sane (optional but recommended)
        scale_kind="numeric",
        has_band=True
    ),
]

METRIC_CFG_BY_KEY = {m["key"]: m for m in METRICS}

# ============================================================
# EXCLUDE UNWANTED METRICS ENTIRELY (won't show in menu/inventory)
# ============================================================
EXCLUDE_METRIC_TEXT_RX = [
    r"\bassociated\s+access\s+point\b",
    r"\bbluetooth\s+coverage\b",
    r"\bsurvey\s+routes\s+and\s+access\s+points\b",
    r"\binterferers\b",
]

# ============================================================
# LOCATIONS + BANDS (NO ALL / NO UNKNOWN)
# ============================================================
LOCATIONS = [
    ("Upper Floor",  [r"upper\s*floor", r"upper\s*room", r"upstairs"]),
    ("Lower Floor",  [r"lower\s*floor", r"living\s*room", r"livingroom"]),
    ("Ground Floor", [r"ground\s*floor", r"groundfloor", r"kitchen", r"kithchen"]),
]

BANDS = [("2.4 GHz", []), ("5 GHz", [])]

# ============================================================
# SMALL UTILS
# ============================================================
def ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p

def unique_dir(base: str) -> str:
    if not os.path.exists(base):
        return base
    k = 2
    while True:
        p = f"{base}_{k}"
        if not os.path.exists(p):
            return p
        k += 1

def slug(s: str) -> str:
    s = (s or "").lower().strip()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_")

def norm_for_match(s: str) -> str:
    s = Path(s).name
    s = os.path.splitext(s)[0]
    s = (s or "").lower().replace("\u00A0", " ")
    s = re.sub(r"[_\-]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def log(msg: str):
    print(msg)

def dlog(msg: str):
    if DEBUG:
        print("[DEBUG]", msg)

def is_scale_file(path: str) -> bool:
    return "_scale" in Path(path).name.lower()

def device_from_filename(path: str) -> str:
    stem = Path(path).stem
    stem = re.sub(r"_scale(_\d+)?$", "", stem, flags=re.IGNORECASE)
    if "_" in stem:
        return stem.split("_", 1)[0]
    return stem.split(" ", 1)[0]

def parse_band_from_text(path_or_name: str) -> str | None:
    s = norm_for_match(path_or_name)
    m = BAND_RX.search(s)
    if not m:
        return None
    g = m.group(1).replace(" ", "")
    return "2.4 GHz" if g.startswith("2.4") or g.startswith("24") else "5 GHz"

def detect_band_from_name(filename: str) -> str:
    b = parse_band_from_text(filename)
    return b if b else "Unknown"

def parse_caption_part(filename: str) -> str:
    stem = Path(filename).stem
    stem = re.sub(r"_scale(_\d+)?$", "", stem, flags=re.IGNORECASE)
    if "_" in stem:
        return stem.split("_", 1)[1]
    return stem

def parse_metric_text(filename: str) -> str | None:
    cap = parse_caption_part(filename)
    m = re.search(r"^\s*(?P<metric>.+?)\s+for\s+", cap, flags=re.IGNORECASE)
    if not m:
        return None
    return m.group("metric").strip()

def metric_key_from_filename(filename: str) -> str | None:
    """
    - Collapses 'g4ar <metric>' => same metric
    - Drops excluded metrics
    - IMPORTANT: excludes Signal Strength Secondary/Tertiary when you pick Signal Strength
    """
    t = (parse_metric_text(filename) or "").strip().lower()
    if not t:
        return None

    t = re.sub(r"[_\-]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()

    # normalize prefix noise
    t = re.sub(r"^g4ar\s+", "", t)

    # drop unwanted metrics completely
    for rx in EXCLUDE_METRIC_TEXT_RX:
        if re.search(rx, t):
            return None

    # --- Signal Strength: main only (exclude secondary/tertiary)
    if re.search(r"\bsignal\s+strength\b", t):
        if re.search(r"\bsecondary\b|\btertiary\b", t):
            return None
        return "signal_strength_main"

    if re.search(r"\bsnr\b", t) or re.search(r"signal\s+to\s+noise", t):
        return "snr"
    if re.search(r"\bnoise\b", t):
        return "noise"
    if re.search(r"\bthroughput\b", t):
        return "throughput"
    if re.search(r"\bdata\s+rate\b", t):
        return "data_rate"
    if re.search(r"\bchannel\s+utilization\b", t):
        return "channel_utilization"
    if re.search(r"\bchannel\s+interference\b", t):
        return "channel_interference"
    if re.search(r"\bchannel\s+width\b", t):
        return "channel_width"
    if re.search(r"\bnetwork\s+health\b", t):
        return "network_health"
    if re.search(r"\bnetwork\s+issues\b", t):
        return "network_issues"
    if re.search(r"\bnumber\s+of\s+aps\b", t):
        return "number_of_aps"
    if re.search(r"\bspectrum\s+channel\s+power\b", t):
        return "spectrum_channel_power"

    return None

# ============================================================
# MATCHERS
# ============================================================
def matches_location(filename: str, loc_key: str) -> bool:
    s = norm_for_match(filename)
    for name, patterns in LOCATIONS:
        if name == loc_key:
            return any(re.search(pat, s) for pat in patterns)
    return False

def matches_band(filename: str, band_key: str) -> bool:
    band = parse_band_from_text(filename)
    if band is None:
        return False
    return band == band_key

# ============================================================
# SCALE IMAGE MATCHING (STRICT + BAND-SAFE)
# ============================================================
def find_matching_scale(main_path: str) -> str:
    """
    Best-effort strict pairing:
      1) exact stem + _scale*
      2) stem-start + _scale
    Band-safe:
      - If heatmap band exists, prefer same-band scale filename.
      - If only bandless scales exist, only allow if exactly ONE candidate.
      - Otherwise: error (forces correct naming instead of silently wrong pairing)
    """
    p = Path(main_path)
    folder = p.parent
    stem = p.stem
    main_band = parse_band_from_text(p.name)

    candidates = []
    for ext in IMG_EXTS:
        candidates.extend(folder.glob(f"{stem}_scale*{ext}"))

    if not candidates:
        pref = (stem + "_scale").lower()
        for q in folder.iterdir():
            if q.is_file() and q.suffix.lower() in IMG_EXTS and q.stem.lower().startswith(pref):
                candidates.append(q)

    if not candidates:
        raise FileNotFoundError(f"Scale image not found for heatmap:\n{main_path}")

    if main_band is not None:
        exact = []
        bandless = []
        for c in candidates:
            c_band = parse_band_from_text(c.name)
            if c_band == main_band:
                exact.append(c)
            elif c_band is None:
                bandless.append(c)

        if exact:
            candidates = exact
        else:
            if len(bandless) == 1:
                candidates = bandless
            else:
                raise FileNotFoundError(
                    "Scale band mismatch/ambiguous.\n"
                    f"Heatmap: {p.name}\n"
                    f"Heatmap band: {main_band}\n"
                    f"Candidates:\n" + "\n".join([c.name for c in candidates])
                )

    candidates = sorted(candidates, key=lambda x: len(x.name))
    return str(candidates[0])

# ============================================================
# IMAGE IO
# ============================================================
def read_image_rgb(path: str) -> np.ndarray:
    im = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if im is None:
        raise FileNotFoundError(f"Could not read: {path}")
    if im.ndim == 2:
        return cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
    if im.shape[2] == 4:
        return cv2.cvtColor(im, cv2.COLOR_BGRA2RGB)
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

def save_gray(mask_u8: np.ndarray, out_file: str):
    cv2.imwrite(out_file, mask_u8)
    return out_file

def save_rgb(rgb: np.ndarray, out_file: str):
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(out_file, bgr)
    return out_file


# ---------------- OCR (robust) ----------------
OCR_MINMAX_CACHE = {}  # scale_path -> (mn, mx)

TESS_TIMEOUT_SEC = 2.0   # prevents Tesseract hanging forever
OCR_UPSCALE = 8          # resolution boost for small text

def _auto_upscale(h: int, target_px: int = 110, lo: int = 2, hi: int = 14) -> int:
    """
    Choose an upscale factor so small legend text becomes readable.
    target_px ~ target crop height after scaling.
    """
    return int(np.clip(target_px / max(h, 1), lo, hi))


def _tess_data(img, config: str):
    """Safe pytesseract.image_to_data with timeout."""
    try:
        return pytesseract.image_to_data(
            img, config=config, output_type=Output.DICT, timeout=TESS_TIMEOUT_SEC
        )
    except Exception:
        return {"text": [], "conf": []}


def _normalize_signs(s: str) -> str:
    return (s.replace("−", "-").replace("–", "-").replace("—", "-")
              .replace("_", "-").replace("＋", "+"))


def _normalize_ops(s: str) -> str | None:
    s = (s or "").replace("≤", "<=").replace("≥", ">=")
    s = s.replace("=<", "<=").replace("=>", ">=")
    s = re.sub(r"[^<>=]", "", s.strip())
    if "<" in s and "=" in s: return "<="
    if ">" in s and "=" in s: return ">="
    if "<" in s: return "<"
    if ">" in s: return ">"
    return None


def _parse_signed_floats(txt: str) -> list[float]:
    txt = _normalize_signs(txt)
    matches = re.findall(r"([+\-]{0,4})\s*(\d+(?:\.\d+)?)", txt)
    vals = []
    for signs, digits in matches:
        v = float(digits)
        if "-" in signs:
            v = -v
        vals.append(v)
    if not vals:
        vals = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", txt)]
    return vals

def _parse_unsigned_numbers(txt: str) -> list[float]:
    """
    Digits-only extraction.
    Also adds a merged candidate for split blocks like '2 500' -> 2500.
    """
    parts = re.findall(r"\d+(?:\.\d+)?", txt or "")
    vals = [float(p) for p in parts]

    # merge split integer chunks: 2 + 500 => 2500, 5 + 85 => 585
    if len(parts) >= 2 and all("." not in p for p in parts):
        try:
            vals.append(float(int("".join(parts))))
        except Exception:
            pass
    return vals


def _fix_db_outliers(val: float | None, lo=-200, hi=200) -> float | None:
    if val is None:
        return None
    if lo <= val <= hi:
        return float(val)
    sign = -1 if val < 0 else 1
    a = abs(val)
    for div in (10, 100):
        cand = sign * (math.floor(a / div))
        if lo <= cand <= hi:
            return float(cand)
    return float(val)


def _mask_color_bar_keep_text(bgr: np.ndarray) -> np.ndarray:
    """
    Remove saturated colored pixels (legend bar) but DO NOT erase dark text/edges.
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    _, S, V = cv2.split(hsv)

    colored = (S > 35) & (V > 60)
    keep_dark = (V < 170)
    kill = colored & (~keep_dark)

    out = bgr.copy()
    out[kill] = (255, 255, 255)

    near_white = (V > 240) & (S < 25)
    out[near_white] = (255, 255, 255)
    return out


def _crop_digits_only(bgr_crop: np.ndarray) -> np.ndarray:
    """
    Best-effort crop that tries to keep only digit cluster (drops Mb/s).
    If it fails, returns original crop.
    """
    bgr = _mask_color_bar_keep_text(bgr_crop)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    up = cv2.resize(gray, None, fx=4, fy=4, interpolation=cv2.INTER_LANCZOS4)
    thr = cv2.threshold(up, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    thr = cv2.morphologyEx(thr, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)

    n, _, stats, _ = cv2.connectedComponentsWithStats(thr, connectivity=8)
    H, W = thr.shape[:2]

    boxes = []
    for i in range(1, n):
        x, y, w, h, a = stats[i].tolist()
        if a < 120 or h < 0.20 * H:
            continue
        boxes.append((x, y, w, h))

    if not boxes:
        return bgr_crop

    boxes = sorted(boxes, key=lambda t: t[0])
    gap = int(0.15 * W)

    groups = [[boxes[0]]]
    last_end = boxes[0][0] + boxes[0][2]
    for (x, y, w, h) in boxes[1:]:
        if x - last_end > gap:
            groups.append([(x, y, w, h)])
        else:
            groups[-1].append((x, y, w, h))
        last_end = max(last_end, x + w)

    def gscore(g):
        hs = [b[3] for b in g]
        areas = [b[2] * b[3] for b in g]
        return (float(np.median(hs)), float(np.sum(areas)))

    digit_group = max(groups, key=gscore)

    x1 = min(b[0] for b in digit_group)
    y1 = min(b[1] for b in digit_group)
    x2 = max(b[0] + b[2] for b in digit_group)
    y2 = max(b[1] + b[3] for b in digit_group)

    pad = 18
    x1 = max(0, x1 - pad); y1 = max(0, y1 - pad)
    x2 = min(W, x2 + pad); y2 = min(H, y2 + pad)

    crop_up = up[y1:y2, x1:x2]
    crop = cv2.resize(crop_up, (0, 0), fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)


def _preprocess_variants_for_ocr(bgr: np.ndarray, upscale: int = OCR_UPSCALE) -> list[np.ndarray]:
    """
    OCR variants (ensemble):
      - dark-text segmentation (thin + thick)
      - blackhat text isolation
      - CLAHE+sharpen Otsu/adaptive
      - NEW: background flattening (divide by blurred background) + Otsu
    """
    bgr = _mask_color_bar_keep_text(bgr)

    bgr_u = cv2.resize(bgr, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_LANCZOS4)
    gray = cv2.cvtColor(bgr_u, cv2.COLOR_BGR2GRAY)

    # add border (helps Tesseract not clip edge digits)
    bgr_u = cv2.copyMakeBorder(bgr_u, 18, 18, 18, 18, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    gray  = cv2.copyMakeBorder(gray,  18, 18, 18, 18, cv2.BORDER_CONSTANT, value=255)

    outs = []

    k = np.ones((3, 3), np.uint8)

    # ---------------- NEW) BACKGROUND FLATTENING ----------------
    # Great for dark digits sitting on a gradient legend background
    bg = cv2.GaussianBlur(gray, (0, 0), 25)
    flat = cv2.divide(gray, bg, scale=255)
    flat = cv2.normalize(flat, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    flat_otsu = cv2.threshold(flat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    if flat_otsu.mean() < 127:
        flat_otsu = 255 - flat_otsu
    flat_otsu = cv2.morphologyEx(flat_otsu, cv2.MORPH_CLOSE, k, iterations=1)

    outs.append(flat)        # grayscale flattened
    outs.append(flat_otsu)   # binarized flattened

    # ---------------- A) DARK TEXT SEGMENTATION (THIN + THICK) ----------------
    hsv_u = cv2.cvtColor(bgr_u, cv2.COLOR_BGR2HSV)
    V = hsv_u[:, :, 2]
    V = cv2.medianBlur(V, 3)

    dark_inv = cv2.threshold(V, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    thin = cv2.morphologyEx(dark_inv, cv2.MORPH_OPEN, k, iterations=1)
    thin = cv2.morphologyEx(thin, cv2.MORPH_CLOSE, k, iterations=1)
    thin_text = np.full_like(thin, 255); thin_text[thin > 0] = 0
    outs.append(thin_text)

    thick = cv2.dilate(thin, k, iterations=1)
    thick_text = np.full_like(thick, 255); thick_text[thick > 0] = 0
    outs.append(thick_text)

    # ---------------- B) BLACKHAT (TEXT ISOLATION) ----------------
    bh_k = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 9))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, bh_k)
    blackhat = cv2.normalize(blackhat, None, 0, 255, cv2.NORM_MINMAX)

    bh_bin = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    bh_bin = cv2.morphologyEx(bh_bin, cv2.MORPH_CLOSE, k, iterations=1)
    if bh_bin.mean() < 127:
        bh_bin = 255 - bh_bin
    outs.append(bh_bin)

    # ---------------- C) CLAHE + SHARPEN ----------------
    try:
        gray_dn = cv2.fastNlMeansDenoising(gray, None, h=15, templateWindowSize=7, searchWindowSize=21)
    except Exception:
        gray_dn = gray

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    g = clahe.apply(gray_dn)

    blur = cv2.GaussianBlur(g, (0, 0), 1.6)
    sharp = cv2.addWeighted(g, 1.8, blur, -0.8, 0)

    t1 = cv2.threshold(sharp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    t2 = cv2.threshold(sharp, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    outs.extend([t1, t2])

    a1 = cv2.adaptiveThreshold(sharp, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 41, 11)
    outs.append(a1)

    # reconnect small gaps + ensure black text on white
    fixed = []
    for img in outs:
        img2 = cv2.morphologyEx(img, cv2.MORPH_CLOSE, k, iterations=1)
        if img2.mean() < 127:
            img2 = 255 - img2
        fixed.append(img2)

    # include grayscale too (sometimes best)
    fixed.append(sharp)

    return fixed

def _tight_text_bbox_from_gray(gray: np.ndarray) -> tuple[int, int, int, int] | None:
    thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    thr = cv2.morphologyEx(thr, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8), iterations=1)
    thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)

    n, _, stats, _ = cv2.connectedComponentsWithStats(thr, connectivity=8)
    H, W = gray.shape[:2]

    boxes = []
    for i in range(1, n):
        x, y, ww, hh, a = stats[i].tolist()
        if a < 40:
            continue

        # NEW: drop skinny full-height vertical bars (legend border)
        if ww <= max(2, int(0.02 * W)) and hh >= int(0.80 * H):
            continue

        boxes.append((x, y, x + ww, y + hh))

    if not boxes:
        return None

    x1 = min(b[0] for b in boxes); y1 = min(b[1] for b in boxes)
    x2 = max(b[2] for b in boxes); y2 = max(b[3] for b in boxes)

    pad = 10
    x1 = max(0, x1 - pad); y1 = max(0, y1 - pad)
    x2 = min(W, x2 + pad); y2 = min(H, y2 + pad)
    return (x1, y1, x2, y2)

def _crop_numeric_focus_remove_operator(bgr_crop: np.ndarray, expected_sign: str = "any") -> np.ndarray:
    """
    Removes operator-like junk (≥, ≤) by cropping to the left edge of the 'main' text cluster.
    Keeps a little left padding to preserve a minus sign if expected_sign is negative.
    """
    if bgr_crop is None or bgr_crop.size == 0:
        return bgr_crop

    gray = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2GRAY)

    thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    thr = cv2.morphologyEx(thr, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8), iterations=1)
    thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)

    n, _, stats, _ = cv2.connectedComponentsWithStats(thr, connectivity=8)
    H, W = gray.shape[:2]
    if n <= 1:
        return bgr_crop

    # Only consider components that are reasonably "text-like"
    comps = []
    for i in range(1, n):
        x, y, w, h, a = stats[i].tolist()
        if a < 12:
            continue
        if h < int(0.20 * H):  # ignore tiny specks
            continue
        comps.append((i, x, y, w, h, a))

    if not comps:
        return bgr_crop

    max_area = max(c[-1] for c in comps)

    # Keep "main" components (digits/letters), operator is usually much smaller
    keep = []
    for (_, x, y, w, h, a) in comps:
        if a >= 0.25 * max_area:
            keep.append((x, x + w))

    if not keep:
        return bgr_crop

    x_min = min(k[0] for k in keep)

    # Preserve minus sign if needed (minus is small area, near digits)
    expand = int(0.10 * W) if expected_sign == "negative" else int(0.04 * W)
    x_start = max(0, x_min - expand)

    return bgr_crop[:, x_start:, :]

def detect_minus_sign(bgr_crop: np.ndarray) -> bool:
    """
    Detects a minus sign "-" by finding a thin horizontal stroke
    near the left side of the numeric text.
    Works on tiny legend labels.
    """
    if bgr_crop is None or bgr_crop.size == 0:
        return False

    gray = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2GRAY)

    # Binarize text (dark) vs background (bright)
    thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    H, W = thr.shape[:2]

    # Only search in the left half-ish (minus is before digits)
    thr = thr[:, :max(10, int(0.65 * W))]

    # Emphasize horizontal strokes
    k_w = max(8, W // 18)  # tune 16-24 if needed
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_w, 1))
    horiz = cv2.morphologyEx(thr, cv2.MORPH_OPEN, kernel, iterations=1)

    # Clean small noise
    horiz = cv2.morphologyEx(horiz, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8), iterations=1)

    n, _, stats, _ = cv2.connectedComponentsWithStats(horiz, connectivity=8)
    for i in range(1, n):
        x, y, w, h, a = stats[i].tolist()

        # minus-like: thin and wide
        if a < 10:
            continue
        if h <= max(2, H // 10) and w >= max(6, W // 25) and (w / max(1, h)) >= 3.0:
            # also avoid detecting random underline near bottom
            if y < 0.75 * H:
                return True

    return False

# ============================================================
# Segmentation-based signed OCR fallback (from ocr.py, simplified)
# Paste BELOW _crop_numeric_focus_remove_operator() and ABOVE _ocr_best_numeric()
# ============================================================

def _seg_preprocess_binary_for_segmentation(bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # normalize gradient background (good for legend bars)
    bg = cv2.GaussianBlur(gray, (0, 0), 25)
    norm = cv2.divide(gray, bg, scale=255)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    norm = clahe.apply(norm)

    thr = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    thr = cv2.dilate(thr, np.ones((2, 2), np.uint8), iterations=1)
    thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8), iterations=1)
    return thr

def _seg_contour_boxes(thr: np.ndarray) -> list[tuple[int,int,int,int]]:
    contours, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w*h < 20:
            continue
        boxes.append((x, y, w, h))
    return boxes

def _seg_group_boxes_into_blocks(boxes, W, H, gap_frac=0.12, pad=6):
    if not boxes:
        return []
    boxes = sorted(boxes, key=lambda b: b[0])
    gap_thresh = int(gap_frac * W)

    groups = []
    cur = [boxes[0]]
    last_end = boxes[0][0] + boxes[0][2]

    for (x,y,w,h) in boxes[1:]:
        if x - last_end > gap_thresh:
            groups.append(cur)
            cur = [(x,y,w,h)]
        else:
            cur.append((x,y,w,h))
        last_end = max(last_end, x+w)
    groups.append(cur)

    out = []
    for g in groups:
        xs = [b[0] for b in g]
        ys = [b[1] for b in g]
        x2 = [b[0]+b[2] for b in g]
        y2 = [b[1]+b[3] for b in g]
        x1,y1,x22,y22 = min(xs), min(ys), max(x2), max(y2)
        x1 = max(0, x1-pad); y1 = max(0, y1-pad)
        x22 = min(W, x22+pad); y22 = min(H, y22+pad)
        out.append((x1,y1,x22,y22))
    return out

def _seg_normalize_signs(s: str) -> str:
    return (s.replace("−","-").replace("–","-").replace("—","-")
              .replace("_","-").replace("＋","+"))

def _seg_normalize_ops(s: str) -> str | None:
    s = (s or "").replace("≤", "<=").replace("≥", ">=")
    s = s.replace("=<", "<=").replace("=>", ">=")
    s = re.sub(r"[^<>=]", "", s.strip())
    if "<" in s and "=" in s: return "<="
    if ">" in s and "=" in s: return ">="
    if "<" in s: return "<"
    if ">" in s: return ">"
    return None

def _seg_parse_signed_candidates(txt: str) -> list[float]:
    txt = _seg_normalize_signs(txt)
    # allow decimals too
    matches = re.findall(r"([+\-]{0,3})\s*(\d+(?:\.\d+)?)", txt)
    vals = []
    for signs, digits in matches:
        v = float(digits)
        if "-" in signs:
            v = -v
        vals.append(v)
    if not vals:
        vals = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", txt)]
    return vals

def _seg_ocr_operator(bgr_crop: np.ndarray) -> str | None:
    gray = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2GRAY)
    thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    if thr.mean() < 127:
        thr = 255 - thr

    cfgs = [
        "--oem 1 --psm 7 -c tessedit_char_whitelist=<>=",
        "--oem 1 --psm 8 -c tessedit_char_whitelist=<>=",
        "--oem 1 --psm 13 -c tessedit_char_whitelist=<>=",
    ]
    for cfg in cfgs:
        t = pytesseract.image_to_string(thr, config=cfg).strip()
        op = _seg_normalize_ops(t)
        if op:
            return op
    return None

def _seg_ocr_signed_number(bgr_crop: np.ndarray) -> float | None:
    gray = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

    thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    if thr.mean() < 127:
        thr = 255 - thr

    blur = cv2.GaussianBlur(gray, (0,0), 1.0)
    sharp = cv2.addWeighted(gray, 1.6, blur, -0.6, 0)

    variants = [thr, sharp]
    cfgs = [
        "--oem 1 --psm 7  -c tessedit_char_whitelist=+-0123456789.",
        "--oem 1 --psm 8  -c tessedit_char_whitelist=+-0123456789.",
        "--oem 1 --psm 13 -c tessedit_char_whitelist=+-0123456789.",
    ]

    best = None
    best_len = -1
    for img in variants:
        for cfg in cfgs:
            t = pytesseract.image_to_string(img, config=cfg).strip()
            vals = _seg_parse_signed_candidates(t)
            if not vals:
                continue
            # choose candidate with most digits
            cand = max(vals, key=lambda n: len(str(abs(n)).replace(".","")))
            L = len(str(abs(cand)).replace(".",""))
            if L > best_len:
                best = cand
                best_len = L
    return None if best is None else float(best)

def seg_extract_op_and_value_from_crop(bgr_crop: np.ndarray, side: str) -> tuple[str | None, float | None]:
    """
    Takes your already-cropped label zone (left or right) and:
      - finds the edge text block
      - splits operator block vs number block (if present)
      - OCR signed number from number block
    """
    thr = _seg_preprocess_binary_for_segmentation(bgr_crop)
    H, W = thr.shape[:2]

    boxes = _seg_contour_boxes(thr)
    blocks = _seg_group_boxes_into_blocks(boxes, W, H, gap_frac=0.10, pad=6)
    if not blocks:
        return None, None

    # edge block: leftmost or rightmost
    x1,y1,x2,y2 = blocks[0] if side == "left" else blocks[-1]
    block_crop = bgr_crop[y1:y2, x1:x2]

    thr2 = _seg_preprocess_binary_for_segmentation(block_crop)
    H2, W2 = thr2.shape[:2]
    sub_boxes = _seg_contour_boxes(thr2)
    subs = _seg_group_boxes_into_blocks(sub_boxes, W2, H2, gap_frac=0.12, pad=6)

    op = None
    number_crop = block_crop

    if len(subs) >= 2:
        ox1,oy1,ox2,oy2 = subs[0]
        nx1,ny1,nx2,ny2 = subs[-1]
        op_crop = block_crop[oy1:oy2, ox1:ox2]
        number_crop = block_crop[ny1:ny2, nx1:nx2]
        op = _seg_ocr_operator(op_crop)

    val = _seg_ocr_signed_number(number_crop)
    return op, val

from collections import defaultdict

def _ocr_best_numeric(bgr_crop: np.ndarray, side: str, metric_cfg: dict) -> tuple[float | None, dict]:
    cfgs = [
        "--oem 1 --psm 7 -c tessedit_char_whitelist=+-0123456789. "
        "-c classify_bln_numeric_mode=1 -c load_system_dawg=0 -c load_freq_dawg=0",
        "--oem 1 --psm 8 -c tessedit_char_whitelist=+-0123456789. "
        "-c classify_bln_numeric_mode=1 -c load_system_dawg=0 -c load_freq_dawg=0",
        "--oem 1 --psm 6 -c tessedit_char_whitelist=+-0123456789. "
        "-c classify_bln_numeric_mode=1 -c load_system_dawg=0 -c load_freq_dawg=0",
    ]

    expected_sign = metric_cfg.get("expected_sign", "any")
    range_hint = metric_cfg.get("range_hint", None)

    def plausible(v: float) -> bool:
        if abs(v) > 100000:
            return False
        if expected_sign == "positive" and v < 0:
            return False
        # STRICT: negative metrics must be < 0 (reject 0)
        if expected_sign == "negative" and v >= 0:
            return False
        if range_hint is not None:
            lo, hi = range_hint
            if v < lo - 1e-6 or v > hi + 1e-6:
                return False
        return True

    # operator removal focus crop first
    focus = _crop_numeric_focus_remove_operator(bgr_crop, expected_sign=expected_sign)
    minus_present = detect_minus_sign(focus)
    seg_op, seg_val = seg_extract_op_and_value_from_crop(focus, side=side)
    crops = [("focus", focus), ("orig", bgr_crop)]

    # digits-only crop based on focus
    try:
        c2 = _crop_digits_only(focus)
        if c2 is not None and c2.size > 0:
            crops.append(("digits", c2))
    except Exception:
        pass

    votes = defaultdict(lambda: {"count": 0, "w": 0.0, "best_conf": -1.0, "best_txt": "", "src": ""})

    for src, crop in crops:
        up = _auto_upscale(crop.shape[0])
        for img in _preprocess_variants_for_ocr(crop, upscale=up):
            for cfg in cfgs:
                d = _tess_data(img, cfg)

                tokens, confs = [], []
                for t, c in zip(d.get("text", []), d.get("conf", [])):
                    t = (t or "").strip()
                    if not t:
                        continue
                    try:
                        cf = float(c)
                    except Exception:
                        continue
                    if cf < 35:
                        continue
                    tokens.append(t)
                    confs.append(cf)

                txt = " ".join(tokens).strip()
                if not txt:
                    continue

                # ---- THIS MUST BE HERE (txt exists here) ----
                raw = _parse_signed_floats(txt)
                vals = []
                for v in raw:
                    if plausible(v):
                        vals.append(v)
                    # recovery if '-' got dropped
                    if expected_sign == "negative" and v > 0 and plausible(-v):
                        vals.append(-v)

                if not vals:
                    continue
                # --------------------------------------------

                cand = min(vals) if side == "left" else max(vals)

                # ✅ PLACE IT HERE (right after cand is chosen)
                cand = float(cand)
                if expected_sign == "negative":
                    # dBm/noise should always be negative
                    cand = -abs(cand)
                elif expected_sign == "positive":
                    # only flip for positive metrics if we really saw a minus sign
                    if minus_present:
                        cand = -abs(cand)
                else:
                    # expected_sign == "any"
                    if minus_present:
                        cand = -abs(cand)

                conf = float(np.mean(confs)) if confs else 0.0

                v = float(cand)
                votes[v]["count"] += 1
                votes[v]["w"] += max(0.0, conf)
                if conf > votes[v]["best_conf"]:
                    votes[v]["best_conf"] = conf
                    votes[v]["best_txt"] = txt
                    votes[v]["src"] = src

    if not votes:
        return None, {"conf": -1.0, "txt": "", "src": ""}
    # ✅ segmentation fallback vote (helps signed negatives like -60, -90)
    if seg_val is not None and plausible(float(seg_val)):
        sv = float(seg_val)
        votes[sv]["count"] += 3
        votes[sv]["w"] += 500.0
        if votes[sv]["best_conf"] < 99:
            votes[sv]["best_conf"] = 99
            votes[sv]["best_txt"] = f"SEG:{seg_op or ''}{seg_val}"
            votes[sv]["src"] = "segmented"

    # keep your positive left guard
    if side == "left" and expected_sign == "positive":
        if 0.0 in votes or 1.0 in votes:
            pick = 0.0 if 0.0 in votes else 1.0
            meta = votes[pick]
            return pick, {"conf": meta["best_conf"], "txt": meta["best_txt"], "src": meta["src"], "vote": meta}

    best_v, meta = max(votes.items(), key=lambda kv: (kv[1]["count"], kv[1]["w"], kv[1]["best_conf"]))
    return best_v, {"conf": meta["best_conf"], "txt": meta["best_txt"], "src": meta["src"], "vote": meta}


def _ocr_best_operator(bgr_crop: np.ndarray) -> tuple[str | None, dict]:
    cfgs = [
        "--oem 1 --psm 10 -c tessedit_char_whitelist=<>=",
        "--oem 1 --psm 7 -c tessedit_char_whitelist=<>=",
        "--oem 1 --psm 8 -c tessedit_char_whitelist=<>=",
    ]

    best_op = None
    best_conf = -1.0
    best_txt = ""

    up = _auto_upscale(bgr_crop.shape[0])
    for img in _preprocess_variants_for_ocr(bgr_crop, upscale=up):
        for cfg in cfgs:
            d = _tess_data(img, cfg)

            tokens, confs = [], []
            for t, c in zip(d.get("text", []), d.get("conf", [])):
                t = (t or "").strip()
                if not t:
                    continue
                try:
                    cf = float(c)
                except Exception:
                    continue
                if cf < 0:
                    continue
                tokens.append(t)
                confs.append(cf)

            txt = " ".join(tokens).strip()
            op = _normalize_ops(txt)
            conf = float(np.mean(confs)) if confs else -1.0

            if op and conf > best_conf:
                best_conf = conf
                best_op = op
                best_txt = txt

    return best_op, {"conf": best_conf, "txt": best_txt}


# ============================================================
# OCR (Scale min/max) — preprocess + data extraction (Ekahau legends)
# Drop this whole block into your notebook and call:
#   mn, mx = ocr_minmax_from_scale_path(scale_path, metric_cfg, debug_dir=..., tag=...)
# ============================================================

import os, re, json, math
import cv2
import numpy as np
import pytesseract

# ----------------------------
# SETTINGS
# ----------------------------
OCR_MINMAX_CACHE = {}     # scale_path -> (mn, mx)
TESS_TIMEOUT_SEC = 2.5    # one OCR call is ~1–2s (process startup dominates)
DET_UPSCALE = 8           # scale text is tiny (~30px). Upscale for block detection.
BLOCK_GAP_FRAC = 0.05     # split [op][number][unit] blocks inside a label crop


# ============================================================
# Small text helpers
# ============================================================
def _tess_string(img, config: str) -> str:
    """Safe pytesseract OCR with timeout."""
    try:
        return pytesseract.image_to_string(img, config=config, timeout=TESS_TIMEOUT_SEC) or ""
    except Exception:
        return ""

def _normalize_signs(s: str) -> str:
    return (s.replace("−", "-").replace("–", "-").replace("—", "-")
              .replace("_", "-").replace("＋", "+"))

def _parse_signed_numbers(txt: str) -> list[float]:
    """
    Extract signed numbers robustly.
    Handles cases like "2-60" -> [2, -60]
    """
    txt = _normalize_signs(txt)
    matches = re.findall(r"([+\-]{0,4})\s*(\d+(?:\.\d+)?)", txt)
    out = []
    for signs, digits in matches:
        v = float(digits)
        if "-" in signs:
            v = -v
        out.append(v)
    if not out:
        out = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", txt)]
    return out

def _candidate_digit_len(v: float) -> int:
    """Prefer -90 over -0, 2500 over 250, etc."""
    v = float(v)
    a = abs(v)
    if abs(a - round(a)) < 1e-6:
        return len(str(int(round(a))))
    s = f"{a:.6f}".rstrip("0").rstrip(".")
    return len(re.sub(r"\D", "", s))

def _fix_db_outliers(val: float | None, lo=-200, hi=200) -> float | None:
    """
    If OCR does something like -900 or -600, try to recover (-90, -60).
    Safe-ish for dB/dBm-like metrics.
    """
    if val is None:
        return None
    try:
        val = float(val)
    except Exception:
        return None
    if lo <= val <= hi:
        return float(val)

    sign = -1 if val < 0 else 1
    a = abs(val)
    for div in (10, 100):
        cand = sign * (math.floor(a / div))
        if lo <= cand <= hi:
            return float(cand)
    return float(val)


# ============================================================
# Legend masking (remove colored bar, keep dark text)
# ============================================================
def _mask_color_bar_keep_text(bgr: np.ndarray) -> np.ndarray:
    """
    Whiten saturated bar pixels while keeping dark text/edges.
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    S = hsv[:, :, 1]
    V = hsv[:, :, 2]

    colored = (S > 35) & (V > 60)
    keep_dark = (V < 170)
    kill = colored & (~keep_dark)

    out = bgr.copy()
    out[kill] = (255, 255, 255)

    near_white = (V > 240) & (S < 25)
    out[near_white] = (255, 255, 255)
    return out


# ============================================================
# Segmentation utilities (split operator / number / unit)
# ============================================================
def _seg_preprocess_binary(bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # flatten gradient background (legend bars often have gradients)
    bg = cv2.GaussianBlur(gray, (0, 0), 25)
    norm = cv2.divide(gray, bg, scale=255)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    norm = clahe.apply(norm)

    thr = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    thr = cv2.dilate(thr, np.ones((2, 2), np.uint8), iterations=1)
    thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8), iterations=1)
    return thr

def _seg_contour_boxes(thr: np.ndarray) -> list[tuple[int, int, int, int]]:
    contours, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w * h < 20:
            continue
        boxes.append((x, y, w, h))
    return boxes

def _seg_group_boxes_into_blocks(
    boxes: list[tuple[int, int, int, int]],
    W: int,
    gap_frac: float = BLOCK_GAP_FRAC,
    pad: int = 6
) -> list[tuple[int, int, int, int]]:
    """
    Group nearby component boxes into horizontal "blocks".
    Returns list of (x1,y1,x2,y2) in block coordinates.
    """
    if not boxes:
        return []
    boxes = sorted(boxes, key=lambda b: b[0])
    gap_thresh = int(gap_frac * W)

    groups = []
    cur = [boxes[0]]
    last_end = boxes[0][0] + boxes[0][2]

    for (x, y, w, h) in boxes[1:]:
        if x - last_end > gap_thresh:
            groups.append(cur)
            cur = [(x, y, w, h)]
        else:
            cur.append((x, y, w, h))
        last_end = max(last_end, x + w)
    groups.append(cur)

    out = []
    for g in groups:
        xs = [b[0] for b in g]
        ys = [b[1] for b in g]
        x2 = [b[0] + b[2] for b in g]
        y2 = [b[1] + b[3] for b in g]
        x1, y1, x22, y22 = min(xs), min(ys), max(x2), max(y2)
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x22 = min(W, x22 + pad)
        y22 = y22 + pad
        out.append((x1, y1, x22, y22))
    return out


# ============================================================
# 1) Detect the LEFT/RIGHT endpoint label blocks on the scale image
# ============================================================
def detect_left_right_label_blocks(scale_bgr: np.ndarray, det_upscale: int = DET_UPSCALE):
    """
    Returns:
      left_label_bgr, right_label_bgr, dbg
    where each label crop is already upscaled.
    """
    bgr = _mask_color_bar_keep_text(scale_bgr)
    bgr_u = cv2.resize(bgr, None, fx=det_upscale, fy=det_upscale, interpolation=cv2.INTER_LANCZOS4)
    gray = cv2.cvtColor(bgr_u, cv2.COLOR_BGR2GRAY)

    # flatten + enhance
    bg = cv2.GaussianBlur(gray, (0, 0), 25)
    flat = cv2.divide(gray, bg, scale=255)
    flat = cv2.normalize(flat, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    flat = clahe.apply(flat)

    thr = cv2.threshold(flat, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    thr = cv2.morphologyEx(thr, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=2)

    H, W = thr.shape[:2]
    n, _, stats, _ = cv2.connectedComponentsWithStats(thr, connectivity=8)

    boxes = []
    for i in range(1, n):
        x, y, w, h, a = stats[i].tolist()
        if a < 50:
            continue
        if h < int(0.30 * H):
            continue
        # drop skinny vertical legend borders
        if w <= max(2, int(0.02 * W)) and h >= int(0.80 * H):
            continue
        boxes.append((x, y, w, h))

    if not boxes:
        return None, None, {"thr": thr, "blocks": []}

    blocks = _seg_group_boxes_into_blocks(boxes, W, gap_frac=0.10, pad=int(0.02 * W))
    blocks = sorted(blocks, key=lambda b: b[0])

    left = blocks[0]
    right = blocks[-1]

    # extend a bit so we don't clip operator signs
    pad_extra = int(0.05 * W)

    lx1, ly1, lx2, ly2 = left
    rx1, ry1, rx2, ry2 = right

    lx1 = 0
    lx2 = min(W, lx2 + pad_extra)

    rx1 = max(0, rx1 - pad_extra)
    rx2 = W

    left_crop = bgr_u[ly1:ly2, lx1:lx2].copy()
    right_crop = bgr_u[ry1:ry2, rx1:rx2].copy()

    return left_crop, right_crop, {"thr": thr, "blocks": blocks, "bgr_u": bgr_u}


# ============================================================
# 2) Split a label into operator + number (+ unit)
# ============================================================
def extract_op_and_number_crops(label_bgr: np.ndarray):
    """
    Robust split:
      - Detect blocks
      - Identify operator blocks by OCRing ONLY <>=
      - Identify numeric blocks by OCRing ONLY digits/sign
      - Merge consecutive numeric-ish blocks (fixes 585->85, 2500->2000)
      - Ignore trailing unit blocks (Mb/s, dBm, %, dB)
    Returns: (op_crop_or_None, number_crop_bgr)
    """
    if label_bgr is None or label_bgr.size == 0:
        return None, label_bgr

    thr = _seg_preprocess_binary(label_bgr)
    H, W = thr.shape[:2]

    boxes = _seg_contour_boxes(thr)
    blocks = _seg_group_boxes_into_blocks(boxes, W, gap_frac=BLOCK_GAP_FRAC, pad=10)
    blocks = sorted(blocks, key=lambda b: b[0])

    if len(blocks) <= 1:
        return None, label_bgr

    def crop_block(b):
        x1,y1,x2,y2 = b
        return label_bgr[y1:y2, x1:x2].copy()

    def ocr_block_op(bgr):
        g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        t = pytesseract.image_to_string(
            g, config="--oem 1 --psm 10 -c tessedit_char_whitelist=<>=", timeout=TESS_TIMEOUT_SEC
        ).strip()
        return _normalize_ops(t)

    def ocr_block_has_digit(bgr):
        g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        t = pytesseract.image_to_string(
            g, config="--oem 1 --psm 10 -c tessedit_char_whitelist=+-0123456789.", timeout=TESS_TIMEOUT_SEC
        )
        return bool(re.search(r"\d", t or ""))

    op_crop = None
    i = 0

    # 1) Optional leading operator block (≤/≥ sometimes becomes <=/>=)
    first = crop_block(blocks[0])
    op = ocr_block_op(first)
    if op is not None:
        op_crop = first
        i = 1

    # 2) Merge consecutive numeric blocks (this fixes 5+85 => 585, 2+500 => 2500)
    num_blocks = []
    while i < len(blocks):
        b = crop_block(blocks[i])

        if ocr_block_has_digit(b):
            num_blocks.append(blocks[i])
            i += 1
            continue

        # if we already started collecting digits, stop when we hit a non-digit block (unit)
        if num_blocks:
            break

        # if we haven't started digits yet, skip junk
        i += 1

    if not num_blocks:
        return op_crop, label_bgr

    x1 = min(b[0] for b in num_blocks)
    y1 = min(b[1] for b in num_blocks)
    x2 = max(b[2] for b in num_blocks)
    y2 = max(b[3] for b in num_blocks)

    # pad so we don’t clip a leading minus
    pad = int(0.06 * W)
    x1 = max(0, x1 - pad)
    x2 = min(W, x2 + pad)

    num_crop = label_bgr[y1:y2, x1:x2].copy()
    return op_crop, num_crop


# ============================================================
# 3) Numeric OCR preprocess + single-pass OCR
# ============================================================
def preprocess_for_numeric_ocr(bgr: np.ndarray, target_h_px: int = 180) -> np.ndarray:
    """
    Returns a clean binary image (black text on white) ready for OCR.
    """
    if bgr is None or bgr.size == 0:
        return None

    h = bgr.shape[0]
    upscale = int(np.clip(target_h_px / max(h, 1), 2, 14))
    bgr_u = cv2.resize(bgr, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_LANCZOS4)

    bgr_u = _mask_color_bar_keep_text(bgr_u)
    gray = cv2.cvtColor(bgr_u, cv2.COLOR_BGR2GRAY)
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

    bg = cv2.GaussianBlur(gray, (0, 0), 25)
    flat = cv2.divide(gray, bg, scale=255)
    flat = cv2.normalize(flat, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    flat = clahe.apply(flat)

    thr = cv2.threshold(flat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    # ensure black text on white
    if thr.mean() < 127:
        thr = 255 - thr

    thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)
    thr = cv2.copyMakeBorder(thr, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255)
    return thr

def ocr_number_from_crop(num_crop_bgr: np.ndarray, expected_sign: str = "any", range_hint=None):
    """
    One label => one number.
    Uses minimal tesseract calls (psm7, fallback psm8).
    """
    if num_crop_bgr is None or num_crop_bgr.size == 0:
        return None, {"txt": ""}

    thr = preprocess_for_numeric_ocr(num_crop_bgr, target_h_px=180)
    if thr is None:
        return None, {"txt": ""}

    cfg7 = "--oem 1 --psm 7 -c tessedit_char_whitelist=+-0123456789. -c classify_bln_numeric_mode=1 -c load_system_dawg=0 -c load_freq_dawg=0"
    cfg8 = "--oem 1 --psm 8 -c tessedit_char_whitelist=+-0123456789. -c classify_bln_numeric_mode=1 -c load_system_dawg=0 -c load_freq_dawg=0"

    txt = _tess_string(thr, cfg7).strip()
    vals = _parse_signed_numbers(txt)

    if not vals:
        txt2 = _tess_string(thr, cfg8).strip()
        vals = _parse_signed_numbers(txt2)
        txt = (txt + " | " + txt2).strip()

    if not vals:
        return None, {"txt": txt}

    # normalize sign expectations
    cands = []
    for v in vals:
        v = float(v)
        if expected_sign == "negative":
            v = -abs(v)
        elif expected_sign == "positive":
            v = abs(v)

        # plausibility guards
        if abs(v) > 100000:
            continue
        if expected_sign == "negative":
            if v >= 0:
                continue
            if abs(v) > 200:   # avoids ≥40 -> 240 and similar
                continue
        if expected_sign == "positive" and v < 0:
            continue

        if range_hint is not None:
            lo, hi = range_hint
            if v < lo - 1e-6 or v > hi + 1e-6:
                continue

        cands.append(v)

    if not cands:
        return None, {"txt": txt, "raw": vals}

    # pick the candidate with most digits (avoid "-0")
    cands.sort(key=lambda x: (_candidate_digit_len(x), abs(x)), reverse=True)
    best = float(cands[0])
    return best, {"txt": txt, "raw": vals, "cands": cands}


# ============================================================
# 4) Public function: OCR min/max from a scale image
# ============================================================
def ocr_minmax_from_scale_path(scale_path: str, metric_cfg: dict, debug_dir: str | None = None, tag: str = ""):
    """
    Returns (mn, mx) as floats.
    - Uses block detection to find left/right endpoint label areas.
    - Splits operator/number/unit blocks to OCR only the numeric block.
    - Uses caching per scale_path.
    """
    if scale_path in OCR_MINMAX_CACHE:
        return OCR_MINMAX_CACHE[scale_path]

    bgr = cv2.imread(scale_path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Could not read scale image: {scale_path}")

    expected_sign = metric_cfg.get("expected_sign", "any")
    range_hint = metric_cfg.get("range_hint", None)
    fixed_style = metric_cfg.get("fixed_style", None)

    L, R, det_dbg = detect_left_right_label_blocks(bgr, det_upscale=DET_UPSCALE)
    if L is None or R is None:
        raise ValueError(f"OCR label detection failed (left/right blocks not found): {scale_path}")

    # isolate numeric crops (avoid operator => '2' issues)
    _, Lnum = extract_op_and_number_crops(L)
    _, Rnum = extract_op_and_number_crops(R)

    def pick_best(a, a_meta, b, b_meta):
        """Pick value with more digits; if tie, bigger abs. Keeps meta of chosen."""
        if a is None and b is None:
            return None, {"txt": "", "picked": "none", "num": a_meta, "full": b_meta}
        if a is None:
            return b, {"picked": "full", "num": a_meta, "full": b_meta}
        if b is None:
            return a, {"picked": "num", "num": a_meta, "full": b_meta}

        da = _candidate_digit_len(a)
        db = _candidate_digit_len(b)

        if db > da:
            return b, {"picked": "full", "num": a_meta, "full": b_meta}
        if da > db:
            return a, {"picked": "num", "num": a_meta, "full": b_meta}

        # tie-breaker: prefer larger magnitude (helps 2500 vs 2000, 90 vs 80)
        if abs(b) > abs(a):
            return b, {"picked": "full", "num": a_meta, "full": b_meta}
        return a, {"picked": "num", "num": a_meta, "full": b_meta}


    # --- OCR both ways for each side ---
    mn_num, mn_num_meta = ocr_number_from_crop(Lnum, expected_sign=expected_sign, range_hint=range_hint)
    mx_num, mx_num_meta = ocr_number_from_crop(Rnum, expected_sign=expected_sign, range_hint=range_hint)

    mn_full, mn_full_meta = ocr_number_from_crop(L, expected_sign=expected_sign, range_hint=range_hint)
    mx_full, mx_full_meta = ocr_number_from_crop(R, expected_sign=expected_sign, range_hint=range_hint)

    # --- pick best ---
    mn, mn_meta = pick_best(mn_num, mn_num_meta, mn_full, mn_full_meta)
    mx, mx_meta = pick_best(mx_num, mx_num_meta, mx_full, mx_full_meta)

    # optional outlier fix for dBm-ish metrics
    if fixed_style == "dbm" or expected_sign == "negative":
        mn = _fix_db_outliers(mn, lo=-200, hi=200)
        mx = _fix_db_outliers(mx, lo=-200, hi=200)

    if mn is None or mx is None:
        raise ValueError(
            f"OCR failed for endpoints on scale: {scale_path}\n"
            f"left_meta={mn_meta}\n"
            f"right_meta={mx_meta}"
        )

    mn = float(mn)
    mx = float(mx)
    if mn > mx:
        mn, mx = mx, mn

    OCR_MINMAX_CACHE[scale_path] = (mn, mx)

    # optional debugging artifacts
    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)
        try:
            cv2.imwrite(os.path.join(debug_dir, f"{tag}ocr_label_left.png"), L)
            cv2.imwrite(os.path.join(debug_dir, f"{tag}ocr_label_right.png"), R)
            cv2.imwrite(os.path.join(debug_dir, f"{tag}ocr_num_left.png"), Lnum)
            cv2.imwrite(os.path.join(debug_dir, f"{tag}ocr_num_right.png"), Rnum)
            with open(os.path.join(debug_dir, f"{tag}ocr_minmax.json"), "w") as f:
                json.dump(
                    {
                        "scale": os.path.basename(scale_path),
                        "mn": mn,
                        "mx": mx,
                        "expected_sign": expected_sign,
                        "range_hint": range_hint,
                        "left": mn_meta,
                        "right": mx_meta,
                        "det_blocks": det_dbg.get("blocks", []),
                    },
                    f,
                    indent=2,
                )
        except Exception:
            pass

    return mn, mx
    # ---------- end NEW ----------

    # detect bar x0/x1
    try:
        _, x0, x1, _ = detect_bar_and_colors(
            scale_rgb,
            strip_y_frac=(0.0, 1.0) if tiny else (0.75, 0.98),
            debug_dir=None,
            tag=tag
        )
    except Exception:
        x0, x1 = int(w * 0.20), int(w * 0.80)

    fixed_style = metric_cfg.get("fixed_style", None)
    expected_sign = metric_cfg.get("expected_sign", "any")

    strip_fracs = [(0.0, 1.0)] if tiny else [(0.60, 0.95), (0.65, 0.98), (0.55, 0.90)]
    pad_fracs   = [0.22, 0.28, 0.35] if tiny else [0.14, 0.18, 0.24, 0.30]

    best = {
        "score": -1,
        "conf_sum": -1.0,
        "mn": None,
        "mx": None,
        "meta": {}
    }

    for (ya, yb) in strip_fracs:
        yA = int(h * ya)
        yB = int(h * yb)
        strip = bgr[yA:yB, :, :]

        for pf in pad_fracs:
            pad = max(22, int(pf * w), int(2.0 * h) if tiny else 0)

            # --- NEW: widen zones on tiny legend strips so labels never get clipped ---
            metric_key = metric_cfg.get("key", "")
            if tiny:
                # throughput/data_rate labels are long (e.g., "980 Mb/s"), give them more space
                min_zone_frac = 0.45 if metric_key in ("throughput", "data_rate") else 0.35
            else:
                min_zone_frac = 0.25

            min_zone_w = int(min_zone_frac * w)

            # initial bar-anchored windows
            lx1 = max(0, int(x0 - 1.8 * pad)); lx2 = min(w, int(x0 + 0.8 * pad))
            rx1 = max(0, int(x1 - 0.8 * pad)); rx2 = min(w, int(x1 + 1.8 * pad))

            # force edges on tiny strips
            if tiny:
                lx1 = 0
                rx2 = w

            # enforce minimum widths (this is the key fix)
            if (lx2 - lx1) < min_zone_w:
                lx1 = max(0, lx2 - min_zone_w)
            if (rx2 - rx1) < min_zone_w:
                rx1 = max(0, rx2 - min_zone_w)

            Lz = strip[:, lx1:lx2, :]
            Rz = strip[:, rx1:rx2, :]

            def tighten(zone_bgr: np.ndarray):
                if tiny:
                    return zone_bgr  # don't over-tighten tiny (30px) legends
                g = cv2.cvtColor(zone_bgr, cv2.COLOR_BGR2GRAY)
                bb = _tight_text_bbox_from_gray(g)
                if bb is None:
                    return zone_bgr
                x1b, y1b, x2b, y2b = bb
                return zone_bgr[y1b:y2b, x1b:x2b]

            L = tighten(Lz)
            R = tighten(Rz)

            def op_and_val(crop_bgr: np.ndarray, side: str):
                if crop_bgr.size == 0:
                    return None, None, {"op": {}, "val": {}}

                Wc = crop_bgr.shape[1]
                op_crop = crop_bgr[:, :max(1, int(0.30 * Wc))]
                op, op_dbg = _ocr_best_operator(op_crop)

                # DO NOT cut here; minus lives near the start for negative numbers
                val, val_dbg = _ocr_best_numeric(crop_bgr, side=side, metric_cfg=metric_cfg)
                return op, val, {"op": op_dbg, "val": val_dbg}

            lop, mn, dbgL = op_and_val(L, "left")
            rop, mx, dbgR = op_and_val(R, "right")

            if fixed_style == "dbm" or expected_sign == "negative":
                mn = _fix_db_outliers(mn, lo=-200, hi=200)
                mx = _fix_db_outliers(mx, lo=-200, hi=200)

            score = int(mn is not None) + int(mx is not None)
            conf_sum = float(dbgL["val"].get("conf", -1.0)) + float(dbgR["val"].get("conf", -1.0))

            if (score > best["score"]) or (score == best["score"] and conf_sum > best["conf_sum"]):
                best = {
                    "score": score,
                    "conf_sum": conf_sum,
                    "mn": mn,
                    "mx": mx,
                    "meta": {
                        "strip_y0": yA, "strip_y1": yB,
                        "pad": pad,
                        "bar_x0": int(x0), "bar_x1": int(x1),
                        "left_op": lop, "right_op": rop,
                        "dbgL": dbgL, "dbgR": dbgR,
                        "lx1": int(lx1), "lx2": int(lx2),
                        "rx1": int(rx1), "rx2": int(rx2),
                    }
                }

            if best["score"] == 2:
                break
        if best["score"] == 2:
            break

    mn = best["mn"]
    mx = best["mx"]
    if mn is None or mx is None:
        if DEBUG and debug_dir:
            ensure_dir(debug_dir)
            with open(os.path.join(debug_dir, f"{tag}ocr_failed_meta.json"), "w") as f:
                json.dump(best["meta"], f, indent=2)
        raise ValueError(f"OCR failed for min/max on scale: {scale_path} (mn={mn}, mx={mx})")

    mn = float(mn); mx = float(mx)
    if mn > mx:
        mn, mx = mx, mn

    if DEBUG and debug_dir:
        ensure_dir(debug_dir)
        yA = best["meta"]["strip_y0"]; yB = best["meta"]["strip_y1"]
        strip = bgr[yA:yB, :, :]
        cv2.imwrite(os.path.join(debug_dir, f"{tag}ocr_strip.png"), strip)

        lx1 = best["meta"]["lx1"]; lx2 = best["meta"]["lx2"]
        rx1 = best["meta"]["rx1"]; rx2 = best["meta"]["rx2"]
        cv2.imwrite(os.path.join(debug_dir, f"{tag}ocr_left_zone.png"),  strip[:, lx1:lx2, :])
        cv2.imwrite(os.path.join(debug_dir, f"{tag}ocr_right_zone.png"), strip[:, rx1:rx2, :])

        with open(os.path.join(debug_dir, f"{tag}ocr_minmax.json"), "w") as f:
            json.dump({**best["meta"], "mn": mn, "mx": mx}, f, indent=2)

    OCR_MINMAX_CACHE[scale_path] = (mn, mx)
    dlog(f"OCR mn/mx: mn={mn}, mx={mx} | scale={os.path.basename(scale_path)}")
    return mn, mx
# ---------------- end OCR ----------------

# ============================================================
# LEGEND BAR COLOR EXTRACTION (DISCRETE + CONTINUOUS)
# ============================================================
def extract_colorbar_bbox(scale_rgb: np.ndarray, strip_y_frac=(0.70, 0.98)):
    """
    Find x-range where the legend bar exists by scanning a horizontal strip near the bottom.
    Returns (x0, x1, y0, y1) where y0/y1 are strip bounds in the ORIGINAL image.
    """
    h, w, _ = scale_rgb.shape
    y0 = int(h * strip_y_frac[0])
    y1 = int(h * strip_y_frac[1])
    strip = scale_rgb[y0:y1, :, :]

    hsv = cv2.cvtColor(strip, cv2.COLOR_RGB2HSV)
    S = hsv[:, :, 1].astype(np.float32)
    V = hsv[:, :, 2].astype(np.float32)

    col_score = np.median(((V < 245) | (S > 8)).astype(np.uint8), axis=0)
    bar_mask = col_score > 0.05

    xs = np.where(bar_mask)[0]
    if len(xs) == 0:
        raise ValueError("Could not detect legend bar area.")
    x0, x1 = int(xs[0]), int(xs[-1])
    return x0, x1, y0, y1

def _bar_strip(scale_rgb: np.ndarray, strip_y_frac=(0.75, 0.98)):
    h, w, _ = scale_rgb.shape
    y0 = int(h * strip_y_frac[0])
    y1 = int(h * strip_y_frac[1])
    strip = scale_rgb[y0:y1, :, :]
    return strip, y0, y1

def detect_bar_and_colors(scale_rgb: np.ndarray, strip_y_frac=(0.75, 0.98),
                          major_min_frac=0.008, debug_dir: str | None = None, tag: str = ""):
    """
    Returns:
      - bar_rgb: 1D array (width x 3) median color across strip
      - x0,x1: bar region
      - major_colors_df: detected major colors with their x_med positions
    """
    strip, y0, y1 = _bar_strip(scale_rgb, strip_y_frac=strip_y_frac)
    bar_rgb = np.median(strip, axis=0).astype(np.uint8)  # shape: (W,3)

    hsv = cv2.cvtColor(strip, cv2.COLOR_RGB2HSV)
    medS = np.median(hsv[:, :, 1], axis=0)
    medV = np.median(hsv[:, :, 2], axis=0)
    bar_mask = (medV < 250) | (medS > 5)
    xs = np.where(bar_mask)[0]
    if len(xs) == 0:
        raise ValueError("Could not detect legend bar area.")
    x0, x1 = int(xs[0]), int(xs[-1])

    bar_rgb_cut = bar_rgb[x0:x1+1]
    bar_w = bar_rgb_cut.shape[0]

    uniq, counts = np.unique(bar_rgb_cut, axis=0, return_counts=True)
    min_count = max(5, int(bar_w * major_min_frac))
    major = counts >= min_count

    records = []
    for c_rgb, cnt in zip(uniq[major], counts[major]):
        idxs = np.where((bar_rgb_cut == c_rgb).all(axis=1))[0]
        records.append({
            "step_rgb": tuple(int(v) for v in c_rgb),
            "count": int(cnt),
            "x_med": float(np.median(idxs)),   # within bar_rgb_cut coordinates
        })
    if records:
        steps = pd.DataFrame(records).sort_values("x_med").reset_index(drop=True)
    else:
        steps = pd.DataFrame(columns=["step_rgb", "count", "x_med"])

    if DEBUG and debug_dir:
        mask_img = (bar_mask.astype(np.uint8) * 255)[None, :]
        mask_img = np.repeat(mask_img, 60, axis=0)
        save_gray(mask_img, os.path.join(debug_dir, f"{tag}legend_bar_mask.png"))

        viz = np.zeros((40, bar_w, 3), dtype=np.uint8)
        viz[:, :, :] = bar_rgb_cut[None, :, :]
        save_rgb(viz, os.path.join(debug_dir, f"{tag}legend_bar_rgb.png"))

    return bar_rgb_cut, x0, x1, steps

# ============================================================
# NUMERIC SCALE MODEL (DISCRETE or CONTINUOUS fallback)
# ============================================================
def rgb_to_lab(rgb_arr_u8: np.ndarray) -> np.ndarray:
    """rgb_arr_u8 shape (N,3) -> LAB float32 shape (N,3)"""
    lab = cv2.cvtColor(rgb_arr_u8[None, :, :], cv2.COLOR_RGB2LAB)[0].astype(np.float32)
    return lab

def deltaE_lab(lab1: np.ndarray, lab2: np.ndarray) -> np.ndarray:
    """Euclidean distance in LAB (approx deltaE)"""
    return np.linalg.norm(lab1 - lab2, axis=1)

@dataclass
class NumericScaleModel:
    kind: str  # "discrete" | "continuous"
    steps: pd.DataFrame
    tree: cKDTree
    sample_lab: np.ndarray
    sample_pos: np.ndarray
    mn: float
    mx: float
    reverse_lr: bool

def build_numeric_scale_model(
    scale_rgb: np.ndarray,
    metric_cfg: dict,
    reverse_lr: bool,
    mn: float,
    mx: float,
    debug_dir: str | None = None,
    tag: str = ""
) -> NumericScaleModel:
    """
    Same legend logic as before, but NO OCR:
    - Uses user-supplied mn/mx for numeric value scaling.
    """
    # detect bar + major colors
    bar_rgb_cut, x0, x1, major_steps = detect_bar_and_colors(
        scale_rgb,
        debug_dir=debug_dir,
        tag=tag
    )
    n_major = len(major_steps)

    # # ------------------------------------------------------------
    # # FORCE PRESET BINS (for fixed Ekahau scales like SNR)
    # # ------------------------------------------------------------
    # preset = metric_cfg.get("preset_edges", None)
    # if preset is not None and metric_cfg.get("force_preset_bins", False):
    #     edges = [float(x) for x in preset]
    #     n_bins = len(edges) - 1

    #     bar_w = bar_rgb_cut.shape[0]

    #     xs = ((np.arange(n_bins) + 0.5) / n_bins * (bar_w - 1)).astype(int)

    #     step_rgb = []
    #     for x in xs:
    #         a = max(0, x - 2)
    #         b = min(bar_w, x + 3)
    #         rgb = np.median(bar_rgb_cut[a:b], axis=0).astype(np.uint8)
    #         step_rgb.append(tuple(int(v) for v in rgb))

    #     steps = pd.DataFrame({
    #         "step_rgb": step_rgb,
    #         "value_min": edges[:-1],
    #         "value_max": edges[1:],
    #     })

    #     if reverse_lr:
    #         steps = steps.iloc[::-1].reset_index(drop=True)

    #     mids = []
    #     for a, b in zip(steps["value_min"], steps["value_max"]):
    #         if np.isneginf(a) and np.isfinite(b):
    #             mids.append(float(b))
    #         elif np.isposinf(b) and np.isfinite(a):
    #             mids.append(float(a))
    #         else:
    #             mids.append(float((a + b) / 2))
    #     steps["value_mid"] = mids

    #     step_rgb_arr = np.array(steps["step_rgb"].tolist(), dtype=np.uint8)
    #     lab = rgb_to_lab(step_rgb_arr)
    #     tree = cKDTree(lab)

    #     finite = [x for x in edges if np.isfinite(x)]
    #     mn_rep = float(min(finite)) if finite else 0.0
    #     mx_rep = float(max(finite)) if finite else 40.0

    #     return NumericScaleModel(
    #         kind="discrete",
    #         steps=steps,
    #         tree=tree,
    #         sample_lab=lab,
    #         sample_pos=np.linspace(0, 1, len(steps)).astype(np.float32),
    #         mn=mn_rep,
    #         mx=mx_rep,
    #         reverse_lr=reverse_lr
    #     )


    # --- DISCRETE path ---
    if n_major >= 2:
        n_bins = n_major

        preset = metric_cfg.get("preset_edges", None)
        if preset is not None and (len(preset) - 1) == n_bins:
            edges = [float(x) for x in preset]
            edge_mode = "preset"
        else:
            if metric_cfg.get("expected_sign") == "negative":
                finite = np.linspace(mn, mx, n_bins - 1).astype(float).tolist()
                edges = [-np.inf] + finite + [np.inf]
            else:
                edges = np.linspace(mn, mx, n_bins + 1).astype(float).tolist()
            edge_mode = "user"

        steps = major_steps.copy().reset_index(drop=True)
        if reverse_lr:
            steps = steps.iloc[::-1].reset_index(drop=True)

        expected = len(edges) - 1
        if len(steps) != expected:
            dlog(f"{tag} discrete mismatch: steps={len(steps)} edges bins={expected} -> switching to continuous")
            n_major = 0  # force continuous
        else:
            steps["value_min"] = edges[:-1]
            steps["value_max"] = edges[1:]
            mids = []
            for a, b in zip(steps["value_min"], steps["value_max"]):
                if np.isneginf(a) and np.isfinite(b):
                    mids.append(float(b))
                elif np.isposinf(b) and np.isfinite(a):
                    mids.append(float(a))
                else:
                    mids.append(float((a + b) / 2))
            steps["value_mid"] = mids

            step_rgb_arr = np.array(steps["step_rgb"].tolist(), dtype=np.uint8)
            lab = rgb_to_lab(step_rgb_arr)
            tree = cKDTree(lab)

            if DEBUG and debug_dir:
                with open(os.path.join(debug_dir, f"{tag}numeric_discrete_meta.json"), "w") as f:
                    json.dump(
                        dict(kind="discrete", mn=float(mn), mx=float(mx), n_bins=n_bins, edge_mode=edge_mode,
                             edges=[float(x) if np.isfinite(x) else (float("inf") if np.isposinf(x) else -float("inf")) for x in edges]),
                        f, indent=2
                    )

            sample_pos = np.linspace(0, 1, len(steps)).astype(np.float32)
            return NumericScaleModel(
                kind="discrete",
                steps=steps,
                tree=tree,
                sample_lab=lab,
                sample_pos=sample_pos,
                mn=float(mn),
                mx=float(mx),
                reverse_lr=reverse_lr
            )

    # --- CONTINUOUS fallback ---
    bar_w = bar_rgb_cut.shape[0]
    xs = np.linspace(0, bar_w - 1, N_SAMPLES_CONTINUOUS).astype(int)
    sample_rgb = bar_rgb_cut[xs].astype(np.uint8)

    uniq_rgb = []
    uniq_pos = []
    last = None
    for i, rgb in enumerate(sample_rgb):
        tup = tuple(int(v) for v in rgb)
        if last is None or tup != last:
            uniq_rgb.append(rgb)
            pos = float(xs[i] / max(1, (bar_w - 1)))
            uniq_pos.append(pos)
            last = tup

    sample_rgb_u8 = np.array(uniq_rgb, dtype=np.uint8)
    sample_pos = np.array(uniq_pos, dtype=np.float32)

    if reverse_lr:
        sample_pos = 1.0 - sample_pos

    sample_lab = rgb_to_lab(sample_rgb_u8)
    tree = cKDTree(sample_lab)

    edges = np.linspace(mn, mx, N_BINS_CONTINUOUS + 1).astype(float).tolist()
    mids = [float((a + b) / 2) for a, b in zip(edges[:-1], edges[1:])]

    mid_pos = np.linspace(0.5 / N_BINS_CONTINUOUS, 1 - 0.5 / N_BINS_CONTINUOUS, N_BINS_CONTINUOUS)
    if reverse_lr:
        mid_pos = 1.0 - mid_pos
    mid_x = (mid_pos * (bar_w - 1)).astype(int)
    step_rgb = [tuple(int(v) for v in bar_rgb_cut[x]) for x in mid_x]

    steps = pd.DataFrame({
        "step_rgb": step_rgb,
        "value_min": edges[:-1],
        "value_max": edges[1:],
        "value_mid": mids
    })

    if DEBUG and debug_dir:
        with open(os.path.join(debug_dir, f"{tag}numeric_continuous_meta.json"), "w") as f:
            json.dump(
                dict(kind="continuous", mn=float(mn), mx=float(mx),
                     n_samples=int(len(sample_pos)), n_bins=int(N_BINS_CONTINUOUS)),
                f, indent=2
            )
        viz = np.zeros((40, bar_w, 3), dtype=np.uint8)
        viz[:, :, :] = bar_rgb_cut[None, :, :]
        save_rgb(viz, os.path.join(debug_dir, f"{tag}legend_bar_rgb_full.png"))

    return NumericScaleModel(
        kind="continuous",
        steps=steps,
        tree=tree,
        sample_lab=sample_lab,
        sample_pos=sample_pos,
        mn=float(mn),
        mx=float(mx),
        reverse_lr=reverse_lr
    )

# ============================================================
# CATEGORICAL SCALE MODEL (NO OCR: labels are Cat1..CatN)
# ============================================================
def _clean_cat_label(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", " ", s)
    s = s.replace("=>", ">=").replace("≥", ">=")
    return s.strip(" |[](){}")

@dataclass
class CategoricalScaleModel:
    steps: pd.DataFrame
    tree: cKDTree
    step_lab: np.ndarray

def build_categorical_scale_model(scale_rgb: np.ndarray, reverse_lr: bool,
                                  debug_dir: str | None = None, tag: str = "") -> CategoricalScaleModel:
    bar_rgb_cut, x0, x1, major_steps = detect_bar_and_colors(scale_rgb, debug_dir=debug_dir, tag=tag)
    if len(major_steps) < 2:
        raise ValueError("Categorical legend detection failed (too few colors).")

    steps = major_steps.copy().reset_index(drop=True)
    if reverse_lr:
        steps = steps.iloc[::-1].reset_index(drop=True)

    # NO OCR: assign stable placeholder labels
    labels = [f"Cat{i+1}" for i in range(len(steps))]
    steps["label"] = labels

    step_rgb_arr = np.array(steps["step_rgb"].tolist(), dtype=np.uint8)
    step_lab = rgb_to_lab(step_rgb_arr)
    tree = cKDTree(step_lab)

    if DEBUG and debug_dir:
        with open(os.path.join(debug_dir, f"{tag}categorical_meta.json"), "w") as f:
            json.dump(dict(labels=labels), f, indent=2)

    return CategoricalScaleModel(steps=steps, tree=tree, step_lab=step_lab)

# ============================================================
# HEX EXTRACTION + MAPPING
# ============================================================
def extract_hexagons(heatmap_rgb: np.ndarray,
                     canny1=30, canny2=100,
                     edge_dilate=1,
                     roi_sat_thresh=15,
                     roi_val_thresh=250,
                     area_min_factor=0.75,
                     area_max_factor=1.35):
    bgr = cv2.cvtColor(heatmap_rgb, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (3, 3), 0)

    edges = cv2.Canny(gray_blur, canny1, canny2)
    kernel = np.ones((3, 3), np.uint8)
    if edge_dilate > 0:
        edges = cv2.dilate(edges, kernel, iterations=edge_dilate)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)

    hsv = cv2.cvtColor(heatmap_rgb, cv2.COLOR_RGB2HSV)
    S = hsv[:, :, 1]
    V = hsv[:, :, 2]

    bg = (V >= 245) & (S <= 20)
    dark = (V <= 25)
    roi = ((~bg) & (~dark)).astype(np.uint8) * 255

    interior = cv2.bitwise_and(roi, cv2.bitwise_not(edges))

    num, labels, stats, cents = cv2.connectedComponentsWithStats(interior, connectivity=8)
    areas = stats[1:, cv2.CC_STAT_AREA].astype(np.int32)

    big = areas[areas > 200]
    if len(big) == 0:
        raise ValueError("No components found. Try adjusting Canny/ROI thresholds.")

    med_area = float(np.median(big))
    a_min = int(med_area * area_min_factor)
    a_max = int(med_area * area_max_factor)
    keep_ids = np.where((areas >= a_min) & (areas <= a_max))[0] + 1

    records = []
    for lbl_id in keep_ids:
        x, y, w, h, area = stats[lbl_id].tolist()
        cx, cy = cents[lbl_id].tolist()

        sub = labels[y:y+h, x:x+w]

        m = (sub == lbl_id).astype(np.uint8)
        m_er = cv2.erode(m, np.ones((3, 3), np.uint8), iterations=2).astype(bool)

        patch = heatmap_rgb[y:y+h, x:x+w]

        pixels = patch[m_er]
        if pixels.size == 0:
            pixels = patch[m.astype(bool)]

        med_rgb = np.median(pixels, axis=0)

        records.append({
            "label_id": int(lbl_id),
            "cx": float(cx), "cy": float(cy),
            "r": float(med_rgb[0]), "g": float(med_rgb[1]), "b": float(med_rgb[2]),
            "area": int(area),
        })

    df = pd.DataFrame(records)
    debug = {"edges": edges, "roi": roi, "interior": interior}
    return df, debug

def assign_row_major_ids(df: pd.DataFrame, y_gap_min=2.0) -> pd.DataFrame:
    ys = np.sort(df["cy"].to_numpy())
    diffs = np.diff(ys)
    row_gaps = diffs[diffs > y_gap_min]
    if len(row_gaps) == 0:
        tmp = df.sort_values(["cy","cx"]).reset_index(drop=True)
        tmp["row"] = 0
        tmp["col"] = np.arange(len(tmp))
        tmp.insert(0, "hex_id", np.arange(1, len(tmp)+1))
        return tmp

    row_step = float(np.median(row_gaps))
    y0 = float(df["cy"].min())

    out = df.copy()
    out["row"] = np.round((out["cy"] - y0) / row_step).astype(int)
    out = out.sort_values(["row","cx"]).reset_index(drop=True)
    out["col"] = out.groupby("row").cumcount()
    out.insert(0, "hex_id", np.arange(1, len(out)+1))
    return out

def _hex_sat(rgb_arr: np.ndarray) -> np.ndarray:
    rgb_u8 = np.clip(rgb_arr, 0, 255).astype(np.uint8)
    hsv = cv2.cvtColor(rgb_u8[None, :, :], cv2.COLOR_RGB2HSV)[0]
    return hsv[:, 1].astype(np.int32)

def map_hex_to_numeric(df_hex: pd.DataFrame, model: NumericScaleModel, debug_dir: str | None = None, tag: str = "") -> pd.DataFrame:
    rgb = df_hex[["r","g","b"]].to_numpy().astype(np.uint8)
    lab = rgb_to_lab(rgb)

    dist, idx = model.tree.query(lab, k=1)
    dist = dist.astype(float)
    idx = idx.astype(int)

    out = df_hex.copy()
    out["deltaE"] = dist

    hsv = cv2.cvtColor(rgb[None, :, :], cv2.COLOR_RGB2HSV)[0]
    S = hsv[:, 1].astype(np.int32)
    V = hsv[:, 2].astype(np.int32)
    is_bg = (V > 245) & (S < 10)

    step_rgb = np.array(model.steps["step_rgb"].tolist(), dtype=np.uint8) if "step_rgb" in model.steps.columns else None
    legend_sat_min = int(np.min(_hex_sat(step_rgb))) if step_rgb is not None and len(step_rgb) else 255
    allow_low_sat = legend_sat_min < SAT_MIN_REJECT

    sat = _hex_sat(rgb)
    out["sat"] = sat

    sat_reject = (sat < SAT_MIN_REJECT) & (not allow_low_sat)
    out["rejected"] = is_bg | (out["deltaE"] > DELTAE_REJECT) | sat_reject

    if model.kind == "discrete":
        out["scale_idx"] = idx
        steps = model.steps.reset_index(drop=True)
        out = out.reset_index(drop=True).join(
            steps.loc[out["scale_idx"], ["step_rgb","value_min","value_max","value_mid"]].reset_index(drop=True)
        )
        out.loc[out["rejected"], ["value_min","value_max","value_mid"]] = np.nan

    else:
        pos = model.sample_pos[idx]
        val = model.mn + pos * (model.mx - model.mn)

        steps = model.steps.reset_index(drop=True)
        edges = np.array([float(x) for x in steps["value_min"].tolist()] + [float(steps["value_max"].iloc[-1])], dtype=float)
        bin_idx = np.digitize(val, edges, right=False) - 1
        bin_idx = np.clip(bin_idx, 0, len(steps)-1)

        out["scale_idx"] = bin_idx.astype(int)
        out["value_mid"] = val

        out = out.reset_index(drop=True).join(
            steps.loc[out["scale_idx"], ["step_rgb","value_min","value_max"]].reset_index(drop=True)
        )
        out.loc[out["rejected"], ["value_min","value_max","value_mid"]] = np.nan

    if DEBUG and debug_dir:
        rej_rate = float(out["rejected"].mean()) if len(out) else 0.0
        with open(os.path.join(debug_dir, f"{tag}mapping_numeric_stats.json"), "w") as f:
            json.dump(dict(
                n_hex=int(len(out)),
                rejected_rate=rej_rate,
                deltaE_median=float(np.nanmedian(out["deltaE"])) if len(out) else None,
                sat_median=float(np.nanmedian(out["sat"])) if len(out) else None,
                kind=model.kind
            ), f, indent=2)

    return out

def map_hex_to_categorical(df_hex: pd.DataFrame, model: CategoricalScaleModel, debug_dir: str | None = None, tag: str = "") -> pd.DataFrame:
    rgb = df_hex[["r","g","b"]].to_numpy().astype(np.uint8)
    lab = rgb_to_lab(rgb)

    dist, idx = model.tree.query(lab, k=1)
    out = df_hex.copy()
    out["deltaE"] = dist.astype(float)
    out["scale_idx"] = idx.astype(int)

    hsv = cv2.cvtColor(rgb[None, :, :], cv2.COLOR_RGB2BGR)[0]  # keep original behavior? (was HSV from RGB)
    hsv = cv2.cvtColor(rgb[None, :, :], cv2.COLOR_RGB2HSV)[0]
    S = hsv[:, 1].astype(np.int32)
    V = hsv[:, 2].astype(np.int32)
    is_bg = (V > 245) & (S < 10)

    step_rgb = np.array(model.steps["step_rgb"].tolist(), dtype=np.uint8)
    legend_sat_min = int(np.min(_hex_sat(step_rgb))) if len(step_rgb) else 255
    allow_low_sat = legend_sat_min < SAT_MIN_REJECT

    sat = _hex_sat(rgb)
    out["sat"] = sat
    sat_reject = (sat < SAT_MIN_REJECT) & (not allow_low_sat)

    out["rejected"] = is_bg | (out["deltaE"] > DELTAE_REJECT) | sat_reject

    steps = model.steps.reset_index(drop=True)
    out = out.reset_index(drop=True).join(
        steps.loc[out["scale_idx"], ["step_rgb","label"]].reset_index(drop=True)
    )
    out.loc[out["rejected"], ["label"]] = ""

    if DEBUG and debug_dir:
        with open(os.path.join(debug_dir, f"{tag}mapping_cat_stats.json"), "w") as f:
            json.dump(dict(
                n_hex=int(len(out)),
                rejected_rate=float(out["rejected"].mean()) if len(out) else 0.0,
                deltaE_median=float(np.nanmedian(out["deltaE"])) if len(out) else None,
            ), f, indent=2)

    return out

# ============================================================
# ANNOTATION
# ============================================================
def annotate_hexes_id_value_clear(img_rgb: np.ndarray,
                                  df: pd.DataFrame,
                                  out_file: str,
                                  value_col="value_mid",
                                  font_scale=0.32,
                                  thickness=1,
                                  pad_x=3, pad_y=2,
                                  box_alpha=0.65,
                                  border_px=1):
    bgr = cv2.cvtColor(img_rgb.copy(), cv2.COLOR_RGB2BGR)
    font = cv2.FONT_HERSHEY_SIMPLEX
    h, w = bgr.shape[:2]

    for _, r in df.iterrows():
        cx = int(round(r["cx"]))
        cy = int(round(r["cy"]))

        v = r.get(value_col, np.nan)
        if pd.isna(v):
            val_str = "NA"
        else:
            val_str = f"{int(round(float(v)))}" if abs(float(v) - round(float(v))) < 1e-6 else f"{float(v):.2f}"

        text = f"({int(r['hex_id'])}:{val_str})"

        lum = 0.2126*r["r"] + 0.7152*r["g"] + 0.0722*r["b"]
        if lum >= 128:
            text_color, outline_color = (0, 0, 0), (255, 255, 255)
            box_color, border_color = (255, 255, 255), (0, 0, 0)
        else:
            text_color, outline_color = (255, 255, 255), (0, 0, 0)
            box_color, border_color = (0, 0, 0), (255, 255, 255)

        (tw, th), base = cv2.getTextSize(text, font, font_scale, thickness)
        x = cx - tw // 2
        y = cy + th // 2

        x1 = max(0, x - pad_x)
        y1 = max(0, y - th - pad_y)
        x2 = min(w - 1, x + tw + pad_x)
        y2 = min(h - 1, y + base + pad_y)

        roi = bgr[y1:y2, x1:x2].astype(np.float32)
        box = np.full_like(roi, box_color, dtype=np.float32)
        bgr[y1:y2, x1:x2] = ((1 - box_alpha) * roi + box_alpha * box).astype(np.uint8)

        if border_px > 0:
            cv2.rectangle(bgr, (x1, y1), (x2, y2), border_color, border_px)

        cv2.putText(bgr, text, (x, y), font, font_scale, outline_color, thickness+2, cv2.LINE_AA)
        cv2.putText(bgr, text, (x, y), font, font_scale, text_color, thickness, cv2.LINE_AA)

    cv2.imwrite(out_file, bgr)
    return out_file

# ============================================================
# HISTOGRAMS
# ============================================================
def _fmt_num(x: float) -> str:
    if np.isneginf(x): return "-inf"
    if np.isposinf(x): return "inf"
    if abs(x - round(x)) < 1e-6:
        return str(int(round(x)))
    return f"{x:.2f}".rstrip("0").rstrip(".")

def save_histogram_colored(df_vals: pd.DataFrame, steps: pd.DataFrame, out_file: str, axis_label: str):
    df_use = df_vals.dropna(subset=["value_mid"]).copy()
    if df_use.empty:
        plt.figure(figsize=(12, 4))
        plt.title("Value Distribution by Range (no valid values)")
        plt.tight_layout()
        plt.savefig(out_file, dpi=220, bbox_inches="tight")
        plt.close()
        return out_file

    counts = df_use.groupby(["value_min","value_max"]).size().reset_index(name="count")
    dist = steps[["value_min","value_max","step_rgb"]].merge(counts, on=["value_min","value_max"], how="left")
    dist["count"] = dist["count"].fillna(0).astype(int)
    dist["pct"] = dist["count"] / max(1, len(df_use)) * 100.0

    labels = []
    for a, b in zip(dist["value_min"], dist["value_max"]):
        if np.isneginf(a) and np.isfinite(b):
            labels.append(f"≤ {_fmt_num(b)}")
        elif np.isposinf(b) and np.isfinite(a):
            labels.append(f"≥ {_fmt_num(a)}")
        else:
            labels.append(f"{_fmt_num(a)} to {_fmt_num(b)}")

    colors = [tuple(np.array(rgb)/255.0) for rgb in dist["step_rgb"]]

    plt.figure(figsize=(12, 4))
    x = np.arange(len(dist))
    plt.bar(x, dist["pct"], color=colors, edgecolor="white", linewidth=1.0)
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylabel("Percentage (%)")
    plt.xlabel(axis_label)
    plt.title("Value Distribution by Range")
    plt.tight_layout()
    plt.savefig(out_file, dpi=220, bbox_inches="tight")
    plt.close()
    return out_file

def save_histogram_categorical(df_vals: pd.DataFrame, steps: pd.DataFrame, out_file: str, axis_label: str):
    df_use = df_vals[df_vals["label"].astype(str).str.len() > 0].copy()
    if df_use.empty:
        plt.figure(figsize=(12, 4))
        plt.title("Value Distribution by Category (no valid values)")
        plt.tight_layout()
        plt.savefig(out_file, dpi=220, bbox_inches="tight")
        plt.close()
        return out_file

    counts = df_use.groupby(["scale_idx","label"]).size().reset_index(name="count")
    counts["pct"] = counts["count"] / max(1, len(df_use)) * 100.0
    counts = counts.sort_values("scale_idx").reset_index(drop=True)

    colors = []
    for i in counts["scale_idx"].tolist():
        rgb = steps.loc[int(i), "step_rgb"]
        colors.append(tuple(np.array(rgb)/255.0))

    plt.figure(figsize=(12, 4))
    x = np.arange(len(counts))
    plt.bar(x, counts["pct"], color=colors, edgecolor="white", linewidth=1.0)
    plt.xticks(x, counts["label"].tolist(), rotation=45, ha="right")
    plt.ylabel("Percentage (%)")
    plt.xlabel(axis_label)
    plt.title("Value Distribution by Category")
    plt.tight_layout()
    plt.savefig(out_file, dpi=220, bbox_inches="tight")
    plt.close()
    return out_file

# ============================================================
# CSV OUTPUT
# ============================================================
CSV_COLS = ["hex_id","label_id","cx","cy","r","g","b","area","deltaE","scale_idx","value","row","col"]

def write_clean_output_csv(df_vals: pd.DataFrame, out_path: str, is_categorical: bool):
    df = df_vals.copy()

    if is_categorical:
        df["value"] = df.get("label", "").astype(str)
    else:
        df["value"] = pd.to_numeric(df.get("value_mid", np.nan), errors="coerce")

    for c in CSV_COLS:
        if c not in df.columns:
            if c == "value" and is_categorical:
                df[c] = ""
            else:
                df[c] = np.nan

    df = df[CSV_COLS]
    df.to_csv(out_path, index=False)
    return out_path

# ============================================================
# COMPARISON OUTPUTS (NUMERIC ONLY)
# ============================================================
def comparison_outputs(df_band: pd.DataFrame, out_dir: str, metric_cfg: dict, band_name: str):
    ensure_dir(out_dir)
    outs = []
    axis_label = metric_cfg["axis_label"]
    metric_name = metric_cfg["name"]

    summary = df_band.groupby("device")["value"].agg(["count","mean","median","std","min","max"]).reset_index()
    p_summary = os.path.join(out_dir, f"summary_{slug(metric_cfg['key'])}_{slug(band_name)}.csv")
    summary.to_csv(p_summary, index=False)
    outs.append(p_summary)

    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df_band, x="device", y="value", hue="device", dodge=False, palette="Set2")
    plt.legend([], [], frameon=False)
    plt.title(f"{metric_name} Distribution by Device - {band_name}")
    plt.xlabel("Device")
    plt.ylabel(axis_label)
    plt.xticks(rotation=45, ha="right")
    plt.grid(True, axis="y")
    if metric_cfg.get("fixed_style") == "dbm":
        plt.ylim(-100, -20)
        plt.yticks(np.arange(-100, -19, 10))
    plt.tight_layout()
    p_box = os.path.join(out_dir, f"boxplot_{slug(metric_cfg['key'])}_{slug(band_name)}.png")
    plt.savefig(p_box, dpi=220, bbox_inches="tight")
    plt.close()
    outs.append(p_box)

    plt.figure(figsize=(10, 6))
    for dev in df_band["device"].unique():
        sns.kdeplot(data=df_band[df_band["device"] == dev], x="value", fill=True, alpha=0.3, label=dev)
    plt.title(f"Density of {metric_name} by Device - {band_name}")
    plt.xlabel(axis_label)
    plt.ylabel("Density")
    plt.legend(title="Device")
    plt.grid(True)
    if metric_cfg.get("fixed_style") == "dbm":
        plt.xlim(-100, -20)
        plt.xticks(np.arange(-100, -19, 10))
        plt.ylim(0.0, 0.09)
    plt.tight_layout()
    p_den = os.path.join(out_dir, f"density_{slug(metric_cfg['key'])}_{slug(band_name)}.png")
    plt.savefig(p_den, dpi=220, bbox_inches="tight")
    plt.close()
    outs.append(p_den)

    return outs

# ============================================================
# UI HELPERS (NO COUNTS / NO ALL)
# ============================================================
def pick(prompt, options):
    print(prompt)
    for i, opt in enumerate(options, 1):
        print(f"{i}. {opt}")
    while True:
        s = input("Enter choice number: ").strip()
        if s.isdigit() and 1 <= int(s) <= len(options):
            return int(s) - 1
        print("Invalid choice, try again.\n")

def scan_heatmaps(input_root: str) -> list[str]:
    all_imgs = []
    for ext in IMG_EXTS:
        all_imgs.extend(glob.glob(os.path.join(input_root, "**", f"*{ext}"), recursive=True))
    heatmaps = [p for p in all_imgs if not is_scale_file(p)]
    return sorted(heatmaps)

def build_inventory(heatmaps: list[str]) -> list[str]:
    found = set()
    for p in heatmaps:
        fn = os.path.basename(p)
        k = metric_key_from_filename(fn)
        if k and (k in METRIC_CFG_BY_KEY):
            found.add(k)
    ordered = [m["key"] for m in METRICS if m["key"] in found]
    return ordered

# ==============================
# Streamlit/VSCode entrypoints
# ==============================

from pathlib import Path

def scan_heatmaps_extracted(extracted_root: str) -> list[str]:
    root = Path(extracted_root)
    if not root.exists():
        raise FileNotFoundError(f"Extracted folder not found: {extracted_root}")

    imgs = []
    for ext in IMG_EXTS:
        imgs.extend(root.rglob(f"*{ext}"))

    heatmaps = []
    for p in imgs:
        if "_zips" in p.parts:
            continue
        if is_scale_file(str(p)):
            continue
        heatmaps.append(str(p))

    return sorted(heatmaps)


def run_ocr_generate_csv(
    extracted_root: str,
    csv_out_root: str,
    *,
    debug: bool = False,
    max_heatmaps: int | None = None
) -> dict:
    global DEBUG
    DEBUG = bool(debug)

    out_dir = Path(csv_out_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    heatmaps = scan_heatmaps_extracted(extracted_root)
    if max_heatmaps is not None:
        heatmaps = heatmaps[: int(max_heatmaps)]

    meta_rows = []
    failed = []

    for hm_path in heatmaps:
        hm_name = os.path.basename(hm_path)
        hm_stem = Path(hm_name).stem

        try:
            metric_key = metric_key_from_filename(hm_name)
            if not metric_key:
                continue
            if metric_key not in ALLOWED_METRIC_KEYS:
                continue

            metric_cfg = METRIC_CFG_BY_KEY.get(metric_key)
            if not metric_cfg:
                continue

            scale_path = find_matching_scale(hm_path)

            hm_rgb = read_image_rgb(hm_path)
            scale_rgb = read_image_rgb(scale_path)

            if metric_key == "snr":
                df_hex, _ = extract_hexagons(hm_rgb, roi_sat_thresh=6, roi_val_thresh=254)
            else:
                df_hex, _ = extract_hexagons(hm_rgb)

            df_hex = assign_row_major_ids(df_hex)

            out_csv = out_dir / f"{hm_stem}_output.csv"

            if metric_cfg.get("scale_kind") == "categorical":
                cat_model = build_categorical_scale_model(scale_rgb, reverse_lr=False)
                df_vals = map_hex_to_categorical(df_hex, cat_model)
                write_clean_output_csv(df_vals, str(out_csv), is_categorical=True)

                meta_rows.append({
                    "heatmap": hm_name,
                    "metric_key": metric_key,
                    "scale_kind": "categorical",
                    "heatmap_path": hm_path,
                    "scale_path": scale_path,
                    "csv": str(out_csv),
                    "n_bins": int(len(cat_model.steps)),
                })

            else:
                mn, mx = ocr_minmax_from_scale_path(scale_path, metric_cfg, debug_dir=None, tag="")
                num_model = build_numeric_scale_model(scale_rgb, metric_cfg, reverse_lr=False, mn=mn, mx=mx)
                df_vals = map_hex_to_numeric(df_hex, num_model)
                write_clean_output_csv(df_vals, str(out_csv), is_categorical=False)

                meta_rows.append({
                    "heatmap": hm_name,
                    "metric_key": metric_key,
                    "scale_kind": "numeric",
                    "numeric_kind": num_model.kind,
                    "mn": float(num_model.mn),
                    "mx": float(num_model.mx),
                    "heatmap_path": hm_path,
                    "scale_path": scale_path,
                    "csv": str(out_csv),
                    "n_bins": int(len(num_model.steps)),
                })

        except Exception as e:
            failed.append({"heatmap": hm_name, "error": str(e)})

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
    }
    