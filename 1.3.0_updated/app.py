import io
import os
import re
import sys
import shutil
import zipfile
import subprocess
from pathlib import Path
from typing import Dict, List, Optional
from charts import render_mesh_compare_apex_dashboard, render_rvr_apex_dashboard
from ai_report_generator import streamlit_report_card


import streamlit as st

from docx_extractor_v2 import process_many_docx_local
from ocr_csv_generator_v2 import run_ocr_generate_csv
from local_kb import LocalKnowledgeBase, answer_with_context
from metadata_utils import clean_router_name


# -----------------------------
# Page setup + styling
# -----------------------------
st.set_page_config(page_title="WiFi Site Survey", layout="wide")

st.markdown(
    """
<style>
div[data-testid="stAppViewContainer"] { background: #0b1220; }
.block-container { padding-top: 1.6rem; padding-bottom: 2.2rem; max-width: 1200px; }
header[data-testid="stHeader"] { background: transparent; }
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }

.hero {
  background: radial-gradient(1200px 500px at 20% 0%, rgba(90, 120, 255, 0.28), transparent 60%),
              radial-gradient(1200px 500px at 80% 20%, rgba(45, 220, 200, 0.16), transparent 60%),
              linear-gradient(180deg, rgba(255,255,255,0.06), rgba(255,255,255,0.02));
  border: 1px solid rgba(255,255,255,0.10);
  border-radius: 22px;
  padding: 1.4rem 1.5rem;
  box-shadow: 0 20px 50px rgba(0,0,0,0.35);
  margin-bottom: 1.1rem;
}
.hero h1 { margin: 0; color: rgba(255,255,255,0.95); font-size: 2.1rem; line-height: 1.1; }
.hero p  { margin: 0.35rem 0 0 0; color: rgba(255,255,255,0.70); font-size: 1.02rem; }

.card {
  background: rgba(255,255,255,0.06);
  border: 1px solid rgba(255,255,255,0.10);
  border-radius: 18px;
  padding: 1.1rem 1.1rem 0.8rem 1.1rem;
  box-shadow: 0 14px 40px rgba(0,0,0,0.25);
}
.card-title {
  display:flex; align-items:center; justify-content:space-between; gap:1rem;
  margin-bottom: 0.7rem;
}
.card-title h2 {
  margin:0; color: rgba(255,255,255,0.92); font-size: 1.25rem;
}
.subtle { color: rgba(255,255,255,0.66); font-size: 0.95rem; }

.step {
  display:inline-flex; align-items:center; justify-content:center;
  min-width: 32px; height: 32px; padding: 0 10px;
  border-radius: 999px;
  background: rgba(255,255,255,0.10);
  border: 1px solid rgba(255,255,255,0.14);
  color: rgba(255,255,255,0.88);
  font-size: 0.88rem;
}

.chips { display:flex; flex-wrap:wrap; gap: 10px; margin-top: 0.75rem; }
.chip {
  background: rgba(255,255,255,0.07);
  border: 1px solid rgba(255,255,255,0.10);
  color: rgba(255,255,255,0.80);
  border-radius: 999px;
  padding: 8px 12px;
  font-size: 0.9rem;
}

div[data-testid="stFileUploaderDropzone"] {
  border-radius: 16px !important;
  border: 1px dashed rgba(255,255,255,0.18) !important;
  background: rgba(255,255,255,0.05) !important;
}
label, .stMarkdown, .stText, .stCaption { color: rgba(255,255,255,0.82) !important; }

.stButton > button, div[data-testid="stDownloadButton"] > button {
  border-radius: 14px !important;
  border: 1px solid rgba(255,255,255,0.16) !important;
  background: rgba(255,255,255,0.10) !important;
  color: rgba(255,255,255,0.92) !important;
  padding: 0.55rem 0.9rem !important;
}
.stButton > button:hover, div[data-testid="stDownloadButton"] > button:hover {
  background: rgba(255,255,255,0.14) !important;
}

pre {
  background: rgba(255,255,255,0.05) !important;
  border: 1px solid rgba(255,255,255,0.10) !important;
  border-radius: 14px !important;
}

div[data-testid="stExpander"] {
  background: rgba(255,255,255,0.03);
  border: 1px solid rgba(255,255,255,0.10);
  border-radius: 16px;
}
</style>
""",
    unsafe_allow_html=True,
)


# -----------------------------
# Paths + constants
# -----------------------------
RUNS_DIR = Path("runs")
RUNS_DIR.mkdir(exist_ok=True)

PARAM_SCRIPT = Path("parameter_vs_range.py")
COMPARE_SCRIPT = Path("comparison.py")

PARAM_FOLDER_TO_DISPLAY: Dict[str, str] = {
    "signal_strength": "Signal Strength",
    "signal_strength_main": "Signal Strength",
    "secondary_signal_strength": "Secondary Signal Strength",
    "tertiary_signal_strength": "Tertiary Signal Strength",
    "snr": "SNR",
    "noise": "Noise",
    "data_rate": "Data Rate",
    "throughput": "Throughput",
    "channel_utilization": "Channel Utilization",
    "channel_interference": "Channel Interference",
    "channel_width": "Channel Width",
    "spectrum_channel_power": "Spectrum Channel Power",
    "network_health": "Network Health",
    "network_issues": "Network Issues",
    "number_of_aps": "Number of APs",
    "number_of_access_points": "Number of Access Points",
    "associated_access_points": "Associated Access Points",
    "bluetooth_coverage": "Bluetooth Coverage",
}
PARAM_DISPLAY_TO_FOLDER: Dict[str, str] = {
    v: k for k, v in PARAM_FOLDER_TO_DISPLAY.items() if k != "signal_strength_main"
}
ALL_PARAM_DISPLAY_OPTIONS = [
    "Signal Strength",
    "Secondary Signal Strength",
    "Tertiary Signal Strength",
    "SNR",
    "Noise",
    "Data Rate",
    "Throughput",
    "Channel Utilization",
    "Channel Interference",
    "Channel Width",
    "Spectrum Channel Power",
    "Network Health",
    "Network Issues",
    "Number of Access Points",
]


# -----------------------------
# Helpers
# -----------------------------
def safe_name(text: str) -> str:
    text = (text or "").strip()
    text = Path(text).stem
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("._-")
    return text or "router"


def normalize_text(text: str) -> str:
    text = (text or "").replace("\u00A0", " ").strip().lower()
    text = text.replace("–", "-").replace("—", "-")
    text = re.sub(r"[_\-]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def router_dir(router_name: str) -> Path:
    return RUNS_DIR / safe_name(router_name)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def reset_dir(path: Path) -> Path:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def zip_folder_bytes(folder: Path) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in folder.rglob("*"):
            if p.is_file():
                zf.write(p, arcname=str(p.relative_to(folder)))
    buf.seek(0)
    return buf.read()


def zip_to_path(folder: Path, zip_path: Path) -> Path:
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in folder.rglob("*"):
            if p.is_file():
                zf.write(p, arcname=str(p.relative_to(folder)))
    return zip_path


def count_files(folder: Optional[Path], exts=None) -> int:
    if not folder or not folder.exists():
        return 0
    if not exts:
        return sum(1 for p in folder.rglob("*") if p.is_file())
    exts = {e.lower() for e in exts}
    return sum(1 for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in exts)


def write_uploaded_file(uploaded_file, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(uploaded_file.getbuffer())


def guess_router_from_docx(files) -> str:
    if not files:
        return ""
    stem = Path(files[0].name).stem
    return clean_router_name(stem) or safe_name(stem.split()[0])


def guess_router_from_esx(files, master_name: Optional[str] = None) -> str:
    if not files:
        return ""
    preferred = None
    if master_name:
        for f in files:
            if f.name != master_name:
                preferred = f.name
                break
    if not preferred:
        preferred = files[0].name
    stem = Path(preferred).stem
    return clean_router_name(stem) or safe_name(stem.split()[0])


def canonical_metric_folder(metric_text: str) -> Optional[str]:
    t = normalize_text(metric_text)

    if "secondary signal strength" in t:
        return "secondary_signal_strength"
    if "tertiary signal strength" in t:
        return "tertiary_signal_strength"
    if "signal strength" in t and "secondary" not in t and "tertiary" not in t:
        return "signal_strength"

    if t == "snr" or ("signal" in t and "noise ratio" in t):
        return "snr"
    if t == "noise":
        return "noise"
    if "throughput" in t:
        return "throughput"
    if "data rate" in t:
        return "data_rate"
    if "channel utilization" in t:
        return "channel_utilization"
    if "channel interference" in t:
        return "channel_interference"
    if "channel width" in t:
        return "channel_width"
    if "spectrum channel power" in t:
        return "spectrum_channel_power"
    if "network health" in t:
        return "network_health"
    if "network issues" in t:
        return "network_issues"
    if "number of aps" in t or "number of access points" in t:
        return "number_of_access_points"
    if "associated access point" in t:
        return "associated_access_points"
    if "bluetooth coverage" in t:
        return "bluetooth_coverage"

    return None


def metric_folder_from_csv_name(filename: str) -> Optional[str]:
    stem = Path(filename).stem
    if stem.lower().endswith("_output"):
        stem = stem[:-7]

    # Expected pattern: <router>_<metric> for <floor> on <band> band_output.csv
    if "_" in stem:
        _, rest = stem.split("_", 1)
    else:
        rest = stem

    metric_text = rest.split(" for ", 1)[0]
    return canonical_metric_folder(metric_text)


def organize_csv_outputs(csv_root: Path) -> Dict[str, int]:
    """
    Copy CSV files into metric-named subfolders under csv_root.
    Keeps original root-level files like _index.csv in place.
    """
    ensure_dir(csv_root)

    all_csvs = [
        p for p in csv_root.rglob("*.csv")
        if p.is_file()
    ]

    for csv_file in all_csvs:
        if csv_file.parent != csv_root:
            continue
        if csv_file.name.startswith("_"):
            continue
        metric_folder = metric_folder_from_csv_name(csv_file.name)
        if not metric_folder:
            continue
        target_dir = ensure_dir(csv_root / metric_folder)
        target_path = target_dir / csv_file.name
        shutil.copy2(csv_file, target_path)

    summary: Dict[str, int] = {}
    for d in sorted([p for p in csv_root.iterdir() if p.is_dir()]):
        n = len([p for p in d.glob("*.csv") if p.is_file()])
        if n > 0:
            summary[d.name] = n
    return summary


def discover_metric_dirs(csv_root: Path) -> List[str]:
    if not csv_root.exists():
        return []
    dirs = []
    for d in sorted([p for p in csv_root.iterdir() if p.is_dir()]):
        if any(p.suffix.lower() == ".csv" for p in d.glob("*.csv")):
            dirs.append(d.name)
    return dirs


def metric_label(folder_name: str) -> str:
    return PARAM_FOLDER_TO_DISPLAY.get(folder_name, folder_name.replace("_", " ").title())


def copy_metric_csvs(src_metric_dir: Path, dst_dir: Path, limit: int = 0) -> int:
    ensure_dir(dst_dir)
    csvs = sorted([p for p in src_metric_dir.glob("*.csv") if p.is_file()])
    if limit and limit > 0:
        csvs = csvs[:limit]
    for p in csvs:
        shutil.copy2(p, dst_dir / p.name)
    return len(csvs)


def collect_plot_pngs(out_dir: Path) -> List[Path]:
    pngs: List[Path] = []
    for sub in ["plots_percent", "plots_actual", "plots"]:
        p = out_dir / sub
        if p.exists():
            pngs.extend(sorted(p.rglob("*.png")))
    return pngs

def collect_compare_plot_pngs(out_dir: Path) -> List[Path]:
    if not out_dir.exists():
        return []
    return sorted(out_dir.rglob("*.png"))


def patch_parameter_script(
    source_script: Path,
    patched_script: Path,
    param_display: str,
    input_dir: Path,
    out_dir: Path,
    zip_path: Path,
) -> Path:
    text = source_script.read_text(encoding="utf-8", errors="ignore")

    # Force selected parameter
    text = re.sub(
        r"(?m)^PARAM_NAME\s*=.*$",
        f"PARAM_NAME = {param_display!r}",
        text,
        count=1,
    )

    # Force input/output paths
    text = re.sub(
        r"(?m)^INPUT_DIR\s*=.*$",
        f"INPUT_DIR = {str(input_dir)!r}",
        text,
        count=1,
    )

    text = re.sub(
        r"(?m)^OUT_BASE\s*=.*$",
        f"OUT_BASE = {str(out_dir)!r}",
        text,
        count=1,
    )

    # Optional zip path replacement if present in script
    text = re.sub(
        r"(?m)^ZIP_PATH\s*=.*$",
        f"ZIP_PATH = {str(zip_path)!r}",
        text,
        count=1,
    )

    patched_script.parent.mkdir(parents=True, exist_ok=True)
    patched_script.write_text(text, encoding="utf-8")
    return patched_script

def patch_comparison_script(
    source_script: Path,
    patched_script: Path,
) -> Path:
    text = source_script.read_text(encoding="utf-8", errors="ignore")
    patched_script.parent.mkdir(parents=True, exist_ok=True)
    patched_script.write_text(text, encoding="utf-8")
    return patched_script


# -----------------------------
# Header
# -----------------------------
st.markdown(
    """
<div class="hero">
  <h1>WiFi Site Survey Automation</h1>
  <p>DOCX → Images + CSV → Parameter vs Range</p>
</div>
""",
    unsafe_allow_html=True,
)

existing_router_dirs = sorted([p.name for p in RUNS_DIR.iterdir() if p.is_dir()])

top_left, top_right = st.columns([2, 2])
with top_left:
    selected_existing_router = st.selectbox(
        "Load existing router folder (optional)",
        [""] + existing_router_dirs,
        index=0,
    )
with top_right:
    default_router_name = st.session_state.get("router_name", "")
    if selected_existing_router:
        default_router_name = selected_existing_router
    router_name_input = st.text_input("Router folder name", value=default_router_name)

router_name = safe_name(router_name_input or selected_existing_router or "")
if router_name:
    st.session_state["router_name"] = router_name

current_router_dir = router_dir(router_name) if router_name else None
docx_input_dir = current_router_dir / "docx_inputs" if current_router_dir else None
extracted_dir = current_router_dir / "extracted" if current_router_dir else None
csv_root_dir = current_router_dir / "csv_outputs" if current_router_dir else None
esx_input_dir = current_router_dir / "esx_inputs" if current_router_dir else None
rvr_inputs_root = current_router_dir / "rvr_inputs" if current_router_dir else None
rvr_outputs_root = current_router_dir / "rvr_outputs" if current_router_dir else None
compare_outputs_root = current_router_dir / "compare_outputs" if current_router_dir else None

chips = []
chips.append(f"Router: {router_name}" if router_name else "Router: —")
chips.append(f"Extracted images: {count_files(extracted_dir, exts={'.png','.jpg','.jpeg','.webp','.bmp','.tif','.tiff'})}")
chips.append(f"CSV metrics: {len(discover_metric_dirs(csv_root_dir)) if csv_root_dir and csv_root_dir.exists() else 0}")
chips.append(f"RVR outputs: {len([p for p in (rvr_outputs_root.iterdir() if rvr_outputs_root and rvr_outputs_root.exists() else []) if p.is_dir()])}")
chips.append(f"Compare outputs: {len([p for p in (compare_outputs_root.iterdir() if compare_outputs_root and compare_outputs_root.exists() else []) if p.is_dir()])}")

st.markdown(
    '<div class="chips">' + "".join([f'<div class="chip">{c}</div>' for c in chips]) + "</div>",
    unsafe_allow_html=True,
)

st.write("")


# -----------------------------
# Step 1 — DOCX to images + CSV
# -----------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown(
    """
<div class="card-title">
  <h2><span class="step">Step 1</span> DOCX to Images + CSV</h2>
</div>
<div class="subtle">
  Upload DOCX survey reports, extract the images, and generate grouped CSV outputs in one run.
  Outputs go to <code>runs/&lt;router_name&gt;/extracted</code> and <code>runs/&lt;router_name&gt;/csv_outputs/&lt;metric&gt;</code>.
  Re-running overwrites this router’s extracted and CSV data.
</div>
""",
    unsafe_allow_html=True,
)

docx_files = st.file_uploader(
    "DOCX report file(s)",
    type=["docx"],
    accept_multiple_files=True,
    key="docx_uploader",
)

with st.expander("Options", expanded=False):
    selected_param_displays = st.multiselect(
        "Parameters to process",
        ALL_PARAM_DISPLAY_OPTIONS,
        default=ALL_PARAM_DISPLAY_OPTIONS,
        key="docx_selected_param_displays",
    )
    max_heatmaps = st.number_input("Max heatmaps (0 = no limit)", min_value=0, value=0, step=1, key="ocr_max_heatmaps")
    debug = st.checkbox("Debug mode", value=False, key="ocr_debug")
    also_master_zip = st.checkbox("Create one master ZIP during extraction", value=False, key="extract_master_zip")
    offer_extracted_zip = st.checkbox("Offer extracted image ZIP download", value=True, key="extract_offer_zip")
    offer_csv_zip = st.checkbox("Offer CSV ZIP download", value=True, key="ocr_offer_zip")

selected_param_keys = [PARAM_DISPLAY_TO_FOLDER.get(name, "signal_strength") for name in selected_param_displays]

run_docx_pipeline = st.button("Run DOCX → Images + CSV", width="stretch", key="run_docx_pipeline_btn")

if run_docx_pipeline:
    working_router = router_name or guess_router_from_docx(docx_files)
    if not working_router:
        st.error("Enter a router name or upload at least one DOCX.")
        st.stop()

    if not docx_files:
        st.error("Upload at least one DOCX.")
        st.stop()

    if not selected_param_keys:
        st.error("Select at least one parameter to process.")
        st.stop()

    st.session_state["router_name"] = working_router
    run_dir = router_dir(working_router)
    docx_in = reset_dir(run_dir / "docx_inputs")
    extracted = reset_dir(run_dir / "extracted")
    csv_root = reset_dir(run_dir / "csv_outputs")

    # Clear downstream folders too, so same-router reruns don't keep stale RvR/compare data
    for downstream in ["rvr_inputs", "rvr_outputs", "compare_inputs", "compare_outputs"]:
        p = run_dir / downstream
        if p.exists():
            shutil.rmtree(p)

    with st.spinner("Extracting images and generating CSVs..."):
        docx_paths: List[str] = []
        for f in docx_files:
            out_path = docx_in / f.name
            out_path.write_bytes(f.getbuffer())
            docx_paths.append(str(out_path))

        process_many_docx_local(
            docx_paths,
            out_root=str(extracted),
            download_per_docx_zip=False,
            also_make_master_zip=also_master_zip,
            selected_parameters=selected_param_keys,
        )

        res = run_ocr_generate_csv(
            extracted_root=str(extracted),
            csv_out_root=str(csv_root),
            max_heatmaps=(None if max_heatmaps == 0 else int(max_heatmaps)),
            debug=debug,
            selected_parameters=selected_param_keys,
        )

    img_count = count_files(extracted, exts={".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"})
    metric_summary = organize_csv_outputs(csv_root)
    st.session_state["last_ocr_result"] = res or {}

    selected_labels = ", ".join(selected_param_displays)
    st.success(
        f"Pipeline complete for: {selected_labels}. Images found: {img_count} | "
        f"CSVs: {res.get('processed', 0) if isinstance(res, dict) else 0} | "
        f"Failed: {res.get('failed_count', 0) if isinstance(res, dict) else 0}"
    )

    sum_left, sum_right = st.columns(2)
    with sum_left:
        st.write("Extracted images folder:")
        st.code(str(extracted))
    with sum_right:
        st.write("CSV output folder:")
        st.code(str(csv_root))

    st.caption(f"Selected parameters: {selected_labels}")

    if metric_summary:
        st.write("Metric folders created:")
        for folder_name, count in metric_summary.items():
            st.write(f"- {metric_label(folder_name)}: {count} CSV file(s)")
    else:
        st.warning("No metric subfolders were created. Check OCR output naming.")

    if isinstance(res, dict) and res.get("index_csv"):
        st.write("Index CSV:")
        st.code(str(res["index_csv"]))

    if isinstance(res, dict) and res.get("failed_csv"):
        st.warning("Failures log:")
        st.code(str(res["failed_csv"]))

    download_cols = st.columns(2)
    if offer_extracted_zip:
        with download_cols[0]:
            extracted_zip_name = f"{working_router}_extracted_images.zip"
            st.download_button(
                "Download extracted images ZIP",
                data=zip_folder_bytes(extracted),
                file_name=extracted_zip_name,
                mime="application/zip",
                width="stretch",
                key="download_extracted_zip",
            )
    if offer_csv_zip:
        with download_cols[1]:
            csv_zip_name = f"{working_router}_csv_outputs.zip"
            st.download_button(
                "Download CSV ZIP",
                data=zip_folder_bytes(csv_root),
                file_name=csv_zip_name,
                mime="application/zip",
                width="stretch",
                key="download_csv_zip",
            )

st.markdown("</div>", unsafe_allow_html=True)

st.write("")


# -----------------------------
# Step 2 — Parameter vs Range
# -----------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown(
    """
<div class="card-title">
  <h2><span class="step">Step 2</span> Parameter vs Range</h2>
</div>
<div class="subtle">
Uses one selected metric folder from Step 1, plus Master/Router ESX files and optional <code>site_geometry.json</code>.
Outputs go to <code>runs/&lt;router_name&gt;/rvr_outputs/&lt;metric&gt;</code>.
</div>
""",
    unsafe_allow_html=True,
)

if not PARAM_SCRIPT.exists():
    st.error(f"Missing {PARAM_SCRIPT.name} at project root.")
else:
    metric_dirs = discover_metric_dirs(csv_root_dir) if csv_root_dir else []
    metric_folders_for_step3: List[str] = []

    st.caption("Upload ESX files, pick the metric to run, then the app prepares a temporary input bundle from the Step 1 CSV output and launches parameter_vs_range.py.")

    esx_files = st.file_uploader(
        "Upload ESX file(s)",
        type=["esx"],
        accept_multiple_files=True,
        key="rvr_esx_all",
    )

    master_choice = None
    router_choices: List[str] = []

    if esx_files:
        esx_names = [f.name for f in esx_files]
        master_choice = st.selectbox("Select Master ESX", esx_names, index=0, key="rvr_master_choice")
        default_router_files = [n for n in esx_names if n != master_choice]
        router_choices = st.multiselect(
            "Select Router ESX file(s)",
            options=[n for n in esx_names if n != master_choice],
            default=default_router_files,
            key="rvr_router_choices",
        )

    use_csv_from_step2 = st.checkbox(
        "Use CSVs from Step 1 output",
        value=True,
        key="rvr_use_step2_csv",
    )

    if use_csv_from_step2:
        if metric_dirs:
            metric_folders_for_step3 = st.multiselect(
                "Select parameter folder(s)",
                metric_dirs,
                default=metric_dirs[:1],
                format_func=metric_label,
                key="rvr_metric_folder_select",
            )
        else:
            st.info("No metric folders found yet in csv_outputs. Run Step 1 first.")
    else:
        selected_param_displays = st.multiselect(
            "Select parameter(s)",
            ALL_PARAM_DISPLAY_OPTIONS,
            default=ALL_PARAM_DISPLAY_OPTIONS[:1],
            key="manual_param_display",
        )
        metric_folders_for_step3 = [PARAM_DISPLAY_TO_FOLDER.get(name, "signal_strength") for name in selected_param_displays]

    uploaded_csvs = None
    if not use_csv_from_step2:
        uploaded_csvs = st.file_uploader(
            "Upload band_output CSV(s)",
            type=["csv"],
            accept_multiple_files=True,
            key="rvr_csvs",
        )

    site_geom = st.file_uploader(
        "site_geometry.json (optional)",
        type=["json"],
        accept_multiple_files=False,
        key="rvr_site_geom",
    )

    with st.expander("Advanced", expanded=False):
        copy_csv_limit = st.number_input("Copy max CSVs from selected metric folder (0 = no limit)", min_value=0, value=0, step=1, key="rvr_csv_limit")
        show_full_logs = st.checkbox("Show full logs", value=False, key="rvr_show_logs")
        offer_rvr_zip = st.checkbox("Offer RvR ZIP download", value=True, key="rvr_offer_zip")

    run_rvr = st.button("Run Parameter vs Range", width="stretch", key="run_rvr_btn")

    if run_rvr:
        working_router = router_name or guess_router_from_esx(esx_files, master_choice)
        if not working_router:
            st.error("Enter a router name or upload ESX files.")
            st.stop()

        if not esx_files:
            st.error("Upload at least one ESX file.")
            st.stop()

        if not master_choice:
            st.error("Select a Master ESX.")
            st.stop()

        if not metric_folders_for_step3:
            st.error("Choose at least one parameter folder.")
            st.stop()

        st.session_state["router_name"] = working_router
        run_dir = router_dir(working_router)
        esx_store = ensure_dir(run_dir / "esx_inputs")
        esx_master_store = reset_dir(esx_store / "master")
        esx_router_store = reset_dir(esx_store / "routers")

        esx_by_name = {f.name: f for f in esx_files}
        write_uploaded_file(esx_by_name[master_choice], esx_master_store / master_choice)
        for rn in router_choices:
            write_uploaded_file(esx_by_name[rn], esx_router_store / rn)

        router_site_geom = run_dir / "site_geometry.json"
        if site_geom:
            write_uploaded_file(site_geom, router_site_geom)

        results = []
        for metric_folder_for_step3 in metric_folders_for_step3:
            metric_rvr_input = reset_dir(run_dir / "rvr_inputs" / metric_folder_for_step3)
            metric_rvr_out = reset_dir(run_dir / "rvr_outputs" / metric_folder_for_step3)

            write_uploaded_file(esx_by_name[master_choice], metric_rvr_input / master_choice)
            for rn in router_choices:
                write_uploaded_file(esx_by_name[rn], metric_rvr_input / rn)
            if router_site_geom.exists():
                shutil.copy2(router_site_geom, metric_rvr_input / "site_geometry.json")

            copied_csv_count = 0
            if use_csv_from_step2:
                src_metric_dir = run_dir / "csv_outputs" / metric_folder_for_step3
                if not src_metric_dir.exists():
                    results.append({"metric": metric_folder_for_step3, "ok": False, "log_text": f"No CSV folder found for {metric_label(metric_folder_for_step3)}.", "metric_rvr_out": metric_rvr_out, "final_zip": None})
                    continue
                copied_csv_count = copy_metric_csvs(src_metric_dir, metric_rvr_input, limit=int(copy_csv_limit))
                if copied_csv_count == 0:
                    results.append({"metric": metric_folder_for_step3, "ok": False, "log_text": "Selected metric folder contains no CSVs.", "metric_rvr_out": metric_rvr_out, "final_zip": None})
                    continue
            else:
                if not uploaded_csvs:
                    st.error("Upload at least one CSV.")
                    st.stop()
                for f in uploaded_csvs:
                    write_uploaded_file(f, metric_rvr_input / f.name)
                    copied_csv_count += 1

            param_display = metric_label(metric_folder_for_step3)
            generated_dir = ensure_dir(run_dir / "_generated")
            patched_param_script = generated_dir / f"parameter_vs_range__{metric_folder_for_step3}.py"
            intended_zip = run_dir / f"rvr_full_output_{metric_folder_for_step3}.zip"
            patch_parameter_script(
                PARAM_SCRIPT,
                patched_param_script,
                param_display=param_display,
                input_dir=metric_rvr_input,
                out_dir=metric_rvr_out,
                zip_path=intended_zip,
            )

            log_path = run_dir / f"rvr_run_{metric_folder_for_step3}.log"
            env = os.environ.copy()
            env["RVR_INPUT_DIR"] = str(metric_rvr_input)
            env["RVR_OUT_BASE"] = str(metric_rvr_out)
            env["RVR_PARAM_NAME"] = param_display
            cmd = [sys.executable, str(patched_param_script)]

            with st.spinner(f"Running parameter_vs_range.py for {param_display}..."):
                proc = subprocess.run(
                    cmd,
                    cwd=str(Path.cwd()),
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )

            log_text = (
                "===== STDOUT =====\n"
                + (proc.stdout or "")
                + "\n\n===== STDERR =====\n"
                + (proc.stderr or "")
            )
            log_path.write_text(log_text, encoding="utf-8")
            ok = proc.returncode == 0
            final_zip = intended_zip if intended_zip.exists() else zip_to_path(metric_rvr_out, run_dir / f"{metric_folder_for_step3}_rvr_outputs.zip")
            results.append({
                "metric": metric_folder_for_step3,
                "param_display": param_display,
                "ok": ok,
                "log_text": log_text,
                "log_path": log_path,
                "metric_rvr_out": metric_rvr_out,
                "final_zip": final_zip,
                "copied_csv_count": copied_csv_count,
            })

        last_result = results[-1] if results else None
        if last_result:
            st.session_state["last_rvr_result"] = {
                "ok": all(r.get("ok") for r in results),
                "returncode": 0 if all(r.get("ok") for r in results) else 1,
                "log_path": str(last_result.get("log_path", "")),
                "rvr_output": str(last_result.get("metric_rvr_out", "")),
                "parameter": last_result.get("param_display", ""),
            }

        for result in results:
            if result["ok"]:
                st.success(f"RvR completed: {result['param_display']}")
            else:
                st.error(f"RvR failed: {metric_label(result['metric'])}")

            with st.expander(f"Logs — {metric_label(result['metric'])}", expanded=not result["ok"]):
                shown = result["log_text"] if show_full_logs else (result["log_text"][-8000:] if len(result["log_text"]) > 8000 else result["log_text"])
                st.code(shown)

            if offer_rvr_zip and result.get("final_zip") and Path(result["final_zip"]).exists():
                zip_path = Path(result["final_zip"])
                st.download_button(
                    f"Download RvR ZIP — {metric_label(result['metric'])}",
                    data=zip_path.read_bytes(),
                    file_name=zip_path.name,
                    mime="application/zip",
                    width="stretch",
                    key=f"download_rvr_{result['metric']}",
                )

            pngs = collect_plot_pngs(result["metric_rvr_out"])
            if pngs:
                with st.expander(f"Preview plots — {metric_label(result['metric'])}", expanded=False):
                    show = pngs[:12]
                    cols = st.columns(3)
                    for i, p in enumerate(show):
                        with cols[i % 3]:
                            st.image(str(p), caption=p.name, width="stretch")
            else:
                st.caption(f"No plot previews were found yet for {metric_label(result['metric'])}.")

st.markdown("</div>", unsafe_allow_html=True)

st.write("")

# if current_router_dir is not None and rvr_outputs_root is not None:
#     render_rvr_apex_dashboard(current_router_dir, rvr_outputs_root)
# else:
#     st.markdown('<div class="card">', unsafe_allow_html=True)
#     st.markdown(
#         """
#         <div class="card-title">
#           <h2><span class="step">Step 3A</span> Interactive ApexCharts</h2>
#         </div>
#         <div class="subtle">Choose or load a router folder first.</div>
#         """,
#         unsafe_allow_html=True,
#     )
#     st.markdown("</div>", unsafe_allow_html=True)


st.write("")
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown(
    """
<div class="card-title">
  <h2><span class="step">Step 3</span> Mesh vs No Mesh Comparison</h2>
</div>
<div class="subtle">
Upload WITH MESH CSVs, WITHOUT MESH CSVs, one shared <code>site_geometry.json</code>, and one or more ESX files.
Outputs go to <code>runs/&lt;router_name&gt;/compare_outputs/&lt;metric&gt;</code>.
</div>
""",
    unsafe_allow_html=True,
)

if not COMPARE_SCRIPT.exists():
    st.error(f"Missing {COMPARE_SCRIPT.name} at project root.")
else:
    compare_param_displays = st.multiselect(
        "Metrics",
        ALL_PARAM_DISPLAY_OPTIONS,
        default=[ALL_PARAM_DISPLAY_OPTIONS[0]],
        key="compare4_param_displays",
    )
    if not compare_param_displays:
        st.warning("Select at least one metric.")
        st.stop()
    compare_metric_folders = [PARAM_DISPLAY_TO_FOLDER.get(p, "signal_strength") for p in compare_param_displays]

    up_left, up_right = st.columns(2)

    with up_left:
        compare_with_csvs = st.file_uploader(
            "Upload WITH MESH CSV(s)",
            type=["csv"],
            accept_multiple_files=True,
            key="compare4_with_csvs",
        )
        compare_with_docx = st.file_uploader(
            "Upload WITH MESH DOCX report(s)",
            type=["docx"],
            accept_multiple_files=True,
            key="compare4_with_docx",
        )

    with up_right:
        compare_without_csvs = st.file_uploader(
            "Upload WITHOUT MESH CSV(s)",
            type=["csv"],
            accept_multiple_files=True,
            key="compare4_without_csvs",
        )
        compare_without_docx = st.file_uploader(
            "Upload WITHOUT MESH DOCX report(s)",
            type=["docx"],
            accept_multiple_files=True,
            key="compare4_without_docx",
        )

    compare_esx_files = st.file_uploader(
        "Upload ESX file(s)",
        type=["esx"],
        accept_multiple_files=True,
        key="compare4_esx_files",
    )

    compare_site_geom = st.file_uploader(
        "Upload site_geometry.json",
        type=["json"],
        accept_multiple_files=False,
        key="compare4_site_geom",
    )

    with st.expander("Advanced", expanded=False):
        offer_compare_zip = st.checkbox(
            "Offer comparison ZIP download",
            value=True,
            key="compare4_offer_zip",
        )
        show_compare_logs = st.checkbox(
            "Show full logs",
            value=False,
            key="compare4_show_logs",
        )

    run_compare = st.button(
        "Run Mesh vs No Mesh Comparison",
        width="stretch",
        key="compare4_run_compare_btn",
    )

    if run_compare:
        if not router_name:
            st.error("Choose a router folder name first.")
            st.stop()

        if not compare_with_csvs:
            st.error("Upload WITH MESH CSV(s).")
            st.stop()

        if not compare_without_csvs:
            st.error("Upload WITHOUT MESH CSV(s).")
            st.stop()

        if not compare_esx_files:
            st.error("Upload one or more ESX files.")
            st.stop()

        if not compare_site_geom:
            st.error("Upload site_geometry.json.")
            st.stop()

        run_dir = router_dir(router_name)
        compare_inputs_root = ensure_dir(run_dir / "compare_inputs")
        compare_shared_dir = ensure_dir(run_dir / "compare_inputs" / "_shared")
        compare_esx_store = reset_dir(compare_shared_dir / "esx")
        site_geom_path = compare_shared_dir / "site_geometry.json"

        for f in compare_esx_files:
            write_uploaded_file(f, compare_esx_store / f.name)

        write_uploaded_file(compare_site_geom, site_geom_path)

        # Handle DOCX uploads
        compare_with_docx_dir = ensure_dir(compare_shared_dir / "with_mesh_docx")
        compare_without_docx_dir = ensure_dir(compare_shared_dir / "without_mesh_docx")

        for f in compare_with_docx:
            write_uploaded_file(f, compare_with_docx_dir / f.name)

        for f in compare_without_docx:
            write_uploaded_file(f, compare_without_docx_dir / f.name)

        generated_dir = ensure_dir(run_dir / "_generated")

        all_success = True
        all_logs = []

        for i, (param_display, metric_folder) in enumerate(zip(compare_param_displays, compare_metric_folders)):
            st.subheader(f"Processing {param_display}...")

            compare_with_dir = reset_dir(compare_inputs_root / "with_mesh" / metric_folder)
            compare_without_dir = reset_dir(compare_inputs_root / "without_mesh" / metric_folder)
            compare_outputs_dir = reset_dir(run_dir / "compare_outputs" / metric_folder)

            copied_with = 0
            copied_without = 0

            for f in compare_with_csvs:
                write_uploaded_file(f, compare_with_dir / f.name)
                copied_with += 1

            for f in compare_without_csvs:
                write_uploaded_file(f, compare_without_dir / f.name)
                copied_without += 1

            patched_compare_script = generated_dir / f"comparison__{metric_folder}.py"
            patch_comparison_script(COMPARE_SCRIPT, patched_compare_script)

            compare_zip_path = run_dir / f"mesh_compare_{metric_folder}.zip"
            compare_log_path = run_dir / f"compare_run_{metric_folder}.log"

            env = os.environ.copy()
            env["COMPARE_WITH_DIR"] = str(compare_with_dir)
            env["COMPARE_WITHOUT_DIR"] = str(compare_without_dir)
            env["COMPARE_OUT_DIR"] = str(compare_outputs_dir)
            env["COMPARE_ZIP_PATH"] = str(compare_zip_path)
            env["COMPARE_SITE_GEOM"] = str(site_geom_path)
            env["COMPARE_ESX_DIR"] = str(compare_esx_store)
            env["COMPARE_PARAM_NAME"] = param_display
            env["COMPARE_WITH_DOCX_DIR"] = str(compare_with_docx_dir)
            env["COMPARE_WITHOUT_DOCX_DIR"] = str(compare_without_docx_dir)

            cmd = [sys.executable, str(patched_compare_script)]

            st.info(f"Running: {' '.join(cmd)}")
            st.caption(f"With mesh CSVs    = {copied_with}")
            st.caption(f"Without mesh CSVs = {copied_without}")
            st.caption(f"Metric            = {param_display}")
            st.caption(f"Output            = {compare_outputs_dir}")

            with st.spinner(f"Running mesh vs no mesh comparison for {param_display}..."):
                proc = subprocess.run(
                    cmd,
                    cwd=str(Path.cwd()),
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )

            log_text = (
                f"===== {param_display} STDOUT =====\n"
                + (proc.stdout or "")
                + "\n\n===== STDERR =====\n"
                + (proc.stderr or "")
            )
            compare_log_path.write_text(log_text, encoding="utf-8")
            all_logs.append(log_text)

            ok = (proc.returncode == 0)
            if ok:
                st.success(f"{param_display} comparison completed successfully.")
            else:
                st.error(f"{param_display} comparison failed (exit code {proc.returncode}).")
                all_success = False

        if all_success:
            st.success("All mesh vs no mesh comparisons completed successfully.")
        else:
            st.error("Some comparisons failed.")

        with st.expander("Logs", expanded=not all_success):
            combined_logs = "\n\n".join(all_logs)
            shown = combined_logs if show_compare_logs else (combined_logs[-8000:] if len(combined_logs) > 8000 else combined_logs)
            st.code(shown)

        if offer_compare_zip:
            zip_files = list(run_dir.glob("mesh_compare_*.zip"))
            if zip_files:
                for zip_path in zip_files:
                    st.download_button(
                        f"Download {zip_path.stem} ZIP",
                        data=zip_path.read_bytes(),
                        file_name=zip_path.name,
                        mime="application/zip",
                        width="stretch",
                    )

        # Show previews for all metrics
        for param_display, metric_folder in zip(compare_param_displays, compare_metric_folders):
            compare_outputs_dir = run_dir / "compare_outputs" / metric_folder
            if compare_outputs_dir.exists():
                with st.expander(f"Preview {param_display} comparison plots", expanded=False):
                    pngs = collect_compare_plot_pngs(compare_outputs_dir)
                    if pngs:
                        cols = st.columns(3)
                        for i, p in enumerate(pngs[:12]):
                            with cols[i % 3]:
                                st.image(str(p), caption=p.name, width="stretch")
                    else:
                        st.caption("No comparison plots were found.")

st.markdown("</div>", unsafe_allow_html=True)

st.write("")

if current_router_dir is not None and rvr_outputs_root is not None:
    render_rvr_apex_dashboard(
        current_router_dir,
        rvr_outputs_root,
        step_label="Step 4A",
        title="Interactive Graph — RvR",
        subtle="Reads the curve table generated by <code>parameter_vs_range.py</code> and renders an interactive RvR chart.",
    )
else:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="card-title">
          <h2><span class="step">Step 4A</span> Interactive Graph — RvR</h2>
        </div>
        <div class="subtle">Choose or load a router folder first.</div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

st.write("")

if compare_outputs_root is not None:
    render_mesh_compare_apex_dashboard(
        compare_outputs_root,
        step_label="Step 4B",
        title="Interactive Graph — Mesh vs No Mesh",
        subtle="Reads the comparison curve table generated by <code>comparison.py</code> and renders an interactive mesh-vs-no-mesh chart.",
    )
else:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="card-title">
          <h2><span class="step">Step 4B</span> Interactive Graph — Mesh vs No Mesh</h2>
        </div>
        <div class="subtle">Choose or load a router folder first.</div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

st.write("")

# ─────────────────────────────────────────────────────────────────────────────
# Step 5 — AI Report Generator
# ─────────────────────────────────────────────────────────────────────────────
streamlit_report_card(
    current_router_dir=current_router_dir,
    rvr_outputs_root=rvr_outputs_root,
    extracted_root=extracted_dir,
    step_label="Step 5",
)

st.write("")
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown(
    """
<div class="card-title">
  <h2><span class="step">Step 7</span> Local Knowledge Base + Chat</h2>
</div>
<div class="subtle">Build a local searchable index from extracted manifests, OCR indexes, and curve tables, then ask natural-language questions about the survey data.</div>
""",
    unsafe_allow_html=True,
)

if current_router_dir is None:
    st.info("Choose or load a router folder first.")
else:
    kb_path = current_router_dir / "survey_kb.json"
    col_a, col_b = st.columns([1, 2])
    with col_a:
        build_kb = st.button("Build / Refresh KB", key="build_kb_btn", width="stretch")
    with col_b:
        question = st.text_input("Ask about coverage, throughput, floors, bands, or router ranking", key="kb_question")

    if build_kb:
        kb = LocalKnowledgeBase.build_from_run_folder(current_router_dir)
        kb.save(kb_path)
        st.success(f"Knowledge base saved with {len(kb.chunks)} chunks.")

    if question:
        if not kb_path.exists():
            kb = LocalKnowledgeBase.build_from_run_folder(current_router_dir)
            kb.save(kb_path)
        kb = LocalKnowledgeBase.load(kb_path)
        answer, chunks = answer_with_context(question, kb)
        st.write(answer)
        if chunks:
            with st.expander("Retrieved context", expanded=False):
                for i, chunk in enumerate(chunks, start=1):
                    st.markdown(f"**[{i}] {chunk.title}**")
                    st.caption(chunk.source_path)
                    st.code(chunk.text[:1200])

st.markdown("</div>", unsafe_allow_html=True)
