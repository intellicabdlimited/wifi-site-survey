import io
import os
import re
import sys
import shutil
import zipfile
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

import streamlit as st

from docx_extractor import process_many_docx_local
from ocr_csv_generator import run_ocr_generate_csv


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
    "Number of APs",
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
    return safe_name(stem.split()[0])


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
    return safe_name(stem.split()[0])


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


# -----------------------------
# Header
# -----------------------------
st.markdown(
    """
<div class="hero">
  <h1>WiFi Site Survey Automation</h1>
  <p>Extract → OCR → Parameter vs Range</p>
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

chips = []
chips.append(f"Router: {router_name}" if router_name else "Router: —")
chips.append(f"Extracted images: {count_files(extracted_dir, exts={'.png','.jpg','.jpeg','.webp','.bmp','.tif','.tiff'})}")
chips.append(f"CSV metrics: {len(discover_metric_dirs(csv_root_dir)) if csv_root_dir and csv_root_dir.exists() else 0}")
chips.append(f"RVR outputs: {len([p for p in (rvr_outputs_root.iterdir() if rvr_outputs_root and rvr_outputs_root.exists() else []) if p.is_dir()])}")

st.markdown(
    '<div class="chips">' + "".join([f'<div class="chip">{c}</div>' for c in chips]) + "</div>",
    unsafe_allow_html=True,
)

st.write("")


# -----------------------------
# Step 1 + Step 2
# -----------------------------
col1, col2 = st.columns(2, gap="large")


# -----------------------------
# Step 1 — DOCX extraction
# -----------------------------
with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(
        """
<div class="card-title">
  <h2><span class="step">Step 1</span> Extract from DOCX</h2>
</div>
<div class="subtle">Writes to <code>runs/&lt;router_name&gt;/extracted</code>. Re-running overwrites this router’s extracted data.</div>
""",
        unsafe_allow_html=True,
    )

    docx_files = st.file_uploader(
        "DOCX file(s)",
        type=["docx"],
        accept_multiple_files=True,
        key="docx_uploader",
    )

    with st.expander("Options", expanded=False):
        also_master_zip = st.checkbox("Create one master ZIP", value=False, key="extract_master_zip")
        offer_extracted_zip = st.checkbox("Offer extracted ZIP download", value=False, key="extract_offer_zip")

    run_extract = st.button("Run extraction", use_container_width=True, key="run_extract_btn")

    if run_extract:
        working_router = router_name or guess_router_from_docx(docx_files)
        if not working_router:
            st.error("Enter a router name or upload at least one DOCX.")
            st.stop()

        if not docx_files:
            st.error("Upload at least one DOCX.")
            st.stop()

        st.session_state["router_name"] = working_router
        run_dir = router_dir(working_router)
        docx_in = reset_dir(run_dir / "docx_inputs")
        extracted = reset_dir(run_dir / "extracted")

        # Clear downstream folders too, so same-router reruns don't keep stale OCR/RvR data
        for downstream in ["csv_outputs", "rvr_inputs", "rvr_outputs"]:
            p = run_dir / downstream
            if p.exists():
                shutil.rmtree(p)

        with st.spinner("Extracting..."):
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
            )

        img_count = count_files(extracted, exts={".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"})
        st.success(f"Extraction complete. Images found: {img_count}")
        st.code(str(extracted))

        if offer_extracted_zip:
            data = zip_folder_bytes(extracted)
            st.download_button(
                "Download extracted.zip",
                data=data,
                file_name=f"{working_router}_extracted.zip",
                mime="application/zip",
                use_container_width=True,
            )

    st.markdown("</div>", unsafe_allow_html=True)


# -----------------------------
# Step 2 — OCR to CSV
# -----------------------------
with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(
        """
<div class="card-title">
  <h2><span class="step">Step 2</span> OCR to CSV</h2>
</div>
<div class="subtle">Reads from <code>runs/&lt;router_name&gt;/extracted</code> and writes grouped CSVs to <code>runs/&lt;router_name&gt;/csv_outputs/&lt;metric&gt;</code>.</div>
""",
        unsafe_allow_html=True,
    )

    extracted_ready = bool(extracted_dir and extracted_dir.exists())
    if not extracted_ready:
        st.info("Choose a router folder and complete Step 1 first.")
        st.button("Run OCR + CSV generation", disabled=True, use_container_width=True, key="disabled_ocr")
    else:
        max_heatmaps = st.number_input("Max heatmaps (0 = no limit)", min_value=0, value=0, step=1, key="ocr_max_heatmaps")

        with st.expander("Options", expanded=False):
            debug = st.checkbox("Debug mode", value=False, key="ocr_debug")
            offer_csv_zip = st.checkbox("Offer CSV ZIP download", value=True, key="ocr_offer_zip")

        run_ocr = st.button("Run OCR + CSV generation", use_container_width=True, key="run_ocr_btn")

        if run_ocr:
            assert current_router_dir is not None
            csv_root = reset_dir(current_router_dir / "csv_outputs")

            with st.spinner("Running OCR..."):
                res = run_ocr_generate_csv(
                    extracted_root=str(current_router_dir / "extracted"),
                    csv_out_root=str(csv_root),
                    max_heatmaps=(None if max_heatmaps == 0 else int(max_heatmaps)),
                    debug=debug,
                )

            metric_summary = organize_csv_outputs(csv_root)
            st.session_state["last_ocr_result"] = res or {}

            st.success(
                f"OCR complete. CSVs: {res.get('processed', 0) if isinstance(res, dict) else 0} | "
                f"Failed: {res.get('failed_count', 0) if isinstance(res, dict) else 0}"
            )
            st.code(str(csv_root))

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

            if offer_csv_zip:
                data = zip_folder_bytes(csv_root)
                st.download_button(
                    "Download csv_outputs.zip",
                    data=data,
                    file_name=f"{router_name}_csv_outputs.zip",
                    mime="application/zip",
                    use_container_width=True,
                )

    st.markdown("</div>", unsafe_allow_html=True)

st.write("")


# -----------------------------
# Step 3 — Parameter vs Range
# -----------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown(
    """
<div class="card-title">
  <h2><span class="step">Step 3</span> Parameter vs Range</h2>
</div>
<div class="subtle">
Uses one selected metric folder from Step 2, plus Master/Router ESX files and optional <code>site_geometry.json</code>.
Outputs go to <code>runs/&lt;router_name&gt;/rvr_outputs/&lt;metric&gt;</code>.
</div>
""",
    unsafe_allow_html=True,
)

if not PARAM_SCRIPT.exists():
    st.error(f"Missing {PARAM_SCRIPT.name} at project root.")
else:
    metric_dirs = discover_metric_dirs(csv_root_dir) if csv_root_dir else []
    metric_folder_for_step3: Optional[str] = None

    st.caption("Upload ESX files, pick the metric to run, then the app prepares a temporary input bundle and launches parameter_vs_range.py.")

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
        "Use CSVs from Step 2 output",
        value=True,
        key="rvr_use_step2_csv",
    )

    if use_csv_from_step2:
        if metric_dirs:
            metric_folder_for_step3 = st.selectbox(
                "Select parameter folder",
                metric_dirs,
                format_func=metric_label,
                key="rvr_metric_folder_select",
            )
        else:
            st.info("No metric folders found yet in csv_outputs. Run Step 2 first.")
    else:
        selected_param_display = st.selectbox(
            "Select parameter",
            ALL_PARAM_DISPLAY_OPTIONS,
            index=0,
            key="manual_param_display",
        )
        metric_folder_for_step3 = PARAM_DISPLAY_TO_FOLDER.get(selected_param_display, "signal_strength")

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

    run_rvr = st.button("Run Parameter vs Range", use_container_width=True, key="run_rvr_btn")

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

        if not metric_folder_for_step3:
            st.error("Choose a parameter folder.")
            st.stop()

        st.session_state["router_name"] = working_router
        run_dir = router_dir(working_router)
        esx_store = ensure_dir(run_dir / "esx_inputs")
        esx_master_store = reset_dir(esx_store / "master")
        esx_router_store = reset_dir(esx_store / "routers")
        metric_rvr_input = reset_dir(run_dir / "rvr_inputs" / metric_folder_for_step3)
        metric_rvr_out = reset_dir(run_dir / "rvr_outputs" / metric_folder_for_step3)

        # Persist ESX files in esx_inputs and copy them into this metric's rvr_inputs
        esx_by_name = {f.name: f for f in esx_files}

        write_uploaded_file(esx_by_name[master_choice], esx_master_store / master_choice)
        write_uploaded_file(esx_by_name[master_choice], metric_rvr_input / master_choice)

        for rn in router_choices:
            write_uploaded_file(esx_by_name[rn], esx_router_store / rn)
            write_uploaded_file(esx_by_name[rn], metric_rvr_input / rn)

        # Site geometry: upload wins, otherwise reuse existing router-level file if present
        router_site_geom = run_dir / "site_geometry.json"
        if site_geom:
            write_uploaded_file(site_geom, router_site_geom)
            shutil.copy2(router_site_geom, metric_rvr_input / "site_geometry.json")
        elif router_site_geom.exists():
            shutil.copy2(router_site_geom, metric_rvr_input / "site_geometry.json")

        copied_csv_count = 0
        if use_csv_from_step2:
            src_metric_dir = run_dir / "csv_outputs" / metric_folder_for_step3
            if not src_metric_dir.exists():
                st.error(f"No CSV folder found for {metric_label(metric_folder_for_step3)}.")
                st.stop()
            copied_csv_count = copy_metric_csvs(src_metric_dir, metric_rvr_input, limit=int(copy_csv_limit))
            if copied_csv_count == 0:
                st.error("Selected metric folder contains no CSVs.")
                st.stop()
        else:
            if not uploaded_csvs:
                st.error("Upload at least one CSV.")
                st.stop()
            for f in uploaded_csvs:
                write_uploaded_file(f, metric_rvr_input / f.name)
                copied_csv_count += 1

        param_display = metric_label(metric_folder_for_step3)

        # Temporary patched runner so the original parameter_vs_range.py remains unchanged
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

        st.info(f"Running: {' '.join(cmd)}")
        st.caption(f"Router folder  = {run_dir}")
        st.caption(f"Parameter      = {param_display}")
        st.caption(f"RVR_INPUT_DIR  = {metric_rvr_input}")
        st.caption(f"RVR_OUT_BASE   = {metric_rvr_out}")
        st.caption(f"Master ESX     = {master_choice}")
        st.caption(f"Router ESX     = {len(router_choices)} file(s)")
        st.caption(f"CSVs provided  = {copied_csv_count}")

        with st.spinner("Running parameter_vs_range.py..."):
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

        ok = (proc.returncode == 0)
        st.session_state["last_rvr_result"] = {
            "ok": ok,
            "returncode": proc.returncode,
            "log_path": str(log_path),
            "rvr_input": str(metric_rvr_input),
            "rvr_output": str(metric_rvr_out),
            "parameter": param_display,
        }

        final_zip = intended_zip if intended_zip.exists() else None
        if final_zip is None:
            fallback_zip = run_dir / f"{metric_folder_for_step3}_rvr_outputs.zip"
            final_zip = zip_to_path(metric_rvr_out, fallback_zip)

        if ok:
            st.success("RvR completed successfully.")
        else:
            st.error(f"RvR failed (exit code {proc.returncode}). Check logs below.")

        with st.expander("Logs", expanded=not ok):
            shown = log_text if show_full_logs else (log_text[-8000:] if len(log_text) > 8000 else log_text)
            st.code(shown)

        if offer_rvr_zip and final_zip and final_zip.exists():
            st.download_button(
                "Download RvR ZIP",
                data=final_zip.read_bytes(),
                file_name=final_zip.name,
                mime="application/zip",
                use_container_width=True,
            )

        pngs = collect_plot_pngs(metric_rvr_out)
        if pngs:
            with st.expander("Preview plots", expanded=False):
                show = pngs[:12]
                cols = st.columns(3)
                for i, p in enumerate(show):
                    with cols[i % 3]:
                        st.image(str(p), caption=p.name, use_container_width=True)
        else:
            st.caption("No plot previews were found yet in plots_percent / plots_actual.")

st.markdown("</div>", unsafe_allow_html=True)


# -----------------------------
# Session tools
# -----------------------------
st.write("")
with st.expander("Session tools", expanded=False):
    if st.button("Clear session state", use_container_width=True):
        for key in ["router_name", "last_ocr_result", "last_rvr_result"]:
            st.session_state.pop(key, None)
        st.rerun()