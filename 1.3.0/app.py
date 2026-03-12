import io
import os
import re
import sys
import shutil
import zipfile
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import streamlit as st

from docx_extractor import process_many_docx_local
from ocr_csv_generator import run_ocr_generate_csv
from local_ai import (
    LocalAIConfig,
    DEFAULT_LOCAL_AI_API_KEY,
    DEFAULT_LOCAL_AI_BASE_URL,
    DEFAULT_LOCAL_AI_EXTRA_INSTRUCTIONS,
    DEFAULT_LOCAL_AI_MAX_TOKENS,
    DEFAULT_LOCAL_AI_MODEL,
    DEFAULT_LOCAL_AI_PROVIDER,
    DEFAULT_LOCAL_AI_TEMPERATURE,
    DEFAULT_LOCAL_AI_TIMEOUT_SEC,
    build_knowledge_base,
    chat_with_knowledge_base,
    generate_metric_graph_reports,
    generate_router_overall_report,
    probe_local_ai,
    warm_local_ai,
)


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
  margin-bottom: 1.2rem;
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


def safe_name(text: str) -> str:
    text = (text or "").strip()
    text = Path(text).stem
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("._-")
    return text or "router"


def normalize_text(text: str) -> str:
    text = (text or "").replace("\u00A0", " ").strip().lower()
    text = text.replace("–", "-").replace("—", "-")
    text = re.sub(r"[_\\-]+", " ", text)
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
    if "_" in stem:
        _, rest = stem.split("_", 1)
    else:
        rest = stem
    metric_text = rest.split(" for ", 1)[0]
    return canonical_metric_folder(metric_text)


def organize_csv_outputs(csv_root: Path) -> Dict[str, int]:
    ensure_dir(csv_root)
    all_csvs = [p for p in csv_root.rglob("*.csv") if p.is_file()]
    for csv_file in all_csvs:
        if csv_file.parent != csv_root:
            continue
        if csv_file.name.startswith("_"):
            continue
        metric_folder = metric_folder_from_csv_name(csv_file.name)
        if not metric_folder:
            continue
        target_dir = ensure_dir(csv_root / metric_folder)
        shutil.copy2(csv_file, target_dir / csv_file.name)

    summary: Dict[str, int] = {}
    for d in sorted([p for p in csv_root.iterdir() if p.is_dir()]):
        n = len([p for p in d.glob("*.csv") if p.is_file()])
        if n > 0:
            summary[d.name] = n
    return summary


def discover_metric_dirs(csv_root: Optional[Path]) -> List[str]:
    if not csv_root or not csv_root.exists():
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


def patch_parameter_script(source_script: Path, patched_script: Path, param_display: str, input_dir: Path, out_dir: Path, zip_path: Path) -> Path:
    text = source_script.read_text(encoding="utf-8", errors="ignore")
    text = re.sub(r"(?m)^PARAM_NAME\s*=.*$", f"PARAM_NAME = {param_display!r}", text, count=1)
    text = re.sub(r"(?m)^INPUT_DIR\s*=.*$", f"INPUT_DIR = {str(input_dir)!r}", text, count=1)
    text = re.sub(r"(?m)^OUT_BASE\s*=.*$", f"OUT_BASE = {str(out_dir)!r}", text, count=1)
    text = re.sub(r"(?m)^ZIP_PATH\s*=.*$", f"ZIP_PATH = {str(zip_path)!r}", text, count=1)
    patched_script.parent.mkdir(parents=True, exist_ok=True)
    patched_script.write_text(text, encoding="utf-8")
    return patched_script


def copy_uploaded_metric_csvs(uploaded_files, dst_dir: Path, target_metric: str, limit: int = 0) -> int:
    ensure_dir(dst_dir)
    matched = []
    for f in uploaded_files or []:
        if metric_folder_from_csv_name(f.name) == target_metric:
            matched.append(f)
    if limit and limit > 0:
        matched = matched[:limit]
    for f in matched:
        write_uploaded_file(f, dst_dir / f.name)
    return len(matched)


def zip_selected_paths(paths: List[Path], zip_path: Path, root_dir: Optional[Path] = None) -> Path:
    if zip_path.exists():
        zip_path.unlink()

    def _arcname(p: Path) -> str:
        if root_dir is not None:
            try:
                return str(p.relative_to(root_dir))
            except Exception:
                pass
        return p.name

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in paths:
            if not path.exists():
                continue
            if path.is_file():
                zf.write(path, arcname=_arcname(path))
                continue
            for p in path.rglob("*"):
                if p.is_file():
                    zf.write(p, arcname=_arcname(p))
    return zip_path


def metric_output_dirs(rvr_outputs_root: Optional[Path]) -> List[str]:
    if not rvr_outputs_root or not rvr_outputs_root.exists():
        return []
    return sorted([p.name for p in rvr_outputs_root.iterdir() if p.is_dir()])


def build_ai_cfg_from_state() -> LocalAIConfig:
    return LocalAIConfig(
        provider=DEFAULT_LOCAL_AI_PROVIDER,
        model=DEFAULT_LOCAL_AI_MODEL,
        base_url=DEFAULT_LOCAL_AI_BASE_URL,
        api_key=DEFAULT_LOCAL_AI_API_KEY,
        temperature=DEFAULT_LOCAL_AI_TEMPERATURE,
        max_tokens=DEFAULT_LOCAL_AI_MAX_TOKENS,
        timeout_sec=DEFAULT_LOCAL_AI_TIMEOUT_SEC,
        extra_instructions=st.session_state.get(
            "local_ai_extra",
            DEFAULT_LOCAL_AI_EXTRA_INSTRUCTIONS,
        ),
    )


st.markdown(
    """
<div class="hero">
  <h1>WiFi Site Survey Automation</h1>
  <p>Extract → OCR → Parameter vs Range → Local AI Graph Reasoning → Local KB Chat</p>
</div>
""",
    unsafe_allow_html=True,
)

existing_router_dirs = sorted([p.name for p in RUNS_DIR.iterdir() if p.is_dir()])
left, right = st.columns([2, 2])
with left:
    selected_existing_router = st.selectbox("Load existing router folder (optional)", [""] + existing_router_dirs, index=0)
with right:
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
ai_reports_root = current_router_dir / "ai_reports" if current_router_dir else None

chips = [
    f"Router: {router_name}" if router_name else "Router: —",
    f"Extracted images: {count_files(extracted_dir, exts={'.png','.jpg','.jpeg','.webp','.bmp','.tif','.tiff'})}",
    f"CSV metrics: {len(discover_metric_dirs(csv_root_dir))}",
    f"RVR metrics: {len(metric_output_dirs(rvr_outputs_root))}",
    f"AI files: {count_files(ai_reports_root)}",
]
st.markdown('<div class="chips">' + ''.join([f'<div class="chip">{c}</div>' for c in chips]) + '</div>', unsafe_allow_html=True)

st.write("")
col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(
        """
<div class="card-title"><h2><span class="step">Step 1</span> Extract from DOCX</h2></div>
<div class="subtle">Writes to <code>runs/&lt;router_name&gt;/extracted</code>. Re-running overwrites this router’s extracted data.</div>
""",
        unsafe_allow_html=True,
    )
    docx_files = st.file_uploader("DOCX file(s)", type=["docx"], accept_multiple_files=True, key="docx_uploader")
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
        for downstream in ["csv_outputs", "rvr_inputs", "rvr_outputs", "ai_reports", "_generated"]:
            p = run_dir / downstream
            if p.exists():
                shutil.rmtree(p)
        with st.spinner("Extracting..."):
            docx_paths: List[str] = []
            for f in docx_files:
                out_path = docx_in / f.name
                out_path.write_bytes(f.getbuffer())
                docx_paths.append(str(out_path))
            process_many_docx_local(docx_paths, out_root=str(extracted), download_per_docx_zip=False, also_make_master_zip=also_master_zip)
        img_count = count_files(extracted, exts={".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"})
        st.success(f"Extraction complete. Images found: {img_count}")
        st.code(str(extracted))
        if offer_extracted_zip:
            st.download_button(
                "Download extracted.zip",
                data=zip_folder_bytes(extracted),
                file_name=f"{working_router}_extracted.zip",
                mime="application/zip",
                use_container_width=True,
            )
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(
        """
<div class="card-title"><h2><span class="step">Step 2</span> OCR to CSV</h2></div>
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
            if (current_router_dir / "rvr_inputs").exists():
                shutil.rmtree(current_router_dir / "rvr_inputs")
            if (current_router_dir / "rvr_outputs").exists():
                shutil.rmtree(current_router_dir / "rvr_outputs")
            if (current_router_dir / "ai_reports").exists():
                shutil.rmtree(current_router_dir / "ai_reports")
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
                st.download_button(
                    "Download csv_outputs.zip",
                    data=zip_folder_bytes(csv_root),
                    file_name=f"{router_name}_csv_outputs.zip",
                    mime="application/zip",
                    use_container_width=True,
                )
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown(
    """
<div class="card-title"><h2><span class="step">Step 3</span> Parameter vs Range</h2></div>
<div class="subtle">This step only generates the comparative graphs and supporting tables. Local AI is handled separately in the next steps.</div>
""",
    unsafe_allow_html=True,
)

if not PARAM_SCRIPT.exists():
    st.error(f"Missing {PARAM_SCRIPT.name} at project root.")
else:
    metric_dirs = discover_metric_dirs(csv_root_dir)
    selected_metric_folders: List[str] = []
    st.caption("Upload the master floor-plan ESX, router ESX files, then choose one or more parameters. The app runs parameter_vs_range.py once per selected metric.")
    esx_files = st.file_uploader("Upload ESX file(s)", type=["esx"], accept_multiple_files=True, key="rvr_esx_all")
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
    use_csv_from_step2 = st.checkbox("Use CSVs from Step 2 output", value=True, key="rvr_use_step2_csv")
    if use_csv_from_step2:
        if metric_dirs:
            default_metrics = metric_dirs[: min(3, len(metric_dirs))]
            selected_metric_folders = st.multiselect(
                "Select parameter folder(s)",
                metric_dirs,
                default=default_metrics,
                format_func=metric_label,
                key="rvr_metric_folder_multi",
            )
        else:
            st.info("No metric folders found yet in csv_outputs. Run Step 2 first.")
    else:
        selected_param_displays = st.multiselect("Select parameter(s)", ALL_PARAM_DISPLAY_OPTIONS, default=ALL_PARAM_DISPLAY_OPTIONS[:1], key="manual_param_display_multi")
        selected_metric_folders = [PARAM_DISPLAY_TO_FOLDER.get(name, "signal_strength") for name in selected_param_displays]
    uploaded_csvs = None
    if not use_csv_from_step2:
        uploaded_csvs = st.file_uploader("Upload band_output CSV(s)", type=["csv"], accept_multiple_files=True, key="rvr_csvs")
    site_geom = st.file_uploader("site_geometry.json (optional)", type=["json"], accept_multiple_files=False, key="rvr_site_geom")
    with st.expander("Run options", expanded=False):
        copy_csv_limit = st.number_input("Copy max CSVs per selected metric folder (0 = no limit)", min_value=0, value=0, step=1, key="rvr_csv_limit")
        show_full_logs = st.checkbox("Show full logs", value=False, key="rvr_show_logs")
        offer_rvr_zip = st.checkbox("Offer batch ZIP download", value=True, key="rvr_offer_zip")
    run_rvr = st.button("Run selected parameters", use_container_width=True, key="run_rvr_btn")
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
        if not selected_metric_folders:
            st.error("Choose at least one parameter folder.")
            st.stop()
        if not use_csv_from_step2 and not uploaded_csvs:
            st.error("Upload at least one CSV when Step 2 outputs are not used.")
            st.stop()
        st.session_state["router_name"] = working_router
        run_dir = router_dir(working_router)
        esx_store = ensure_dir(run_dir / "esx_inputs")
        esx_master_store = reset_dir(esx_store / "master")
        esx_router_store = reset_dir(esx_store / "routers")
        if (run_dir / "ai_reports").exists():
            shutil.rmtree(run_dir / "ai_reports")
        esx_by_name = {f.name: f for f in esx_files}
        write_uploaded_file(esx_by_name[master_choice], esx_master_store / master_choice)
        for rn in router_choices:
            write_uploaded_file(esx_by_name[rn], esx_router_store / rn)
        router_site_geom = run_dir / "site_geometry.json"
        if site_geom:
            write_uploaded_file(site_geom, router_site_geom)
        progress = st.progress(0.0)
        status_box = st.empty()
        batch_results: List[Dict[str, object]] = []
        for idx, metric_folder in enumerate(selected_metric_folders, start=1):
            param_display = metric_label(metric_folder)
            status_box.info(f"Running {idx}/{len(selected_metric_folders)}: {param_display}")
            metric_rvr_input = reset_dir(run_dir / "rvr_inputs" / metric_folder)
            metric_rvr_out = reset_dir(run_dir / "rvr_outputs" / metric_folder)
            shutil.copy2(esx_master_store / master_choice, metric_rvr_input / master_choice)
            for rn in router_choices:
                src_router = esx_router_store / rn
                if src_router.exists():
                    shutil.copy2(src_router, metric_rvr_input / rn)
            if router_site_geom.exists():
                shutil.copy2(router_site_geom, metric_rvr_input / "site_geometry.json")
            copied_csv_count = 0
            if use_csv_from_step2:
                src_metric_dir = run_dir / "csv_outputs" / metric_folder
                if src_metric_dir.exists():
                    copied_csv_count = copy_metric_csvs(src_metric_dir, metric_rvr_input, limit=int(copy_csv_limit))
            else:
                copied_csv_count = copy_uploaded_metric_csvs(uploaded_csvs, metric_rvr_input, metric_folder, limit=int(copy_csv_limit))
            if copied_csv_count == 0:
                batch_results.append({
                    "metric_folder": metric_folder,
                    "parameter": param_display,
                    "ok": False,
                    "returncode": None,
                    "copied_csv_count": 0,
                    "error": "No CSVs matched this metric.",
                    "log_text": "",
                    "metric_output": str(metric_rvr_out),
                })
                progress.progress(idx / len(selected_metric_folders))
                continue
            generated_dir = ensure_dir(run_dir / "_generated")
            patched_param_script = generated_dir / f"parameter_vs_range__{metric_folder}.py"
            intended_zip = run_dir / f"rvr_full_output_{metric_folder}.zip"
            patch_parameter_script(PARAM_SCRIPT, patched_param_script, param_display, metric_rvr_input, metric_rvr_out, intended_zip)
            log_path = run_dir / f"rvr_run_{metric_folder}.log"
            env = os.environ.copy()
            env["RVR_INPUT_DIR"] = str(metric_rvr_input)
            env["RVR_OUT_BASE"] = str(metric_rvr_out)
            env["RVR_PARAM_NAME"] = param_display
            proc = subprocess.run([sys.executable, str(patched_param_script)], cwd=str(Path.cwd()), env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            log_text = "===== STDOUT =====\n" + (proc.stdout or "") + "\n\n===== STDERR =====\n" + (proc.stderr or "")
            log_path.write_text(log_text, encoding="utf-8")
            final_zip = intended_zip if intended_zip.exists() else zip_to_path(metric_rvr_out, run_dir / f"{metric_folder}_rvr_outputs.zip")
            batch_results.append({
                "metric_folder": metric_folder,
                "parameter": param_display,
                "ok": proc.returncode == 0,
                "returncode": proc.returncode,
                "copied_csv_count": copied_csv_count,
                "log_path": str(log_path),
                "log_text": log_text,
                "metric_input": str(metric_rvr_input),
                "metric_output": str(metric_rvr_out),
                "final_zip": str(final_zip) if final_zip and final_zip.exists() else "",
            })
            progress.progress(idx / len(selected_metric_folders))
        status_box.empty()
        progress.empty()
        st.session_state["last_rvr_result"] = {"router_name": working_router, "batch_results": batch_results}
        ok_count = sum(1 for r in batch_results if r.get("ok"))
        fail_count = len(batch_results) - ok_count
        if ok_count:
            st.success(f"Completed {ok_count} parameter run(s). Failed: {fail_count}.")
        else:
            st.error("No selected parameter completed successfully.")
        if batch_results:
            st.dataframe(pd.DataFrame([
                {
                    "Parameter": r.get("parameter"),
                    "Status": "OK" if r.get("ok") else "Failed",
                    "CSV files": r.get("copied_csv_count"),
                    "Return code": r.get("returncode"),
                }
                for r in batch_results
            ]), use_container_width=True, hide_index=True)
        bundle_paths = [run_dir / "rvr_outputs"]
        for r in batch_results:
            if r.get("log_path"):
                bundle_paths.append(Path(str(r.get("log_path"))))
        if offer_rvr_zip:
            batch_zip = zip_selected_paths(bundle_paths, run_dir / f"rvr_batch_bundle_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip", root_dir=run_dir)
            if batch_zip.exists():
                st.download_button("Download batch ZIP", data=batch_zip.read_bytes(), file_name=batch_zip.name, mime="application/zip", use_container_width=True)
        for result_item in batch_results:
            title = f"{result_item.get('parameter')} — {'OK' if result_item.get('ok') else 'Failed'}"
            with st.expander(title, expanded=not bool(result_item.get("ok"))):
                st.caption(f"Input folder: {result_item.get('metric_input', '—')}")
                st.caption(f"Output folder: {result_item.get('metric_output', '—')}")
                final_zip_str = str(result_item.get("final_zip", ""))
                if final_zip_str and Path(final_zip_str).exists():
                    p = Path(final_zip_str)
                    st.download_button(f"Download {result_item.get('parameter')} ZIP", data=p.read_bytes(), file_name=p.name, mime="application/zip", use_container_width=True, key=f"download_{result_item.get('metric_folder')}")
                pngs = collect_plot_pngs(Path(str(result_item.get("metric_output"))))
                if pngs:
                    cols = st.columns(3)
                    for i, p in enumerate(pngs[:9]):
                        with cols[i % 3]:
                            st.image(str(p), caption=p.name, use_container_width=True)
                shown = str(result_item.get("log_text", ""))
                if not show_full_logs and len(shown) > 8000:
                    shown = shown[-8000:]
                if shown:
                    st.code(shown)
st.markdown("</div>", unsafe_allow_html=True)

st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown(
    """
<div class="card-title"><h2><span class="step">Step 4</span> Local AI graph summaries</h2></div>
<div class="subtle">This step reads the generated graph outputs, creates one report per graph with its graph embedded, one metric-level summary per parameter, and one consolidated overall report across the selected routers, floors, and bands.</div>
""",
    unsafe_allow_html=True,
)

available_rvr_metrics = metric_output_dirs(rvr_outputs_root)
if not router_name or not current_router_dir:
    st.info("Choose or create a router folder first.")
else:
    st.caption("Local AI runtime settings are controlled internally in local_ai.py. This step generates one reasoning summary for each graph and stores the outputs under ai_reports.")

    cfg = build_ai_cfg_from_state()

    st.info("Temperature, max output tokens, timeout, API key, provider, model, and base URL are controlled internally for end-user simplicity.")

    st.code(
        "\n".join(
            [
                f"Provider      : {cfg.provider}",
                f"Base URL      : {cfg.base_url}",
                f"Model         : {cfg.model}",
                f"Temperature   : {cfg.temperature}",
                f"Max tokens    : {cfg.max_tokens}",
                f"Timeout       : {'wait indefinitely' if cfg.timeout_sec is None else f'{cfg.timeout_sec} seconds'}",
                f"API key       : {'set' if cfg.api_key else 'blank'}",
            ]
        )
    )

    st.text_area(
        "Extra instructions for local AI",
        value=st.session_state.get(
            "local_ai_extra",
            DEFAULT_LOCAL_AI_EXTRA_INSTRUCTIONS,
        ),
        height=120,
        key="local_ai_extra",
    )
    cfg = build_ai_cfg_from_state()

    col_a, col_b = st.columns([1, 2])
    with col_a:
        if st.button("Test local AI connection", key="local_ai_test_btn", use_container_width=True):
            ok, msg = probe_local_ai(cfg)
            if ok:
                st.success(msg)
            else:
                st.error(msg)
    with col_b:
        st.caption("Recommended for Ollama: run `ollama serve` and pull your model before using Step 4 or Step 5.")

    if available_rvr_metrics:
        default_ai_metrics = available_rvr_metrics[: min(3, len(available_rvr_metrics))]
        ai_metric_selection = st.multiselect(
            "Select generated metric outputs to summarize",
            options=available_rvr_metrics,
            default=default_ai_metrics,
            format_func=metric_label,
            key="ai_metric_selection",
        )
        offer_ai_zip = st.checkbox("Offer AI reports ZIP download", value=True, key="offer_ai_zip")
        run_ai = st.button("Run local AI graph summaries", use_container_width=True, key="run_local_ai_graphs")
        if run_ai:
            if not ai_metric_selection:
                st.error("Choose at least one generated metric output.")
                st.stop()

            ok, msg = probe_local_ai(cfg)
            if not ok:
                st.error(msg)
                st.stop()

            try:
                warm_local_ai(cfg)
                st.success("Local model preloaded and kept warm for the batch run.")
            except Exception as exc:
                st.warning(f"Could not preload the local model: {type(exc).__name__}: {exc}. Continuing anyway.")

            ai_root = reset_dir(current_router_dir / "ai_reports")
            progress = st.progress(0.0)
            status = st.empty()
            ai_batch_results: List[Dict[str, object]] = []
            for idx, metric_folder in enumerate(ai_metric_selection, start=1):
                status.info(f"Summarizing {idx}/{len(ai_metric_selection)}: {metric_label(metric_folder)}")
                metric_out = current_router_dir / "rvr_outputs" / metric_folder
                metric_ai_root = ai_root / metric_folder
                try:
                    result = generate_metric_graph_reports(
                        router_name=router_name,
                        metric_display=metric_label(metric_folder),
                        metric_output_dir=metric_out,
                        ai_output_root=metric_ai_root,
                        cfg=cfg,
                    )
                    ai_batch_results.append({"metric_folder": metric_folder, "ok": True, "result": result})
                except Exception as exc:
                    ai_batch_results.append({"metric_folder": metric_folder, "ok": False, "error": f"{type(exc).__name__}: {exc}"})
                progress.progress(idx / len(ai_metric_selection))
            overall_report = None
            ok_results = [item.get("result") for item in ai_batch_results if item.get("ok") and item.get("result")]
            if ok_results:
                status.info("Building consolidated overall report...")
                try:
                    overall_report = generate_router_overall_report(
                        router_name=router_name,
                        metric_results=ok_results,
                        ai_output_root=ai_root / "_overall",
                        cfg=cfg,
                    )
                except Exception as exc:
                    overall_report = {"error": f"{type(exc).__name__}: {exc}"}

            progress.empty()
            status.empty()
            st.session_state["last_ai_result"] = {"router_name": router_name, "batch_results": ai_batch_results, "overall_report": overall_report}
            ok_count = sum(1 for item in ai_batch_results if item.get("ok"))
            st.success(f"Local AI finished. Successful metrics: {ok_count}/{len(ai_batch_results)}")
            summary_df = []
            for item in ai_batch_results:
                graph_count = 0
                if item.get("ok") and item.get("result"):
                    graph_count = len(item["result"].get("graph_reports", []))
                summary_df.append({
                    "Metric": metric_label(str(item.get("metric_folder"))),
                    "Status": "OK" if item.get("ok") else "Failed",
                    "Graph summaries": graph_count,
                })
            st.dataframe(pd.DataFrame(summary_df), use_container_width=True, hide_index=True)
            if overall_report and overall_report.get("text"):
                with st.expander("Overall consolidated report", expanded=True):
                    st.markdown(str(overall_report.get("text")))
                    overall_docx = Path(str(overall_report.get("files", {}).get("docx", "")))
                    if overall_docx.exists():
                        st.download_button(
                            "Download overall report DOCX",
                            data=overall_docx.read_bytes(),
                            file_name=overall_docx.name,
                            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                            use_container_width=True,
                        )
            if offer_ai_zip:
                ai_zip = zip_selected_paths(
                    [ai_root],
                    current_router_dir / f"ai_reports_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                    root_dir=current_router_dir,
                )
                st.download_button("Download AI reports ZIP", data=ai_zip.read_bytes(), file_name=ai_zip.name, mime="application/zip", use_container_width=True)
            for item in ai_batch_results:
                title = f"{metric_label(str(item.get('metric_folder')))} — {'OK' if item.get('ok') else 'Failed'}"
                with st.expander(title, expanded=not bool(item.get("ok"))):
                    if not item.get("ok"):
                        st.error(str(item.get("error")))
                        continue
                    result = item["result"]
                    overview = result.get("overview")
                    if overview and overview.get("text"):
                        st.markdown(str(overview.get("text")))
                        overview_docx = Path(str(overview.get("files", {}).get("docx", "")))
                        if overview_docx.exists():
                            st.download_button(
                                f"Download {metric_label(str(item.get('metric_folder')))} overview DOCX",
                                data=overview_docx.read_bytes(),
                                file_name=overview_docx.name,
                                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                                use_container_width=True,
                                key=f"overview_docx_{item.get('metric_folder')}",
                            )
                    graph_reports = result.get("graph_reports", [])
                    st.caption(f"Graph summaries generated: {len(graph_reports)}")
                    for graph_item in graph_reports[:8]:
                        with st.expander(graph_item.get("plot_name", "Graph"), expanded=False):
                            st.markdown(str(graph_item.get("text", "")))
                            src_plot = Path(str(graph_item.get("plot_path", "")))
                            if src_plot.exists():
                                st.image(str(src_plot), caption=src_plot.name, use_container_width=True)
                            graph_docx = Path(str(graph_item.get("files", {}).get("docx", "")))
                            if graph_docx.exists():
                                st.download_button(
                                    f"Download {graph_item.get('plot_name', 'graph')} DOCX",
                                    data=graph_docx.read_bytes(),
                                    file_name=graph_docx.name,
                                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                                    use_container_width=True,
                                    key=f"graph_docx_{item.get('metric_folder')}_{graph_item.get('plot_name')}",
                                )
    else:
        st.info("No parameter-vs-range outputs found yet. Complete Step 3 first.")
st.markdown("</div>", unsafe_allow_html=True)

st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown(
    """
<div class="card-title"><h2><span class="step">Step 5</span> Local AI knowledge-base chat</h2></div>
<div class="subtle">Ask questions against the local knowledge base built from your generated AI reports, graph outputs, logs, and CSV summaries.</div>
""",
    unsafe_allow_html=True,
)

if not router_name or not current_router_dir:
    st.info("Choose or create a router folder first.")
else:
    kb_key = f"kb_docs_{router_name}"
    hist_key = f"kb_chat_history_{router_name}"
    if hist_key not in st.session_state:
        st.session_state[hist_key] = []
    kb_count = len(st.session_state.get(kb_key, []))
    top1, top2, top3 = st.columns([1, 1, 2])
    with top1:
        if st.button("Build or refresh knowledge base", use_container_width=True, key="build_kb_btn"):
            st.session_state[kb_key] = build_knowledge_base(current_router_dir)
            kb_count = len(st.session_state[kb_key])
            st.success(f"Knowledge base ready. Chunks: {kb_count}")
    with top2:
        if st.button("Clear chat", use_container_width=True, key="clear_kb_chat_btn"):
            st.session_state[hist_key] = []
            st.rerun()
    with top3:
        st.caption(f"Knowledge-base chunks in memory: {kb_count}")
    question = st.text_area("Ask about the generated outputs", value="", height=110, key="kb_question")
    ask_kb = st.button("Ask local AI", use_container_width=True, key="ask_kb_btn")
    if ask_kb:
        if not question.strip():
            st.error("Enter a question first.")
            st.stop()
        cfg = build_ai_cfg_from_state()
        ok, msg = probe_local_ai(cfg)
        if not ok:
            st.error(msg)
            st.stop()
        try:
            warm_local_ai(cfg)
        except Exception:
            pass
        if kb_key not in st.session_state or not st.session_state.get(kb_key):
            st.session_state[kb_key] = build_knowledge_base(current_router_dir)
        history = st.session_state.get(hist_key, [])
        with st.spinner("Querying local knowledge base..."):
            chat_result = chat_with_knowledge_base(
                router_name=router_name,
                router_dir=current_router_dir,
                question=question,
                cfg=cfg,
                history=history,
                kb_docs=st.session_state.get(kb_key, []),
            )
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": str(chat_result.get("answer", "")), "sources": chat_result.get("sources", [])})
        st.session_state[hist_key] = history
    history = st.session_state.get(hist_key, [])
    if history:
        for idx, msg in enumerate(history):
            if msg.get("role") == "user":
                with st.chat_message("user"):
                    st.markdown(str(msg.get("content", "")))
            else:
                with st.chat_message("assistant"):
                    st.markdown(str(msg.get("content", "")))
                    sources = msg.get("sources", []) or []
                    if sources:
                        st.caption("Sources used: " + ", ".join(map(str, sources[:8])))
    else:
        st.info("Build the knowledge base, then ask about floors, bands, best/worst routers, graph reasoning, missing data, or anything grounded in the generated outputs.")
st.markdown("</div>", unsafe_allow_html=True)

st.write("")
with st.expander("Session tools", expanded=False):
    if st.button("Clear session state", use_container_width=True):
        for key in list(st.session_state.keys()):
            if key.startswith("kb_docs_") or key.startswith("kb_chat_history_"):
                st.session_state.pop(key, None)
        for key in [
            "router_name",
            "last_ocr_result",
            "last_rvr_result",
            "last_ai_result",
            "local_ai_extra",
        ]:
            st.session_state.pop(key, None)
        st.rerun()