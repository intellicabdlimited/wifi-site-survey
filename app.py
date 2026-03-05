import io
import os
import sys
import shutil
import zipfile
import subprocess
from datetime import datetime
from uuid import uuid4
from pathlib import Path

import streamlit as st

from docx_extractor import process_many_docx_local
from ocr_csv_generator import run_ocr_generate_csv

# -----------------------------
# Page setup + minimal styling
# -----------------------------
st.set_page_config(page_title="WiFi Site Survey Automation", layout="wide")

st.markdown(
    """
<style>
/* Layout */
div[data-testid="stAppViewContainer"] { background: #0b1220; }
.block-container { padding-top: 1.6rem; padding-bottom: 2.2rem; max-width: 1200px; }

/* Hide Streamlit chrome (optional) */
header[data-testid="stHeader"] { background: transparent; }
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }

/* Hero */
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

/* Cards */
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

/* Step badge */
.step {
  display:inline-flex; align-items:center; justify-content:center;
  min-width: 32px; height: 32px; padding: 0 10px;
  border-radius: 999px;
  background: rgba(255,255,255,0.10);
  border: 1px solid rgba(255,255,255,0.14);
  color: rgba(255,255,255,0.88);
  font-size: 0.88rem;
}

/* Metric chips */
.chips { display:flex; flex-wrap:wrap; gap: 10px; margin-top: 0.75rem; }
.chip {
  background: rgba(255,255,255,0.07);
  border: 1px solid rgba(255,255,255,0.10);
  color: rgba(255,255,255,0.80);
  border-radius: 999px;
  padding: 8px 12px;
  font-size: 0.9rem;
}

/* Inputs */
div[data-testid="stFileUploaderDropzone"] {
  border-radius: 16px !important;
  border: 1px dashed rgba(255,255,255,0.18) !important;
  background: rgba(255,255,255,0.05) !important;
}
label, .stMarkdown, .stText, .stCaption { color: rgba(255,255,255,0.82) !important; }

/* Buttons */
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

/* Code blocks */
pre {
  background: rgba(255,255,255,0.05) !important;
  border: 1px solid rgba(255,255,255,0.10) !important;
  border-radius: 14px !important;
}

/* Expanders */
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
# Paths + helpers
# -----------------------------
RUNS_DIR = Path("runs")
RUNS_DIR.mkdir(exist_ok=True)

PARAM_SCRIPT = Path("parameter_vs_range.py")  # <- your big RvR script here

def zip_folder(folder: Path) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in folder.rglob("*"):
            if p.is_file():
                z.write(p, arcname=str(p.relative_to(folder)))
    buf.seek(0)
    return buf.read()

def count_files(folder: Path, exts=None) -> int:
    if not folder or not folder.exists():
        return 0
    if not exts:
        return sum(1 for p in folder.rglob("*") if p.is_file())
    exts = {e.lower() for e in exts}
    return sum(1 for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in exts)

def ensure_run_dir() -> Path:
    last_run_dir = st.session_state.get("last_run_dir")
    if last_run_dir:
        rd = Path(last_run_dir)
        rd.mkdir(parents=True, exist_ok=True)
        return rd
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid4().hex[:6]
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    st.session_state["last_run_dir"] = str(run_dir)
    return run_dir

def safe_write_uploaded_file(uploaded_file, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(uploaded_file.getbuffer())

def copy_csvs(src_dir: Path, dst_dir: Path, limit: int = 0) -> int:
    dst_dir.mkdir(parents=True, exist_ok=True)
    all_csvs = sorted([p for p in src_dir.rglob("*.csv") if p.is_file()])
    if limit and limit > 0:
        all_csvs = all_csvs[:limit]
    n = 0
    for p in all_csvs:
        shutil.copy2(p, dst_dir / p.name)
        n += 1
    return n

def zip_to_path(folder: Path, zip_path: Path):
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in folder.rglob("*"):
            if p.is_file():
                z.write(p, arcname=str(p.relative_to(folder)))

# -----------------------------
# Header + run picker
# -----------------------------
st.markdown(
    """
<div class="hero">
  <h1>WiFi Site Survey Automation</h1>
  <p>Extract → OCR → Parameter vs Range (on one page).</p>
</div>
""",
    unsafe_allow_html=True,
)

runs = sorted([p for p in RUNS_DIR.iterdir() if p.is_dir()], reverse=True)
run_options = ["Current session"] + [p.name for p in runs]

top_left, top_right = st.columns([3, 2])
with top_right:
    selected_run = st.selectbox("Load a previous run (optional)", run_options, index=0)

if selected_run != "Current session":
    picked = RUNS_DIR / selected_run
    st.session_state["last_run_dir"] = str(picked)
    st.session_state["last_extracted_dir"] = str(picked / "extracted")
    # not forcing csv dir; user may have old run

last_run_dir = st.session_state.get("last_run_dir")
last_extracted_dir = st.session_state.get("last_extracted_dir")
last_csv_dir = st.session_state.get("last_csv_dir")
last_ocr = st.session_state.get("last_ocr_result", {})
last_rvr = st.session_state.get("last_rvr_result", {})

chips = []
chips.append(f"Run: {Path(last_run_dir).name}" if last_run_dir else "Run: —")
chips.append(f"Images: {count_files(Path(last_extracted_dir), exts={'.png','.jpg','.jpeg'})}" if last_extracted_dir else "Images: 0")
chips.append(f"CSVs: {last_ocr.get('processed', 0)}")
chips.append(f"OCR Failed: {last_ocr.get('failed_count', 0)}")
chips.append(f"RvR: {'OK' if last_rvr.get('ok') else '—'}")

st.markdown(
    '<div class="chips">' + "".join([f'<div class="chip">{c}</div>' for c in chips]) + "</div>",
    unsafe_allow_html=True
)
st.write("")

# -----------------------------
# Step 1 + Step 2 side-by-side
# -----------------------------
col1, col2 = st.columns(2, gap="large")

# -----------------------------
# Step 1 — Extract
# -----------------------------
with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(
        """
<div class="card-title">
  <h2><span class="step">Step 1</span> Extract from DOCX</h2>
</div>
<div class="subtle">Upload DOCX files. Outputs are stored under runs/&lt;run_id&gt;/extracted.</div>
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
        also_master_zip = st.checkbox("Create one master ZIP (slower)", value=False, key="opt_master_zip")
        offer_extracted_zip = st.checkbox("Offer extracted ZIP download", value=False, key="opt_extracted_zip")

    run_extract = st.button("Run extraction", use_container_width=True, key="run_extract_btn")

    if run_extract:
        if not docx_files:
            st.error("Upload at least one DOCX.")
        else:
            run_dir = ensure_run_dir()
            in_dir = run_dir / "inputs"
            extracted_dir = run_dir / "extracted"
            in_dir.mkdir(parents=True, exist_ok=True)
            extracted_dir.mkdir(parents=True, exist_ok=True)

            with st.spinner("Extracting..."):
                docx_paths = []
                for f in docx_files:
                    p = in_dir / f.name
                    p.write_bytes(f.getbuffer())
                    docx_paths.append(str(p))

                process_many_docx_local(
                    docx_paths,
                    out_root=str(extracted_dir),
                    download_per_docx_zip=False,
                    also_make_master_zip=also_master_zip,
                )

            st.session_state["last_run_dir"] = str(run_dir)
            st.session_state["last_extracted_dir"] = str(extracted_dir)
            st.session_state.pop("last_ocr_result", None)
            st.session_state.pop("last_csv_dir", None)
            st.session_state.pop("last_rvr_result", None)

            img_count = count_files(extracted_dir, exts={".png", ".jpg", ".jpeg"})
            st.success(f"Done. Images found: {img_count}")
            st.code(str(extracted_dir))

            if offer_extracted_zip:
                with st.spinner("Preparing ZIP..."):
                    data = zip_folder(extracted_dir)
                st.download_button(
                    "Download extracted.zip",
                    data=data,
                    file_name=f"extracted_{run_dir.name}.zip",
                    mime="application/zip",
                    use_container_width=True,
                )

    st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------
# Step 2 — OCR → CSV
# -----------------------------
with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(
        """
<div class="card-title">
  <h2><span class="step">Step 2</span> OCR to CSV</h2>
</div>
<div class="subtle">Uses the most recent extracted folder from Step 1 (or loaded run).</div>
""",
        unsafe_allow_html=True,
    )

    extracted_dir_state = st.session_state.get("last_extracted_dir")

    if not extracted_dir_state:
        st.info("Complete Step 1 first (or load a previous run).")
        st.button("Run OCR + CSV generation", disabled=True, use_container_width=True)
    else:
        extracted_dir = Path(extracted_dir_state)
        csv_dir = extracted_dir.parent / "csv_outputs"
        csv_dir.mkdir(parents=True, exist_ok=True)

        max_heatmaps = st.number_input("Max heatmaps (0 = no limit)", min_value=0, value=0, step=1, key="ocr_max_heatmaps")

        with st.expander("Options", expanded=False):
            debug = st.checkbox("Debug mode", value=False, key="ocr_debug")
            offer_csv_zip = st.checkbox("Offer CSV ZIP download", value=True, key="ocr_zip")

        run_ocr = st.button("Run OCR + CSV generation", use_container_width=True, key="run_ocr_btn")

        if run_ocr:
            with st.spinner("Running OCR..."):
                res = run_ocr_generate_csv(
                    extracted_root=str(extracted_dir),
                    csv_out_root=str(csv_dir),
                    max_heatmaps=(None if max_heatmaps == 0 else int(max_heatmaps)),
                    debug=debug,
                )

            st.session_state["last_csv_dir"] = str(csv_dir)
            st.session_state["last_ocr_result"] = res
            st.session_state.pop("last_rvr_result", None)

            st.success(f"Done. CSVs: {res.get('processed', 0)} | Failed: {res.get('failed_count', 0)}")
            st.code(str(csv_dir))

            if res.get("index_csv"):
                st.write("Index CSV:")
                st.code(str(res["index_csv"]))

            if res.get("failed_csv"):
                st.warning("Failures log:")
                st.code(str(res["failed_csv"]))

            if offer_csv_zip:
                with st.spinner("Preparing ZIP..."):
                    data = zip_folder(csv_dir)
                st.download_button(
                    "Download csv_outputs.zip",
                    data=data,
                    file_name="csv_outputs.zip",
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
Runs <code>parameter_vs_range.py</code> with a run-scoped input folder.
Upload multiple <code>.esx</code> files, select one as Master, and choose which ones are router ESX files.
</div>
""",
    unsafe_allow_html=True,
)

if not PARAM_SCRIPT.exists():
    st.error(f"Missing {PARAM_SCRIPT.name} at project root. Put it next to app.py.")
else:
    st.caption("Inputs needed: Master .esx + Router .esx + band_output CSVs + optional site_geometry.json (recommended).")

    esx_files = st.file_uploader(
        "Upload ESX file(s) (multiple allowed)",
        type=["esx"],
        accept_multiple_files=True,
        key="rvr_esx_all",
    )

    master_choice = None
    router_choices = []

    if esx_files:
        esx_names = [f.name for f in esx_files]
        master_choice = st.selectbox("Select Master ESX", esx_names, index=0, key="rvr_master_choice")

        default_router = [n for n in esx_names if n != master_choice]
        router_choices = st.multiselect(
            "Select Router ESX file(s)",
            options=[n for n in esx_names if n != master_choice],
            default=default_router,
            key="rvr_router_choices",
        )

    use_csv_from_step2 = st.checkbox(
        "Use CSVs from Step 2 output (runs/<run>/csv_outputs)",
        value=True,
        key="rvr_use_step2_csv",
    )

    uploaded_csvs = None
    if not use_csv_from_step2:
        uploaded_csvs = st.file_uploader(
            "Upload band_output CSV(s) (required if not using Step 2)",
            type=["csv"],
            accept_multiple_files=True,
            key="rvr_csvs",
        )

    site_geom = st.file_uploader(
        "site_geometry.json (optional, avoids clicking anchors)",
        type=["json"],
        accept_multiple_files=False,
        key="rvr_site_geom",
    )

    with st.expander("Advanced", expanded=False):
        out_subdir = st.text_input("Output subfolder name", value="rvr_outputs", key="rvr_out_subdir")
        copy_csv_limit = st.number_input("Copy max CSVs from Step 2 (0 = no limit)", min_value=0, value=0, step=1, key="rvr_csv_limit")
        show_full_logs = st.checkbox("Show full logs", value=False, key="rvr_show_logs")

    run_rvr = st.button("Run Parameter vs Range", use_container_width=True, key="run_rvr_btn")

    if run_rvr:
        if not esx_files:
            st.error("Upload at least one .esx file.")
            st.stop()
        if not master_choice:
            st.error("Select a Master ESX.")
            st.stop()

        run_dir = ensure_run_dir()
        rvr_inputs = run_dir / "rvr_inputs"
        rvr_out = run_dir / out_subdir
        rvr_inputs.mkdir(parents=True, exist_ok=True)
        rvr_out.mkdir(parents=True, exist_ok=True)

        # Write ESX files: only include selected master + selected routers
        esx_by_name = {f.name: f for f in esx_files}

        # Master (required)
        safe_write_uploaded_file(esx_by_name[master_choice], rvr_inputs / master_choice)

        # Routers (optional)
        for rn in router_choices:
            safe_write_uploaded_file(esx_by_name[rn], rvr_inputs / rn)

        # Optional site geometry
        if site_geom:
            safe_write_uploaded_file(site_geom, rvr_inputs / "site_geometry.json")

        # CSVs
        copied_csv_count = 0
        if use_csv_from_step2:
            csv_dir_state = st.session_state.get("last_csv_dir")
            if not csv_dir_state or not Path(csv_dir_state).exists():
                st.error("No Step-2 CSV folder found. Run Step 2 first or uncheck 'Use CSVs from Step 2' and upload CSVs.")
                st.stop()

            copied_csv_count = copy_csvs(Path(csv_dir_state), rvr_inputs, limit=int(copy_csv_limit))
            if copied_csv_count == 0:
                st.error("Step-2 CSV folder exists but contains no .csv files.")
                st.stop()
        else:
            if not uploaded_csvs:
                st.error("Upload at least one CSV (or enable using Step 2 CSVs).")
                st.stop()
            for f in uploaded_csvs:
                safe_write_uploaded_file(f, rvr_inputs / f.name)
                copied_csv_count += 1

        # Run subprocess
        log_path = run_dir / "rvr_run.log"
        env = os.environ.copy()
        env["RVR_INPUT_DIR"] = str(rvr_inputs)
        env["RVR_OUT_BASE"] = str(rvr_out)

        cmd = [sys.executable, str(PARAM_SCRIPT)]
        st.info(f"Running: {' '.join(cmd)}")
        st.caption(f"RVR_INPUT_DIR = {rvr_inputs}")
        st.caption(f"RVR_OUT_BASE  = {rvr_out}")
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
            "rvr_inputs": str(rvr_inputs),
            "rvr_out": str(rvr_out),
        }

        # Prefer the script-produced zip if it exists in project root
        produced_zip = Path("rvr_full_output.zip")
        run_zip = run_dir / f"rvr_full_output_{run_dir.name}.zip"

        if produced_zip.exists():
            try:
                shutil.copy2(produced_zip, run_zip)
            except Exception:
                run_zip.write_bytes(produced_zip.read_bytes())
        else:
            # Fallback: zip the output folder so user still gets an artifact
            fallback_zip = run_dir / f"rvr_outputs_{run_dir.name}.zip"
            zip_to_path(rvr_out, fallback_zip)
            run_zip = fallback_zip

        if ok:
            st.success("RvR completed successfully.")
        else:
            st.error(f"RvR failed (exit code {proc.returncode}). Check logs below.")

        with st.expander("Logs", expanded=not ok):
            shown = log_text if show_full_logs else (log_text[-8000:] if len(log_text) > 8000 else log_text)
            st.code(shown)

        if run_zip.exists():
            st.download_button(
                "Download RvR ZIP",
                data=run_zip.read_bytes(),
                file_name=run_zip.name,
                mime="application/zip",
                use_container_width=True,
            )

        # Preview plots (if any)
        plots_dir = rvr_out / "plots"
        if plots_dir.exists():
            pngs = sorted(list(plots_dir.rglob("*.png")))
            if pngs:
                with st.expander("Preview plots", expanded=False):
                    show = pngs[:12]
                    cols = st.columns(3)
                    for i, p in enumerate(show):
                        with cols[i % 3]:
                            st.image(str(p), caption=p.name, use_container_width=True)
            else:
                st.info("RvR outputs exist, but no PNG plots were found.")
        else:
            st.caption("Plot preview will appear after a successful run (outputs/plots).")

st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------
# Session tools
# -----------------------------
st.write("")
with st.expander("Session tools", expanded=False):
    if st.button("Clear session state", use_container_width=True):
        st.session_state.pop("last_run_dir", None)
        st.session_state.pop("last_extracted_dir", None)
        st.session_state.pop("last_csv_dir", None)
        st.session_state.pop("last_ocr_result", None)
        st.session_state.pop("last_rvr_result", None)
        st.rerun()