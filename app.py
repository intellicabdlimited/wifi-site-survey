# import io
# import zipfile
# from datetime import datetime
# from uuid import uuid4
# from pathlib import Path

# import streamlit as st

# from docx_extractor import process_many_docx_local   # renamed module
# from ocr_csv_generator import run_ocr_generate_csv   # new module

# st.set_page_config(page_title="WiFi Site Survey Automation", layout="wide")
# st.title("WiFi Site Survey Automation")

# RUNS_DIR = Path("runs")
# RUNS_DIR.mkdir(exist_ok=True)

# tab1, tab2 = st.tabs(["1) DOCX Extract", "2) OCR → CSV"])

# def zip_folder(folder: Path) -> bytes:
#     buf = io.BytesIO()
#     with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_STORED) as z:
#         for p in folder.rglob("*"):
#             if p.is_file():
#                 z.write(p, arcname=str(p.relative_to(folder)))
#     buf.seek(0)
#     return buf.read()

# with tab1:
#     st.header("DOCX → Extract heatmaps/scales")

#     docx_files = st.file_uploader(
#         "Upload one or more DOCX files",
#         type=["docx"],
#         accept_multiple_files=True
#     )

#     also_master_zip = st.checkbox("Create one master ZIP (slower)", value=False)
#     run_extract = st.button("Run extraction")

#     if run_extract:
#         if not docx_files:
#             st.error("Please upload at least one DOCX.")
#         else:
#             run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid4().hex[:6]
#             run_dir = RUNS_DIR / run_id
#             in_dir = run_dir / "inputs"
#             extracted_dir = run_dir / "extracted"
#             in_dir.mkdir(parents=True, exist_ok=True)
#             extracted_dir.mkdir(parents=True, exist_ok=True)

#             with st.spinner("Extracting images from DOCX..."):
#                 docx_paths = []
#                 for f in docx_files:
#                     p = in_dir / f.name
#                     p.write_bytes(f.getbuffer())
#                     docx_paths.append(str(p))

#                 zip_paths = process_many_docx_local(
#                     docx_paths,
#                     out_root=str(extracted_dir),
#                     download_per_docx_zip=False,
#                     also_make_master_zip=also_master_zip
#                 )

#             st.session_state["last_run_dir"] = str(run_dir)
#             st.session_state["last_extracted_dir"] = str(extracted_dir)

#             st.success("Extraction complete.")
#             st.write("Saved extracted files to:", str(extracted_dir))

#             # optional: offer a ZIP of the extracted folder
#             if st.checkbox("Download extracted folder as ZIP (can be big / slower)", value=False):
#                 data = zip_folder(extracted_dir)
#                 st.download_button(
#                     "Download extracted.zip",
#                     data=data,
#                     file_name=f"extracted_{run_id}.zip",
#                     mime="application/zip",
#                 )

# with tab2:
#     st.header("OCR → Generate CSVs")

#     extracted_dir = st.session_state.get("last_extracted_dir")
#     if not extracted_dir:
#         st.info("Run Tab-1 first so we have extracted images to OCR.")
#     else:
#         extracted_dir = Path(extracted_dir)
#         csv_dir = extracted_dir.parent / "csv_outputs"
#         csv_dir.mkdir(parents=True, exist_ok=True)

#         max_heatmaps = st.number_input("Max heatmaps (optional)", min_value=0, value=0, step=1)
#         run_ocr = st.button("Run OCR + CSV generation")

#         if run_ocr:
#             with st.spinner("Running OCR and generating CSVs..."):
#                 res = run_ocr_generate_csv(
#                     extracted_root=str(extracted_dir),
#                     csv_out_root=str(csv_dir),
#                     max_heatmaps=(None if max_heatmaps == 0 else int(max_heatmaps)),
#                     debug=False,
#                 )

#             st.success(f"Done. CSVs created: {res['processed']} | Failed: {res['failed_count']}")
#             st.write("CSV output folder:", str(csv_dir))
#             st.write("Index CSV:", res["index_csv"])
#             if res["failed_csv"]:
#                 st.warning(f"Some failures logged: {res['failed_csv']}")

#             # optional download
#             if st.checkbox("Download CSV folder as ZIP", value=True):
#                 data = zip_folder(csv_dir)
#                 st.download_button(
#                     "Download csv_outputs.zip",
#                     data=data,
#                     file_name="csv_outputs.zip",
#                     mime="application/zip",
#                 )

import io
import zipfile
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
# Storage helpers
# -----------------------------
RUNS_DIR = Path("runs")
RUNS_DIR.mkdir(exist_ok=True)

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

# -----------------------------
# Header + run picker (simple)
# -----------------------------
st.markdown(
    """
<div class="hero">
  <h1>WiFi Site Survey Automation</h1>
  <p>Two-step workflow: extract images from DOCX, then OCR to CSV.</p>
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

# Session state references
last_run_dir = st.session_state.get("last_run_dir")
last_extracted_dir = st.session_state.get("last_extracted_dir")
last_csv_dir = st.session_state.get("last_csv_dir")
last_ocr = st.session_state.get("last_ocr_result", {})

chips = []
chips.append(f"Run: {Path(last_run_dir).name}" if last_run_dir else "Run: —")
if last_extracted_dir:
    chips.append(f"Images: {count_files(Path(last_extracted_dir), exts={'.png','.jpg','.jpeg'})}")
else:
    chips.append("Images: 0")
chips.append(f"CSVs: {last_ocr.get('processed', 0)}")
chips.append(f"Failed: {last_ocr.get('failed_count', 0)}")

st.markdown('<div class="chips">' + "".join([f'<div class="chip">{c}</div>' for c in chips]) + "</div>", unsafe_allow_html=True)
st.write("")

# -----------------------------
# Two-step grid
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
        label_visibility="visible",
    )

    with st.expander("Options", expanded=False):
        also_master_zip = st.checkbox("Create one master ZIP (slower)", value=False)
        offer_extracted_zip = st.checkbox("Offer extracted ZIP download", value=False)

    run_extract = st.button("Run extraction", use_container_width=True, key="run_extract_btn")

    if run_extract:
        if not docx_files:
            st.error("Upload at least one DOCX.")
        else:
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid4().hex[:6]
            run_dir = RUNS_DIR / run_id
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

            img_count = count_files(extracted_dir, exts={".png", ".jpg", ".jpeg"})
            st.success(f"Done. Images found: {img_count}")
            st.code(str(extracted_dir))

            if offer_extracted_zip:
                with st.spinner("Preparing ZIP..."):
                    data = zip_folder(extracted_dir)
                st.download_button(
                    "Download extracted.zip",
                    data=data,
                    file_name=f"extracted_{run_id}.zip",
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

        max_heatmaps = st.number_input("Max heatmaps (0 = no limit)", min_value=0, value=0, step=1)

        with st.expander("Options", expanded=False):
            debug = st.checkbox("Debug mode", value=False)
            offer_csv_zip = st.checkbox("Offer CSV ZIP download", value=True)

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

# -----------------------------
# Tiny footer actions
# -----------------------------
st.write("")
with st.expander("Session tools", expanded=False):
    if st.button("Clear session state", use_container_width=True):
        st.session_state.pop("last_run_dir", None)
        st.session_state.pop("last_extracted_dir", None)
        st.session_state.pop("last_csv_dir", None)
        st.session_state.pop("last_ocr_result", None)
        st.rerun()