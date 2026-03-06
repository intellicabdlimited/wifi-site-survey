# WiFi Site Survey Automation

## Overview

This project automates a three-step workflow for WiFi site survey analysis:

1. Extract images and related assets from DOCX survey files
2. Run OCR on survey heatmaps and convert them into CSV outputs
3. Run parameter-vs-range analysis using ESX files, CSV outputs, and optional site geometry

The goal is to move the work from a notebook-style process into a cleaner local workflow that can run inside VS Code with a Streamlit interface.

The app is designed for practical use. You upload your files, run each step in order, and download the generated outputs. It also keeps each run inside its own folder so results are easier to trace and review later.

---

## What the app does

The main app is `app.py`, a Streamlit interface that combines the whole pipeline on one page.

### Step 1: Extract from DOCX

This step reads one or more `.docx` files and extracts images into a run-specific folder.

Typical use:
- Upload survey report DOCX files
- Extract heatmap images and related files
- Store everything under a run folder such as `runs/<run_id>/extracted`

### Step 2: OCR to CSV

This step processes the extracted heatmap images and converts them into CSV outputs.

Typical use:
- Read the heatmap and matching scale image
- Detect the metric from the filename
- Run OCR on the legend to get the numeric range when needed
- Extract hexagon values from the heatmap
- Save one output CSV per heatmap
- Save `_index.csv` and `_failed.csv` for tracking

### Step 3: Parameter vs Range

This step runs `parameter_vs_range.py` using:
- one master ESX file
- one or more router ESX files
- band output CSV files
- optional `site_geometry.json`

Typical use:
- Compare router survey data by parameter and range
- Build output plots and packaged results
- Download a ZIP containing all generated outputs

---

## Project structure

A typical project layout looks like this:

```text
project_root/
│
├── app.py
├── docx_extractor.py
├── ocr_csv_generator.py
├── parameter_vs_range.py
├── README.md
│
├── runs/
│   └── <run_id>/
│       ├── inputs/
│       ├── extracted/
│       ├── csv_outputs/
│       ├── rvr_inputs/
│       ├── rvr_outputs/
│       └── rvr_run.log
│
└── ss_results/            # optional, depending on older OCR script usage
```

You may also see files such as:
- `_index.csv`
- `_failed.csv`
- generated PNG plots
- ZIP archives of outputs

---

## Core files

### `app.py`

This is the Streamlit application.

It handles:
- page layout
- file uploads
- run folder creation
- calling the extractor
- calling OCR to generate CSV files
- launching `parameter_vs_range.py` with subprocess
- storing simple run information in Streamlit session state

### `docx_extractor.py`

This file is responsible for reading DOCX files and extracting images locally.

Expected imported function:

```python
process_many_docx_local(...)
```

### `ocr_csv_generator.py`

This file scans the extracted image folder, matches heatmaps with scale images, performs OCR, maps heatmap values, and writes CSV outputs.

Expected imported function:

```python
run_ocr_generate_csv(...)
```

### `parameter_vs_range.py`

This file runs the final parameter-vs-range logic.

The Streamlit app launches it as a separate process and passes input/output paths through environment variables.

---

## Requirements

You need Python 3.10 or newer in most cases.

Main Python packages used by the project:

```bash
pip install streamlit python-docx pytesseract opencv-python-headless scipy pandas matplotlib seaborn numpy
```

Depending on how your extractor is implemented, you may also need:

```bash
pip install pillow lxml
```

### Tesseract OCR

This project depends on Tesseract for OCR.

#### Ubuntu or Debian

```bash
sudo apt update
sudo apt install tesseract-ocr
```

#### Windows

Install Tesseract and make sure the executable is added to your system PATH.

If it is not in PATH, set it manually in Python, for example:

```python
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
```

#### Check that it works

Run this in a terminal:

```bash
tesseract --version
```

If this command fails, OCR in Step 2 will also fail.

---

## How to run the app

From the project root:

```bash
streamlit run app.py
```

Then open the local URL shown in the terminal.

---

## Recommended workflow

Use the app in this order.

### 1. Run extraction

Upload one or more DOCX files in Step 1 and click **Run extraction**.

Expected result:
- a new run folder is created under `runs/`
- extracted images are saved under `runs/<run_id>/extracted`

### 2. Run OCR

After extraction, go to Step 2 and click **Run OCR + CSV generation**.

Expected result:
- CSV outputs are saved under `runs/<run_id>/csv_outputs`
- `_index.csv` summarizes successful items
- `_failed.csv` lists failed items, if any

### 3. Run parameter vs range

In Step 3, upload or select the required files and run the final analysis.

Expected result:
- run-scoped inputs are copied into `runs/<run_id>/rvr_inputs`
- outputs are written under `runs/<run_id>/rvr_outputs`
- a ZIP file is created for download
- logs are saved to `runs/<run_id>/rvr_run.log`

---

## Step 3 input guide

This is the part that usually causes the most confusion.

### Required files

#### 1. Master ESX

This is the floor-plan reference ESX.

Use the file that contains the base floor plans or the site layout that other survey files should align to.

Examples:
- `actual floor plans.esx`
- a site layout ESX containing the master floor drawings

#### 2. Router ESX files

These are survey files for each router or device.

Examples:
- `KVD21 February 2026 Survey (No APs).esx`
- `Sagemcom February 2026 Survey (No APs).esx`
- `TMO-G4AR February 2026 Survey (No APs).esx`
- `TMO-G4SE February 2026 Survey (No APs).esx`
- `TMO-G5AR February 2026 Survey (No APs).esx`

#### 3. Band output CSV files

These are the CSV files produced by Step 2, or uploaded manually if you are not using Step 2 output directly.

Examples:
- `TMO-G5AR_Data Rate for Ground Floor on 2.4 GHz band_output.csv`
- `TMO-G5AR_SNR for Lower Floor on 5 GHz band_output.csv`

#### 4. Optional `site_geometry.json`

This file is strongly recommended.

It stores anchor points and geometry mapping so the script does not need to rebuild or guess floor alignment every time.

### Notes

- One file must be selected as the **Master ESX**
- The remaining ESX files can be selected as **Router ESX** files
- Step 3 does not work properly with only the master file if router survey data is missing
- If you do not use Step 2 CSVs automatically, you must upload the correct CSV files yourself

---

## How the run folders work

Each run is stored under a unique folder name such as:

```text
runs/20260306_153045_ab12cd/
```

This helps you keep input files, intermediate outputs, and final outputs together.

Typical layout:

```text
runs/<run_id>/
├── inputs/
├── extracted/
├── csv_outputs/
├── rvr_inputs/
├── rvr_outputs/
└── rvr_run.log
```

This design is useful because:
- each run is isolated
- you can reload older runs in the app
- debugging becomes easier
- outputs are easier to archive and share

---

## Session state behavior in Streamlit

The app stores some values in `st.session_state`, such as:
- the last run directory
- the last extracted directory
- the last CSV directory
- the previous OCR result summary
- the previous parameter-vs-range result summary

This means rerunning the script does not automatically remove the previous session data.

That is expected behavior in Streamlit.

If you want a clean state, use the **Clear session state** button in the app, or explicitly clear `st.session_state` in code.

Example:

```python
for k in list(st.session_state.keys()):
    del st.session_state[k]
st.rerun()
```

Important: clearing session state does not delete files already written under `runs/`. Those folders remain on disk until you remove them manually.

---

## Why SNR may fail in VS Code even when Colab works

This is one of the most common issues in this workflow.

Possible reasons include:

### 1. Tesseract is not installed locally

Colab installs Tesseract explicitly. Your VS Code environment may not have it installed or available in PATH.

### 2. Local OCR environment differs from Colab

Even if the Python code is the same, the local machine may differ in:
- Tesseract version
- OpenCV version
- image processing behavior
- available fonts or rendering behavior
- file path handling

### 3. Duplicate or outdated OCR code

If your local `ocr_csv_generator.py` contains repeated blocks, partial rewrites, or mixed versions from Colab and VS Code, SNR may fail because the local file is not actually running the same logic as the notebook version.

### 4. Heatmap filtering is too strict

SNR heatmaps sometimes need slightly different extraction thresholds than other metrics. Your code already tries to handle that with a special branch for SNR, but the local version may still need the same exact logic as the working Colab script.

### 5. Scale image matching fails silently for some items

If the wrong scale image is matched, OCR min/max detection becomes unreliable and the CSV step can fail.

### 6. The working Colab script is cleaner than the local script

In several cases, the local file becomes a mix of copied notebook blocks, older experiments, and repeated sections. That can lead to behavior differences even when the main logic looks similar.

---

## Troubleshooting checklist

### OCR step fails

Check the following:

1. Tesseract is installed and available from terminal
2. The heatmap image and matching `_scale` image both exist
3. Filenames still follow the expected naming pattern
4. `metric_key_from_filename()` correctly detects the metric
5. `find_matching_scale()` is returning the right scale file
6. `_failed.csv` contains the real error message

### SNR specifically fails

Check the following:

1. Compare the local `ocr_csv_generator.py` against the working Colab version
2. Confirm that the SNR branch is still present:

```python
if metric_key == "snr":
    df_hex, _ = extract_hexagons(hm_rgb, roi_sat_thresh=6, roi_val_thresh=254)
else:
    df_hex, _ = extract_hexagons(hm_rgb)
```

3. Enable debug mode and inspect scale crops and OCR outputs
4. Confirm that Tesseract can read SNR scale labels locally
5. Make sure the local file does not contain duplicate function definitions that override the intended version

### Step 3 produces no output ZIP or plots

Check the following:

1. `parameter_vs_range.py` exists in the project root
2. The master ESX file is selected correctly
3. Router ESX files are selected correctly
4. CSV inputs actually exist in `rvr_inputs`
5. `site_geometry.json` is present when the script expects it
6. `rvr_run.log` contains the real runtime error

### Error like `TypeError: 'NoneType' object is not callable`

This usually happens when notebook-only display code is still present in a local script.

For example, Colab or Jupyter code like this should not be used directly in a plain Python subprocess script:

```python
display(IPyImage(filename=p))
```

In a VS Code or Streamlit workflow, replace notebook display calls with file saving and Streamlit preview logic.

---

## Notes on parameter selection in Step 3

Depending on your current `parameter_vs_range.py` version, the selected parameter may still be controlled inside the script rather than from the app.

If the output always shows one parameter such as `Data Rate`, that usually means the script is still hardcoded internally.

To make Step 3 fully interactive, the app and script should support a parameter selection flow such as:
- SNR
- Data Rate
- Throughput
- Signal Strength
- Channel Utilization
- Spectrum Channel Power

The app can pass that parameter into the script through:
- an environment variable
- a command-line argument
- a generated wrapper file

If that is not yet implemented, the README should reflect the current script behavior rather than promising a parameter picker that does not exist.

---

## Output files you should expect

### Step 1 output

Typical output:
- extracted PNG or JPG images
- optional ZIP of extracted assets

### Step 2 output

Typical output:
- one CSV per heatmap
- `_index.csv`
- `_failed.csv` if any items fail

### Step 3 output

Typical output:
- plots
- summary files
- final ZIP package
- `rvr_run.log`

---

## Example local setup

```bash
git clone <your-repo>
cd <your-repo>
python -m venv .venv
source .venv/bin/activate
pip install streamlit pytesseract opencv-python-headless scipy pandas matplotlib seaborn numpy python-docx
sudo apt install tesseract-ocr
streamlit run app.py
```

On Windows, activate the virtual environment differently and install Tesseract separately.

---

## Development notes

When maintaining this project, try to keep the logic separated by responsibility:

- `docx_extractor.py` for extraction only
- `ocr_csv_generator.py` for OCR and CSV generation only
- `parameter_vs_range.py` for parameter/range analysis only
- `app.py` for user interaction and orchestration only

This makes debugging much easier.

It is also a good idea to avoid copying the same OCR functions multiple times into the same file. Repeated definitions can silently override earlier code and make debugging difficult.

---

## Known limitations

- OCR quality depends heavily on legend clarity and local Tesseract setup
- Some metrics are more stable than others
- File naming conventions matter a lot for automatic matching
- Old run folders are not deleted automatically
- Streamlit reruns do not automatically clear prior session state
- Notebook-only display code does not translate directly into subprocess-based local execution

---

## Suggested maintenance practice

When something works in Colab but fails locally, use this order:

1. Verify local dependencies
2. Verify Tesseract installation
3. Compare the exact working Colab code with the local file
4. Remove duplicate or outdated code blocks
5. Test one metric at a time
6. Use debug output to inspect OCR crops and failure points

This is usually faster than trying random changes across the whole pipeline.

---

## Final note

This project works best when each step is validated separately.

Start with extraction.
Then confirm OCR outputs for one metric.
Then move to the parameter-vs-range stage.

That approach keeps the pipeline manageable and makes failures much easier to understand.
