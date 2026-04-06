WiFi Site Survey Automation

Overview

WiFi Site Survey Automation is a local Streamlit application for processing Wi-Fi survey deliverables and generating comparative analysis outputs. The project is built around Ekahau-style survey exports and supports the following workflow:

1. Extract heatmap and scale images from DOCX survey reports.
2. Convert extracted heatmaps into structured CSV data using OCR and color-scale reconstruction.
3. Generate parameter-vs-range (RvR) plots using ESX geometry and floor alignment.
4. Compare with-mesh and without-mesh survey outputs.
5. Render interactive dashboards for both RvR and mesh comparison outputs.
6. Generate DOCX-based comparative reports, optionally enriched with a local LLM through Ollama.
7. Build a local searchable knowledge base for natural-language querying of survey results.

The application is intended to run locally and keeps all processing on the user’s machine.



 Key Features

 1. DOCX Survey Extraction

The project can extract survey heatmaps and related scale images from DOCX files. It uses caption parsing and filename metadata normalization to identify:

* router name,
* metric type,
* floor,
* frequency band,
* asset role such as heatmap or scale.

The extraction stage also generates a manifest file that downstream OCR and reporting stages use.

 2. OCR-Based CSV Generation

The OCR pipeline converts extracted survey images into structured CSV outputs. It supports both:

* numeric scales, such as signal strength, SNR, throughput, and noise,
* categorical scales, where the value must be inferred from a discrete legend.

The OCR stage performs:

* scale image pairing,
* OCR of min/max values,
* color bar interpretation,
* hexagon extraction from heatmaps,
* mapping of detected color regions to numeric or categorical values.

 3. Range-vs-Parameter Analysis

The RvR workflow uses CSV outputs and ESX geometry to calculate distance-aware trends. It produces comparative plots showing how a selected metric behaves with increasing distance from the DUT or AP reference position.

This stage depends on:

* CSV outputs from the OCR stage,
* master and router ESX files,
* a `site_geometry.json` definition.

 4. Mesh vs No Mesh Comparison

The comparison workflow takes two sets of survey CSVs:

* with mesh,
* without mesh.

It aligns them using shared site geometry and generates comparative outputs in an RvR-style format.

 5. Interactive Dashboards

The application includes interactive dashboards implemented in Streamlit and ApexCharts for:

* router-to-router RvR analysis,
* with-mesh vs without-mesh comparison.

 6. AI Report Generation

The report generator creates a DOCX comparative report using generated curve tables and extracted visual assets. If Ollama is available locally, the report can include AI-assisted summary bullets. Without Ollama, the code falls back to deterministic summaries.

 7. Local Knowledge Base and Chat

The project can build a local knowledge base from:

* extraction manifests,
* OCR indexes,
* curve tables,
* comparison outputs.

Users can then ask natural-language questions about routers, floors, bands, metrics, or scenario differences.



 Tech Stack

* Python
* Streamlit for the web UI
* OpenCV for image processing
* Tesseract OCR via `pytesseract`
* Pandas / NumPy / SciPy / scikit-learn for data processing and ranking
* python-docx for report generation
* Matplotlib / Seaborn for plot generation
* Ollama for optional local LLM summaries



 Main Modules

 `app.py`

Primary Streamlit entry point. It provides the full UI and orchestrates the processing workflow.

The active application is implemented in the later portion of the file. The file also contains large blocks of older commented code that appear to be retained as historical copies.

 `docx_extractor.py`

Legacy DOCX extraction logic for harvesting images from survey reports.

 `docx_extractor_v2.py`

Enhanced extraction wrapper that reuses the legacy image extraction path but adds metadata manifests for downstream OCR and reporting.

 `ocr_csv_generator.py`

Core OCR and heatmap-to-CSV conversion engine. This is one of the main processing modules.

 `ocr_csv_generator_v2.py`

Wrapper around the OCR engine that uses extraction manifests, filters jobs by selected parameters, and writes index and failure logs.

 `parameter_vs_range.py`

Generates range-vs-parameter plots and related outputs using site geometry and CSV inputs.

 `comparison.py`

Compares with-mesh and without-mesh survey outputs for a selected metric.

 `pipeline_runner.py`

Patches and runs `parameter_vs_range.py` programmatically for multiple metrics.

 `site_geometry_builder.py`

Builds `site_geometry.json` from ESX floor plans and user-selected anchor points. Also attempts to infer DUT positions from router ESX files.

 `ai_report_generator.py`

Creates DOCX comparative reports and supports optional AI summarization using Ollama.

 `local_kb.py`

Builds a searchable local knowledge base from extracted and generated artifacts and answers user questions using either Ollama or a lexical fallback.

 `metadata_utils.py`

Canonicalizes router names, bands, floors, and metric labels, and parses caption or filename metadata.

 `asset_registry.py`

Indexes extracted heatmap and scale assets for use in report generation and chart linking.

 `charts/apex_rvr.py`

Renders interactive RvR and mesh-comparison dashboards in Streamlit.



 Supported Metrics

The code explicitly supports the following metrics:

* Signal Strength
* Secondary Signal Strength
* Tertiary Signal Strength
* SNR
* Noise
* Data Rate
* Throughput
* Channel Utilization
* Channel Interference
* Channel Width
* Spectrum Channel Power
* Network Health
* Network Issues
* Number of Access Points

The application normalizes user-facing metric labels into canonical internal metric keys.



 Application Workflow

 Step 1 — DOCX to Images + CSV

The user uploads one or more DOCX survey reports. The application:

* stores the uploaded DOCX files,
* extracts heatmap and scale images,
* writes extraction manifests,
* runs OCR on extracted survey maps,
* writes grouped CSV outputs by metric.

Outputs are stored under:

* `runs/<router_name>/docx_inputs`
* `runs/<router_name>/extracted`
* `runs/<router_name>/csv_outputs/<metric>`

This stage also produces:

* `_index.csv`
* `_failed.csv` if OCR jobs fail

 Step 2 — Parameter vs Range

Using one selected metric from Step 1, plus ESX files and site geometry, the application runs `parameter_vs_range.py` to generate RvR outputs.

Outputs are stored under:

* `runs/<router_name>/rvr_outputs/<metric>`

 Step 3 — Mesh vs No Mesh Comparison

The user uploads:

* WITH MESH CSVs,
* WITHOUT MESH CSVs,
* ESX files,
* shared site geometry.

The application then runs `comparison.py` for the selected metric and stores results under:

* `runs/<router_name>/compare_outputs/<metric>`

 Step 4A — Interactive Graph: RvR

Displays interactive charts generated from Step 2 curve tables.

 Step 4B — Interactive Graph: Mesh vs No Mesh

Displays interactive charts generated from Step 3 comparison tables.

 Step 5 — AI Report Generator

Builds a DOCX comparative report from generated outputs and extracted survey assets.

 Step 6 — Local Knowledge Base + Chat

Builds a local index over survey outputs and answers questions using retrieved context.



 Project Structure

```text
.
├── app.py
├── ai_report_generator.py
├── asset_registry.py
├── comparison.py
├── commands_to_run_wifisitesurvey.txt
├── docx_extractor.py
├── docx_extractor_v2.py
├── local_kb.py
├── metadata_utils.py
├── ocr_csv_generator.py
├── ocr_csv_generator_v2.py
├── parameter_vs_range.py
├── pipeline_runner.py
├── requirements.txt
├── site_geometry_builder.py
├── charts/
│   ├── __init__.py
│   └── apex_rvr.py
└── __pycache__/
```



 Installation

Create a virtual environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

 Python dependencies

From `requirements.txt`:

* matplotlib
* numpy
* opencv-python-headless
* pandas
* pillow
* pytesseract
* python-docx
* requests
* scikit-learn
* scipy
* seaborn
* streamlit

 System dependencies

The included command notes indicate the project expects:

* Python 3
* Tesseract OCR
* OpenGL runtime libraries for OpenCV
* optionally `python3-tk` for popup-based point selection
* optionally Ollama for local AI summaries



 Running the Application

Launch the Streamlit app:

```bash
streamlit run app.py
```

The provided helper file `commands_to_run_wifisitesurvey.txt` also documents optional Ollama setup.



 Ollama Integration

Ollama is not required for the main extraction, OCR, plotting, or comparison workflows. It is used for:

* AI-generated report bullets,
* local knowledge-base answers.

The code expects a local Ollama endpoint at:

```text
http://localhost:11434
```

and defaults to model:

```text
gemma3:4b
```

If Ollama is unavailable, the report and KB logic can fall back to non-AI summaries.



 Important Input Types

 DOCX Reports

The extraction pipeline is designed around DOCX survey reports containing:

* embedded heatmaps,
* scale images,
* captions that identify metric, floor, and band.

 ESX Files

ESX files are used for:

* master floor plan extraction,
* AP/DUT position inference,
* geometry alignment,
* range-vs-parameter calculations.

 `site_geometry.json`

This file defines the floor alignment and DUT placement needed for multi-floor range calculations and mesh comparisons.

It can be:

* uploaded,
* reused from a prior run,
* created interactively using popup floor-plan windows.



 Output Structure

For each router name, the project creates a run folder:

```text
runs/<router_name>/
├── docx_inputs/
├── extracted/
├── csv_outputs/
├── rvr_inputs/
├── rvr_outputs/
├── compare_inputs/
├── compare_outputs/
└── survey_kb.json
```

 Extraction outputs

Under `extracted/`, the project stores extracted survey assets and manifest CSVs.

 OCR outputs

Under `csv_outputs/<metric>/`, the project stores per-heatmap CSV files plus index and failure logs.

 RvR outputs

Under `rvr_outputs/<metric>/`, the project stores generated plots, ranking data, and curve tables.

 Comparison outputs

Under `compare_outputs/<metric>/`, the project stores with-mesh vs without-mesh plots and comparison tables.

 Report outputs

The AI report generator writes DOCX reports and an accompanying audit file of referenced assets.



 Internal Processing Notes

 Metadata-first workflow

A major design pattern in the code is metadata normalization. Captions, filenames, floors, bands, and router names are standardized before downstream processing.

 Manifest-driven OCR

The v2 OCR flow uses extraction manifests rather than loose image scanning whenever possible. This improves pairing of heatmaps and scale images.

 Script patching for metric execution

The app and `pipeline_runner.py` dynamically patch `parameter_vs_range.py` so the same script can be run repeatedly for different metrics and input/output paths.

 Geometry alignment

`site_geometry_builder.py` builds a multi-floor alignment model by asking the user to click anchor points on floor images. It then combines those transforms with inferred DUT locations from router ESX files.

 Asset registry

The reporting system uses `AssetRegistry` to match heatmaps and scales to a given:

* router,
* metric,
* floor,
* band.

This allows the final report to include the most relevant extracted survey artifacts.



 Suggested Use Case

This project is most suitable when you have:

* one or more Wi-Fi survey DOCX reports,
* ESX exports for floor plans and AP placement,
* a need to compare routers or survey scenarios,
* a need to generate plots, tables, and local narrative summaries from survey artifacts.



 Minimal Run Sequence

A typical user flow is:

1. Start the Streamlit app.
2. Upload survey DOCX files.
3. Run extraction and OCR.
4. Upload ESX files and create or reuse site geometry.
5. Run parameter-vs-range analysis for a metric.
6. Optionally run with-mesh vs without-mesh comparison.
7. Review interactive graphs.
8. Generate a DOCX report.
9. Build the local knowledge base and ask questions.
