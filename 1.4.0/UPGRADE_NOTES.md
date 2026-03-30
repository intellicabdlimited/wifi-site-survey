# Wi-Fi Site Survey Upgrade Notes

## What changed

This upgrade adds a more maintainable metadata-driven layer on top of the existing pipeline.

### 1. Cleaner router naming
A shared `clean_router_name()` function now normalizes router names across extraction, OCR, reporting, and asset lookup.

Examples:
- `KVD21 February 2026 Survey with Mesh Extender` -> `KVD21`
- `TMO-G4AR February 2026 Survey with Mesh Extender` -> `TMO-G4AR`
- `Sagemcom` -> `Sagemcom`

### 2. Exact heatmap + scale pairing
`docx_extractor_v2.py` writes `_extract_manifest.csv` for each extracted DOCX folder.
Each row stores:
- router key
- parameter key
- floor name
- band
- role (`heatmap` or `scale`)
- original caption text
- source DOCX
- exact file path

`ocr_csv_generator_v2.py` now uses this manifest first instead of relying only on filename heuristics.
This is the main fix for band-related mixups and the most likely fix for the throughput 2.4 GHz vs 5 GHz duplication issue.

### 3. Stronger report association
`ai_report_generator.py` now uses `AssetRegistry` for exact router / parameter / floor / band matching.
This replaces the older fuzzy `find_heatmap_for_router()` logic that could pick the wrong file.

### 4. Better DOCX layout
The report now places router heatmaps in larger 2-column cards instead of a very narrow 5-column strip.
This improves readability and reduces Word clipping/cropping problems.

### 5. Multi-parameter Step 3
`app.py` now allows selecting multiple metric folders at once for the RvR stage and runs them sequentially.
A helper `pipeline_runner.py` is also included for batch CLI execution.

### 6. Local knowledge base + chat
`local_kb.py` builds a TF-IDF index from:
- extracted manifests
- OCR index files
- curve tables

The Streamlit app includes a local Q&A section that can answer questions from the generated run data.

### 7. Mesh-vs-no-mesh report mode
The report generator now has a mesh comparison mode.
It reads the comparison curve tables and plots. If scenario-specific extracted manifests are available, it also inserts the matching with-mesh and without-mesh heatmaps.

## Root causes found during code review

### Throughput duplication risk
The old OCR/report path depended heavily on partial filename matching and did not preserve a first-class manifest of `caption -> heatmap -> scale` relationships.
That makes throughput especially vulnerable when multiple floors and both 2.4 GHz / 5 GHz versions exist.

### Report mismatch risk
The old report code matched router assets using:
- first 4 characters of router name
- first 2 words of parameter
- first 2 words of floor
- a simple band substring

That is fragile and can associate the wrong heatmap or scale.

## Files added
- `metadata_utils.py`
- `asset_registry.py`
- `docx_extractor_v2.py`
- `ocr_csv_generator_v2.py`
- `pipeline_runner.py`
- `local_kb.py`
- updated `ai_report_generator.py`
- updated `app.py`

## Recommended next validation steps
1. Re-run Step 1 and verify `_extract_manifest.csv` exists inside each extracted router folder.
2. Re-run Step 2 for throughput and compare `_index.csv` rows for 2.4 GHz vs 5 GHz.
3. Re-run Step 3 for Throughput only and verify the generated curve tables differ by band.
4. Generate one DOCX report and manually inspect 3-4 sections for exact graph/heatmap/scale alignment.
5. For mesh reports, store scenario-specific extracted folders under:
   - `runs/<router>/compare_inputs/with_mesh_extracted/`
   - `runs/<router>/compare_inputs/without_mesh_extracted/`

## Known limits
- The underlying `parameter_vs_range.py` and `comparison.py` plotting engines were not fully rewritten in this pass.
- The mesh report can only embed with/without heatmaps when the scenario-specific extracted folders are present.
- The local KB uses TF-IDF retrieval, not embeddings. It is fully local and lightweight, but it is not semantic-vector RAG.

## Additional final integration fixes
- Extraction keeps the legacy full-image harvesting path and still writes manifest metadata.
- Parameter coverage was expanded so missing heatmap types and CSV outputs are included.
- Report asset selection now prefers OCR `_index.csv` rows when available, tying each report heatmap to the exact image used to generate the CSV and downstream graph.
- Every generated report writes a companion `*_asset_audit.csv` file listing the exact heatmap and scale chosen per router/floor/band/parameter and flagging any missing pair.
