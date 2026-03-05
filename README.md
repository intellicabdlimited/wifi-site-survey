````md
# WiFi Site Survey

A Python project for extracting data from WiFi site survey documents and generating CSV outputs.

## Project Structure

```text
WIFI SITE SURVEY/
├── __pycache__/           # Python bytecode cache (auto-generated)
├── .venv/                 # Local virtual environment (optional)
├── runs/                  # Output artifacts (generated files, logs, exports)
├── .gitignore             # Git ignore rules
├── README.md              # Project documentation (this file)
├── app.py                 # Main entry point / runner
├── docx_extractor.py      # Extracts content from .docx reports
├── ocr_csv_generator.py   # OCR + CSV generation utilities
├── parameter_vs_range.py 
└── requirements.txt       # Python dependencies

````

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

## Notes

* `__pycache__/` and `runs/` may be created automatically during execution.
* Put input files where `app.py` expects them (adjust paths inside `app.py` if needed).

```
```
