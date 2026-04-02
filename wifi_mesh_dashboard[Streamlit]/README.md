# WiFi Mesh Dashboard

Single-page Streamlit dashboard for comparing **with mesh** vs **without mesh** using pre-stored CSV exports.

## Folder structure

```text
wifi_mesh_dashboard/
├── app.py
├── requirements.txt
├── utils/
│   ├── data_loader.py
│   ├── charts.py
│   └── styles.py
└── data/
    └── compare_inputs/
        ├── _shared/
        │   ├── site_geometry.json
        │   └── esx/
        ├── with_mesh/
        │   ├── throughput/
        │   │   └── <router>/*.csv
        │   └── signal_strength/
        │       └── <router>/*.csv
        └── without_mesh/
            ├── throughput/
            │   └── <router>/*.csv
            └── signal_strength/
                └── <router>/*.csv
```

## Run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## If your exports are still flat

Example:

```bash
python scripts/reorganize_flat_exports.py /path/to/flat/throughput data/compare_inputs/with_mesh --copy
python scripts/reorganize_flat_exports.py /path/to/flat/signal_strength data/compare_inputs/without_mesh --copy
```

The script reads the router name from filenames like:

- `TMO-G5AR_Throughput for Lower Floor on 5 GHz band_output.csv`
- `TMO-G5AR_Signal Strength for Lower Floor on 5 GHz band_output.csv`

and places them into:

- `data/compare_inputs/<topology>/throughput/TMO-G5AR/`
- `data/compare_inputs/<topology>/signal_strength/TMO-G5AR/`

## Notes

- The dashboard reads **pre-stored files only**.
- It does not require upload widgets.
- Distance bands are derived from the CSV `col` positions and summarized into eight bands for charting.
