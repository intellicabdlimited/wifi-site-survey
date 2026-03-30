# Final Integration Notes

This package combines the earlier upgrade work with the later extraction and parameter fixes.

## Key guarantees added
- The report generator now prefers `csv_outputs/_index.csv` when selecting heatmaps and scales.
  - That means the report uses the exact heatmap image that produced the OCR CSV used downstream in the graph pipeline.
- The report generator no longer uses fuzzy router/parameter/floor/band matching.
  - Matching is exact after canonical normalization.
- A companion `*_asset_audit.csv` is written beside every generated DOCX report.
  - It lists the exact heatmap and scale selected for each router/floor/band/parameter.
  - Missing pairs are flagged explicitly instead of silently substituting the wrong image.

## Included fixes
- Clean router naming
- Redundant `(Standard)`-style text removed from report headings
- Multi-parameter Step 3 selection
- Mesh vs No-Mesh report mode
- Legacy full-image DOCX extraction restored inside the upgraded pipeline
- Expanded parameter extraction and CSV generation coverage
- Alias handling for `signal_strength` vs `signal_strength_main`
- Alias handling for `number_of_access_points` vs `number_of_aps`
- Improved Word heatmap layout using larger 2-column cards
- Local knowledge base + chat support
