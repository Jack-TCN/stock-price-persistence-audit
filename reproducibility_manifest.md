# Corrected reproducibility map

`evidence_package/combined_model_results.csv` contains the 35,640 supervised and 990 sampled-origin rows. `chronos_full_origin_subset_60_rows.csv` is separate. Model, list and horizon summaries use descriptive statistics; split_boundary_exact.csv has all 1,980 actual conditions. Best/worst extracts retain complete experiment keys. Source paths in derived combined CSVs are portable and numeric fields retain double precision.

`data/artifact_parts.json` lists hashes of the exact input ZIP and all forecast archives. `prepare_artifacts.py` performs path-safe extraction and checks those hashes. Forecast-only verification rebuilds exact declared test selections, raw targets, persistence and all three error measures. Its output explicitly says that checkpoint/scaler bytes were not supplied. Full training produces those objects and enables full artifact checking.

Training/generation source bytes match the frozen run. The public verifier adds relocation and an explicit forecast-only scope without altering the prediction/target/metric tests. Portable run manifests have local path fields redacted and keep the SHA256 of their unredacted originals. `original_verification.json` reports belong to those originals; new verification writes a separately named public report rather than overwriting them. Runtime and execution-timestamp metadata caveats are retained in provenance/.

The main and supplementary PDFs match source/ and use corrected numeric tables. The raw input snapshot is retrospective, lists are selected and overlapping, and source adjustments are not verified as historically available. Shared assets, configurations and dates limit inference. These caveats are scientific limits, not defects hidden by the package verifier.
