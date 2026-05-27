# Reproducibility Manifest

## Study Scope

The study audits absolute stock-price level forecasting against latest-close persistence. It covers 165 dataset-series memberships, corresponding to 133 unique ticker symbols after cross-index overlaps, across Dow 30, Nasdaq large-tech, S&P 100 subset, Hang Seng large-cap, and ETF basket lists.

## Main Evidence Files

- `evidence_package/combined_model_results.csv`: all supervised and sampled-origin Chronos result rows.
- `evidence_package/model_gate_summary.csv`: model-level pass counts, pass rates, RMSE-ratio medians, confidence intervals, and Wilcoxon diagnostics.
- `evidence_package/model_horizon_summary.csv`: model-by-horizon median RMSE ratios.
- `evidence_package/dataset_model_summary.csv`: dataset-by-model summaries.
- `evidence_package/dataset_coverage.csv`: dataset membership counts and date coverage.
- `evidence_package/split_boundary_summary.csv`: nominal chronological split boundaries.
- `evidence_package/best_100_model_rows.csv`: best row-level cases by RMSE ratio.
- `evidence_package/worst_100_model_rows.csv`: worst row-level cases by RMSE ratio.
- `evidence_package/chronos_full_origin_subset_60_rows.csv`: full-origin Chronos subset verification cited in the manuscript.
- `evidence_package/FULL_ORIGIN_SUBSET_SUMMARY.md`: compact summary of the full-origin subset.
- `evidence_package/full_origin_subset_by_dataset.csv`: full-origin subset grouped by dataset.
- `evidence_package/full_origin_subset_by_horizon.csv`: full-origin subset grouped by horizon.

## Source Files

- `source/manuscript.tex`: main manuscript source.
- `source/supplementary_material.tex`: supplement source.
- `source/crossmarket_core_tables.tex`: core manuscript tables.
- `source/crossmarket_supplement_tables.tex`: expanded supplement tables.
- `source/crossmarket_split_appendix_tables.tex`: split-boundary appendix tables.
- `source/figures/cross_market_modern_audit/rmse_ratio_distribution.pdf`: main figure source used by the manuscript.

## Code Files

- `code/scripts/run_cross_market_data_audit.py`: data download and audit stage.
- `code/scripts/run_cross_market_model_audit.py`: supervised model audit stage.
- `code/scripts/run_chronos_zero_shot_audit.py`: Chronos zero-shot audit stage.
- `code/scripts/build_cross_market_evidence_tables.py`: evidence table generation.
- `code/scripts/build_cross_market_figures.py`: figure generation.
- `code/scripts/build_cross_market_supplement_tables.py`: supplement table generation.
- `code/scripts/build_split_appendix.py`: split-boundary appendix generation.

## Verification Targets

The package-level checker `verify_results.py` verifies the headline row counts and Chronos full-origin subset counts:

- total combined result rows: 36,630
- supervised result rows: 35,640
- sampled-origin Chronos rows: 990
- supervised persistence-gate passes: 23
- sampled-origin Chronos passes: 133
- full-origin Chronos subset rows: 60
- full-origin Chronos subset passes: 2

## Archived Release

The public repository has been archived on Zenodo. Use the concept DOI for citation and long-term access:

- https://doi.org/10.5281/zenodo.20412777
