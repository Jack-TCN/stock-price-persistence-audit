# Stock-Price Persistence Audit

This repository contains the public code, manuscript source, and evidence artifacts for:

**Do Selected Neural and Zero-Shot Time-Series Models Beat Persistence in Absolute Stock-Price Forecasting? A Cross-Market Audit**

The study asks whether selected supervised neural models and a zero-shot time-series foundation model beat a latest-close persistence baseline when forecasting absolute stock-price levels across multiple equity-market universes.

## Main Result

Across 165 dataset-series memberships, corresponding to 133 unique ticker symbols after cross-index overlaps, the audit finds that persistence is a hard baseline for absolute price-level forecasting:

- supervised neural-model audit rows: 35,640
- supervised rows beating persistence on all three reported error metrics: 23
- sampled-origin Chronos-T5-small rows: 990
- sampled-origin Chronos rows beating persistence on all three reported error metrics: 133
- full-origin Chronos subset rows: 60
- full-origin Chronos subset rows beating persistence: 2

The manuscript therefore presents a negative, baseline-first audit rather than a claim that a new forecasting model improves stock-price prediction.

## Repository Layout

- `manuscript/`: compiled manuscript PDF.
- `supplement/`: compiled supplementary-material PDF.
- `source/`: LaTeX source, generated table files, and figure assets.
- `evidence_package/`: CSV and markdown artifacts supporting the paper's headline claims.
- `code/scripts/`: scripts used for data audit, supervised model audit, Chronos audit, evidence-table generation, figure generation, and supplement-table generation.
- `verify_results.py`: package-level consistency checker for the headline counts.
- `run_all_reproduce.sh`: conservative reproduction command sketch.
- `reproducibility_manifest.md`: file-by-file map from claims to artifacts.

## Quick Verification

From the repository root:

```bash
python verify_results.py
```

Expected summary:

- combined result rows: 36,630
- dataset-series memberships: 165
- unique tickers: 133
- supervised rows / passes: 35,640 / 23
- sampled-origin Chronos rows / passes: 990 / 133
- full-origin Chronos subset rows / passes: 60 / 2

## Reproduction Notes

The supplied CSV artifacts are sufficient to verify the paper's reported counts and summary claims. Re-running the full experimental pipeline requires market-data access and GPU/CPU resources suitable for repeated rolling-origin model evaluation.

Python package notes are listed in `requirements_submission.txt`. The shell script `run_all_reproduce.sh` documents the intended high-level reproduction order.

## Data and Code Availability

The evidence files in this repository contain derived audit outputs and summary tables. The underlying market data are obtained from public market-data interfaces by ticker and date range; users should respect the terms of the data provider they use when reproducing the audit.

A Zenodo DOI should be minted from a public GitHub release and then cited in the manuscript's Data and Code Availability section.

