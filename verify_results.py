from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
EVIDENCE = ROOT / "evidence_package"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"FAIL: {message}")


def main() -> None:
    combined = pd.read_csv(EVIDENCE / "combined_model_results.csv")
    full_origin = pd.read_csv(EVIDENCE / "chronos_full_origin_subset_60_rows.csv")
    coverage = pd.read_csv(EVIDENCE / "dataset_coverage.csv")

    require(len(combined) == 36630, f"combined rows expected 36630, got {len(combined)}")
    require(combined["ticker"].nunique() == 133, f"unique tickers expected 133, got {combined['ticker'].nunique()}")
    require(int(coverage["assets"].sum()) == 165, f"dataset-series memberships expected 165, got {int(coverage['assets'].sum())}")

    supervised = combined[combined["protocol"].str.contains("supervised", case=False, na=False)]
    chronos = combined[combined["model"].eq("Chronos-T5-small zero-shot")]
    require(len(supervised) == 35640, f"supervised rows expected 35640, got {len(supervised)}")
    require(len(chronos) == 990, f"sampled-origin Chronos rows expected 990, got {len(chronos)}")
    require(int(supervised["passes_latest_rmse_gate"].sum()) == 23, "supervised pass count expected 23")
    require(int(chronos["passes_latest_rmse_gate"].sum()) == 133, "sampled-origin Chronos pass count expected 133")

    require(len(full_origin) == 60, f"full-origin Chronos rows expected 60, got {len(full_origin)}")
    require(int(full_origin["passes_latest_rmse_gate"].sum()) == 2, "full-origin Chronos pass count expected 2")

    median_ratio = full_origin["RMSE_ratio_vs_latest"].median()
    require(1.027 <= median_ratio <= 1.029, f"full-origin median RMSE ratio expected about 1.028, got {median_ratio:.6f}")

    print("PASS: package evidence is internally consistent.")
    print(f"Combined rows: {len(combined)}")
    print(f"Dataset-series memberships: {int(coverage['assets'].sum())}")
    print(f"Unique tickers: {combined['ticker'].nunique()}")
    print(f"Supervised rows/passes: {len(supervised)}/{int(supervised['passes_latest_rmse_gate'].sum())}")
    print(f"Sampled Chronos rows/passes: {len(chronos)}/{int(chronos['passes_latest_rmse_gate'].sum())}")
    print(f"Full-origin Chronos rows/passes: {len(full_origin)}/{int(full_origin['passes_latest_rmse_gate'].sum())}")
    print(f"Full-origin median RMSE ratio: {median_ratio:.4f}")


if __name__ == "__main__":
    main()
